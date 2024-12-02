"""This module implements a wrapper for stacking observations over time.

This wrapper
"""
import logging
from collections import deque
from typing import (
    List, Optional, Tuple, Set, Sequence, Union, Generic, TypeAlias)

import numpy as np
import gymnasium as gym
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    is_breakpoint, array_copyto, multi_array_copyto)
from jiminy_py.tree import flatten_with_path, unflatten_as

from ..bases import (DT_EPS,
                     NestedObs,
                     Act,
                     EngineObsType,
                     InterfaceJiminyEnv,
                     BasePipelineWrapper)
from ..utils import zeros, copy


StackedObs: TypeAlias = NestedObs

LOGGER = logging.getLogger(__name__)


class StackObservation(
        BasePipelineWrapper[StackedObs, Act, NestedObs, Act],
        Generic[NestedObs, Act]):
    """Partially stack observations in a rolling manner.

    This wrapper combines and extends OpenAI Gym wrappers `FrameStack` and
    `FilteredJiminyEnv` to support nested filter keys. It derives from
    `BasePipelineWrapper` rather than `gym.Wrapper`. This means that
    observations can be stacked at observation update period rather than step
    update period. It is also possible to access the stacked observation from
    any block of the environment pipeline, and it will be taken into account
    when calling `evaluate` or `play_interactive`.

    It adds one extra dimension to all the leaves of the original observation
    spaces that must be stacked. In such a case, the first dimension
    corresponds to the individual timesteps (from oldest `0` to latest `-1`).

    .. note::
        The standard container spaces `gym.spaces.Dict` and `gym.spaces.Tuple`
        are both supported. All the stacked leaf spaces must have type
        `gym.spaces.Box`.

    .. note::
        The latest frame of the stacked leaf observations corresponds to their
        latest respective values no matter what. It will frozen when shifted to
        the left.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[NestedObs, Act],
                 *,
                 num_stack: int,
                 nested_filter_keys: Optional[Sequence[
                    Union[Sequence[Union[str, int]], Union[str, int]]]] = None,
                 skip_frames_ratio: int = -1) -> None:
        """
        :param env: Environment to wrap.
        :param num_stack: Number of observation frames to keep stacked.
        :param nested_filter_keys: List of nested observation paths to stack.
                                   If a nested path is not associated with a
                                   leaf, then every leaves starting from this
                                   root path will be stacked.
                                   Optional: All leaves will be stacked by
                                   default.
        :param skip_frames_ratio: Number of observation refresh to skip between
                                  each update of the stacked leaf values. -1 to
                                  update only once per environment step.
                                  Optional: -1 by default.
        """
        # Handling of default argument(s)
        env_observation_space: gym.Space = env.observation_space
        assert isinstance(env_observation_space, gym.spaces.Dict)
        if nested_filter_keys is None:
            nested_filter_keys = (tuple(env_observation_space.keys()),)

        # Make sure all nested keys are stored in sequence
        assert not isinstance(nested_filter_keys, str)
        self.nested_filter_keys: Sequence[Sequence[Union[str, int]]] = []
        for key_nested in nested_filter_keys:
            if isinstance(key_nested, (str, int)):
                key_nested = (key_nested,)
            self.nested_filter_keys.append(tuple(key_nested))

        # Backup some user argument(s)
        self.num_stack = num_stack
        self.skip_frames_ratio = skip_frames_ratio

        # Get all paths associated with leaf values that must be stacked
        self.path_filtered_leaves: Set[Tuple[Union[str, int], ...]] = set()
        for path, _ in flatten_with_path(env.observation_space):
            if any(path[:len(e)] == e for e in self.nested_filter_keys):
                self.path_filtered_leaves.add(path)

        # Make sure that some keys are preserved
        if not self.path_filtered_leaves:
            raise ValueError(
                "At least one observation leaf must be stacked.")

        # Initialize base class
        super().__init__(env)

        # Bind observation of the environment for all non-stacked keys
        observation = copy(self.env.observation)
        observation_leaves = dict(flatten_with_path(observation))
        for path, space in flatten_with_path(self.observation_space):
            if path not in self.path_filtered_leaves:
                continue
            observation_leaves[path] = zeros(space)
        self.observation = unflatten_as(
            observation, observation_leaves.values())

        # Allocate fixed-length deque buffer for each leaf value to stack
        self._frames_leaves: List[deque] = [
            deque(maxlen=self.num_stack) for _ in self.path_filtered_leaves]

        # Define frame update triplet for fast access
        self._frames_update_triplets: Sequence[
            Tuple[deque, np.ndarray, Tuple[np.ndarray, ...]]] = []
        frames_iterator = iter(self._frames_leaves)
        env_observation_leaves = flatten_with_path(env.observation)
        for path, env_observation_leaf in env_observation_leaves:
            if path not in self.path_filtered_leaves:
                continue
            assert isinstance(env_observation_leaf, np.ndarray)
            frames = next(frames_iterator)
            observation_leaf = observation_leaves[path]
            if observation_leaf.ndim < 2:
                observation_leaf = observation_leaf.reshape((-1, 1))
            self._frames_update_triplets.append((
                frames, env_observation_leaf, tuple(observation_leaf)))

        # Initialize some proxies for fast lookup
        self._step_dt = self.env.step_dt

        # Number of stack update that has been skipped since the last one
        self._n_last_stack = -1

        # Whether the stack has been shifted to the left since last update
        self._was_stack_shifted = True

        # Bind action of the base environment
        assert self.action_space.contains(self.env.action)
        self.action = self.env.action

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        # Define leaf observation spaces
        observation_space_leaves = dict(
            flatten_with_path(self.env.observation_space))
        for path, space in observation_space_leaves.items():
            # Skip leaf spaces that must not be stacked
            if path not in self.path_filtered_leaves:
                continue

            # Make sure that stacked leaf spaces derive from `gym.spaces.Box`
            if not (isinstance(space, gym.spaces.Box) and
                    space.dtype is not None and
                    issubclass(space.dtype.type, (np.floating, np.integer))):
                raise TypeError(
                    "Stacked leaf spaces must have type `gym.spaces.Box` "
                    "whose dtype is either `np.floating` or `np.integer`.")

            # Prepend the original bounds with an additional stacking dimension
            low = np.repeat(space.low[np.newaxis], self.num_stack, axis=0)
            high = np.repeat(space.high[np.newaxis], self.num_stack, axis=0)
            observation_space_leaves[path] = gym.spaces.Box(
                low=low, high=high, dtype=space.dtype.type)

        # Define observation space by un-flattening leaf spaces
        self.observation_space = unflatten_as(
            self.env.observation_space, observation_space_leaves.values())

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Make sure observe update is discrete-time
        if self.env.observe_dt <= 0.0:
            raise ValueError(
                "This wrapper does not support time-continuous update.")

        # Check if skip frame ratio is divisor of step update ratio
        if self.skip_frames_ratio > 0:
            step_ratio = round(self._step_dt / self.env.observe_dt)
            frame_ratio = self.skip_frames_ratio + 1
            if (step_ratio // frame_ratio) * frame_ratio != step_ratio:
                LOGGER.warning(
                    "Beware `step_dt // observe_dt` is not a multiple of "
                    "`skip_frames_ratio + 1`.")

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

        # Reset frames to zero
        frames_iterator = iter(self._frames_leaves)
        for path, space in flatten_with_path(self.env.observation_space):
            if path not in self.path_filtered_leaves:
                continue
            frames = next(frames_iterator)
            for _ in range(self.num_stack):
                frames.append(zeros(space))

        # Re-initialize stack state.
        # Note that the initial observation is always stored.
        self._n_last_stack = self.skip_frames_ratio - 1
        self._was_stack_shifted = True

    def refresh_observation(self, measurement: EngineObsType) -> None:
        # Skip update if nothing to do
        if not is_breakpoint(self.stepper_state, self.observe_dt, DT_EPS):
            return

        # Refresh environment observation
        self.env.refresh_observation(measurement)

        # Update stacked observation leaf values if necessary
        update_stack, shift_stack = False, False
        if self.is_simulation_running:
            self._n_last_stack += 1
            if self.skip_frames_ratio < 0:
                if is_breakpoint(self.stepper_state, self._step_dt, DT_EPS):
                    update_stack = True
            elif self._n_last_stack == self.skip_frames_ratio:
                update_stack = True
            shift_stack = self._n_last_stack == 0

        # Backup the nested observation fields to stack.
        # Leaf values are copied to ensure they do not get altered later on.
        for frames, env_obs_leaf, obs_leaf in self._frames_update_triplets:
            if update_stack:
                frames[-1] = env_obs_leaf.copy()
            if shift_stack:
                frames.append(env_obs_leaf)
                multi_array_copyto(obs_leaf, tuple(frames))
            else:
                array_copyto(obs_leaf[-1], env_obs_leaf)

        # Re-initialize number of skipped stack update
        if update_stack:
            self._n_last_stack = -1
            self._was_stack_shifted = True

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to updated in-place.
        """
        self.env.compute_command(action, command)
