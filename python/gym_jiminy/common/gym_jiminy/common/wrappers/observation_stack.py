""" TODO: Write documentation.
"""
from copy import deepcopy
from operator import getitem
from functools import reduce
from collections import deque
from typing import (
    List, Any, Dict, Optional, Tuple, Sequence, Iterator, Union, Generic,
    SupportsFloat)
from typing_extensions import TypeAlias

import numpy as np

import gymnasium as gym

from ..utils import is_breakpoint, zeros, copy, copyto
from ..bases import (DT_EPS,
                     ObsT,
                     ActT,
                     InfoType,
                     EngineObsType,
                     JiminyEnvInterface,
                     BasePipelineWrapper)


StackedObsType: TypeAlias = ObsT


class PartialObservationStack(
        gym.Wrapper,  # [StackedObsType, ActT, ObsT, ActT],
        Generic[ObsT, ActT]):
    """Observation wrapper that partially stacks observations in a rolling
    manner.

    This wrapper combines and extends OpenAI Gym wrappers `FrameStack` and
    `FilteredJiminyEnv` to support nested filter keys.

    It adds one extra dimension to all the leaves of the original observation
    spaces that must be stacked. If so, the first dimension corresponds to the
    individual timesteps (from oldest [0] to latest [-1]).

    .. note::
        The observation space must be `gym.spaces.Dict`, while, ultimately,
        stacked leaf fields must be `gym.spaces.Box`.
    """
    def __init__(self,
                 env: gym.Env[ObsT, ActT],
                 num_stack: int,
                 nested_filter_keys: Optional[
                     Sequence[Union[Sequence[str], str]]] = None,
                 **kwargs: Any):
        """
        :param env: Environment to wrap.
        :param nested_filter_keys: List of nested observation fields to stack.
                                   Those fields does not have to be leaves. If
                                   not, then every leaves fields from this root
                                   will be stacked.
        :param num_stack: Number of observation frames to partially stack.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # pylint: disable=unused-argument

        # Sanitize user arguments if necessary
        assert isinstance(env.observation_space, gym.spaces.Dict)
        if nested_filter_keys is None:
            nested_filter_keys = list(
                env.observation_space.keys())  # type: ignore[attr-defined]

        # Backup user argument(s)
        self.nested_filter_keys: List[List[str]] = list(
            list(fields) for fields in nested_filter_keys)
        self.num_stack = num_stack

        # Initialize base wrapper.
        # Note that `gym.Wrapper` automatically binds the action/observation to
        # the one of the environment if not overridden explicitly.
        super().__init__(env)  # Do not forward extra arguments, if any

        # Get the leaf fields to stack
        def _get_branches(root: Any) -> Iterator[List[str]]:
            if isinstance(root, gym.spaces.Dict):
                for field, node in root.spaces.items():
                    if isinstance(node, gym.spaces.Dict):
                        for path in _get_branches(node):
                            yield [field] + path
                    else:
                        yield [field]

        self.leaf_fields_list: List[List[str]] = []
        for fields in self.nested_filter_keys:
            root_field = reduce(getitem,  # type: ignore[arg-type]
                                fields, self.env.observation_space)
            if isinstance(root_field, gym.spaces.Dict):
                leaf_paths = _get_branches(root_field)
                self.leaf_fields_list += [fields + path for path in leaf_paths]
            else:
                self.leaf_fields_list.append(fields)

        # Compute stacked observation space
        self.observation_space = deepcopy(self.env.observation_space)
        for fields in self.leaf_fields_list:
            assert isinstance(self.observation_space, gym.spaces.Dict)
            root_space = reduce(getitem,  # type: ignore[arg-type]
                                fields[:-1],  self.observation_space)
            space = root_space[fields[-1]]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(
                    "Stacked leaf fields must be associated with "
                    "`gym.spaces.Box` space")
            low = np.repeat(space.low[np.newaxis], self.num_stack, axis=0)
            high = np.repeat(space.high[np.newaxis], self.num_stack, axis=0)
            assert space.dtype is not None
            assert issubclass(space.dtype.type, (np.floating, np.integer))
            root_space[fields[-1]] = gym.spaces.Box(
                low=low, high=high, dtype=space.dtype.type)

        # Bind observation of the environment for all keys but the stacked ones
        if isinstance(self.env, JiminyEnvInterface):
            self.observation = copy(self.env.observation)
            for fields in self.leaf_fields_list:
                assert isinstance(self.observation_space, gym.spaces.Dict)
                root_obs = reduce(getitem, fields[:-1], self.observation)
                space = reduce(getitem,  # type: ignore[arg-type]
                               fields, self.observation_space)
                root_obs[fields[-1]] = zeros(space)
        else:
            # Fallback to classical memory allocation
            self.observation = zeros(self.observation_space)

        # Allocate internal frames buffers
        self._frames: List[deque] = [
            deque(maxlen=self.num_stack) for _ in self.leaf_fields_list]

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        # Reset frames to zero
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            assert isinstance(self.env.observation_space, gym.spaces.Dict)
            leaf_space = reduce(getitem,  # type: ignore[arg-type]
                                fields, self.env.observation_space)
            for _ in range(self.num_stack):
                frames.append(zeros(leaf_space))

    def refresh_observation(self, measurement: ObsT) -> None:
        """ TODO: Write documentation.
        """
        # Copy measurement if impossible to bind memory in the first place
        if not isinstance(self.env, JiminyEnvInterface):
            copyto(self.observation, measurement)

        # Backup the nested observation fields to stack.
        # Leaf values are copied to ensure they do not get altered later on.
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(getitem,  # type: ignore[arg-type]
                              fields, measurement)
            assert isinstance(leaf_obs, np.ndarray)
            frames.append(leaf_obs.copy())

        # Update nested fields of the observation by the stacked ones
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(getitem, fields, self.observation)
            assert isinstance(leaf_obs, np.ndarray)
            leaf_obs[:] = frames

    def step(self,
             action: ActT
             ) -> Tuple[StackedObsType, SupportsFloat, bool, bool, InfoType]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.refresh_observation(obs)
        return self.observation, reward, terminated, truncated, info

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None,
              ) -> Tuple[StackedObsType, InfoType]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._setup()
        self.refresh_observation(observation)
        return self.observation, info


class StackedJiminyEnv(
        BasePipelineWrapper[StackedObsType, ActT, ObsT, ActT],
        Generic[ObsT, ActT]):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 env: JiminyEnvInterface[ObsT, ActT],
                 skip_frames_ratio: int = 0,
                 **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # Backup some user argument(s)
        self.skip_frames_ratio = skip_frames_ratio

        # Initialize some internal buffers
        self.__n_last_stack = 0

        # Instantiate wrapper
        self.wrapper = PartialObservationStack(env, **kwargs)

        # Initialize base classes
        super().__init__(env, **kwargs)

        # Bind the observation of the wrapper
        self.observation = self.wrapper.observation

        # Bind the action of the environment
        assert self.action_space.contains(env.action)
        self.action = env.action

    def _initialize_action_space(self) -> None:
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        self.observation_space = self.wrapper.observation_space

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Setup wrapper
        self.wrapper._setup()

        # Make sure observe update is discrete-time
        if self.env.observe_dt <= 0.0:
            raise ValueError(
                "This wrapper does not support time-continuous update.")

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

        # Re-initialize some internal buffer(s).
        # Note that the initial observation is always stored.
        self.__n_last_stack = self.skip_frames_ratio - 1

    def refresh_observation(self, measurement: EngineObsType) -> None:
        # Get environment observation
        self.env.refresh_observation(measurement)

        # Update observed features if necessary
        if self.is_simulation_running and is_breakpoint(
                self.stepper_state.t, self.env.observe_dt, DT_EPS):
            self.__n_last_stack += 1
        if self.__n_last_stack == self.skip_frames_ratio:
            self.__n_last_stack = -1
            self.wrapper.refresh_observation(self.env.observation)

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)
