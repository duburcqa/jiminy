""" TODO: Write documentation.
"""
from copy import deepcopy
from functools import reduce
from collections import deque
from typing import Tuple, Dict, Sequence, List, Any, Iterator

import numpy as np

import gym

from ..utils import SpaceDictNested, is_breakpoint, zeros
from ..bases import BasePipelineWrapper


class FilterFrameStack(gym.Wrapper):
    """Observation wrapper that stacks filtered observations in a rolling
    manner.

    It combines and extends OpenAI Gym wrappers `FrameStack` and
    `FilterObservation` to support nested filter keys.

    .. note::
        The observation space must be `gym.spaces.Dict`, while, ultimately,
        stacked leaf fields must be `gym.spaces.Box`.
    """
    def __init__(self,  # pylint: disable=unused-argument
                 env: gym.Env,
                 nested_filter_keys: Sequence[Sequence[str]],
                 num_stack: int,
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
        # Define helper that will be used to determine the leaf fields to stack
        def _get_branches(root: Any) -> Iterator[List[str]]:
            if isinstance(root, gym.spaces.Dict):
                for field, node in root.spaces.items():
                    if isinstance(node, gym.spaces.Dict):
                        for path in _get_branches(node):
                            yield [field] + path
                    else:
                        yield [field]

        # Backup user arguments
        self.nested_filter_keys: List[List[str]] = list(
            map(list, nested_filter_keys))  # type: ignore[arg-type]
        self.num_stack = num_stack

        # Initialize base wrapper
        super().__init__(env)  # Do not forward extra arguments, if any

        # Get the leaf fields to stack
        self.leaf_fields_list: List[List[str]] = []
        for fields in self.nested_filter_keys:
            root_field = reduce(
                lambda d, key: d[key], fields, self.env.observation_space)
            if isinstance(root_field, gym.spaces.Dict):
                leaf_paths = _get_branches(root_field)
                self.leaf_fields_list += [fields + path for path in leaf_paths]
            else:
                self.leaf_fields_list.append(fields)

        # Compute stacked observation space
        self.observation_space = deepcopy(self.env.observation_space)
        for fields in self.leaf_fields_list:
            root_space = reduce(
                lambda d, key: d[key], fields[:-1], self.observation_space)
            space = root_space[fields[-1]]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(
                    "Stacked leaf fields must be associated with "
                    "`gym.spaces.Box` space")
            low = np.repeat(
                space.low[np.newaxis, ...], self.num_stack, axis=0)
            high = np.repeat(
                space.high[np.newaxis, ...], self.num_stack, axis=0)
            root_space.spaces[fields[-1]] = gym.spaces.Box(
                low=low, high=high, dtype=space.dtype)

        # Allocate internal frames buffers
        self._frames: List[deque] = [
            deque(maxlen=self.num_stack) for _ in self.leaf_fields_list]

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        # Initialize the frames by duplicating the original one
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_space = reduce(
                lambda d, key: d[key], fields, self.env.observation_space)
            for _ in range(self.num_stack):
                frames.append(zeros(leaf_space))

    def observation(self, observation: SpaceDictNested) -> SpaceDictNested:
        """ TODO: Write documentation.
        """
        # Replace nested fields of original observation by the stacked ones
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            root_obs = reduce(lambda d, key: d[key], fields[:-1],
                              observation)
            root_obs[fields[-1]] = np.stack(frames)

        # Return the stacked observation
        return observation

    def compute_observation(self, measure: SpaceDictNested) -> SpaceDictNested:
        """ TODO: Write documentation.
        """
        # Backup the nested observation fields to stack
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(lambda d, key: d[key], fields, measure)
            frames.append(leaf_obs.copy())  # Copy to make sure not altered

        # Return the stacked observation
        return self.observation(measure)

    def step(self,
             action: SpaceDictNested
             ) -> Tuple[SpaceDictNested, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        return self.compute_observation(observation), reward, done, info

    def reset(self, **kwargs: Any) -> SpaceDictNested:
        observation = self.env.reset(**kwargs)
        self._setup()
        return self.compute_observation(observation)


class StackedJiminyEnv(BasePipelineWrapper):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 env: gym.Env,
                 skip_frames_ratio: int = 0,
                 **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # Backup some user argument(s)
        self.skip_frames_ratio = skip_frames_ratio

        # Initialize base classes
        super().__init__(env, **kwargs)

        # Instantiate wrapper
        self.wrapper = FilterFrameStack(env, **kwargs)

        # Assertion(s) for type checker
        assert self.env.action_space is not None

        # Define the observation and action spaces
        self.action_space = self.env.action_space
        self.observation_space = self.wrapper.observation_space

        # Initialize some internal buffers
        self.__n_last_stack = 0
        self._action = zeros(self.action_space, dtype=np.float64)
        self._observation = zeros(self.observation_space)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Setup wrapper
        self.wrapper._setup()

        # Re-initialize some internal buffer(s)
        # Note that the initial observation is always stored.
        self.__n_last_stack = self.skip_frames_ratio - 1

        # Compute the observe and control update periods
        self.control_dt = self.env.control_dt
        self.observe_dt = self.env.observe_dt

        # Make sure observe update is discrete-time
        if self.observe_dt <= 0.0:
            raise ValueError(
                "`StackedJiminyEnv` does not support time-continuous update.")

    def refresh_observation(self) -> None:  # type: ignore[override]
        # Get environment observation
        self.env.refresh_observation()

        # Update observed features if necessary
        t = self.stepper_state.t
        if self.simulator.is_simulation_running and \
                is_breakpoint(t, self.observe_dt, self._dt_eps):
            self.__n_last_stack += 1
        if self.__n_last_stack == self.skip_frames_ratio:
            self.__n_last_stack = -1
            self._observation = self.env.get_observation()
            self.wrapper.compute_observation(self._observation)
