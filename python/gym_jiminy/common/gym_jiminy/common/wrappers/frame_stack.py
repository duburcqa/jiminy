""" TODO: Write documentation.
"""
from copy import deepcopy
from functools import reduce
from collections import deque
from typing import (
    List, Any, Dict, Optional, Tuple, Sequence, Iterator, Union, Generic)

import numpy as np

import gymnasium as gym

from ..utils import is_breakpoint, zeros
from ..bases import (DT_EPS,
                     ObsType,
                     ActType,
                     BaseObsType,
                     BaseActType,
                     InfoType,
                     JiminyEnvInterface,
                     BasePipelineWrapper)


StackedObsType = ObsType


class PartialFrameStack(
        gym.Wrapper[StackedObsType, ActType, ObsType, ActType],
        Generic[ObsType, ActType]):
    """Observation wrapper that partially stacks observations in a rolling
    manner.

    It combines and extends OpenAI Gym wrappers `FrameStack` and
    `FilterObservation` to support nested filter keys.

    .. note::
        The observation space must be `gym.spaces.Dict`, while, ultimately,
        stacked leaf fields must be `gym.spaces.Box`.
    """
    def __init__(self,
                 env: gym.Env[ObsType, ActType],
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
            nested_filter_keys = list(env.observation_space.spaces.keys())

        # Backup user argument(s)
        self.nested_filter_keys: List[List[str]] = list(
            list(fields) for fields in nested_filter_keys)
        self.num_stack = num_stack

        # Initialize base wrapper
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
            root_field = reduce(
                lambda d, key: d[key],
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
            root_space = reduce(
                lambda d, key: d[key], fields[:-1], self.observation_space)
            space = root_space[fields[-1]]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(
                    "Stacked leaf fields must be associated with "
                    "`gym.spaces.Box` space")
            low = np.repeat(space.low[np.newaxis], self.num_stack, axis=0)
            high = np.repeat(space.high[np.newaxis], self.num_stack, axis=0)
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
            assert isinstance(self.env.observation_space, gym.spaces.Dict)
            leaf_space = reduce(
                lambda d, key: d[key], fields, self.env.observation_space)
            for _ in range(self.num_stack):
                frames.append(zeros(leaf_space))

    def observation(self, observation: ObsType) -> ObsType:
        """ TODO: Write documentation.
        """
        # Replace nested fields of original observation by the stacked ones
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            root_obs = reduce(lambda d, key: d[key], fields[:-1], observation)

            # Assert(s) for type checker
            assert isinstance(root_obs, dict)

            root_obs[fields[-1]] = np.stack(frames)

        # Return the stacked observation
        return observation

    def compute_observation(self, measurement: ObsType) -> ObsType:
        """ TODO: Write documentation.
        """
        # Backup the nested observation fields to stack
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(lambda d, key: d[key], fields, measurement)

            # Assert(s) for type checker
            assert isinstance(leaf_obs, np.ndarray)

            # Copy to make sure not altered
            frames.append(leaf_obs.copy())

        # Return the stacked observation
        return self.observation(measurement)

    def step(self,
             action: Optional[ActType] = None
             ) -> Tuple[StackedObsType, float, bool, bool, InfoType]:
        observation, reward, done, truncated, info = self.env.step(action)
        return (
            self.compute_observation(observation), reward, done, truncated,
            info)

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None,
              ) -> Tuple[StackedObsType, InfoType]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._setup()
        return self.compute_observation(observation), info


class StackedJiminyEnv(
        BasePipelineWrapper[StackedObsType, ActType, ObsType, ActType],
        Generic[ObsType, ActType]):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 env: JiminyEnvInterface[
                     ObsType, ActType, BaseObsType, BaseActType],
                 skip_frames_ratio: int = 0,
                 **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # Backup some user argument(s)
        self.skip_frames_ratio = skip_frames_ratio

        # Initialize some internal buffers
        self.__n_last_stack = 0

        # Instantiate wrapper
        self.wrapper = PartialFrameStack(env, **kwargs)

        # Initialize base classes
        super().__init__(env, **kwargs)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        self.observation_space = self.wrapper.observation_space

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

    def refresh_observation(self, measurement: ObsType) -> None:
        # Get environment observation
        self.env.refresh_observation(measurement)

        # Update observed features if necessary
        t = self.stepper_state.t
        if self.simulator.is_simulation_running and \
                is_breakpoint(t, self.observe_dt, DT_EPS):
            self.__n_last_stack += 1
        if self.__n_last_stack == self.skip_frames_ratio:
            self.__n_last_stack = -1
            self._observation = self.env.get_observation()
            self.wrapper.compute_observation(self._observation)
