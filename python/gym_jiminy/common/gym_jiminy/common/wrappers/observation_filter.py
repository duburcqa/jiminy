""" TODO: Write documentation.
"""
from collections import OrderedDict
from typing import Sequence, Union, Generic
from typing_extensions import TypeAlias

import numpy as np
import gymnasium as gym

from ..bases import (ObsT,
                     ActT,
                     EngineObsType,
                     BasePipelineWrapper,
                     JiminyEnvInterface)


FilteredObsType: TypeAlias = ObsT


class FilteredJiminyEnv(BasePipelineWrapper[FilteredObsType, ActT, ObsT, ActT],
                        Generic[ObsT, ActT]):
    """TODO: Write documentation.
    """
    def __init__(self,
                 env: JiminyEnvInterface[ObsT, ActT],
                 nested_filter_keys: Sequence[Union[Sequence[str], str]]
                 ) -> None:
        # Make sure that the observation space derives from 'gym.spaces.Dict'
        assert isinstance(env.observation_space, gym.spaces.Dict)

        # Make sure all nested keys are stored in sequence
        self.nested_filter_keys = []
        for key_nested in nested_filter_keys:
            if isinstance(key_nested, str):
                key_nested = (key_nested,)
            self.nested_filter_keys.append(key_nested)

        # Remove redundant nested keys if any
        for i, key_nested in list(enumerate(self.nested_filter_keys))[::-1]:
            for j, path in list(enumerate(self.nested_filter_keys[:i]))[::-1]:
                if path[:len(key_nested)] == key_nested:
                    self.nested_filter_keys.pop(j)
                elif path == key_nested[:len(path)]:
                    self.nested_filter_keys.pop(i)
                    break

        # Bind action of the base environment
        assert self.action_space.contains(env.action)
        self.action = env.action  # type: ignore[assignment]

        # Initialize base class
        super().__init__(env)

        # Bind observation of the environment for all filtered keys
        self.observation = OrderedDict()
        for key_nested in self.nested_filter_keys:
            observation_filtered = self.observation
            observation = self.env.observation
            for key in key_nested[:-1]:
                assert isinstance(observation, dict)
                observation = observation[key]
                observation_filtered = observation_filtered.setdefault(
                    key, OrderedDict())
            assert isinstance(observation, dict)
            observation_filtered[key_nested[-1]] = observation[key_nested[-1]]

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to the base implementation, it configures the controller
        and registers its target to the telemetry.
        """
        # Call base implementation
        super()._setup()

        # Compute the observe and control update periods
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        self.observation_space = gym.spaces.Dict()
        for key_nested in self.nested_filter_keys:
            space_filtered = self.observation_space
            space = self.env.observation_space
            for key in key_nested[:-1]:
                assert isinstance(space, gym.spaces.Dict)
                space = space[key]
                space_filtered = space_filtered.spaces.setdefault(
                    key, gym.spaces.Dict())  # type: ignore[assignment]
            assert isinstance(space, gym.spaces.Dict)
            space_filtered[key_nested[-1]] = space[key_nested[-1]]

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It simply forwards the command computed by the wrapped environment
        without any processing.
        """
        self.env.refresh_observation(measurement)

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)
