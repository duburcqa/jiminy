""" TODO: Write documentation.
"""
from collections import OrderedDict
from typing import Sequence, Union, Generic
from typing_extensions import TypeAlias

import gymnasium as gym

from ..bases import (ObsT,
                     ActT,
                     JiminyEnvInterface,
                     BaseTransformObservation)


FilteredObsType: TypeAlias = ObsT


class FilterObservation(BaseTransformObservation[FilteredObsType, ObsT, ActT],
                        Generic[ObsT, ActT]):
    """Filter nested observation space.

    This wrapper does not nothing but providing an observation only exposing
    a subset of all the branches and leaves of the original observation space.
    For flattening the observation space after filtering, you should wrap the
    environment with `FlattenObservation` as yet another layer.
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

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.

        It gathers a subset of all the branches and leaves of the original
        observation space without any further processing.
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

    def transform_observation(self) -> None:
        """No-op transform since the transform observation is sharing memory
        with the wrapped one since it is just a partial view.
        """
