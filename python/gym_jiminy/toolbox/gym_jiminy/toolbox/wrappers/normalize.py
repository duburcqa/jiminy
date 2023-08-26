""" TODO: Write documentation.
"""
from typing import TypeVar, cast

import numpy as np
import numba as nb

import gymnasium as gym


ObsT = TypeVar("ObsT")


@nb.jit(nopython=True, inline='always')
def _normalize(value: np.ndarray,
               mean: np.ndarray,
               scale: np.ndarray) -> np.ndarray:
    """Element-wise normalization of array.

    :param value: Un-normalized data.
    :param mean: mean.
    :param scale: scale.
    """
    return (value - mean) / scale


@nb.jit(nopython=True, inline='always')
def _denormalize(value: np.ndarray,
                 mean: np.ndarray,
                 scale: np.ndarray) -> np.ndarray:
    """Reverse element-wise normalization of array.

    :param value: Normalized data.
    :param mean: mean.
    :param scale: scale.
    """
    return mean + value * scale


class NormalizeAction(gym.ActionWrapper):
    """Normalize action space without clipping.
    """
    def __init__(self, env: gym.Env[ObsT, np.ndarray]) -> None:
        # Make sure that the action space derives from 'gym.spaces.Box'
        assert isinstance(env.action_space, gym.spaces.Box)

        # Make sure that it is bounded
        low, high = env.action_space.low, env.action_space.high
        assert all(np.all(np.isfinite(val)) for val in (low, high)), \
               "Action space must have finite bounds."

        # Assert that it has floating-point dtype
        assert env.action_space.dtype is not None
        dtype = env.action_space.dtype.type
        assert issubclass(dtype, np.floating)

        # Initialize base class
        super().__init__(env)

        # Define the action space
        self._action_mean = (high + low) / 2.0
        self._action_scale = (high - low) / 2.0
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=dtype)

        # Copy 'mirror_mat' attribute if specified
        if hasattr(env.action_space, "mirror_mat"):
            self.action_space.mirror_mat = (  # type: ignore[attr-defined]
                env.action_space.mirror_mat)

    def action(self, action: np.ndarray) -> np.ndarray:
        return _denormalize(action, self._action_mean, self._action_scale)


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observation space without clipping.
    """
    def __init__(self, env: gym.Env[ObsT, np.ndarray]) -> None:
        # Make sure that the action space derives from 'gym.spaces.Box'
        assert isinstance(env.observation_space, gym.spaces.Box)
        env_observation_space = cast(gym.spaces.Box, env.observation_space)

        # Make sure that it is bounded
        low, high = env_observation_space.low, env_observation_space.high
        assert all(np.all(np.isfinite(val)) for val in (low, high)), \
               "Observation space must have finite bounds."

        # Assert that it has floating-point dtype
        assert env_observation_space.dtype is not None
        dtype = env_observation_space.dtype.type
        assert issubclass(dtype, np.floating)

        # Initialize base class
        super().__init__(env)

        # Define the observation space
        self._observation_mean = (high + low) / 2.0
        self._observation_scale = (high - low) / 2.0
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env_observation_space.shape, dtype=dtype)

        # Copy 'mirror_mat' attribute if specified
        if hasattr(env_observation_space, "mirror_mat"):
            self.observation_space.mirror_mat = (  # type: ignore[attr-defined]
                env_observation_space.mirror_mat)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return _normalize(observation,
                          self._observation_mean,
                          self._observation_scale)
