""" TODO: Write documentation.
"""
from typing import TypeVar

import numpy as np

import gymnasium as gym


ObsT = TypeVar("ObsT")


class NormalizeAction(gym.ActionWrapper):
    """Normalize action space without clipping, contrary to usual
    implementations.
    """
    def __init__(self, env: gym.Env[ObsT, np.ndarray]) -> None:
        # Make sure that the action space derives from 'gym.spaces.Box'
        assert isinstance(env.action_space, gym.spaces.Box)
        base_action_space = env.action_space

        # Make sure that it is bounded
        low, high = base_action_space.low, base_action_space.high
        assert all(np.all(np.isfinite(val)) for val in (low, high)), \
               "Action space must have finite bounds."

        # Assert that it has floating-point dtype
        assert base_action_space.dtype is not None
        dtype = base_action_space.dtype.type
        assert issubclass(dtype, np.floating)

        # Initialize base class
        super().__init__(env)

        # Define the action space
        self._action_orig_mean = (high + low) / 2.0
        self._action_orig_dev = (high - low) / 2.0
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=base_action_space.shape, dtype=dtype)

        # Copy 'mirror_mat' attribute if specified
        if hasattr(base_action_space, "mirror_mat"):
            self.action_space.mirror_mat = (  # type: ignore[attr-defined]
                base_action_space.mirror_mat)

    def action(self, action: np.ndarray) -> np.ndarray:
        return self._action_orig_mean + action * self._action_orig_dev
