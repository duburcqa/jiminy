""" TODO: Write documentation.
"""
import numpy as np
import gym


class NormalizeAction(gym.ActionWrapper):
    """Normalize action space without clipping, contrary to usual
    implementations.
    """
    def __init__(self, env: gym.Env) -> None:
        assert (np.all(np.isfinite(env.action_space.low)) and
                np.all(np.isfinite(env.action_space.high))), \
               "Action space must have finite bounds."
        super().__init__(env)
        low = self.env.action_space.low
        high = self.env.action_space.high
        self._action_orig_mean = (high + low) / 2.0
        self._action_orig_dev = (high - low) / 2.0
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape,
            dtype=env.action_space.dtype)
        if hasattr(env.action_space, "mirror_mat"):
            self.action_space.mirror_mat = env.action_space.mirror_mat

    def action(self, action: np.ndarray) -> np.ndarray:
        return self._action_orig_mean + action * self._action_orig_dev
