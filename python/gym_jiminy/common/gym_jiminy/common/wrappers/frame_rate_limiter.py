""" TODO: Write documentation.
"""
import time
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np

import gym

from jiminy_py.viewer import sleep

from ..utils import DataNested
from ..bases import BasePipelineWrapper
from ..envs import BaseJiminyEnv


class FrameRateLimiter(gym.Wrapper):
    """Limit the rendering framerate of an environment to a given threshold,
    which is typically useful if human rendering is enabled.

    .. note::
        This wrapper only affects `render`, not `replay`.

    .. warning::
        This wrapper is only compatible with `BasePipelineWrapper` and
        `BaseJiminyEnv` as it requires having a `step_dt` attribute.
    """
    def __init__(self,  # pylint: disable=unused-argument
                 env: Union[BasePipelineWrapper, BaseJiminyEnv],
                 speed_ratio: float = 1.0,
                 **kwargs: Any):
        """
        :param env: Environment to wrap.
        :param speed_ratio: Real-time factor.
                            Optional: No time dilation by default (1.0).
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Backup user argument(s)
        assert speed_ratio > 0
        self.speed_ratio = speed_ratio

        # Extract proxies for convenience
        self._step_dt_rel = env.unwrapped.step_dt / speed_ratio

        # Buffer to keep track of the last time `step` method was called
        self._time_prev = 0.0

        # Initialize base wrapper
        super().__init__(env)

    def step(self,
             action: Optional[DataNested] = None
             ) -> Tuple[DataNested, float, bool, Dict[str, Any]]:
        """This method does nothing more than  recording the current time,
        then calling `self.env.step`. See `BaseJiminyEnv.step` for details.

        :param action: Action to perform. `None` to not update the action.

        :returns: Next observation, reward, status of the episode (done or
                  not), plus some extra information
        """
        self._time_prev = time.time()
        return self.env.step(action)

    def render(self,
               mode: Optional[str] = None,
               **kwargs: Any) -> Optional[np.ndarray]:
        """This method does nothing more than calling `self.env.render`, then
        sleeping to cap the framerate at the requested speed ratio. See
        `BaseJiminyEnv.render` for details.

        :param mode: Rendering mode. It can be either 'human' to display the
                     current simulation state, or 'rgb_array' to return a
                     snapshot as an RGB array without showing it on the screen.
                     Optional: 'human' by default if available based on current
                     backend, 'rgb_array' otherwise.
        :param kwargs: Extra keyword arguments to forward to
                       `jiminy_py.simulator.Simulator.render` method.

        :returns: RGB array if 'mode' is 'rgb_array', None otherwise.
        """
        out = self.env.render(mode, **kwargs)
        sleep(self._step_dt_rel - (time.time() - self._time_prev))
        return out
