""" TODO: Write documentation.
"""
import time
from typing import Any, List, Optional, Tuple, Generic, Union

import numpy as np

import gymnasium as gym
from gymnasium.core import RenderFrame

from jiminy_py.viewer import sleep

from ..bases import (
    ObsType, ActType, BaseObsType, BaseActType, InfoType, EnvOrWrapperType)




class FrameRateLimiter(gym.Wrapper[ObsType, ActType, ObsType, ActType],
                       Generic[ObsType, ActType]):
    """Limit the rendering framerate of an environment to a given threshold,
    which is typically useful if human rendering is enabled.

    .. note::
        This wrapper only affects `render`, not `replay`.

    .. warning::
        This wrapper is only compatible with `BasePipelineWrapper` and
        `BaseJiminyEnv` as it requires having a `step_dt` attribute.
    """
    def __init__(self,  # pylint: disable=unused-argument
                 env: EnvOrWrapperType[
                     ObsType, ActType, BaseObsType, BaseActType],
                 speed_ratio: float = 1.0,
                 human_only: bool = True,
                 **kwargs: Any):
        """
        :param env: Environment to wrap.
        :param speed_ratio: Real-time factor.
                            Optional: No time dilation by default (1.0).
        :param human_only: Only limit the framerate for 'human' render mode.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Backup user argument(s)
        assert speed_ratio > 0
        self.speed_ratio = speed_ratio
        self.human_only = human_only

        # Extract proxies for convenience
        self._step_dt_rel = env.unwrapped.step_dt / speed_ratio

        # Buffer to keep track of the last time `step` method was called
        self._time_prev = 0.0

        # Initialize base wrapper
        super().__init__(env)

    def step(self,
             action: Optional[ActType] = None
             ) -> Tuple[ObsType, float, bool, bool, InfoType]:
        """This method does nothing more than  recording the current time,
        then calling `self.env.step`. See `BaseJiminyEnv.step` for details.

        :param action: Action to perform. `None` to not update the action.

        :returns: Next observation, reward, status of the episode (done or
                  not), plus some extra information
        """
        self._time_prev = time.time()
        return self.env.step(action)

    def render(self,
               **kwargs: Any
               ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """This method does nothing more than calling `self.env.render`, then
        sleeping to cap the framerate at the requested speed ratio. See
        `BaseJiminyEnv.render` for details.

        :param kwargs: Extra keyword arguments to forward to
                       `jiminy_py.simulator.Simulator.render` method.

        :returns: RGB array if 'mode' is 'rgb_array', None otherwise.
        """
        out = self.env.render(**kwargs)
        if not self.human_only or out is None:
            sleep(self._step_dt_rel - (time.time() - self._time_prev))
        return out
