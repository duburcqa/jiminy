""" TODO: Write documentation.
"""
import time
from typing import Any, List, Optional, Tuple, Generic, Union, SupportsFloat

import gymnasium as gym
from gymnasium.core import RenderFrame

from jiminy_py.viewer import sleep

from gym_jiminy.common.bases import ObsT, ActT, InfoType, JiminyEnvInterface
from gym_jiminy.common.envs import BaseJiminyEnv


class FrameRateLimiter(gym.Wrapper,  # [ObsT, ActT, ObsT, ActT],
                       Generic[ObsT, ActT]):
    """Limit the rendering framerate of an environment to a given threshold,
    which is typically useful if human rendering is enabled.

    .. note::
        This wrapper only affects `render`, not `replay`.

    .. warning::
        This wrapper is only compatible with `BasePipelineWrapper` and
        `BaseJiminyEnv` as it requires having a `step_dt` attribute.
    """
    def __init__(self,  # pylint: disable=unused-argument
                 env: JiminyEnvInterface[ObsT, ActT],
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
        assert isinstance(env.unwrapped, BaseJiminyEnv)
        self._step_dt_rel = env.unwrapped.step_dt / speed_ratio

        # Buffer to keep track of the last time `step` method was called
        self._time_prev = 0.0

        # Initialize base wrapper
        super().__init__(env)

    def step(self,
             action: ActT) -> Tuple[ObsT, SupportsFloat, bool, bool, InfoType]:
        """This method does nothing more than  recording the current time,
        then calling `self.env.step`. See `BaseJiminyEnv.step` for details.

        :param action: Action to perform. `None` to not update the action.

        :returns: Next observation, reward, status of the episode (done or
                  not), plus some extra information
        """
        self._time_prev = time.time()
        return self.env.step(action)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """This method does nothing more than calling `self.env.render`, then
        sleeping to cap the framerate at the requested speed ratio. See
        `BaseJiminyEnv.render` for details.

        :returns: RGB array if 'render_mode' is 'rgb_array', None otherwise.
        """
        out: Optional[
            Union[RenderFrame, List[RenderFrame]]] = self.env.render()
        if not self.human_only or out is None:
            sleep(self._step_dt_rel - (time.time() - self._time_prev))
        return out
