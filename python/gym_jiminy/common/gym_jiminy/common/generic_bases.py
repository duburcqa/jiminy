""" TODO: Write documentation.
"""
from typing import Optional, Dict, Any

import numpy as np
import gym

import jiminy_py.core as jiminy

from .utils import _clamp, SpaceDictRecursive


class ControlInterface:
    """Controller interface for both controllers and environments.
    """
    controller_dt: Optional[float]
    action_space: Optional[gym.Space]
    _action: Optional[SpaceDictRecursive]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the control interface.

        It only allocates some attributes.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Define some attributes
        self.controller_dt = None
        self.action_space = None
        self._action = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    def _send_command(self,
                      t: float,
                      q: np.ndarray,
                      v: np.ndarray,
                      sensors_data: jiminy.sensorsData,
                      u_command: np.ndarray) -> None:
        """This method implement the callback function required by Jiminy
        Controller to get the command. In practice, it only updates a variable
        shared between C++ and Python to the internal value stored by this
        class.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
        """
        # pylint: disable=unused-argument

        u_command[:] = self.compute_command(self._action)

    # methods to override:
    # ----------------------------

    def _refresh_action_space(self) -> None:
        """Configure the action space.
        """
        raise NotImplementedError

    def compute_command(self,
                        action: SpaceDictRecursive
                        ) -> SpaceDictRecursive:
        """Compute the command sent to the subsequent block.

        :param action: Action to perform.
        """
        raise NotImplementedError

    def compute_reward(self,
                       *args: Any,
                       info: Dict[str, Any],
                       **kwargs: Any) -> float:
        """Compute reward at current episode state.

        By default, it always returns 0.0. It must be overloaded to implement
        a proper reward function.

        .. warning::
            For compatibility with openAI gym API, only the total reward is
            returned. Yet, it is possible to update 'info' by reference if
            one wants to log extra info for monitoring.

        :param args: Extra arguments that may be useful for derived
                     environments, for example `Gym.GoalEnv`.
        :param info: Dictionary of extra information for monitoring.
        :param kwargs: Extra keyword arguments that may be useful for derived
                       environments.

        :returns: Total reward.
        """
        # pylint: disable=no-self-use,unused-argument

        return 0.0

    def compute_reward_terminal(self, info: Dict[str, Any]) -> float:
        """Compute terminal reward at current episode final state.

        .. note::
            Implementation is optional. Not computing terminal reward if not
            overloaded by the user for the sake of efficiency.

        .. warning::
            Similarly to `compute_reward`, 'info' can be updated by reference
            to log extra info for monitoring.

        :param info: Dictionary of extra information for monitoring.

        :returns: Terminal reward.
        """
        raise NotImplementedError


class ObserveInterface:
    """Observer interface for both observers and environments.
    """
    dt: Optional[float]
    observation_space: Optional[gym.Space]
    _observation: Optional[SpaceDictRecursive]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the observation interface.

        It only allocates some attributes.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Define some attributes
        self.dt = None
        self.observation_space = None
        self._observation = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    def _refresh_observation_space(self) -> None:
        """Configure the observation space.
        """
        raise NotImplementedError

    def fetch_obs(self) -> SpaceDictRecursive:
        """Fetch the observation based on the current state of the robot.
        """
        raise NotImplementedError

    def get_obs(self) -> SpaceDictRecursive:
        """Get post-processed observation.

        It clamps the observation to make sure it does not violate the lower
        and upper bounds.

        .. warning::
            In most cases, it is not necessary to overloaded this method, and
            doing so may lead to unexpected behavior if not done carefully.
        """
        return _clamp(self.observation_space, self._observation)
