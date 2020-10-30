from typing import Optional

import gym

from .utils import SpaceDictRecursive


class ControlInterface:
    """Controller interface for both controllers and environments.
    """
    action_space: Optional[gym.Space]
    controller_dt: Optional[float]

    def __init__(self) -> None:
        """Initialize the control interface.

        It only allocates some attributes.
        """
        # Define some attributes
        self.action_space = None
        self.controller_dt = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__()

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


class ObserveInterface:
    """Observer interface for both observers and environments.
    """
    observation_space: Optional[gym.Space]
    dt: Optional[float]

    def __init__(self) -> None:
        """Initialize the observation interface.

        It only allocates some attributes.
        """
        # Define some attributes
        self.observation_space = None
        self.dt = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__()

    def _refresh_observation_space(self) -> None:
        """Configure the observation space.
        """
        raise NotImplementedError

    def _fetch_obs(self) -> SpaceDictRecursive:
        """Fetch the observation based on the current state of the robot.
        """
        raise NotImplementedError

    def get_obs(self) -> SpaceDictRecursive:
        """Get the post-processed observation.
        """
        raise NotImplementedError
