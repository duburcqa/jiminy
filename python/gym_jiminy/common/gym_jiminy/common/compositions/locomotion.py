"""Rewards mainly relevant for locomotion tasks on floating-base robots.
"""
from typing import Sequence

import numpy as np

from ..bases import InterfaceJiminyEnv, BaseQuantityReward
from ..quantities import AverageOdometryVelocity

from .generic import radial_basis_function


class OdometryVelocityReward(BaseQuantityReward):
    """Reward the agent for tracking a non-stationary target odometry velocity.

    The error transform in a normalized reward to maximize by applying RBF
    kernel on the error. The reward will be 0.0 if the error cancels out
    completely and less than 0.01 above the user-specified cutoff threshold.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 target: Sequence[float],
                 cutoff: float) -> None:
        """
        :param target: Initial target average odometry velocity (vX, vY, vYaw).
                       The target can be updated in necessary by calling
                       `set_target`.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self._target = np.asarray(target)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_odometry_velocity",
            (AverageOdometryVelocity, {}),
            self._transform,
            is_normalized=True,
            is_terminal=False)

    @property
    def target(self) -> np.ndarray:
        """Get current target odometry velocity.
        """
        return self._target

    @target.setter
    def target(self, target: Sequence[float]) -> None:
        """Set current target odometry velocity.
        """
        self._target = np.asarray(target)

    def _transform(self, value: np.ndarray) -> float:
        """Apply Radial Base Function transform to the residual error between
        the current and target average odometry velocity.

        .. note::
            The user must call `set_target` method before `compute_reward` to
            update the target odometry velocity if non-stationary.

        :param value: Current average odometry velocity.
        """
        return radial_basis_function(value - self.target, self.cutoff)
