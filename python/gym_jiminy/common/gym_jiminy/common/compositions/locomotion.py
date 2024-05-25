"""Rewards mainly relevant for locomotion tasks on floating-base robots.
"""
from operator import sub
from functools import partial

from ..bases import (
    InterfaceJiminyEnv, BaseQuantityReward, QuantityEvalMode, StateQuantity)
from ..quantities import (
    MaskedQuantity, UnaryOpQuantity, BinaryOpQuantity, AverageOdometryVelocity,
    ZeroMomentPoint)

from .generic import radial_basis_function


class TrackingBaseHeightReward(BaseQuantityReward):
    """Reward the agent for tracking the height of the floating base of the
    robot wrt some reference trajectory.

    A reference trajectory must be selected before evaluating this reward
    otherwise an exception will be risen. See `DatasetTrajectoryQuantity` and
    `AbstractQuantity` documentations for details.

    The error transform in a normalized reward to maximize by applying RBF
    kernel on the error. The reward will be 0.0 if the error cancels out
    completely and less than 0.01 above the user-specified cutoff threshold.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float) -> None:
        """
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_base_height",
            (BinaryOpQuantity, dict(
                quantity_left=(MaskedQuantity, dict(
                    quantity=(UnaryOpQuantity, dict(
                        quantity=(StateQuantity, dict(
                            mode=QuantityEvalMode.TRUE)),
                        op=lambda state: state.q)),
                    key=(2,))),
                quantity_right=(MaskedQuantity, dict(
                    quantity=(UnaryOpQuantity, dict(
                        quantity=(StateQuantity, dict(
                            mode=QuantityEvalMode.REFERENCE)),
                        op=lambda state: state.q)),
                    key=(2,))),
                op=sub)),
            partial(radial_basis_function, cutoff=self.cutoff, ord=2),
            is_normalized=True,
            is_terminal=False)


class TrackingOdometryVelocityReward(BaseQuantityReward):
    """Reward the agent for tracking the odometry velocity wrt some reference
    trajectory.

    A reference trajectory must be selected before evaluating this reward
    otherwise an exception will be risen. See `DatasetTrajectoryQuantity` and
    `AbstractQuantity` documentations for details.

    The error transform in a normalized reward to maximize by applying RBF
    kernel on the error. The reward will be 0.0 if the error cancels out
    completely and less than 0.01 above the user-specified cutoff threshold.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float) -> None:
        """
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_odometry_velocity",
            (BinaryOpQuantity, dict(
                quantity_left=(AverageOdometryVelocity, dict(
                    mode=QuantityEvalMode.TRUE)),
                quantity_right=(AverageOdometryVelocity, dict(
                    mode=QuantityEvalMode.REFERENCE)),
                op=sub)),
            partial(radial_basis_function, cutoff=self.cutoff, ord=2),
            is_normalized=True,
            is_terminal=False)
