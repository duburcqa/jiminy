"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
from operator import sub
from functools import partial
from typing import Optional

from ..bases import (
    InfoType, InterfaceJiminyEnv, AbstractReward, BaseQuantityReward,
    QuantityEvalMode)
from ..quantities import BinaryOpQuantity, ActuatedJointPositions

from .mixin import radial_basis_function


class SurviveReward(AbstractReward):
    """Constant positive reward equal to 1.0 systematically, unless the current
    state of the environment is the terminal state. In which case, the value
    0.0 is returned instead.
    """

    def __init__(self, env: InterfaceJiminyEnv) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        """
        super().__init__(env, "reward_survive")

    @property
    def is_terminal(self) -> Optional[bool]:
        return False

    @property
    def is_normalized(self) -> bool:
        return True

    def compute(self, terminated: bool, info: InfoType) -> Optional[float]:
        """Return a constant positive reward equal to 1.0 no matter what.
        """
        return 1.0


class TrackingActuatedJointPositionsReward(BaseQuantityReward):
    """Reward the agent for tracking the position of all the actuated joints of
    the robot wrt some reference trajectory.

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
            "reward_actuated_joint_positions",
            (BinaryOpQuantity, dict(
                quantity_left=(ActuatedJointPositions, dict(
                    mode=QuantityEvalMode.TRUE)),
                quantity_right=(ActuatedJointPositions, dict(
                    mode=QuantityEvalMode.REFERENCE)),
                op=sub)),
            partial(radial_basis_function, cutoff=self.cutoff, ord=2),
            is_normalized=True,
            is_terminal=False)
