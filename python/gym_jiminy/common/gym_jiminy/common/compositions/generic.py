"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
from operator import sub
from functools import partial
from typing import Optional, Callable, TypeVar

from ..bases import (
    InfoType, QuantityCreator, InterfaceJiminyEnv, AbstractReward,
    BaseQuantityReward, QuantityEvalMode)
from ..quantities import BinaryOpQuantity, ActuatedJointPositions

from .mixin import radial_basis_function


ValueT = TypeVar('ValueT')


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


class BaseTrackingReward(BaseQuantityReward):
    """Base class from which to derive reward defined as a difference between
    the current and reference value of a given quantity.

    A reference trajectory must be selected before evaluating this reward
    otherwise an exception will be risen. See `DatasetTrajectoryQuantity` and
    `AbstractQuantity` documentations for details.

    The error transform in a normalized reward to maximize by applying RBF
    kernel on the error. The reward will be 0.0 if the error cancels out
    completely and less than 0.01 above the user-specified cutoff threshold.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 quantity_creator: Callable[
                    [QuantityEvalMode], QuantityCreator[ValueT]],
                 cutoff: float,
                 *,
                 op: Callable[[ValueT, ValueT], ValueT] = sub,
                 order: int = 2) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the reward. This name will be used as key
                     for storing current value of the reward in 'info', and to
                     add the underlying quantity to the set of already managed
                     quantities by the environment. As a result, it must be
                     unique otherwise an exception will be raised.
        :param quantity_creator: Any callable taking a quantity evaluation mode
                                 as input argument and return a tuple gathering
                                 the class of the underlying quantity to use as
                                 reward after some post-processing, plus all
                                 its constructor keyword-arguments except
                                 environment 'env', parent 'parent.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        :param op: Any callable taking the true and reference values of the
                   quantity as input argument and returning the difference
                   between them, considering the algebra defined by their Lie
                   Group. The basic subtraction operator `operator.sub` is
                   appropriate for Euclidean.
                   Optional: `operator.sub` by default.
        :param order: Order of Lp-Norm that will be used as distance metric.
                      Optional: 2 by default.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            name,
            (BinaryOpQuantity, dict(
                quantity_left=quantity_creator(QuantityEvalMode.TRUE),
                quantity_right=quantity_creator(QuantityEvalMode.REFERENCE),
                op=op)),
            partial(radial_basis_function, cutoff=self.cutoff, order=order),
            is_normalized=True,
            is_terminal=False)


class TrackingActuatedJointPositionsReward(BaseTrackingReward):
    """Reward the agent for tracking the position of all the actuated joints of
    the robot wrt some reference trajectory.

    .. seealso::
        See `BaseTrackingReward` documentation for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_actuated_joint_positions",
            lambda mode: (ActuatedJointPositions, dict(mode=mode)),
            cutoff)
