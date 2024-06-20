"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
from operator import sub
from functools import partial
from typing import Optional, Sequence, Callable, TypeVar, Union

import numpy as np
import numba as nb

from ..bases import (
    InfoType, QuantityCreator, InterfaceJiminyEnv, AbstractReward,
    QuantityReward, QuantityEvalMode, QuantityTermination)
from ..quantities import (
    StackedQuantity, UnaryOpQuantity, BinaryOpQuantity, ActuatedJointsPosition)

from .mixin import radial_basis_function


ValueT = TypeVar('ValueT')

ArrayOrScalar = Union[np.ndarray, float, int]
ArrayLikeOrScalar = Union[ArrayOrScalar, Sequence[Union[float, int]]]


class SurviveReward(AbstractReward):
    """Reward the agent for surviving, ie make episodes last as long as
    possible by avoiding triggering termination conditions.

    Constant positive reward equal to 1.0 systematically, unless the current
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


class TrackingQuantityReward(QuantityReward):
    """Base class from which to derive reward defined as a difference between
    the current and reference value of a given quantity.

    A reference trajectory must be selected before evaluating this reward
    otherwise an exception will be risen. See `DatasetTrajectoryQuantity` and
    `AbstractQuantity` documentations for details.

    The error is transformed in a normalized reward to maximize by applying RBF
    kernel on the error. The reward will be 0.0 if the error cancels out
    completely and less than 'CUTOFF_ESP' above the user-specified cutoff
    threshold.
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
                                 reward after some post-processing, plus any
                                 keyword-arguments of its constructor except
                                 'env' and 'parent'.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        :param op: Any callable taking the true and reference values of the
                   quantity as input argument and returning the difference
                   between them, considering the algebra defined by their Lie
                   Group. The basic subtraction operator `operator.sub` is
                   appropriate for the Euclidean space.
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


class TrackingActuatedJointPositionsReward(TrackingQuantityReward):
    """Reward the agent for tracking the position of all the actuated joints of
    the robot wrt some reference trajectory.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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
            lambda mode: (ActuatedJointsPosition, dict(mode=mode)),
            cutoff)


class DriftTrackingQuantityTermination(QuantityTermination):
    """Base class to derive termination condition from the difference between
    the current and reference drift of a given quantity.

    The drift is defined as the difference between the most recent and oldest
    values of a time series. In this case, a variable-length horizon bounded by
    'max_stack' is considered.

    All elements must be within bounds for at least one time step in the fixed
    horizon. If so, then the episode continues, otherwise it is either
    truncated or terminated according to 'is_truncation' constructor argument.
    This only applies after the end of a grace period. Before that, the episode
    continues no matter what.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 quantity_creator: Callable[
                    [QuantityEvalMode], QuantityCreator[ArrayOrScalar]],
                 low: Optional[ArrayLikeOrScalar],
                 high: Optional[ArrayLikeOrScalar],
                 max_stack: int,
                 grace_period: float = 0.0,
                 *,
                 op: Callable[
                    [ArrayOrScalar, ArrayOrScalar], ArrayOrScalar] = sub,
                 is_truncation: bool = False,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the termination condition. This name will
                     be used as key for storing the current episode state from
                     the perspective of this specific condition in 'info', and
                     to add the underlying quantity to the set of already
                     managed quantities by the environment. As a result, it
                     must be unique otherwise an exception will be raised.
        :param quantity_creator: Any callable taking a quantity evaluation mode
                                 as input argument and return a tuple gathering
                                 the class of the underlying quantity to use as
                                 reward after some post-processing, plus any
                                 keyword-arguments of its constructor except
                                 'env' and 'parent'.
        :param low: Lower bound below which termination is triggered.
        :param high: Upper bound above which termination is triggered.
        :param max_stack: Horizon over which values of the quantity will be
                          stacked if desired. 1 to disable.
                          Optional: 1 by default.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param op: Any callable taking the true and reference values of the
                   quantity as input argument and returning the difference
                   between them, considering the algebra defined by their Lie
                   Group. The basic subtraction operator `operator.sub` is
                   appropriate for Euclidean space.
                   Optional: `operator.sub` by default.
        :param is_truncation: Whether the episode should be considered
                              terminated or truncated whenever the termination
                              condition is triggered.
                              Optional: False by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # pylint: disable=unnecessary-lambda-assignment

        # Backup user argument(s)
        self.max_stack = max_stack
        self.op = op

        # Define drift of quantity
        stack_creator = lambda mode: (StackedQuantity, dict(  # noqa: E731
            quantity=quantity_creator(mode),
            max_stack=max_stack))
        delta_creator = lambda mode: (BinaryOpQuantity, dict(  # noqa: E731
            quantity_left=(UnaryOpQuantity, dict(
                quantity=stack_creator(mode),
                op=lambda stack: stack[-1])),
            quantity_right=(UnaryOpQuantity, dict(
                quantity=stack_creator(mode),
                op=lambda stack: stack[0])),
            op=op))

        # Add drift quantity to the set of quantities managed by environment
        drift_tracking_quantity = (BinaryOpQuantity, dict(
                quantity_left=delta_creator(QuantityEvalMode.TRUE),
                quantity_right=delta_creator(QuantityEvalMode.REFERENCE),
                op=sub))

        # Call base implementation
        super().__init__(env,
                         name,
                         drift_tracking_quantity,  # type: ignore[arg-type]
                         low,
                         high,
                         grace_period,
                         is_truncation=is_truncation,
                         is_training_only=is_training_only)


class ShiftTrackingQuantityTermination(QuantityTermination):
    """Base class to derive termination condition from the shift between the
    current and reference values of a given quantity.

    The shift is defined as the minimum time-aligned distance (L2-norm of the
    difference) between two multivariate time series. In this case, a
    variable-length horizon bounded by 'max_stack' is considered.

    All elements must be within bounds for at least one time step in the fixed
    horizon. If so, then the episode continues, otherwise it is either
    truncated or terminated according to 'is_truncation' constructor argument.
    This only applies after the end of a grace period. Before that, the episode
    continues no matter what.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 quantity_creator: Callable[
                    [QuantityEvalMode], QuantityCreator[ArrayOrScalar]],
                 thr: float,
                 max_stack: int,
                 grace_period: float = 0.0,
                 *,
                 op: Callable[[np.ndarray, np.ndarray], np.ndarray] = sub,
                 is_truncation: bool = False,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the termination condition. This name will
                     be used as key for storing the current episode state from
                     the perspective of this specific condition in 'info', and
                     to add the underlying quantity to the set of already
                     managed quantities by the environment. As a result, it
                     must be unique otherwise an exception will be raised.
        :param quantity_creator: Any callable taking a quantity evaluation mode
                                 as input argument and return a tuple gathering
                                 the class of the underlying quantity to use as
                                 reward after some post-processing, plus any
                                 keyword-arguments of its constructor except
                                 'env' and 'parent'.
        :param thr: Termination is triggered if the shift exceeds this
                    threshold.
        :param max_stack: Horizon over which values of the quantity will be
                          stacked if desired. 1 to disable.
                          Optional: 1 by default.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param op: Any callable taking the true and reference stacked values of
                   the quantity as input argument and returning the difference
                   between them, considering the algebra defined by their Lie
                   Group. True and reference values are stacked in contiguous
                   N-dimension arrays along the first axis, namely the first
                   dimension gathers individual timesteps. For instance, the
                   common subtraction operator `operator.sub` is appropriate
                   for Euclidean space.
                   Optional: `operator.sub` by default.
        :param order: Order of Lp-Norm that will be used as distance metric.
        :param is_truncation: Whether the episode should be considered
                              terminated or truncated whenever the termination
                              condition is triggered.
                              Optional: False by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # pylint: disable=unnecessary-lambda-assignment

        # Backup user argument(s)
        self.max_stack = max_stack
        self.op = op

        # Define jit-able minimum distance between two time series
        @nb.jit(nopython=True, cache=True)
        def min_norm(values: np.ndarray) -> float:
            """Compute the minimum Euclidean norm over all timestamps of a
            multivariate time series.

            :param values: Time series as a N-dimensional array whose last
                           dimension corresponds to individual timestamps over
                           a finite horizon. The value at each timestamp will
                           be regarded as a 1D vector for computing their
                           Euclidean norm.
            """
            num_times = values.shape[-1]
            values_squared_flat = np.square(values).reshape((-1, num_times))
            return np.sqrt(np.min(np.sum(values_squared_flat, axis=0)))

        self._min_norm = min_norm

        # Define drift of quantity
        stack_creator = lambda mode: (StackedQuantity, dict(  # noqa: E731
            quantity=quantity_creator(mode),
            max_stack=max_stack,
            as_array=True))

        # Add drift quantity to the set of quantities managed by environment
        shift_tracking_quantity = (BinaryOpQuantity, dict(
            quantity_left=stack_creator(QuantityEvalMode.TRUE),
            quantity_right=stack_creator(QuantityEvalMode.REFERENCE),
            op=self._compute_min_distance))

        # Call base implementation
        super().__init__(env,
                         name,
                         shift_tracking_quantity,  # type: ignore[arg-type]
                         None,
                         thr,
                         grace_period,
                         is_truncation=is_truncation,
                         is_training_only=is_training_only)

    def _compute_min_distance(self,
                              left: np.ndarray,
                              right: np.ndarray) -> float:
        """Compute the minimum time-aligned Euclidean distance between two
        multivariate time series kept in sync.

        Internally, the time-aligned difference between the two time series
        will first be computed according to the user-specified binary operator
        'op'. The classical Euclidean norm of the difference is then computed
        over all timestamps individually and the minimum value is returned.

        :param left: Time series as a N-dimensional array whose first dimension
                     corresponds to individual timestamps over a finite
                     horizon. The value at each timestamp will be regarded as a
                     1D vector for computing their Euclidean norm. It will be
                     passed as left-hand side of the binary operator 'op'.
        :param right: Time series as a N-dimensional array with the exact same
                      shape as 'left'. See 'left' for details. It will be
                      passed as right-hand side of the binary operator 'op'.
        """
        return self._min_norm(self.op(left, right))
