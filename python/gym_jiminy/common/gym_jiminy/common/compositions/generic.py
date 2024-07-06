"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
from operator import sub
from functools import partial
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, TypeVar

import numpy as np
import numba as nb
import pinocchio as pin

from ..bases import (
    InfoType, QuantityCreator, InterfaceJiminyEnv, InterfaceQuantity,
    QuantityEvalMode, AbstractReward, QuantityReward,
    AbstractTerminationCondition, QuantityTermination)
from ..bases.compositions import ArrayOrScalar, ArrayLikeOrScalar
from ..quantities import (
    EnergyGenerationMode, StackedQuantity, UnaryOpQuantity, BinaryOpQuantity,
    MultiActuatedJointKinematic, AverageMechanicalPowerConsumption)

from .mixin import radial_basis_function


ValueT = TypeVar('ValueT')


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
        :param order: Order of L^p-norm that will be used as distance metric.
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
            lambda mode: (MultiActuatedJointKinematic, dict(
                kinematic_level=pin.KinematicLevel.POSITION,
                is_motor_side=False,
                mode=mode)),
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
                 horizon: float,
                 grace_period: float = 0.0,
                 *,
                 op: Callable[
                    [ArrayOrScalar, ArrayOrScalar], ArrayOrScalar] = sub,
                 post_fn: Optional[Callable[
                    [ArrayOrScalar], ArrayOrScalar]] = None,
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
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the drift.
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
        :apram post_fn: Optional callable taking the true and reference drifts
                        of the quantity as input argument and returning some
                        post-processed value to which bound checking will be
                        applied. None to skip post-processing entirely.
                        Optional: None by default.
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

        # Convert horizon in stack length, assuming constant env timestep
        max_stack = max(int(np.ceil(horizon / env.step_dt)), 1)

        # Backup user argument(s)
        self.max_stack = max_stack
        self.op = op
        self.post_fn = post_fn

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
            op=self._compute_drift_error))

        # Call base implementation
        super().__init__(env,
                         name,
                         drift_tracking_quantity,  # type: ignore[arg-type]
                         low,
                         high,
                         grace_period,
                         is_truncation=is_truncation,
                         is_training_only=is_training_only)

    def _compute_drift_error(self,
                             left: np.ndarray,
                             right: np.ndarray) -> ArrayOrScalar:
        """Compute the difference between the true and reference drift over
        a given horizon, then apply some post-processing on it if requested.

        :param left: True value of the drift as a N-dimensional array.
        :param right: Reference value of the drift as a N-dimensional array.
        """
        diff = left - right
        if self.post_fn is not None:
            return self.post_fn(diff)
        return diff


class ShiftTrackingQuantityTermination(QuantityTermination[np.ndarray]):
    """Base class to derive termination condition from the shift between the
    current and reference values of a given quantity.

    The shift is defined as the minimum time-aligned distance (L^2-norm of the
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
                 horizon: float,
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
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the shift.
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
        :param order: Order of L^p-norm that will be used as distance metric.
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

        # Convert horizon in stack length, assuming constant env timestep
        max_stack = max(int(np.ceil(horizon / env.step_dt)), 1)

        # Backup user argument(s)
        self.max_stack = max_stack
        self.op = op

        # Jit-able method computing minimum distance between two time series
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
            mode='slice',
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
                         np.array(thr),
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


@dataclass(unsafe_hash=True)
class _MultiActuatedJointBoundDistance(
        InterfaceQuantity[Tuple[np.ndarray, np.ndarray]]):
    """Distance of the actuated joints from their respective lower and upper
    mechanical stops.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                position=(MultiActuatedJointKinematic, dict(
                    kinematic_level=pin.KinematicLevel.POSITION,
                    is_motor_side=False,
                    mode=QuantityEvalMode.TRUE))),
            auto_refresh=False)

        # Lower and upper bounds of the actuated joints
        self.position_low, self.position_high = np.array([]), np.array([])

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Initialize the actuated joint position indices
        self.position.initialize()
        position_indices = self.position.kinematic_indices

        # Refresh mechanical joint position indices
        position_limit_low = self.env.robot.pinocchio_model.lowerPositionLimit
        self.position_low = position_limit_low[position_indices]
        position_limit_high = self.env.robot.pinocchio_model.upperPositionLimit
        self.position_high = position_limit_high[position_indices]

    def refresh(self) -> Tuple[np.ndarray, np.ndarray]:
        position = self.position.get()
        return (position - self.position_low, self.position_high - position)


class MechanicalSafetyTermination(AbstractTerminationCondition):
    """Discouraging the agent from hitting the mechanical stops by immediately
    terminating the episode if the articulated joints approach them at
    excessive speed.

    Hitting the lower and upper mechanical stops is inconvenient but forbidding
    it completely is not desirable as it induces safety margins that constrain
    the problem too strictly. This is particularly true when the maximum motor
    torque becomes increasingly limited and PD controllers are being used for
    low-level motor control, which turns out to be the case in most instances.
    Overall, such an hard constraint would impede performance while completing
    the task successfully remains the highest priority. Still, the impact
    velocity must be restricted to prevent destructive damage. It is
    recommended to estimate an acceptable thresholdfrom real experimental data.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 position_margin: float,
                 velocity_max: float,
                 grace_period: float = 0.0,
                 *,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param position_margin: Distance of actuated joints from their
                                respective mechanical bounds below which
                                their speed is being watched.
        :param velocity_max: Maximum velocity above which further approaching
                             the mechanical stops triggers termination when
                             watched for being close from them.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Backup user argument(s)
        self.position_margin = position_margin
        self.velocity_max = velocity_max

        # Call base implementation
        super().__init__(
            env,
            "termination_mechanical_safety",
            grace_period,
            is_truncation=False,
            is_training_only=is_training_only)

        # Add quantity to the set of quantities managed by the environment
        self.env.quantities["_".join((self.name, "position_delta"))] = (
            _MultiActuatedJointBoundDistance, {})
        self.env.quantities["_".join((self.name, "velocity"))] = (
            MultiActuatedJointKinematic, dict(
                kinematic_level=pin.KinematicLevel.VELOCITY,
                is_motor_side=False))

        # Keep track of the underlying quantities
        registry = self.env.quantities.registry
        self.position_delta = registry["_".join((self.name, "position_delta"))]
        self.velocity = registry["_".join((self.name, "velocity"))]

    def __del__(self) -> None:
        try:
            for field in ("position_delta", "velocity"):
                if hasattr(self, field):
                    del self.env.quantities["_".join((self.name, field))]
        except Exception:   # pylint: disable=broad-except
            # This method must not fail under any circumstances
            pass

    def compute(self, info: InfoType) -> bool:
        """Evaluate the termination condition.

        The underlying quantity is first evaluated. The episode continues if
        its value is within bounds, otherwise the episode is either truncated
        or terminated according to 'is_truncation'.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Evaluate the quantity
        position_delta_low, position_delta_high = self.position_delta.get()
        velocity = self.velocity.get()

        # Check if the robot is going to hit the mechanical stops at high speed
        is_done = any(
            (position_delta_low < self.position_margin) &
            (velocity < - self.velocity_max))
        is_done |= any(
            (position_delta_high < self.position_margin) &
            (velocity > self.velocity_max))
        return is_done


class MechanicalPowerConsumptionTermination(QuantityTermination):
    """Terminate the episode immediately if the average mechanical power
    consumption is too high.

    High power consumption is undesirable as it means that the motion is
    suboptimal and probably unnatural and fragile. Moreover, it helps to
    accommodate hardware capability to avoid motor overheating while increasing
    battery autonomy and lifespan. Finally, it may be necessary to deal with
    some hardware limitations on max power drain.
    """
    def __init__(
            self,
            env: InterfaceJiminyEnv,
            max_power: float,
            horizon: float,
            generator_mode: EnergyGenerationMode = EnergyGenerationMode.CHARGE,
            grace_period: float = 0.0,
            *,
            is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param max_power: Maximum average mechanical power consumption applied
                          on any of the contact points or collision bodies
                          above which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the average.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Backup user argument(s)
        self.max_power = max_power
        self.horizon = horizon
        self.generator_mode = generator_mode

        # Call base implementation
        super().__init__(
            env,
            "termination_power_consumption",
            (AverageMechanicalPowerConsumption, dict(  # type: ignore[arg-type]
                horizon=self.horizon,
                generator_mode=self.generator_mode)),
            None,
            self.max_power,
            grace_period,
            is_truncation=False,
            is_training_only=is_training_only)


class ShiftTrackingMotorPositionsTermination(ShiftTrackingQuantityTermination):
    """Terminate the episode if the selected reference trajectory is not
    tracked with expected accuracy regarding the actuated joint positions,
    whatever the timestep being considered over some fixed-size sliding window.

    The robot must track the reference if there is no hazard, only applying
    minor corrections to keep balance. Rewarding the agent for doing so is
    not effective as favoring robustness remains more profitable. Indeed, it
    would anticipate disturbances, lowering its current reward to maximize the
    future return, primarily averting termination. Limiting the shift over a
    given horizon allows for large deviations to handle strong pushes.
    Moreover, assuming that the agent is not able to keep track of the time
    flow, which means that only the observation at the current step is provided
    to the agent and o stateful network architecture such as LSTM is being
    used, restricting the shift also urges to do what it takes to get back to
    normal as soon as possible for fear of triggering termination, as it may
    happen any time the deviation is above the maximum acceptable shift,
    irrespective of its scale.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 thr: float,
                 horizon: float,
                 grace_period: float = 0.0,
                 *,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param thr: Maximum shift above which termination is triggered.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the shift.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Call base implementation
        super().__init__(
            env,
            "termination_tracking_motor_positions",
            lambda mode: (MultiActuatedJointKinematic, dict(
                kinematic_level=pin.KinematicLevel.POSITION,
                is_motor_side=False,
                mode=mode)),
            thr,
            horizon,
            grace_period,
            is_truncation=False,
            is_training_only=is_training_only)
