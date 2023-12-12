"""Implementation of basic Proportional-Derivative controller block compatible
with gym_jiminy reinforcement learning pipeline environment design.
"""
import math
from typing import List, Union

import numpy as np
import numba as nb
import gymnasium as gym
from numpy.lib.stride_tricks import as_strided

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto,
    EncoderSensor as encoder)

from ..bases import BaseObsT, JiminyEnvInterface, BaseControllerBlock
from ..utils import fill


# Pre-computed factorial for small integers
INV_FACTORIAL_TABLE = tuple(1.0 / math.factorial(i) for i in range(4))

# Name of the n-th position derivative
N_ORDER_DERIVATIVE_NAMES = ("Position", "Velocity", "Acceleration", "Jerk")

# Command velocity deadband to reduce vibrations and internal efforts
EVAL_DEADBAND = 5.0e-3


@nb.jit(nopython=True, nogil=True, cache=True, inline='always')
def toeplitz(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Numba-compatible implementation of `scipy.linalg.toeplitz` method.

    .. note:
        Special cases are ignored for efficiency, hence the input types
        are more respective than originally.

    .. warning:
        It returns a strided matrix instead of contiguous copy for efficiency.

    .. seealso::
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html

    :param col: First column of the matrix.
    :param row: First row of the matrix.
    """  # noqa: E501  # pylint: disable=line-too-long
    vals = np.concatenate((col[::-1], row[1:]))
    stride = vals.strides[0]  # pylint: disable=E1136
    return as_strided(vals[len(col)-1:],
                      shape=(len(col), len(row)),
                      strides=(-stride, stride))


@nb.jit(nopython=True, nogil=True, cache=True, inline='always')
def integrate_zoh(state_prev: np.ndarray,
                  state_min: np.ndarray,
                  state_max: np.ndarray,
                  dt: float) -> np.ndarray:
    """N-th order exact integration scheme assuming Zero-Order-Hold for the
    highest-order derivative, taking state bounds into account.

    .. warning::
        This method tries its best to keep the state within bounds, but it is
        not always possible if the order is strictly larger than 1. Indeed, the
        bounds of different derivative order may be conflicting. In such a
        case, it gives priority to lower orders.

    :param state_prev: Previous state update, ordered from lowest to highest
                       derivative order, which means:
                       s[i](t) = s[i](t-1) + integ_{t-1}^{t}(s[i+1](t))
    :param state_min: Lower bounds of the state.
    :param state_max: Upper bounds of the state.
    :param dt: Integration delta of time since previous state update.
    """
    # Make sure that dt is not negative
    assert dt >= 0.0, "The integration timestep 'dt' must be positive."

    # Early return if the timestep is too small
    if abs(dt) < 1e-9:
        return state_prev.copy()

    # Compute integration matrix
    dim, size = state_prev.shape
    integ_coeffs = np.array([
        pow(dt, k) * INV_FACTORIAL_TABLE[k] for k in range(dim)])
    integ_matrix = toeplitz(integ_coeffs, np.zeros(dim)).T
    integ_zero = integ_matrix[:, :-1].copy() @ state_prev[:-1]
    integ_drift = integ_matrix[:, -1:]

    # Propagate derivative bounds to compute highest-order derivative bounds
    deriv_min_stack = (state_min - integ_zero) / integ_drift
    deriv_max_stack = (state_max - integ_zero) / integ_drift
    deriv_min, deriv_max = np.full((size,), -np.inf), np.full((size,), np.inf)
    for deriv_min_i, deriv_max_i in zip(deriv_min_stack, deriv_max_stack):
        for k, (deriv_min_k, deriv_max_k) in enumerate(zip(
                deriv_min, deriv_max)):
            if deriv_min_k < deriv_min_i[k] < deriv_max_k:
                deriv_min[k] = deriv_min_i[k]
            if deriv_min_k < deriv_max_i[k] < deriv_max_k:
                deriv_max[k] = deriv_max_i[k]

    # Clip highest-order derivative to ensure every derivative are within
    # bounds if possible, lowest orders in priority otherwise.
    deriv = np.clip(state_prev[-1], deriv_min, deriv_max)

    # Integrate, taking into account clipped highest derivative
    return integ_zero + integ_drift * deriv


@nb.jit(nopython=True, nogil=True, cache=True)
def pd_controller(q_measured: np.ndarray,
                  v_measured: np.ndarray,
                  command_state: np.ndarray,
                  command_state_lower: np.ndarray,
                  command_state_upper: np.ndarray,
                  kp: np.ndarray,
                  kd: np.ndarray,
                  motors_effort_limit: np.ndarray,
                  control_dt: float) -> np.ndarray:
    """Compute command under discrete-time proportional-derivative feedback
    control.

    Internally, it integrates the command state over the controller update
    period in order to obtain the desired motor positions 'q_desired' and
    velocities 'v_desired'. By computing them this way, the desired motor
    positions and velocities can be interpreted as targets should be reached
    right before updating the command once again. The integration takes into
    account some lower and upper bounds that ideally should not be exceeded.
    If not possible, priority is given to consistency of the integration, so
    no clipping of the command state ever occurs. The lower order bounds will
    be satisfied first, which means that position limits are the only one to be
    guaranteed to never be violated.

    The command effort is computed as follows:

        tau = - kp * ((q_measured - q_desired) + kd * (v_measured - v_desired))

    The torque will be held constant until the next controller update.

    .. seealso::
        See `PDController` documentation to get more information, and
        `integrate_zoh` documentation for details about the state integration.

    :param q_measured: Current position of the actuators.
    :param v_measured: Current velocity of the actuators.
    :param command_state: Current command state, namely, all the derivatives of
                          the target motors positions up to order N. The order
                          must be larger than 2 but can be arbitrarily large.
    :param command_state_lower: Lower bound of the command state that should be
                                satisfied if possible, prioritizing lower order
                                derivatives.
    :param command_state_upper: Upper bound of the command state that should be
                                satisfied if possible, prioritizing lower order
                                derivatives.
    :param kp: PD controller position-proportional gain in motor order.
    :param kd: PD controller velocity-proportional gain in motor order.
    :param motors_effort_limit: Maximum effort that the actuators can output.
    :param control_dt: Controller update period. It will be involved in the
                       integration of the command state.
    """
    # Integrate command state
    command_state[:] = integrate_zoh(
        command_state, command_state_lower, command_state_upper, control_dt)

    # Extract targets motors positions and velocities from command state
    q_target, v_target = command_state[:2]

    # Compute the joint tracking error
    q_error, v_error = q_target - q_measured, v_target - v_measured

    # Compute PD command
    u_command = kp * (q_error + kd * v_error)

    # Clip the command motors torques before returning
    return np.clip(u_command, -motors_effort_limit, motors_effort_limit)


def get_encoder_to_motor_map(robot: jiminy.Robot) -> Union[slice, List[int]]:
    """Get the mapping from encoder sensors to motors.

    .. warning::
        If reordering is necessary, then a list of indices is returned, which
        can used to permute encoder sensor data to match the command torque
        vector. Yet, it relies on so-called "fancy" or "advanced" indexing for
        doing so, which means that the returned data is a copy of the original
        data instead of a reference. On the contrary, it reordering is not
        necessary, a slice is returned instead and no copy happens whatsoever.

    :param robot: Jiminy robot for which to compute mapping.

    :returns: A slice if possible, a list of indices otherwise.
    """
    # Define the mapping from motors to encoders
    encoder_to_motor = [-1 for _ in range(robot.nmotors)]
    encoders = [robot.get_sensor(encoder.type, sensor_name)
                for sensor_name in robot.sensors_names[encoder.type]]
    for i, motor_name in enumerate(robot.motors_names):
        motor = robot.get_motor(motor_name)
        for j, sensor in enumerate(encoders):
            assert isinstance(sensor, encoder)
            if motor.joint_idx == sensor.joint_idx:
                encoder_to_motor[sensor.idx] = i
                encoders.pop(j)
                break
        else:
            raise RuntimeError(
                f"No encoder sensor associated with motor '{motor_name}'. "
                "Every actuated joint must have encoder sensors attached.")

    # Try converting it to slice if possible
    if (np.array(encoder_to_motor) == np.arange(robot.nmotors)).all():
        return slice(None)
    return encoder_to_motor


class PDController(
        BaseControllerBlock[np.ndarray, np.ndarray, BaseObsT, np.ndarray]):
    """Low-level Proportional-Derivative controller.

    The action corresponds to a given derivative of the target motors
    positions. All the lower-order derivatives are obtained by integration,
    considering that the action is constant until the next controller update.

    .. note::
        The higher the derivative order of the action, the smoother the command
        motors torques. Thus, a high order is generally beneficial for robotic
        applications. However, it introduces some kind of delay between the
        action and its effects. This phenomenon makes learning more difficult
        but most importantly, it limits the responsiveness of the agent
        and therefore impedes its optimal performance.

    .. note::
        The position and velocity bounds on the command corresponds to the
        joint limits specified by the dynamical model of the robot. Then, lax
        higher-order bounds are extrapolated. In a single timestep of the
        environment, they are chosen to be sufficient either to span the whole
        range of the state derivative directly preceding (ie acceleration
        bounds are inferred from velocity bounds) or to allow reaching the
        command effort limits depending on what is the most restrictive.

    .. warning::
        It must be connected directly to the environment to control without
        any intermediary controllers altering the action space.
    """
    def __init__(self,
                 name: str,
                 env: JiminyEnvInterface[BaseObsT, np.ndarray],
                 *,
                 update_ratio: int = 1,
                 order: int,
                 kp: Union[float, List[float], np.ndarray],
                 kd: Union[float, List[float], np.ndarray],
                 target_position_margin: float = 0.0,
                 target_velocity_limit: float = float("inf")) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
        :param order: Derivative order of the action.
        :param kp: PD controller position-proportional gains in motor order.
        :param kd: PD controller velocity-proportional gains in motor order.
        :param target_position_margin: Minimum distance of the motor target
                                       positions from their respective bounds.
        :param target_velocity_limit: Maximum motor target velocities.
        """
        # Make sure that the specified derivative order is valid
        assert (0 < order < 4), "Derivative order of command out-of-bounds"

        # Make sure the action space of the environment has not been altered
        if env.action_space is not env.unwrapped.action_space:
            raise RuntimeError(
                "Impossible to connect this block on an environment whose "
                "action space has been altered.")

        # Make sure that the number of PD gains matches the number of motors
        try:
            kp = np.broadcast_to(kp, (env.robot.nmotors,))
            kd = np.broadcast_to(kd, (env.robot.nmotors,))
        except ValueError as e:
            raise TypeError(
                "PD gains inconsistent with number of motors.") from e

        # Backup some user argument(s)
        self.order = order
        self.kp = kp
        self.kd = kd

        # Mapping from motors to encoders
        self.encoder_to_motor = get_encoder_to_motor_map(env.robot)

        # Whether stored reference to encoder measurements are already in the
        # same order as the motors, allowing skipping re-ordering entirely.
        self._is_same_order = isinstance(self.encoder_to_motor, slice)

        # Define buffers storing information about the motors for efficiency.
        # Note that even if the robot instance may change from one simulation
        # to another, the observation and action spaces are required to stay
        # the same the whole time. This induces that the motors effort limit
        # must not change unlike the mapping from full state to motors.
        self.motors_effort_limit = env.robot.command_limit[
            env.robot.motors_velocity_idx]

        # Extract the motors target position and velocity bounds from the model
        motors_position_lower: List[float] = []
        motors_position_upper: List[float] = []
        for motor_name in env.robot.motors_names:
            motor = env.robot.get_motor(motor_name)
            joint_type = jiminy.get_joint_type(
                env.robot.pinocchio_model, motor.joint_idx)
            if joint_type == jiminy.JointModelType.ROTARY_UNBOUNDED:
                lower, upper = float("-inf"), float("inf")
            else:
                motor_position_idx = motor.joint_position_idx
                lower = env.robot.position_limit_lower[motor_position_idx]
                upper = env.robot.position_limit_upper[motor_position_idx]
            motors_position_lower.append(lower + target_position_margin)
            motors_position_upper.append(upper - target_position_margin)
        motors_velocity_limit = np.minimum(
            env.robot.velocity_limit[env.robot.motors_velocity_idx],
            target_velocity_limit)
        command_state_lower = [
            np.array(motors_position_lower), -motors_velocity_limit]
        command_state_upper = [
            np.array(motors_position_upper), motors_velocity_limit]

        # Try to infers bounds for higher-order derivatives if necessary.
        # They are tuned to allow for bang-bang control without restriction.
        step_dt = env.step_dt
        for i in range(2, order + 1):
            range_limit = (
                command_state_upper[-1] - command_state_lower[-1]) / step_dt
            effort_limit = self.motors_effort_limit / (
                self.kp * step_dt ** (i - 1) * INV_FACTORIAL_TABLE[i - 1] *
                np.maximum(step_dt / i, self.kd))
            n_order_limit = np.minimum(range_limit, effort_limit)
            command_state_lower.append(-n_order_limit)
            command_state_upper.append(n_order_limit)
        self._command_state_lower = np.stack(command_state_lower, axis=0)
        self._command_state_upper = np.stack(command_state_upper, axis=0)

        # Extract measured motor positions and velocities for fast access
        self.q_measured, self.v_measured = env.sensors_data[encoder.type]

        # Allocate memory for the command state
        self._command_state = np.zeros((order + 1, env.robot.nmotors))

        # Initialize the controller
        super().__init__(name, env, update_ratio)

        # Reference to highest-order derivative for fast access
        self._action = self._command_state[-1]

    def _initialize_state_space(self) -> None:
        """Configure the state space of the controller.

        The state spaces corresponds to all the derivatives of the target
        motors positions up to order N-1.
        """
        self.state_space = gym.spaces.Box(
            low=self._command_state_lower[:-1],
            high=self._command_state_upper[:-1],
            dtype=np.float64)

    def _initialize_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the N-th order derivative of the
        target motors positions.
        """
        self.action_space = gym.spaces.Box(
            low=self._command_state_lower[-1],
            high=self._command_state_upper[-1],
            dtype=np.float64)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Make sure control update is discrete-time
        if self.env.control_dt <= 0.0:
            raise ValueError(
                "This block does not support time-continuous update.")

        # Refresh measured motor positions and velocities proxies
        self.q_measured, self.v_measured = self.env.sensors_data[encoder.type]
        self.q_measured, self.v_measured = self.env.sensors_data[
            encoder.type][:, self.encoder_to_motor]

        # Reset the command state
        fill(self._command_state, 0)

    @property
    def fieldnames(self) -> List[str]:
        return [f"target{N_ORDER_DERIVATIVE_NAMES[self.order]}{name}"
                for name in self.env.robot.motors_names]

    def get_state(self) -> np.ndarray:
        return self._command_state[:-1]

    def compute_command(self, action: np.ndarray) -> np.ndarray:
        """Compute the motor torques using a PD controller.

        It is proportional to the error between the observed motors positions/
        velocities and the target ones.

        :param action: Desired N-th order deriv. of the target motor positions.
        """
        # Re-initialize the command state to the current motor state if the
        # simulation is not running. This must be done here because the
        # command state must be valid prior to calling `refresh_observation`
        # for the first time, which happens at `reset`.
        is_simulation_running = self.env.is_simulation_running
        if not is_simulation_running:
            for i, value in enumerate((self.q_measured, self.v_measured)):
                np.clip(value,
                        self._command_state_lower[i],
                        self._command_state_upper[i],
                        out=self._command_state[i])

        # Skip integrating command and return early if no simulation running.
        # It also checks that the low-level function is already pre-compiled.
        # This is necessary to avoid spurious timeout during first step.
        if not is_simulation_running and pd_controller.signatures:
            return np.zeros_like(action)

        # Update the highest order derivative of the target motor positions to
        # match the provided action.
        array_copyto(self._action, action)

        # Dead band to avoid slow drift of target at rest for evaluation only
        if not self.env.is_training:
            self._action[np.abs(self._action) > EVAL_DEADBAND] = 0.0

        # Extract motor positions and velocity from encoder data
        q_measured, v_measured = self.q_measured, self.v_measured
        if not self._is_same_order:
            q_measured = q_measured[self.encoder_to_motor]
            v_measured = v_measured[self.encoder_to_motor]

        # Compute the motor efforts using PD control
        return pd_controller(
            q_measured,
            v_measured,
            self._command_state,
            self._command_state_lower,
            self._command_state_upper,
            self.kp,
            self.kd,
            self.motors_effort_limit,
            self.control_dt)
