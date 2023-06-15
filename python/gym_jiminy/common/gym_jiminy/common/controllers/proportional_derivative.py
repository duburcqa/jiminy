"""Implementation of basic Proportional-Derivative controller block compatible
with gym_jiminy reinforcement learning pipeline environment design.
"""
import math
from typing import Any, List, Union

import numpy as np
import numba as nb
import gymnasium as gym
from numpy.lib.stride_tricks import as_strided

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    EncoderSensor as encoder)

from ..bases import ObsType, BaseControllerBlock
from ..utils import fill


# Pre-computed factorial for small integers
FACTORIAL_TABLE = tuple(math.factorial(i) for i in range(6))

# Name of the n-th position derivative
N_ORDER_DERIVATIVE_NAMES = [
    "Position", "Velocity", "Acceleration", "Jerk", "Snap", "Crackle", "Pop"]

# Command velocity deadband to reduce vibrations and internal efforts
EVAL_DEADBAND = 5.0e-3


@nb.jit(nopython=True, nogil=True)
def toeplitz(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Numba-compatible implementation of `scipy.linalg.toeplitz` method.

    .. note:
        Special cases are ignored for efficiency, hence the input types
        are more respective than originally.

    .. warning:
        It returns a strided matrix instead of contiguous copy for efficiency.

    :param col: First column of the matrix.
    :param row: First row of the matrix.

    .. seealso::
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """  # noqa: E501  # pylint: disable=line-too-long
    vals = np.concatenate((col[::-1], row[1:]))
    stride = vals.strides[0]  # pylint: disable=E1136
    return as_strided(vals[len(col)-1:],
                      shape=(len(col), len(row)),
                      strides=(-stride, stride))


@nb.jit(nopython=True, nogil=True)
def integrate_zoh(state_prev: np.ndarray,
                  dt: float,
                  state_min: np.ndarray,
                  state_max: np.ndarray) -> np.ndarray:
    """N-order exact integration scheme assuming Zero-Order-Hold for the
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
    # Early return if the timestep is too small
    if abs(dt) < 1e-9:
        return state_prev.copy()

    # Compute integration matrix
    order = len(state_prev)
    integ_coeffs = np.array([
        pow(dt, k) / FACTORIAL_TABLE[k] for k in range(order)])
    integ_matrix = toeplitz(integ_coeffs, np.zeros(order)).T
    integ_zero = integ_matrix[:, :-1].copy() @ state_prev[:-1]
    integ_drift = integ_matrix[:, -1:]

    # Propagate derivative bounds to compute highest-order derivative bounds
    deriv = state_prev[-1]
    deriv_min_stack = (state_min - integ_zero) / integ_drift
    deriv_max_stack = (state_max - integ_zero) / integ_drift
    deriv_min = np.full_like(deriv, fill_value=-np.inf)
    deriv_max = np.full_like(deriv, fill_value=np.inf)
    for deriv_min_i, deriv_max_i in zip(deriv_min_stack, deriv_max_stack):
        deriv_min_i_valid = np.logical_and(
            deriv_min < deriv_min_i, deriv_min_i < deriv_max)
        deriv_min[deriv_min_i_valid] = deriv_min_i[deriv_min_i_valid]
        deriv_max_i_valid = np.logical_and(
            deriv_min < deriv_max_i, deriv_max_i < deriv_max)
        deriv_max[deriv_max_i_valid] = deriv_max_i[deriv_max_i_valid]

    # Clip highest-order derivative to ensure every derivative are withing
    # bounds if possible, lowest orders in priority otherwise.
    deriv = np.minimum(np.maximum(deriv, deriv_min), deriv_max)

    # Integrate, taking into account clipped highest derivative
    return integ_zero + integ_drift * deriv


@nb.jit(nopython=True, nogil=True)
def _compute_command_impl(encoders_data: np.ndarray,
                          command_state: np.ndarray,
                          command_state_lower: np.ndarray,
                          command_state_upper: np.ndarray,
                          pid_kp: np.ndarray,
                          pid_kd: np.ndarray,
                          motor_effort_limit: np.ndarray,
                          control_dt: float,
                          deadband: float) -> np.ndarray:
    """ TODO Write documentation.
    """
    # Estimate position and motor velocity from encoder data
    q_measured, v_measured = encoders_data

    # Integrate command state
    command_state[:] = integrate_zoh(
        command_state, control_dt, command_state_lower, command_state_upper)

    # Dead band to avoid slow drift of target position
    command_state[1] *= np.abs(command_state[1]) > deadband

    # Extract position/velocity targets
    q_target, v_target = command_state[:2]

    # Compute the joint tracking error
    q_error, v_error = q_target - q_measured, v_target - v_measured

    # Compute PD command
    u_command = pid_kp * (q_error + pid_kd * v_error)

    # Clip the command motors torques before returning
    return np.minimum(np.maximum(
        u_command, -motor_effort_limit), motor_effort_limit)


class PDController(BaseControllerBlock[np.ndarray, np.ndarray]):
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

    .. warning::
        It must be connected directly to the environment to control without
        any intermediary controllers.
    """
    def __init__(self,
                 name: str,
                 env: gym.Env[ObsType, np.ndarray],
                 order: int = 1,
                 update_ratio: int = 1,
                 pid_kp: Union[float, List[float], np.ndarray] = 0.0,
                 pid_kd: Union[float, List[float], np.ndarray] = 0.0,
                 soft_bounds_margin: float = 0.0,
                 **kwargs: Any) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param order: Derivative order of the action.
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
        :param pid_kp: PD controller position-proportional gain in motor order.
        :param pid_kd: PD controller velocity-proportional gain in motor order.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Make sure that the specified derivative order is valid
        assert order > 0

        # Backup some user argument(s)
        self.order = order
        self.pid_kp = np.asarray(pid_kp)
        self.pid_kd = np.asarray(pid_kd)

        # Define the mapping from motors to encoders
        self.encoder_to_motor = np.full(
            (env.robot.nmotors,), fill_value=-1, dtype=np.int64)
        encoders = [env.robot.get_sensor(encoder.type, sensor_name)
                    for sensor_name in env.robot.sensors_names[encoder.type]]
        for i, motor_name in enumerate(env.robot.motors_names):
            motor = env.robot.get_motor(motor_name)
            for j, sensor in enumerate(encoders):
                if motor.joint_idx == sensor.joint_idx:
                    self.encoder_to_motor[sensor.idx] = i
                    encoders.pop(j)
                    break
            else:
                raise RuntimeError(
                    f"No encoder sensor associated with motor '{motor_name}'. "
                    "Every actuated joint must have encoder sensors attached.")

        # Define buffers storing information about the motors for efficiency.
        # Note that even if the robot instance may change from one simulation
        # to another, the observation and action spaces are required to stay
        # the same the whole time. This induces that the motors effort limit
        # must not change unlike the mapping from full state to motors.
        self._motors_effort_limit = env.robot.command_limit[
            env.robot.motors_velocity_idx]

        # Compute the lower and upper bounds of the command state.
        # First, extract the lower and upper position and velocity bounds.
        # Then, extrapolate lax higher-order bounds. They are sufficient to
        # span the whole range of the state derivative directly preceding them
        # in a single timestep of the environment, ie acceleration bounds are
        # inferred from velocity bounds.
        motors_position_idx = sum(env.robot.motors_position_idx, [])
        motors_velocity_idx = env.robot.motors_velocity_idx
        command_state_lower = [
            env.robot.position_limit_lower[
                motors_position_idx] + soft_bounds_margin,
            -env.robot.velocity_limit[motors_velocity_idx],
        ]
        command_state_upper = [
            env.robot.position_limit_upper[
                motors_position_idx] - soft_bounds_margin,
            env.robot.velocity_limit[motors_velocity_idx],
        ]
        for _ in range(order - 1):
            n_order_limit = (
                command_state_upper[-1] - command_state_lower[-1]
                ) / env.step_dt
            command_state_lower.append(-n_order_limit)
            command_state_upper.append(n_order_limit)
        self._command_state_lower = np.stack(command_state_lower, axis=0)
        self._command_state_upper = np.stack(command_state_upper, axis=0)

        # Allocate memory for the command state
        self._command_state = np.zeros((order + 1, env.robot.nmotors))

        # Initialize the controller
        super().__init__(name, env, update_ratio)

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

        # Reset the command state
        fill(self._command_state, 0.0)

    def get_fieldnames(self) -> List[str]:
        return [f"target{N_ORDER_DERIVATIVE_NAMES[self.order]}{name}"
                for name in self.robot.motors_names]

    def compute_command(self,
                        action: np.ndarray
                        ) -> np.ndarray:
        """Compute the motor torques using a PD controller.

        It is proportional to the error between the observed motors positions/
        velocities and the target ones.

        :param action: Desired motors positions and velocities as a dictionary.
        """
        # Re-initialize the command state to the current motor state if the
        # simulation is not running. This must be done here because the
        # command state must be valid prior to calling `refresh_observation`
        # for the first time, which happens at `reset`.
        if not self.simulator.is_simulation_running:
            self._command_state[:2] = self.sensors_data[encoder.type]
            np.clip(self._command_state,
                    self._command_state_lower,
                    self._command_state_upper,
                    out=self._command_state)

        # Update the highest order derivative of the target motor positions to
        # match the provided action.
        self._command_state[-1] = action

        # Compute the motor efforts using PD control
        return _compute_command_impl(
            self.sensors_data[encoder.type],
            self._command_state,
            self._command_state_lower,
            self._command_state_upper,
            self.pid_kp,
            self.pid_kd,
            self._motors_effort_limit,
            self.control_dt,
            0.0 if self.env.is_training else EVAL_DEADBAND)
