"""Implementation of basic Proportional-Derivative controller block compatible
with gym_jiminy reinforcement learning pipeline environment design.
"""
import warnings
from typing import List, Union

import numpy as np
import numba as nb
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    EncoderSensor as encoder)

from ..bases import BaseObsT, InterfaceJiminyEnv, BaseControllerBlock
from ..utils import fill


# Name of the n-th position derivative
N_ORDER_DERIVATIVE_NAMES = ("Position", "Velocity", "Acceleration")


@nb.jit(nopython=True, cache=True, inline='always', fastmath=True)
def integrate_zoh(state: np.ndarray,
                  state_min: np.ndarray,
                  state_max: np.ndarray,
                  dt: float,
                  horizon: np.ndarray) -> None:
    """First order exact integration scheme assuming Zero-Order-Hold for the
    velocity, taking state bounds into account.

    :param state: Current state, ordered from lowest to highest derivative
                  order, ie: s[i](t) = s[i](t-1) + integ_{t-1}^{t}(s[i+1](t)).
                  It will be updated in place.
    :param state_min: Lower bounds of the state.
    :param state_max: Upper bounds of the state.
    :param dt: Integration delta of time since previous state update.
    :param horizon: Horizon length to start slowing down before hitting bounds.
    """
    # Make sure that dt is not negative
    assert dt >= 0.0, "The integration timestep 'dt' must be positive."

    # Early return if timestep is too small
    if abs(dt) < 1e-9:
        return

    # Make sure that velocity bounds are satisfied
    state[1] = np.minimum(np.maximum(state[1], state_min[1]), state_max[1])

    # Deduce velocity bounds from position bounds
    velocity_min = (state_min[0] - state[0]) / (horizon * dt)
    velocity_max = (state_max[0] - state[0]) / (horizon * dt)

    # Clip velocity to ensure than position bounds are always satisfied.
    # Note that it may cause velocity bounds to be violated.
    state[1] = np.minimum(np.maximum(state[1], velocity_min), velocity_max)

    # Integrate 1-step ahead, taking into account clipped velocity
    state[0] += dt * state[1]


@nb.jit(nopython=True, cache=True, fastmath=True)
def pd_controller(q_measured: np.ndarray,
                  v_measured: np.ndarray,
                  action: np.ndarray,
                  order: int,
                  command_state: np.ndarray,
                  command_state_lower: np.ndarray,
                  command_state_upper: np.ndarray,
                  kp: np.ndarray,
                  kd: np.ndarray,
                  motors_effort_limit: np.ndarray,
                  control_dt: float,
                  out: np.ndarray) -> np.ndarray:
    """Compute command torques under discrete-time proportional-derivative
    feedback control.

    Internally, it integrates the command state (position, velocity and
    acceleration) at controller update period in order to obtain the desired
    motor positions 'q_desired' and velocities 'v_desired', takes into account
    some lower and upper bounds. By computing them this way, the target motor
    positions and velocities can be interpreted as targets that has be to reach
    right before updating the command once again. Enforcing consistency between
    target position and velocity in such a way is a necessary but insufficient
    condition for the motors to be able to track them.

    The command effort is computed as follows:

        tau = - kp * ((q_measured - q_desired) + kd * (v_measured - v_desired))

    The torque will be held constant until the next controller update.

    .. seealso::
        See `PDController` documentation to get more information, and
        `integrate_zoh` documentation for details about the state integration.

    :param q_measured: Current position of the actuators.
    :param v_measured: Current velocity of the actuators.
    :param action: Desired value of the n-th derivative of the command motor
                   positions at the end of the controller update.
    :param order: Derivative order of the position associated with the action.
    :param command_state: Current command state, namely, all the derivatives of
                          the target motors positions up to order N. The order
                          must be larger than 1 but can be arbitrarily large.
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
    :param horizon: Horizon length to start slowing down before hitting bounds.
    :param out: Pre-allocated memory to store the command motor torques.
    """
    # Update command accelerations based on the action and its derivative order
    if order == 2:
        # The action corresponds to the command motor accelerations
        acceleration = action
    else:
        if order == 0:
            # Compute command velocity
            velocity = (action - command_state[0]) / control_dt

            # Clip command velocity
            velocity = np.minimum(np.maximum(
                velocity, command_state_lower[1]), command_state_upper[1])
        else:
            # The action corresponds to the command motor velocities
            velocity = action

        # Compute command acceleration
        acceleration = (velocity - command_state[1]) / control_dt

    # Clip command acceleration
    command_state[2] = np.minimum(np.maximum(
        acceleration, command_state_lower[2]), command_state_upper[2])

    # Integrate command velocity
    command_state[1] += command_state[2] * control_dt

    # Compute slowdown horizon.
    # It must be as short as possible to avoid altering the user-specified
    # command if not strictly necessary, but long enough to avoid violation of
    # acceleration bounds when hitting bounds.
    horizon = np.maximum(np.abs(command_state[1]) / (
        command_state_upper[2] * max(control_dt, 1-9)), 1.0)

    # Integrate command position, clipping velocity to satisfy state bounds
    integrate_zoh(command_state,
                  command_state_lower,
                  command_state_upper,
                  control_dt,
                  horizon)

    # Extract targets motors positions and velocities from command state
    q_target, v_target = command_state[:2]

    # Compute the joint tracking error
    q_error, v_error = q_target - q_measured, v_target - v_measured

    # Compute PD command
    out[:] = kp * (q_error + kd * v_error)

    # Clip the command motors torques before returning
    out[:] = np.minimum(np.maximum(
        out, -motors_effort_limit), motors_effort_limit)

    return out


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
                for sensor_name in robot.sensor_names[encoder.type]]
    for i, motor_name in enumerate(robot.motor_names):
        motor = robot.get_motor(motor_name)
        for j, sensor in enumerate(encoders):
            assert isinstance(sensor, encoder)
            if motor.joint_index == sensor.joint_index:
                encoder_to_motor[sensor.index] = i
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

    The action is a given derivative of the target motors positions, from
    which the target acceleration will be deduced. The latter is then
    integrated twice using two first-order integrator in cascade, considering
    that the acceleration is constant until the next controller update:

        v_{t+1} = v_{t} + dt * a_{t}
        q_{t+1} = q_{t} + dt * v_{t+1}

    .. note::
        The higher the derivative order of the action, the smoother the command
        motor torques. Thus, a high order is generally beneficial for robotic
        applications. However, it introduces some kind of delay between the
        action and its effects. This phenomenon makes learning more difficult
        but most importantly, it limits the responsiveness of the agent
        and therefore impedes its optimal performance.

    .. note::
        The position and velocity bounds on the command corresponds to the
        joint limits specified by the dynamical model of the robot. Then, lax
        default acceleration bounds are extrapolated. More precisely, they are
        chosen to be sufficient either to span the whole range of velocity or
        to allow reaching the command effort limits depending on what is the
        most restrictive. Position, velocity and acceleration.

    .. warning::
        It must be connected directly to the base environment to control
        without any intermediary controllers altering its action space.
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObsT, np.ndarray],
                 *,
                 update_ratio: int = 1,
                 order: int = 1,
                 kp: Union[float, List[float], np.ndarray],
                 kd: Union[float, List[float], np.ndarray],
                 target_position_margin: float = 0.0,
                 target_velocity_limit: float = float("inf"),
                 target_acceleration_limit: float = float("inf")) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller. -1 to
                             match the simulation timestep of the environment.
        :param order: Derivative order of the action. It accepts position,
                      velocity or acceleration (respectively 0, 1 and 2).
                      Optional: 1 by default.
        :param kp: PD controller position-proportional gains in motor order.
        :param kd: PD controller velocity-proportional gains in motor order.
        :param target_position_margin: Minimum distance of the motor target
                                       positions from their respective bounds.
                                       Optional: 0.0 by default.
        :param target_velocity_limit: Restrict maximum motor target velocities
                                      wrt their hardware specifications.
                                      Optional: "inf" by default.
        :param target_acceleration_limit:
            Restrict maximum motor target accelerations wrt their hardware
            specifications.
            Optional: "inf" by default.
        """
        # Make sure that the specified derivative order is valid
        assert (0 <= order < 3), "Derivative order of command out-of-bounds"

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
        if not self._is_same_order:
            warnings.warn(
                "Consider using the same ordering for encoders and motors for "
                "optimal performance.")

        # Define buffers storing information about the motors for efficiency.
        # Note that even if the robot instance may change from one simulation
        # to another, the observation and action spaces are required to stay
        # the same the whole time. This induces that the motors effort limit
        # must not change unlike the mapping from full state to motors.
        self.motors_effort_limit = env.robot.command_limit[
            env.robot.motor_velocity_indices]

        # Extract the motors target position bounds from the model
        motors_position_lower: List[float] = []
        motors_position_upper: List[float] = []
        for motor_name in env.robot.motor_names:
            motor = env.robot.get_motor(motor_name)
            joint_type = jiminy.get_joint_type(
                env.robot.pinocchio_model, motor.joint_index)
            if joint_type == jiminy.JointModelType.ROTARY_UNBOUNDED:
                lower, upper = float("-inf"), float("inf")
            else:
                motor_position_index = motor.joint_position_index
                lower = env.robot.position_limit_lower[motor_position_index]
                upper = env.robot.position_limit_upper[motor_position_index]
            motors_position_lower.append(lower + target_position_margin)
            motors_position_upper.append(upper - target_position_margin)

        # Extract the motors target velocity bounds
        motors_velocity_limit = np.minimum(
            env.robot.velocity_limit[env.robot.motor_velocity_indices],
            target_velocity_limit)

        # Compute acceleration bounds allowing unrestricted bang-bang control
        range_limit = 2 * motors_velocity_limit / env.step_dt
        effort_limit = self.motors_effort_limit / (
            self.kp * env.step_dt * np.maximum(env.step_dt / 2, self.kd))
        acceleration_limit = np.minimum(
            np.minimum(range_limit, effort_limit), target_acceleration_limit)

        # Compute command state bounds
        self._command_state_lower = np.stack([np.array(motors_position_lower),
                                              -motors_velocity_limit,
                                              -acceleration_limit], axis=0)
        self._command_state_upper = np.stack([np.array(motors_position_upper),
                                              motors_velocity_limit,
                                              acceleration_limit], axis=0)

        # Extract measured motor positions and velocities for fast access.
        # Note that they will be initialized in `_setup` method.
        self.q_measured, self.v_measured = np.array([]), np.array([])

        # Allocate memory for the command state
        self._command_state = np.zeros((3, env.robot.nmotors))

        # Initialize the controller
        super().__init__(name, env, update_ratio)

        # References to command position, velocity and acceleration
        (self._command_position,
         self._command_velocity,
         self._command_accel) = self._command_state

        # Command motor torques buffer for efficiency
        self._u_command = np.array([])

    def _initialize_state_space(self) -> None:
        """Configure the state space of the controller.

        The state spaces corresponds to all the derivatives of the target
        motors positions up to order N-1.
        """
        self.state_space = gym.spaces.Box(
            low=self._command_state_lower[:2],
            high=self._command_state_upper[:2],
            dtype=np.float64)

    def _initialize_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the N-th order derivative of the
        target motors positions.
        """
        self.action_space = gym.spaces.Box(
            low=self._command_state_lower[self.order],
            high=self._command_state_upper[self.order],
            dtype=np.float64)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Make sure control update is discrete-time
        if self.env.control_dt <= 0.0:
            raise ValueError(
                "This block does not support time-continuous update.")

        # Re-initialize pre-allocated memory for command motor torques
        self._u_command = np.zeros((self.env.robot.nmotors,))

        # Refresh measured motor positions and velocities proxies
        self.q_measured, self.v_measured = (
            self.env.sensor_measurements[encoder.type])

        # Reset the command state
        fill(self._command_state, 0)

    @property
    def fieldnames(self) -> List[str]:
        return [f"target{N_ORDER_DERIVATIVE_NAMES[self.order]}{name}"
                for name in self.env.robot.motor_names]

    def get_state(self) -> np.ndarray:
        return self._command_state[:2]

    def compute_command(self, action: np.ndarray) -> np.ndarray:
        """Compute the motor torques using a PD controller.

        It is proportional to the error between the observed motors positions/
        velocities and the target ones.

        .. warning::
            Calling this method manually while a simulation is running is
            forbidden, because it would mess with the controller update period.

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

        # Extract motor positions and velocity from encoder data
        q_measured, v_measured = self.q_measured, self.v_measured
        if not self._is_same_order:
            q_measured = q_measured[self.encoder_to_motor]
            v_measured = v_measured[self.encoder_to_motor]

        # Compute the motor efforts using PD control.
        # The command state must not be updated if no simulation is running.
        return pd_controller(
            q_measured,
            v_measured,
            action,
            self.order,
            self._command_state,
            self._command_state_lower,
            self._command_state_upper,
            self.kp,
            self.kd,
            self.motors_effort_limit,
            self.control_dt if is_simulation_running else 0.0,
            self._u_command)
