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
    EncoderSensor, array_copyto)

from ..bases import BaseObs, InterfaceJiminyEnv, BaseControllerBlock
from ..utils import zeros, fill


# Name of the n-th position derivative
N_ORDER_DERIVATIVE_NAMES = ("Position", "Velocity", "Acceleration")


@nb.jit(nopython=True, cache=True, inline='always', fastmath=True)
def integrate_zoh(state: np.ndarray,
                  state_min: np.ndarray,
                  state_max: np.ndarray,
                  dt: float) -> None:
    """Second order approximate integration scheme assuming Zero-Order-Hold for
    the acceleration, taking position, velocity and acceleration bounds into
    account.

    Internally, it simply chains two first-order integrators in cascade. The
    acceleration will be updated in-place if clipping is necessary to satisfy
    bounds.

    :param state: Current state, ordered from lowest to highest derivative
                  order, ie: s[i](t) = s[i](t-1) + integ_{t-1}^{t}(s[i+1](t)),
                  as a 2D array whose first dimension gathers the 3 derivative
                  orders. It will be updated in-place.
    :param state_min: Lower bounds of the state as a 2D array.
    :param state_max: Upper bounds of the state as a 2D array.
    :param dt: Integration delta of time since previous state update.
    """
    # Make sure that dt is not negative
    assert dt >= 0.0, "Integration backward in time is not supported."

    # Early return if timestep is too small
    if abs(dt) < 1e-9:
        return

    # Split position, velocity and acceleration for convenience
    position, velocity, acceleration = state

    # Loop over motors individually as it is faster than masked vectorization
    _, dim = state.shape
    for i in range(dim):
        # Split position, velocity and acceleration bounds
        position_min, velocity_min, acceleration_min = state_min[:, i]
        position_max, velocity_max, acceleration_max = state_max[:, i]

        # Clip acceleration
        acceleration[i] = min(
            max(acceleration[i], acceleration_min), acceleration_max)

        # Backup the initial velocity to later compute the clipped acceleration
        velocity_prev = velocity[i]

        # Integrate acceleration 1-step ahead
        velocity[i] += acceleration[i] * dt

        # Make sure that "true" velocity bounds are satisfied
        velocity[i] = min(max(velocity[i], velocity_min), velocity_max)

        # Force slowing down early enough to avoid violating acceleration
        # limits when hitting position bounds.
        horizon = max(
            int(abs(velocity_prev) / acceleration_max / dt) * dt, dt)
        position_min_delta = position_min - position[i]
        position_max_delta = position_max - position[i]
        if horizon > dt:
            drift = 0.5 * (horizon * (horizon - dt)) * acceleration_max
            position_min_delta -= drift
            position_max_delta += drift
        velocity_min = position_min_delta / horizon
        velocity_max = position_max_delta / horizon
        velocity[i] = min(max(velocity[i], velocity_min), velocity_max)

        # Velocity after hitting bounds must be cancellable in a single step
        if np.abs(velocity[i]) > dt * acceleration_max:
            velocity_min = - max(
                position_min_delta / velocity[i], dt) * acceleration_max
            velocity_max = max(
                position_max_delta / velocity[i], dt) * acceleration_max
            velocity[i] = min(max(velocity[i], velocity_min), velocity_max)

        # Back-propagate velocity clipping at the acceleration-level
        acceleration[i] = (velocity[i] - velocity_prev) / dt

        # Integrate position 1-step ahead
        position[i] += dt * velocity[i]


@nb.jit(nopython=True, cache=True, fastmath=True)
def pd_controller(q_measured: np.ndarray,
                  v_measured: np.ndarray,
                  command_state: np.ndarray,
                  command_state_lower: np.ndarray,
                  command_state_upper: np.ndarray,
                  kp: np.ndarray,
                  kd: np.ndarray,
                  motors_effort_limit: np.ndarray,
                  control_dt: float,
                  out: np.ndarray) -> None:
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
    :param command_state: Current command state, namely, all the derivatives of
                          the target motors positions up to acceleration order.
    :param command_state_lower: Lower bound of the command state that must be
                                satisfied at all cost.
    :param command_state_upper: Upper bound of the command state that must be
                                satisfied at all cost.
    :param kp: PD controller position-proportional gain in motor order.
    :param kd: PD controller velocity-proportional gain in motor order.
    :param motors_effort_limit: Maximum effort that the actuators can output.
    :param control_dt: Controller update period. It will be involved in the
                       integration of the command state.
    :param out: Pre-allocated memory to store the command motor torques.
    """
    # Integrate target motor positions and velocities
    integrate_zoh(command_state,
                  command_state_lower,
                  command_state_upper,
                  control_dt)

    # Extract targets motors positions and velocities from command state
    q_target, v_target, _ = command_state

    # Compute the joint tracking error
    q_error, v_error = q_target - q_measured, v_target - v_measured

    # Compute PD command
    out[:] = kp * (q_error + kd * v_error)

    # Clip the command motors torques before returning
    out[:] = np.minimum(np.maximum(
        out, -motors_effort_limit), motors_effort_limit)


@nb.jit(nopython=True, cache=True, fastmath=True)
def pd_adapter(action: np.ndarray,
               order: int,
               command_state: np.ndarray,
               command_state_lower: np.ndarray,
               command_state_upper: np.ndarray,
               dt: float,
               out: np.ndarray) -> None:
    """Compute the target motor accelerations that must be held constant for a
    given time interval in order to reach the desired value of some derivative
    of the target motor positions at the end of that interval if possible.

    Internally, it applies backward in time the same integration scheme as the
    PD controller. Knowing the initial and final value of the derivative over
    the time interval, constant target motor accelerations can be uniquely
    deduced. In practice, it consists in successive clipped finite difference
    of that derivative up to acceleration-level.

    :param action: Desired value of the n-th derivative of the command motor
                   positions at the end of the controller update.
    :param order: Derivative order of the position associated with the action.
    :param command_state: Current command state, namely, all the derivatives of
                          the target motors positions up to acceleration order.
    :param command_state_lower: Lower bound of the command state that must be
                                satisfied at all cost.
    :param command_state_upper: Upper bound of the command state that must be
                                satisfied at all cost.
    :param dt: Time interval during which the target motor accelerations will
               be held constant.
    :param out: Pre-allocated memory to store the target motor accelerations.
    """
    # Update command accelerations based on the action and its derivative order
    if order == 2:
        # The action corresponds to the command motor accelerations
        out[:] = action
    else:
        if order == 0:
            # Compute command velocity
            velocity = (action - command_state[0]) / dt

            # Clip command velocity
            velocity = np.minimum(np.maximum(
                velocity, command_state_lower[1]), command_state_upper[1])
        else:
            # The action corresponds to the command motor velocities
            velocity = action

        # Compute command acceleration
        out[:] = (velocity - command_state[1]) / dt


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
    # Loop over all actuated joints and look for their encoder counterpart
    encoder_to_motor_map = [-1,] * robot.nmotors
    encoders = robot.sensors[EncoderSensor.type]
    for motor in robot.motors:
        for i, sensor in enumerate(encoders):
            assert isinstance(sensor, EncoderSensor)
            if motor.index == sensor.motor_index:
                encoder_to_motor_map[sensor.index] = motor.index
                del encoders[i]
                break
        else:
            raise RuntimeError(
                f"No encoder sensor associated with motor '{motor.name}'. "
                "Every actuated joint must have a encoder sensor attached on "
                "motor side.")

    # Try converting it to slice if possible
    if (np.array(encoder_to_motor_map) == np.arange(robot.nmotors)).all():
        return slice(None)

    return encoder_to_motor_map


class PDController(
        BaseControllerBlock[np.ndarray, np.ndarray, BaseObs, np.ndarray]):
    """Low-level Proportional-Derivative controller.

    The action are the target motors accelerations. The latter are integrated
    twice using two first-order integrators in cascade, considering that the
    acceleration is constant until the next controller update:

        v_{t+1} = v_{t} + dt * a_{t}
        q_{t+1} = q_{t} + dt * v_{t+1}

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
                 env: InterfaceJiminyEnv[BaseObs, np.ndarray],
                 *,
                 update_ratio: int = 1,
                 kp: Union[float, List[float], np.ndarray],
                 kd: Union[float, List[float], np.ndarray],
                 joint_position_margin: float = 0.0,
                 joint_velocity_limit: float = float("inf"),
                 joint_acceleration_limit: float = float("inf")) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller. -1 to
                             match the simulation timestep of the environment.
                             Optional: 1 by default.
        :param kp: PD controller position-proportional gains in motor order.
        :param kd: PD controller velocity-proportional gains in motor order.
        :param joint_position_margin: Minimum distance of the joint target
                                      positions from their respective bounds.
                                      Optional: 0.0 by default.
        :param joint_velocity_limit: Restrict maximum joint target velocities
                                     wrt their hardware specifications.
                                     Optional: "inf" by default.
        :param joint_acceleration_limit: Maximum joint target acceleration.
                                         Optional: "inf" by default.
        """
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
        self.kp = kp
        self.kd = kd

        # Mapping from motors to encoders
        self.encoder_to_motor_map = get_encoder_to_motor_map(env.robot)

        # Whether stored reference to encoder measurements are already in the
        # same order as the motors, allowing skipping re-ordering entirely.
        self._is_same_order = isinstance(self.encoder_to_motor_map, slice)
        if not self._is_same_order:
            warnings.warn(
                "Consider using the same ordering for encoders and motors for "
                "optimal performance.")

        # Define buffers storing information about the motors for efficiency.
        # Note that even if the robot instance may change from one simulation
        # to another, the observation and action spaces are required to stay
        # the same the whole time. This induces that the motors effort limit
        # must not change unlike the mapping from full state to motors.
        self.motors_effort_limit = np.array([
            motor.effort_limit for motor in env.robot.motors])

        # Refresh mechanical reduction ratio.
        # FIXME: Is it considered invariant ? If not, it should be refreshed in
        # `_setup`, as done for `DeformationEstimator`.
        encoder_to_joint_ratio = []
        for motor in env.robot.motors:
            motor_options = motor.get_options()
            encoder_to_joint_ratio.append(motor_options["mechanicalReduction"])

        # Define the motors target position bounds
        motors_position_lower = np.array([
            motor.position_limit_lower + ratio * joint_position_margin
            for motor, ratio in zip(env.robot.motors, encoder_to_joint_ratio)])
        motors_position_upper = np.array([
            motor.position_limit_upper - ratio * joint_position_margin
            for motor, ratio in zip(env.robot.motors, encoder_to_joint_ratio)])

        # Define the motors target velocity bounds
        motors_velocity_limit = np.array([
            min(motor.velocity_limit, ratio * joint_velocity_limit)
            for motor, ratio in zip(env.robot.motors, encoder_to_joint_ratio)])

        # Define acceleration bounds allowing unrestricted bang-bang control
        range_limit = 2 * motors_velocity_limit / env.step_dt
        effort_limit = self.motors_effort_limit / (
            self.kp * env.step_dt * np.maximum(env.step_dt / 2, self.kd))
        target_acceleration_limit = np.array([
            ratio * joint_acceleration_limit
            for ratio in encoder_to_joint_ratio])
        acceleration_limit = np.minimum(
            np.minimum(range_limit, effort_limit), target_acceleration_limit)

        # Compute command state bounds
        self._command_state_lower = np.stack([motors_position_lower,
                                              -motors_velocity_limit,
                                              -acceleration_limit], axis=0)
        self._command_state_upper = np.stack([motors_position_upper,
                                              motors_velocity_limit,
                                              acceleration_limit], axis=0)

        # Extract measured motor positions and velocities for fast access.
        # Note that they will be initialized in `_setup` method.
        self.q_measured, self.v_measured = np.array([]), np.array([])

        # Allocate memory for the command state
        self._command_state = np.zeros((3, env.robot.nmotors))

        # Initialize the controller
        super().__init__(name, env, update_ratio)

        # Make sure that the state is within bounds
        self._command_state[:2] = zeros(self.state_space)

        # References to command acceleration for fast access
        self._command_acceleration = self._command_state[2]

    def _initialize_state_space(self) -> None:
        """Configure the state space of the controller.

        The state spaces corresponds to the target motors positions and
        velocities.
        """
        self.state_space = gym.spaces.Box(
            low=self._command_state_lower[:2],
            high=self._command_state_upper[:2],
            dtype=np.float64)

    def _initialize_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the target motors accelerations.
        """
        self.action_space = gym.spaces.Box(
            low=self._command_state_lower[2],
            high=self._command_state_upper[2],
            dtype=np.float64)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Make sure control update is discrete-time
        if self.env.control_dt <= 0.0:
            raise ValueError(
                "This block does not support time-continuous update.")

        # Refresh measured motor positions and velocities proxies
        self.q_measured, self.v_measured = (
            self.env.sensor_measurements[EncoderSensor.type])

        # Reset the command state
        fill(self._command_state, 0)

    @property
    def fieldnames(self) -> List[str]:
        return [f"currentTarget{N_ORDER_DERIVATIVE_NAMES[2]}{motor.name}"
                for motor in self.env.robot.motors]

    def get_state(self) -> np.ndarray:
        return self._command_state[:2]

    def compute_command(self, action: np.ndarray, command: np.ndarray) -> None:
        """Compute the target motor torques using a PD controller.

        It is proportional to the error between the observed motors positions/
        velocities and the target ones.

        .. warning::
            Calling this method manually while a simulation is running is
            forbidden, because it would mess with the controller update period.

        :param action: Desired target motor acceleration.
        :param command: Current motor torques that will be updated in-place.
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
            q_measured = q_measured[self.encoder_to_motor_map]
            v_measured = v_measured[self.encoder_to_motor_map]

        # Update target motor accelerations
        array_copyto(self._command_acceleration, action)

        # Compute the motor efforts using PD control.
        # The command state must not be updated if no simulation is running.
        pd_controller(
            q_measured,
            v_measured,
            self._command_state,
            self._command_state_lower,
            self._command_state_upper,
            self.kp,
            self.kd,
            self.motors_effort_limit,
            self.control_dt if is_simulation_running else 0.0,
            command)


class PDAdapter(
        BaseControllerBlock[np.ndarray, np.ndarray, BaseObs, np.ndarray]):
    """Adapt the action of a lower-level Proportional-Derivative controller
    to be the target motor positions or velocities rather than accelerations.

    The action is the desired value of some derivative of the target motor
    positions. The target motor accelerations are then deduced so as to reach
    this value by the next update of this controller if possible without
    exceeding the position, velocity and acceleration bounds. Finally, these
    target position accelerations are passed to a lower-level PD controller,
    usually running at a higher frequency.

    .. note::
        The higher the derivative order of the action, the smoother the command
        motor torques. Thus, a high order is generally beneficial for robotic
        applications. However, it introduces some kind of delay between the
        action and its effects. This phenomenon limits the responsiveness of
        the agent and therefore impedes its optimal performance. Besides, it
        introduces addition layer of indirection between the action and its
        effect which may be difficult to grasp for the agent. Finally,
        exploration usually consists in addition temporally uncorrelated
        gaussian random process at action-level. The effect of such random
        processes tends to vanish when integrated, making exploration very
        inefficient.
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObs, np.ndarray],
                 *,
                 update_ratio: int = -1,
                 order: int = 1) -> None:
        """
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller. -1 to
                             match the environment step `env.step_dt`.
                             Optional: -1 by default.
        :param order: Derivative order of the action. It accepts position or
                      velocity (respectively 0 or 1).
                      Optional: 1 by default.
        """
        # Make sure that the specified derivative order is valid
        assert order in (0, 1), "Derivative order out-of-bounds"

        # Make sure that a PD controller block is already connected
        controller = env.controller  # type: ignore[attr-defined]
        if not isinstance(controller, PDController):
            raise RuntimeError(
                "This block must be directly connected to a lower-level "
                "`PDController` block.")

        # Backup some user argument(s)
        self.order = order

        # Define some proxies for convenience
        self._pd_controller = controller

        # Initialize the controller
        super().__init__(name, env, update_ratio)

    def _initialize_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the N-th order derivative of the
        target motors positions.
        """
        self.action_space = gym.spaces.Box(
            low=self._pd_controller._command_state_lower[self.order],
            high=self._pd_controller._command_state_upper[self.order],
            dtype=np.float64)

    @property
    def fieldnames(self) -> List[str]:
        return [f"nextTarget{N_ORDER_DERIVATIVE_NAMES[self.order]}{motor.name}"
                for motor in self.env.robot.motors]

    def compute_command(self, action: np.ndarray, command: np.ndarray) -> None:
        """Compute the target motor accelerations from the desired value of
        some derivative of the target motor positions.

        :param action: Desired target motor acceleration.
        :param command: Current motor torques that will be updated in-place.
        """
        pd_adapter(
            action,
            self.order,
            self._pd_controller._command_state,
            self._pd_controller._command_state_lower,
            self._pd_controller._command_state_upper,
            self.control_dt,
            command)
