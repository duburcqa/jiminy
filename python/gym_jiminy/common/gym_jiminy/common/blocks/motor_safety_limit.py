"""Implementation of basic Proportional-Derivative controller block compatible
with gym_jiminy reinforcement learning pipeline environment design.
"""
from typing import List

import numpy as np
import numba as nb

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    EncoderSensor as encoder)

from .proportional_derivative_controller import get_encoder_to_motor_map

from ..bases import (BaseObsT,
                     JiminyEnvInterface,
                     BaseControllerBlock,
                     BasePipelineWrapper,
                     ControlledJiminyEnv)


@nb.jit(nopython=True, nogil=True, cache=True)
def apply_safety_limits(command: np.ndarray,
                        q_measured: np.ndarray,
                        v_measured: np.ndarray,
                        kp: np.ndarray,
                        kd: np.ndarray,
                        motors_soft_position_lower: np.ndarray,
                        motors_soft_position_upper: np.ndarray,
                        motors_velocity_limit: np.ndarray,
                        motors_effort_limit: np.ndarray
                        ) -> np.ndarray:
    """Clip the command torque to ensure safe operation.

    It acts on each actuator independently and only activate close to the
    position or velocity limits. Basically, the idea to the avoid moving faster
    when some prescribed velocity limit or exceeding soft position bounds by
    forcing the command torque to act against it. Still, it may not be enough
    to prevent such issue in practice as the command torque is bounded.

    .. warning::
        All the input arguments must be in motor order, including the measured
        position and velocity of the actuators. In practice, those measurements
        comes from the encoder sensors. If so, then the measurements must be
        re-ordered if necessary to match motor order instead of sensor order.

    .. seealso::
        See `MotorSafetyLimit` documentation for details.

    :param command: Desired command to will be updated in-place if necessary.
    :param q_measured: Current position of the actuators.
    :param v_measured: Current velocity of the actuators.
    :param kp: Scale of the velocity bound triggered by position limits.
    :param kd: Scale of the effort bound triggered by velocity limits.
    :param motors_soft_position_lower:
        Soft lower position limit of the actuators.
    :param motors_soft_position_upper:
        Soft upper position limit of the actuators.
    :param motors_velocity_limit: Maximum velocity of the actuators.
    :param motors_effort_limit: Maximum effort that the actuators can output.
                                The command torque cannot exceed this limits,
                                not even if needed to enforce safe operation.
    """
    # Computes velocity bounds based on margin from soft joint limit if any
    safe_velocity_lower = motors_velocity_limit * np.clip(
        -kp * (q_measured - motors_soft_position_lower), -1.0, 1.0)
    safe_velocity_upper = motors_velocity_limit * np.clip(
        -kp * (q_measured - motors_soft_position_upper), -1.0, 1.0)

    # Computes effort bounds based on velocity and effort bounds
    safe_effort_lower = motors_effort_limit * np.clip(
        -kd * (v_measured - safe_velocity_lower), -1.0, 1.0)
    safe_effort_upper = motors_effort_limit * np.clip(
        -kd * (v_measured - safe_velocity_upper), -1.0, 1.0)

    # Clip command according to safe effort bounds
    return np.clip(command, safe_effort_lower, safe_effort_upper)


class MotorSafetyLimit(
        BaseControllerBlock[np.ndarray, np.ndarray, BaseObsT, np.ndarray]):
    """Safety mechanism primarily designed to prevent hardware damage and
    premature wear, but also to temper violent, sporadic and dangerous motions.

    A velocity limit v+ is enforced by bounding the commanded effort such that
    no effort can be applied to push the joint beyond the velocity limit, and
    a damping effort is applied if the joint is moving at a velocity beyond
    the limit, ie -kd * (v - v+).

    When the joint is near the soft limits x+/-, the velocities are bounded to
    keep the position from crossing the soft limits. The k_position term
    determines the scale of the bound on velocity, ie v+/- = -kp * (x - x+/-).
    These bounds on velocity are the ones determining the bounds on effort.

    .. note::
        The prescribed position and velocity limits may be more respective that
        the actual hardware specification of the robot.

    .. warning::
        It must be connected directly to the environment without any other
        intermediary controller.

    .. seealso::
        See official ROS documentation and implementation for details:
        https://wiki.ros.org/pr2_controller_manager/safety_limits
        https://github.com/PR2/pr2_mechanism/blob/melodic-devel/pr2_mechanism_model/src/joint.cpp
    """  # noqa: E501  # pylint: disable=line-too-long
    def __init__(self,
                 name: str,
                 env: JiminyEnvInterface[BaseObsT, np.ndarray],
                 *,
                 kp: float,
                 kd: float,
                 soft_position_margin: float = 0.0,
                 soft_velocity_max: float = float("inf")) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.

        :param kp: Scale of the velocity bound triggered by position limits.
        :param kd: Scale of the effort bound triggered by velocity limits.
        :param soft_position_margin: Minimum distance of the current motor
                                     positions from their respective bounds
                                     before starting to break.
        :param soft_velocity_max: Maximum velocity of the motor before
                                  starting to break.
        """
        # Make sure that no other controller has been added prior to this block
        env_unwrapped: JiminyEnvInterface = env
        while isinstance(env_unwrapped, BasePipelineWrapper):
            if isinstance(env_unwrapped, ControlledJiminyEnv):
                raise TypeError(
                    "No other control block must be added prior to this one.")
            env_unwrapped = env_unwrapped.env

        # Backup some user argument(s)
        self.kp = kp
        self.kd = kd

        # Define buffers storing information about the motors for efficiency
        motors_position_idx: List[int] = sum(env.robot.motors_position_idx, [])
        self.motors_position_lower = env.robot.position_limit_lower[
            motors_position_idx] + soft_position_margin
        self.motors_position_upper = env.robot.position_limit_upper[
            motors_position_idx] - soft_position_margin
        self.motors_velocity_limit = np.minimum(env.robot.velocity_limit[
            env.robot.motors_velocity_idx], soft_velocity_max)
        self.motors_effort_limit = env.robot.command_limit[
            env.robot.motors_velocity_idx]
        self.motors_effort_limit[
            self.motors_position_lower > self.motors_position_upper] = 0.0

        # Extract measured motor positions and velocities for fast access
        self.q_measured, self.v_measured = env.sensors_data[encoder.type]

        # Mapping from motors to encoders
        self.encoder_to_motor = get_encoder_to_motor_map(env.robot)

        # Whether stored reference to encoder measurements are already in the
        # same order as the motors, allowing skipping re-ordering entirely.
        self._is_same_order = isinstance(self.encoder_to_motor, slice)

        # Initialize the controller
        super().__init__(name, env, 1)

    def _initialize_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the command motors efforts.
        """
        self.action_space = self.env.action_space

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Refresh measured motor positions and velocities proxies
        self.q_measured, self.v_measured = self.env.sensors_data[encoder.type]
        self.q_measured, self.v_measured = self.env.sensors_data[
            encoder.type][:, self.encoder_to_motor]

    @property
    def fieldnames(self) -> List[str]:
        return [f"currentMotorTorque{name}"
                for name in self.env.robot.motors_names]

    def compute_command(self, action: np.ndarray) -> np.ndarray:
        """Apply safety limits to the desired motor torques right before
        sending it to the robot so as to avoid exceeded prescribed position
        and velocity limits.

        :param action: Desired motor torques to apply on the robot.
        """
        # Extract motor positions and velocity from encoder data
        q_measured, v_measured = self.q_measured, self.v_measured
        if not self._is_same_order:
            q_measured = q_measured[self.encoder_to_motor]
            v_measured = v_measured[self.encoder_to_motor]

        # Clip command according to safe effort bounds
        return apply_safety_limits(action,
                                   q_measured,
                                   v_measured,
                                   self.kp,
                                   self.kd,
                                   self.motors_position_lower,
                                   self.motors_position_upper,
                                   self.motors_velocity_limit,
                                   self.motors_effort_limit)
