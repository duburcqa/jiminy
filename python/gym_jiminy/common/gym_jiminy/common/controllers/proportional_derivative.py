"""Implementation of basic Proportional-Derivative controller block compatible
with gym_jiminy reinforcement learning pipeline environment design.
"""
from collections import OrderedDict
from typing import Union, Any, List

import numpy as np
import numba as nb
import gym

from jiminy_py.core import EncoderSensor as encoder

from ..bases import BaseControllerBlock
from ..envs import BaseJiminyEnv
from ..utils import SpaceDictNested, FieldDictNested


@nb.jit(nopython=True, nogil=True)
def _compute_command_impl(q_target: np.ndarray,
                          v_target: np.ndarray,
                          encoders_data: np.ndarray,
                          motor_to_encoder: np.ndarray,
                          pid_kp: np.ndarray,
                          pid_kd: np.ndarray) -> np.ndarray:
    """Implementation of PD control law.

    .. note::
        Used internally by `PDController` to compute command but separated to
        allow precompilation. It is not meant to be called manually.

    :meta private:
    """
    # Estimate position and motor velocity from encoder data
    q_measured, v_measured = encoders_data[:, motor_to_encoder]

    # Compute the joint tracking error
    q_error = q_measured - q_target
    v_error = v_measured - v_target

    # Compute PD command
    return - pid_kp * (q_error + pid_kd * v_error)


class PDController(BaseControllerBlock):
    """Low-level Proportional-Derivative controller.

    .. warning::
        It must be connected directly to the environment to control without
        any intermediary controllers.
    """
    def __init__(self,
                 env: BaseJiminyEnv,
                 update_ratio: int = 1,
                 pid_kp: Union[float, List[float], np.ndarray] = 0.0,
                 pid_kd: Union[float, List[float], np.ndarray] = 0.0,
                 **kwargs: Any) -> None:
        """
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
        :param pid_kp: PD controller position-proportional gain in motor order.
        :param pid_kd: PD controller velocity-proportional gain in motor order.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup some user arguments
        self.pid_kp = np.asarray(pid_kp)
        self.pid_kd = np.asarray(pid_kd)

        # Define the mapping from motors to encoders
        encoder_joints = []
        for name in env.robot.sensors_names[encoder.type]:
            sensor = env.robot.get_sensor(encoder.type, name)
            encoder_joints.append(sensor.joint_name)

        motor_to_encoder = []
        for name in env.robot.motors_names:
            motor = env.robot.get_motor(name)
            motor_joint = motor.joint_name
            encoder_found = False
            for i, encoder_joint in enumerate(encoder_joints):
                if motor_joint == encoder_joint:
                    motor_to_encoder.append(i)
                    encoder_found = True
                    break
            if not encoder_found:
                raise RuntimeError(
                    f"No encoder sensor associated with motor '{name}'. Every "
                    "actuated joint must have an encoder sensor attached.")
        self.motor_to_encoder = np.array(motor_to_encoder)

        # Initialize the controller
        super().__init__(env, update_ratio)

    def _refresh_action_space(self) -> None:
        """Configure the action space of the controller.

        The action spaces corresponds to the position and velocity of motors
        instead of the torque.
        """
        # Extract the position and velocity bounds for the observation space
        sensors_space = self.env._get_sensors_space()
        encoders_space = sensors_space[encoder.type]
        pos_high, vel_high = encoders_space.high
        pos_low, vel_low = encoders_space.low

        # Reorder the position and velocity bounds to match motors order
        pos_high = pos_high[self.motor_to_encoder]
        pos_low = pos_low[self.motor_to_encoder]
        vel_high = vel_high[self.motor_to_encoder]
        vel_low = vel_low[self.motor_to_encoder]

        # Set the action space. Note that it is flattened.
        self.action_space = gym.spaces.Dict(OrderedDict(
            Q=gym.spaces.Box(
                low=pos_low, high=pos_high, dtype=np.float64),
            V=gym.spaces.Box(
                low=vel_low, high=vel_high, dtype=np.float64)))

    def get_fieldnames(self) -> FieldDictNested:
        pos_fieldnames = [f"targetPosition{name}"
                          for name in self.robot.motors_names]
        vel_fieldnames = [f"targetVelocity{name}"
                          for name in self.robot.motors_names]
        return OrderedDict(Q=pos_fieldnames, V=vel_fieldnames)

    def compute_command(self,
                        measure: SpaceDictNested,
                        action: SpaceDictNested
                        ) -> np.ndarray:
        """Compute the motor torques using a PD controller.

        It is proportional to the error between the measured motors positions/
        velocities and the target ones.

        :param measure: Observation of the environment.
        :param action: Desired motors positions and velocities as a dictionary.
        """
        return _compute_command_impl(
            q_target=action['Q'], v_target=action['V'],
            encoders_data=measure['sensors'][encoder.type],
            motor_to_encoder=self.motor_to_encoder,
            pid_kp=self.pid_kp, pid_kd=self.pid_kd)
