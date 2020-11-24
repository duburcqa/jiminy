""" TODO: Write documentation.
"""
from collections import OrderedDict
from typing import Union, Any, List

import numpy as np
import gym

from jiminy_py.core import EncoderSensor as encoder

from ..bases import BaseControllerBlock
from ..envs import BaseJiminyEnv
from ..utils import SpaceDictNested, FieldDictNested


class PDController(BaseControllerBlock):
    """Low-level Proportional-Derivative controller.

    .. warning::
        It must be connected directly to the environment to control without
        intermediary controllers.
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

        self.motor_to_encoder = []
        for name in env.robot.motors_names:
            motor = env.robot.get_motor(name)
            motor_joint = motor.joint_name
            encoder_found = False
            for i, encoder_joint in enumerate(encoder_joints):
                if motor_joint == encoder_joint:
                    self.motor_to_encoder.append(i)
                    encoder_found = True
                    break
            if not encoder_found:
                raise RuntimeError(
                    f"No encoder sensor associated with motor '{name}'. Every "
                    "actuated joint must have an encoder sensor attached.")

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
        self.action_space = gym.spaces.Dict([
            (encoder.fieldnames[0], gym.spaces.Box(
                low=pos_low, high=pos_high, dtype=np.float64)),
            (encoder.fieldnames[1], gym.spaces.Box(
                low=vel_low, high=vel_high, dtype=np.float64))])

    def get_fieldnames(self) -> FieldDictNested:
        pos_fieldnames = [f"targetPosition{name}"
                          for name in self.robot.motors_names]
        vel_fieldnames = [f"targetVelocity{name}"
                          for name in self.robot.motors_names]
        return OrderedDict([(encoder.fieldnames[0], pos_fieldnames),
                            (encoder.fieldnames[1], vel_fieldnames)])

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
        # Update the internal state of the controller
        q_target = action[encoder.fieldnames[0]]
        v_target = action[encoder.fieldnames[1]]

        # Estimate position and motor velocity from encoder data
        encoders_data = measure['sensors'][encoder.type]
        q_measured, v_measured = encoders_data[:, self.motor_to_encoder]

        # Compute the joint tracking error
        q_error = q_measured - q_target
        v_error = v_measured - v_target

        # Compute PD command
        return - self.pid_kp * (q_error + self.pid_kd * v_error)
