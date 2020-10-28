import numpy as np
from typing import Optional, Union, Tuple, Dict, Any

import gym

import jiminy_py.core as jiminy
from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)


SpaceDictRecursive = Union[Dict[str, 'SpaceDictRecursive'], np.ndarray]


# Define universal scale for the observation and action space
FREEFLYER_POS_TRANS_SCALE = 1.0
FREEFLYER_VEL_LIN_SCALE = 1.0
FREEFLYER_VEL_ANG_SCALE = 1.0
JOINT_POS_SCALE = 1.0
JOINT_VEL_SCALE = 1.0
FLEX_VEL_ANG_SCALE = 1.0
MOTOR_EFFORT_SCALE = 100.0
SENSOR_GYRO_SCALE = 1.0
SENSOR_ACCEL_SCALE = 10.0
T_SCALE = 1.0


class ObservationActionNormalization(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """   TODO
        """
        super().__init__(env)
        self.observation_scale = None
        self.action_scale = None

    def _refresh_observation_space(self) -> None:
        """   TODO
        """
        self.observation_scale = {}

        # Define some proxies for convenience
        sensors_data = self.robot.sensors_data
        model_options = self.robot.get_model_options()
        num_sensors = {}
        for sensor in (encoder, effort, contact, force, imu):
            num_sensors[sensor] = len(self.robot.sensors_names[sensor.type])

        # ======= Compute robot configuration and velocity scale =======

        # Extract pre-defined scale from the robot
        position_scale = (self.robot.position_limit_upper -
                          self.robot.position_limit_lower) / 2
        velocity_scale = self.robot.velocity_limit

        # Replace inf bounds by the appropriate scale
        if self.robot.has_freeflyer:
            position_scale[:3] = FREEFLYER_POS_TRANS_SCALE
            velocity_scale[:3] = FREEFLYER_VEL_LIN_SCALE
            velocity_scale[3:6] = FREEFLYER_VEL_ANG_SCALE

        for jointIdx in self.robot.flexible_joints_idx:
            jointVelIdx = self.robot.pinocchio_model.joints[jointIdx].idx_v
            velocity_scale[jointVelIdx + np.arange(3)] = FLEX_VEL_ANG_SCALE

        if not model_options['joints']['enablePositionLimit']:
            position_scale[
                self.robot.rigid_joints_position_idx] = JOINT_POS_SCALE

        if not model_options['joints']['enableVelocityLimit']:
            velocity_scale[
                self.robot.rigid_joints_velocity_idx] = JOINT_VEL_SCALE

        # ======= Compute motors efforts scale =======

        # Extract pre-defined scale from the robot
        effort_scale = self.robot.effort_limit

        # Replace inf bounds by the appropriate scale
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_scale[motor.joint_velocity_idx] = MOTOR_EFFORT_SCALE

        # Keep only the actual motor effort
        effort_scale = effort_scale[self.robot.motors_velocity_idx]

        # ======= Update observation space scale =======

        # Handling of time scale
        if 't' in self.observation_space.spaces.keys():
            self.observation_scale['t'] = T_SCALE

        # Handling of state scale
        if 'state' in self.observation_space.spaces.keys():
            self.observation_scale['state'] = np.concatenate(
                (position_scale, velocity_scale))

        # Handling of sensors data scale
        if 'sensors' in self.observation_space.spaces.keys():
            sensors_space = self.observation_space['sensors']

            # Handling of encoders data scale
            self.observation_scale['sensors'] = {}
            if encoder.type in sensors_space.spaces.keys():
                enc_sensors_scale = np.full(
                    (len(encoder.fieldnames), num_sensors[encoder]),
                    np.inf, dtype=np.float64)

                # Replace inf bounds by the appropriate scale
                for sensor_name in self.robot.sensors_names[encoder.type]:
                    # Get the sensor scaling.
                    # Note that for rotary unbounded encoders, the bounds
                    # cannot be extracted from the configuration vector scale,
                    # but rather are known in advance.
                    sensor = self.robot.get_sensor(encoder.type, sensor_name)
                    sensor_idx = sensor.idx
                    joint = self.robot.pinocchio_model.joints[sensor.joint_idx]
                    if sensor.joint_type == jiminy.joint_t.ROTARY_UNBOUNDED:
                        sensor_position_scale = 2 * np.pi
                    else:
                        sensor_position_scale = position_scale[joint.idx_q]
                    sensor_velocity_scale = velocity_scale[joint.idx_v]

                    # Update the scale accordingly
                    enc_sensors_scale[0, sensor_idx] = sensor_position_scale
                    enc_sensors_scale[1, sensor_idx] = sensor_velocity_scale

                # Set the scale
                self.observation_scale['sensors'][encoder.type] = enc_sensors_scale

            # Handling of IMUs data scale
            if imu.type in sensors_space.spaces.keys():
                imu_sensors_scale = np.zeros(
                    (len(imu.fieldnames), num_sensors[imu]),
                    dtype=sensors_data[imu.type].dtype)

                # Set the quaternion scale
                quat_imu_idx = [field.startswith('Quat')
                                for field in imu.fieldnames]
                imu_sensors_scale[quat_imu_idx] = np.full(
                    (sum(quat_imu_idx), num_sensors[imu]), 1.0)

                # Set the gyroscope scale
                gyro_imu_idx = [field.startswith('Gyro')
                                for field in imu.fieldnames]
                imu_sensors_scale[gyro_imu_idx] = np.full(
                    (sum(gyro_imu_idx), num_sensors[imu]), SENSOR_GYRO_SCALE)

                # Set the accelerometer scale
                accel_imu_idx = [field.startswith('Accel')
                                 for field in imu.fieldnames]
                imu_sensors_scale[accel_imu_idx] = np.full(
                    (sum(accel_imu_idx), num_sensors[imu]), SENSOR_ACCEL_SCALE)

                # Set the scale
                self.observation_scale['sensors'][imu.type] = imu_sensors_scale

            # Handling of Contact sensors data scale
            if contact.type in sensors_space.spaces.keys():
                total_mass = self.robot.pinocchio_data_th.mass[0]
                gravity = - self.robot.pinocchio_model_th.gravity.linear[2]
                total_weight = total_mass * gravity

                contact_sensors_scale = np.full(
                    (len(contact.fieldnames), num_sensors[contact]),
                    total_weight, dtype=sensors_data[contact.type].dtype)

                self.observation_scale['sensors'][contact.type] = \
                    contact_sensors_scale

            # Handling of Force sensors data scale
            if force.type in sensors_space.spaces.keys():
                force_sensors_scale = np.zeros(
                    (len(force.fieldnames), num_sensors[force]),
                    dtype=sensors_data[force.type].dtype)

                total_mass = self.robot.pinocchio_data_th.mass[0]
                gravity = - self.robot.pinocchio_model_th.gravity.linear[2]
                total_weight = total_mass * gravity

                # Set the linear force scale
                lin_force_idx = [field.startswith('F')
                                 for field in force.fieldnames]
                force_sensors_scale[lin_force_idx] = np.full(
                    (sum(lin_force_idx), num_sensors[force]), total_weight)

                # Set the moment scale
                # TODO: Defining the moment scale using 'total_weight' does not
                # really make sense.
                moment_idx = [field.startswith('M')
                              for field in force.fieldnames]
                force_sensors_scale[moment_idx] = np.full(
                    (sum(moment_idx), num_sensors[force]), total_weight)

                self.observation_scale['sensors'][force.type] = \
                    force_sensors_scale

            # Handling of Effort sensors data scale
            if effort.type in sensors_space.spaces.keys():
                effort_sensors_scale = np.zeros(
                    (len(effort.fieldnames), num_sensors[effort]),
                    dtype=sensors_data[effort.type].dtype)

                sensor_list = self.robot.sensors_names[effort.type]
                for sensor_name in sensor_list:
                    sensor = self.robot.get_sensor(effort.type, sensor_name)
                    sensor_idx = sensor.idx
                    motor_idx = sensor.motor_idx
                    effort_sensors_scale[0, sensor_idx] = \
                        effort_scale[motor_idx]

                self.observation_scale['sensors'][effort.type] = effort_scale

        # ======= Update action space scale =======

        self.action_scale = effort_scale

    def _refresh_action_space(self) -> None:
        """   TODO
        """
        # Extract pre-defined scale from the robot
        effort_scale = self.robot.effort_limit

        # Replace inf bounds by the appropriate scale
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_scale[motor.joint_velocity_idx] = MOTOR_EFFORT_SCALE

        # Keep only the actual motor effort
        self.action_scale = effort_scale[self.robot.motors_velocity_idx]

    def reset(self, **kwargs) -> SpaceDictRecursive:
        """   TODO
        """
        obs = self.env.reset(**kwargs)
        self._refresh_learning_spaces_scale()
        obs_n = self.normalize(obs, self.observation_scale)
        return obs_n

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """   TODO

        @param action  Normalized action.
        """
        if action is not None:
            action = self.reverse_normalize(action, self.action_scale)
        obs, reward, done, info = self.env.step(action)
        obs_n = self.normalize(obs, self.observation_scale)
        return obs_n, reward, done, info

    @classmethod
    def normalize(cls,
                  value: SpaceDictRecursive,
                  scale: SpaceDictRecursive) -> SpaceDictRecursive:
        """   TODO
        """
        if isinstance(scale, dict):
            value_n = {}
            for k, v in value.items():
                value_n[k] = cls.normalize(v, scale[k])
            return value_n
        else:
            return value / scale

    @classmethod
    def reverse_normalize(cls,
                          value_n: SpaceDictRecursive,
                          scale:  SpaceDictRecursive) -> SpaceDictRecursive:
        """   TODO
        """
        if isinstance(scale, dict):
            value = {}
            for k, v in value_n.items():
                value[k] = cls.reverse_normalize(v, scale[k])
            return value
        else:
            return value_n * scale
