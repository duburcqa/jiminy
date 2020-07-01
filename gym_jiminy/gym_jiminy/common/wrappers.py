import numpy as np
from functools import reduce
from operator import __mul__

from gym import Wrapper, ObservationWrapper, spaces

from jiminy_py.core import EncoderSensor as enc, \
                           EffortSensor as effort, \
                           ForceSensor as force, \
                           ImuSensor as imu


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


def flatten_observation(space, x=None):
    # Note that it does not preserve dtype
    def _flatten_bounds(space, bounds_type):
        if isinstance(space, spaces.Box):
            if bounds_type == 'high':
                return np.asarray(space.high).flatten()
            else:
                return np.asarray(space.low).flatten()
        elif isinstance(space, spaces.Discrete):
            if bounds_type == 'high':
                return np.one(space.n)
            else:
                return np.zeros(space.n)
        elif isinstance(space, spaces.Tuple):
            return np.concatenate([_flatten_bounds(s, bounds_type)
                                   for s in space.spaces])
        elif isinstance(space, spaces.Dict):
            return np.concatenate([_flatten_bounds(s, bounds_type)
                                   for s in space.spaces.values()])
        elif isinstance(space, spaces.MultiBinary):
            if bounds_type == 'high':
                return np.one(space.n)
            else:
                return np.zeros(space.n)
        elif isinstance(space, spaces.MultiDiscrete):
            if bounds_type == 'high':
                return np.one(reduce(__mul__, space.nvec))
            else:
                return np.zeros(reduce(__mul__, space.nvec))
        else:
            raise NotImplementedError
    if x is None:
        return spaces.Box(low=_flatten_bounds(space, 'low'),
                          high=_flatten_bounds(space, 'high'),
                          dtype=np.float64)
    else:
        return spaces.flatten(space, x)


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_observation(
            self.env.observation_space)

    def observation(self, observation):
        return flatten_observation(
            self.env.observation_space, observation)


class ObservationActionNormalization(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_scale = None
        self.action_scale = None

    def _refresh_learning_spaces_scale(self):
        self.observation_scale = {}

        ## Define some proxies for convenience
        sensors_data = self.engine_py.sensors_data
        model_options = self.robot.get_model_options()

        ## Compute the full robot configuration and velocity scale

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
            position_scale[self.robot.rigid_joints_position_idx] = JOINT_POS_SCALE

        if not model_options['joints']['enableVelocityLimit']:
            velocity_scale[self.robot.rigid_joints_velocity_idx] = JOINT_VEL_SCALE

        ## Compute the robot motor effort scale

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

        ## Handling of time scale
        if 't' in self.observation_space.spaces.keys():
            self.observation_scale['t'] = T_SCALE

        ## Handling of state scale
        if 'state' in self.observation_space.spaces.keys():
            self.observation_scale['state'] = np.concatenate(
                (position_scale, velocity_scale))

        ## Handling of sensors data scale
        if 'sensors' in self.observation_space.spaces.keys():
            sensors_space = self.observation_space['sensors']

            # Handling of encoders data scale
            self.observation_scale['sensors'] = {}
            if enc.type in sensors_space.spaces.keys():
                enc_sensors_scale = np.full(
                    (len(enc.fieldnames), len(self.robot.sensors_names[enc.type])),
                    np.inf, dtype=np.float64)

                # Replace inf bounds by the appropriate scale
                for sensor_name in self.robot.sensors_names[enc.type]:
                    sensor = self.robot.get_sensor(enc.type, sensor_name)
                    sensor_idx = sensor.idx
                    pos_idx = sensor.joint_position_idx
                    vel_idx = sensor.joint_velocity_idx
                    enc_sensors_scale[0, sensor_idx] = position_scale[pos_idx]
                    enc_sensors_scale[1, sensor_idx] = velocity_scale[vel_idx]

                # Set the scale
                self.observation_scale['sensors'][enc.type] = enc_sensors_scale

            # Handling of IMUs data scale
            if imu.type in sensors_space.spaces.keys():
                imu_sensors_scale = np.zeros(
                    (len(imu.fieldnames), len(self.robot.sensors_names[imu.type])),
                    dtype=sensors_data[imu.type].dtype)

                # Set the quaternion scale
                quat_imu_idx = ['Quat' in field for field in imu.fieldnames]
                imu_sensors_scale[quat_imu_idx] = np.full(
                    (sum(quat_imu_idx), len(self.robot.sensors_names[imu.type])),
                    1.0)

                # Set the gyroscope scale
                gyro_imu_idx = ['Gyro' in field for field in imu.fieldnames]
                imu_sensors_scale[gyro_imu_idx] = np.full(
                    (sum(gyro_imu_idx), len(self.robot.sensors_names[imu.type])),
                    SENSOR_GYRO_SCALE)

                # Set the accelerometer scale
                accel_imu_idx = ['Accel' in field for field in imu.fieldnames]
                imu_sensors_scale[accel_imu_idx] = np.full(
                    (sum(accel_imu_idx), len(self.robot.sensors_names[imu.type])),
                    SENSOR_ACCEL_SCALE)

                # Set the scale
                self.observation_scale['sensors'][imu.type] = imu_sensors_scale

            # Handling of Force sensors data scale
            if force.type in sensors_space.spaces.keys():
                total_mass = self.robot.pinocchio_data_th.mass[0]
                gravity = - self.robot.pinocchio_model_th.gravity.linear[2]
                total_weight = total_mass * gravity

                force_sensors_scale = np.full(
                    (len(force.fieldnames), len(self.robot.sensors_names[force.type])),
                    total_weight, dtype=sensors_data[force.type].dtype)

                self.observation_scale['sensors'][force.type] = force_sensors_scale

            # Handling of Effort sensors data scale
            if effort.type in sensors_space.spaces.keys():
                effort_sensors_scale = np.zeros(
                    (len(effort.fieldnames), len(self.robot.sensors_names[effort.type])),
                    dtype=sensors_data[effort.type].dtype)

                sensor_list = self.robot.sensors_names[effort.type]
                for sensor_name in sensor_list:
                    sensor = self.robot.get_sensor(effort.type, sensor_name)
                    sensor_idx = sensor.idx
                    motor_idx = sensor.motor_idx
                    effort_sensors_scale[0, sensor_idx] = effort_scale[motor_idx]

                self.observation_scale['sensors'][effort.type] = effort_scale

        ## Handling of action scale
        self.action_scale = effort_scale

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._refresh_learning_spaces_scale()
        obs_n = self.normalize(obs, self.observation_scale)
        return obs_n

    def step(self, action_n):
        action = self.reverse_normalize(action_n, self.action_scale)
        obs, reward, done, info = self.env.step(action)
        obs_n = self.normalize(obs, self.observation_scale)
        return obs_n, reward, done, info

    @classmethod
    def normalize(cls, value, scale):
        if isinstance(scale, dict):
            value_n = {}
            for k, v in value.items():
                value_n[k] = cls.normalize(v, scale[k])
            return value_n
        else:
            return value / scale

    @classmethod
    def reverse_normalize(cls, value_n, scale):
        if isinstance(scale, dict):
            value = {}
            for k, v in value_n.items():
                value[k] = cls.reverse_normalize(v, scale[k])
            return value
        else:
            return value_n * scale
