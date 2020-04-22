## @file

"""
@package    gym_jiminy

@brief      Package containing python-native helper methods for Gym Jiminy Open Source.
"""

import os
import numpy as np

from gym import core, spaces
from gym.utils import seeding

from jiminy_py import core as jiminy
from jiminy_py.engine_asynchronous import EngineAsynchronous

from . import RenderOutMock


# Define universal bounds for the observation space
FREEFLYER_POS_TRANS_UNIVERSAL_MAX = np.inf
FREEFLYER_VEL_LIN_UNIVERSAL_MAX = 1000.0
FREEFLYER_VEL_ANG_UNIVERSAL_MAX = 10000.0
JOINT_POS_UNIVERSAL_MAX = np.inf
JOINT_VEL_UNIVERSAL_MAX = 100.0
MOTOR_EFFORT_MAX = 1000.0
SENSOR_FORCE_UNIVERSAL_MAX = 100000.0
SENSOR_GYRO_UNIVERSAL_MAX = 100.0
SENSOR_ACCEL_UNIVERSAL_MAX = 10000.0


class RobotJiminyEnv(core.Env):
    """
    @brief      Base class to train a robot in Gym OpenAI using a user-specified
                Python Jiminy engine for physics computations.

                It creates an Gym environment wrapping Jiminy Engine and behaves
                like any other Gym environment.

    @details    The Python Jiminy engine must be completely initialized beforehand,
                which means that the Jiminy Robot and Controller are already setup.
                For now, the only engine available is `EngineAsynchronous`.
    """

    ## Metadata of the environment
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, robot_name : str, engine_py : EngineAsynchronous, dt : float):
        """
        @brief      Constructor

        @param[in]  robot_name  Name of the robot
        @param[in]  engine_py   Python Jiminy engine used for physics computations.
                                It must be completely initialized. For now, the
                                only engine available is `EngineAsynchronous`.
        @param[in]  dt          Desired update period of the simulation

        @return     Instance of the environment.
        """

        # ###################### Configure the learning environment ######################

        ## Name of the robot
        self.robot_name = robot_name

        ## Jiminy engine associated with the robot. It is used for physics computations.
        self.engine_py = engine_py

        ## Update period of the simulation
        self.dt = dt

        ## Configure the action and observation spaces
        self.action_space = None
        self.observation_space = None
        self._refresh_learning_spaces()

        ## State of the robot
        self.state = None
        self._viewer = None

        ## Number of simulation steps performed after having met the stopping criterion
        self.steps_beyond_done = None

        self.seed()

        # ######### Enforce some options by default for the robot and the engine #########

        robot_options = self.engine_py._engine.robot.get_options()
        engine_options = self.engine_py.get_engine_options()

        # Disable completely the telemetry to speed up the simulation
        for field in robot_options["telemetry"].keys():
            robot_options["telemetry"][field] = False
        for field in engine_options["telemetry"].keys():
            engine_options["telemetry"][field] = False

        # Set the position and velocity bounds of the robot
        robot_options["model"]["joints"]["enablePositionLimit"] = True
        robot_options["model"]["joints"]["enableVelocityLimit"] = True

        # Set the effort limits of the motors
        for motor_name in robot_options["motors"].keys():
            robot_options["motors"][motor_name]["enableEffortLimit"] = True

        # Configure the stepper update period and disable max number of iterations
        engine_options["stepper"]["iterMax"] = -1
        engine_options["stepper"]["sensorsUpdatePeriod"] = self.dt
        engine_options["stepper"]["controllerUpdatePeriod"] = self.dt

        self.engine_py._engine.robot.set_options(robot_options)
        self.engine_py.set_engine_options(engine_options)

    def _refresh_learning_spaces(self):
        ## Define some proxies for convenience
        robot = self.engine_py._engine.robot
        enc_t = jiminy.EncoderSensor.type
        effort_t = jiminy.EffortSensor.type
        force_t = jiminy.ForceSensor.type
        imu_t = jiminy.ImuSensor.type

        ## Extract some information from the robot
        position_limit_upper = robot.position_limit_upper
        position_limit_lower = robot.position_limit_lower
        velocity_limit = robot.velocity_limit
        if robot.has_freeflyer:
            position_limit_lower[:3] = -FREEFLYER_POS_TRANS_UNIVERSAL_MAX
            position_limit_upper[:3] = FREEFLYER_POS_TRANS_UNIVERSAL_MAX
            velocity_limit[:3] = FREEFLYER_VEL_LIN_UNIVERSAL_MAX

        ## Action space
        action_low = -robot.effort_limit[robot.motors_velocity_idx]
        action_high = robot.effort_limit[robot.motors_velocity_idx]

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        ## Observation space
        sensor_data = robot.sensors_data

        obs_space_dict_raw = {key: {'min': np.full(value.shape, -np.inf),
                                    'max': np.full(value.shape, np.inf)}
                              for key, value in robot.sensors_data.items()}

        obs_space_dict_raw['state'] = {'min': np.concatenate((position_limit_lower, -velocity_limit)),
                                       'max': np.concatenate((position_limit_upper, velocity_limit))}

        if enc_t in sensor_data.keys(): # Special treatment for the Encoder sensors
            sensor_list = robot.sensors_names[enc_t]
            for sensor_name in sensor_list:
                sensor_idx = robot.get_sensor(enc_t, sensor_name).idx
                if robot.get_model_options()['joints']['enablePositionLimit']:
                    joint_pos_idx = robot.get_sensor(enc_t, sensor_name).joint_position_idx
                    obs_space_dict_raw[enc_t]['min'][0, sensor_idx] = position_limit_lower[joint_pos_idx]
                    obs_space_dict_raw[enc_t]['max'][0, sensor_idx] = position_limit_upper[joint_pos_idx]
                else:
                    obs_space_dict_raw[enc_t]['min'][1, :] = -JOINT_POS_UNIVERSAL_MAX
                    obs_space_dict_raw[enc_t]['max'][1, :] = JOINT_POS_UNIVERSAL_MAX
                if robot.get_model_options()['joints']['enableVelocityLimit']:
                    joint_vel_idx = robot.get_sensor(enc_t, sensor_name).joint_velocity_idx
                    obs_space_dict_raw[enc_t]['min'][1, sensor_idx] = -velocity_limit[joint_vel_idx]
                    obs_space_dict_raw[enc_t]['max'][1, sensor_idx] = velocity_limit[joint_vel_idx]
                else:
                    obs_space_dict_raw[enc_t]['min'][1, :] = -JOINT_VEL_UNIVERSAL_MAX
                    obs_space_dict_raw[enc_t]['max'][1, :] = JOINT_VEL_UNIVERSAL_MAX

        if effort_t in sensor_data.keys(): # Special treatment for the Effort sensors
            sensor_list = robot.sensors_names[effort_t]
            for sensor_name in sensor_list:
                sensor_idx = robot.get_sensor(effort_t, sensor_name).idx
                motor_idx = robot.get_sensor(effort_t, sensor_name).motor_idx
                if not np.isinf(action_low[motor_idx]):
                    obs_space_dict_raw[effort_t]['min'][0, sensor_idx] = action_low[motor_idx]
                    obs_space_dict_raw[effort_t]['max'][0, sensor_idx] = action_high[motor_idx]
                else:
                    obs_space_dict_raw[effort_t]['min'][0, sensor_idx] = -MOTOR_EFFORT_MAX
                    obs_space_dict_raw[effort_t]['min'][0, sensor_idx] = MOTOR_EFFORT_MAX

        if force_t in sensor_data.keys(): # Special treatment for the Force sensors
            obs_space_dict_raw[force_t]['min'][:, :] = -SENSOR_FORCE_UNIVERSAL_MAX
            obs_space_dict_raw[force_t]['max'][:, :] = SENSOR_FORCE_UNIVERSAL_MAX

        if imu_t in sensor_data.keys(): # Special treatment for the IMU sensors
            # The quaternion is normalized
            quat_imu_indices = ['Quat' in field for field in jiminy.ImuSensor.fieldnames]
            obs_space_dict_raw[imu_t]['min'][quat_imu_indices, :] = -1.0
            obs_space_dict_raw[imu_t]['max'][quat_imu_indices, :] = 1.0

            gyro_imu_indices = ['Gyro' in field for field in jiminy.ImuSensor.fieldnames]
            obs_space_dict_raw[imu_t]['min'][gyro_imu_indices, :] = -SENSOR_GYRO_UNIVERSAL_MAX
            obs_space_dict_raw[imu_t]['max'][gyro_imu_indices, :] = SENSOR_GYRO_UNIVERSAL_MAX

            accel_imu_indices = ['Accel' in field for field in jiminy.ImuSensor.fieldnames]
            obs_space_dict_raw[imu_t]['min'][accel_imu_indices, :] = -SENSOR_ACCEL_UNIVERSAL_MAX
            obs_space_dict_raw[imu_t]['max'][accel_imu_indices, :] = SENSOR_ACCEL_UNIVERSAL_MAX

        self.observation_space = spaces.Dict(
            {key: spaces.Box(low=value["min"], high=value["max"], dtype=np.float64)
             for key, value in obs_space_dict_raw.items()})

    def seed(self, seed=None):
        """
        @brief      Specify the seed of the simulation.

        @details    One must reset the simulation after updating the seed because
                    otherwise the behavior is undefined as it is not part of the
                    specification for Python Jiminy engines.

        @param[in]  seed    Desired seed as a Unsigned Integer 32bit
                            Optional: The seed will be randomly generated using np if omitted.

        @return     Updated seed of the simulation
        """
        self.np_random, seed = seeding.np_random(seed)
        self.engine_py.seed(seed)
        self.state = self.engine_py.state
        return [seed]

    def _sample_state(self):
        """
        @brief      Returns a random valid initial state.
        """
        raise NotImplementedError()

    def reset(self):
        """
        @brief      Reset the simulation.

        @details    The initial state is randomly sampled.

        @return     Initial state of the simulation
        """
        self.state = self._sample_state()
        self.engine_py.reset(self.state)
        self.steps_beyond_done = None
        return self._get_obs()

    def render(self, mode=None, lock=None, **kwargs):
        """
        @brief      Render the current state of the robot in Gepetto-viewer.

        @details    Do not suport Multi-Rendering RGB output because it is not
                    possible to create window in new tabs programmatically in
                    Gepetto viewer.

        @param[in]  mode    Unused. Defined for compatibility with Gym OpenAI.
        @param[in]  lock    Unique threading.Lock for every simulation
                            Optional: Only required for parallel rendering

        @return     Fake output for compatibility with Gym OpenAI.
        """

        self.engine_py.render(return_rgb_array=False, lock=lock, **kwargs)
        return RenderOutMock()

    def close(self):
        """
        @brief      Terminate the Python Jiminy engine. Mostly defined for
                    compatibility with Gym OpenAI.
        """
        self.engine_py.close()

    def _get_obs(self):
        """
        @brief      Returns the observation.
        """
        raise NotImplementedError()


class RobotJiminyGoalEnv(RobotJiminyEnv, core.GoalEnv):
    """
    @brief      Base class to train a robot in Gym OpenAI using a user-specified
                Jiminy Engine for physics computations.

                It creates an Gym environment wrapping Jiminy Engine and behaves
                like any other Gym goal-environment.

    @details    The Jiminy Engine must be completely initialized beforehand, which
                means that the Jiminy Robot and Controller are already setup.
    """
    def __init__(self, robot_name : str, engine_py : EngineAsynchronous, dt : float):
        """
        @brief      Constructor

        @param[in]  robot_name  Name of the robot
        @param[in]  engine_py   Python Jiminy engine used for physics computations.
                                It must be completely initialized. For now, the
                                only engine available is `EngineAsynchronous`.
        @param[in]  dt          Desired update period of the simulation

        @return     Instance of the environment.
        """

        ## @var observation_space
        # @copydoc RobotJiminyEnv::observation_space

        super().__init__(robot_name, engine_py, dt)

        ## Current goal
        self.goal = self._sample_goal()

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float64),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype=np.float64),
            observation=self.observation_space)

    def reset(self):
        """
        @brief      Reset the simulation.

        @details    Sample a new goal, then call `RobotJiminyEnv.reset`.
        .
        @remark     See documentation of `RobotJiminyEnv` for details.

        @return     Initial state of the simulation
        """
        self.goal = self._sample_goal().copy()
        return super().reset()

    def _sample_goal(self):
        """
        @brief      Samples a new goal and returns it.
        """
        raise NotImplementedError()
