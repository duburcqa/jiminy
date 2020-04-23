## @file

"""
@package    gym_jiminy

@brief      Package containing python-native helper methods for Gym Jiminy Open Source.
"""

import os
import numpy as np

from gym import core, spaces
from gym.utils import seeding

from pinocchio import neutral
from jiminy_py import core as jiminy
from jiminy_py.core import EncoderSensor as enc, \
                           EffortSensor as effort, \
                           ForceSensor as force, \
                           ImuSensor as imu
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.engine_asynchronous import EngineAsynchronous

from . import RenderOutMock


# Define universal bounds for the observation space
FREEFLYER_POS_TRANS_UNIVERSAL_MAX = 1000.0
FREEFLYER_VEL_LIN_UNIVERSAL_MAX = 1000.0
FREEFLYER_VEL_ANG_UNIVERSAL_MAX = 10000.0
JOINT_POS_UNIVERSAL_MAX = 10000.0
JOINT_VEL_UNIVERSAL_MAX = 100.0
FLEX_VEL_ANG_UNIVERSAL_MAX = 10000.0
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

        ## Jiminy engine associated with the robot (used for physics computations)
        self.engine_py = engine_py
        self.robot = engine_py.robot

        ## Update period of the simulation
        self.dt = dt

        ## Configure the action and observation spaces
        self.action_space = None
        self.observation_space = None
        self._refresh_learning_spaces()

        ## State of the robot
        self.state = None
        self.sensor_data = None
        self._viewer = None

        ## Number of simulation steps performed after having met the stopping criterion
        self.steps_beyond_done = None

        self.seed()

        # ######### Enforce some options by default for the robot and the engine #########

        robot_options = self.robot.get_options()
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

        self.robot.set_options(robot_options)
        self.engine_py.set_engine_options(engine_options)

    def _refresh_learning_spaces(self):
        ## Define some proxies for convenience

        sensor_data = self.robot.sensors_data
        model_options = self.robot.get_model_options()

        ## Extract some information about the robot

        position_limit_upper = self.robot.position_limit_upper
        position_limit_lower = self.robot.position_limit_lower
        velocity_limit = self.robot.velocity_limit
        effort_limit = self.robot.effort_limit

        # Replace inf bounds by the appropriate universal bound for the state space
        if self.robot.has_freeflyer:
            position_limit_lower[:3] = -FREEFLYER_POS_TRANS_UNIVERSAL_MAX
            position_limit_upper[:3] = +FREEFLYER_POS_TRANS_UNIVERSAL_MAX
            velocity_limit[:3] = FREEFLYER_VEL_LIN_UNIVERSAL_MAX
            velocity_limit[3:6] = FREEFLYER_VEL_ANG_UNIVERSAL_MAX

        for jointIdx in self.robot.flexible_joints_idx:
            jointVelIdx = self.robot.pinocchio_model.joints[jointIdx].idx_v
            velocity_limit[jointVelIdx + np.arange(3)] = FLEX_VEL_ANG_UNIVERSAL_MAX

        if not model_options['joints']['enablePositionLimit']:
            position_limit_lower[self.robot.rigid_joints_position_idx] = -JOINT_POS_UNIVERSAL_MAX
            position_limit_upper[self.robot.rigid_joints_position_idx] = +JOINT_POS_UNIVERSAL_MAX

        if not model_options['joints']['enableVelocityLimit']:
            velocity_limit[self.robot.rigid_joints_velocity_idx] = JOINT_VEL_UNIVERSAL_MAX

        # Replace inf bounds by the appropriate universal bound for the action space
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_limit[motor.joint_velocity_idx] = MOTOR_EFFORT_MAX

        ## Action space

        action_low  = -effort_limit[self.robot.motors_velocity_idx]
        action_high = +effort_limit[self.robot.motors_velocity_idx]

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        ## Sensor space

        sensor_space_raw = {key: {'min': np.full(value.shape, -np.inf),
                                  'max': np.full(value.shape, np.inf)}
                            for key, value in self.robot.sensors_data.items()}

        # Replace inf bounds by the appropriate universal bound for the Encoder sensors
        if enc.type in sensor_data.keys():
            sensor_list = self.robot.sensors_names[enc.type]
            for sensor_name in sensor_list:
                sensor_idx = self.robot.get_sensor(enc.type, sensor_name).idx
                pos_idx = self.robot.get_sensor(enc.type, sensor_name).joint_position_idx
                sensor_space_raw[enc.type]['min'][0, sensor_idx] = position_limit_lower[pos_idx]
                sensor_space_raw[enc.type]['max'][0, sensor_idx] = position_limit_upper[pos_idx]
                vel_idx = self.robot.get_sensor(enc.type, sensor_name).joint_velocity_idx
                sensor_space_raw[enc.type]['min'][1, sensor_idx] = -velocity_limit[vel_idx]
                sensor_space_raw[enc.type]['max'][1, sensor_idx] = +velocity_limit[vel_idx]

        # Replace inf bounds by the appropriate universal bound for the Effort sensors
        if effort.type in sensor_data.keys():
            sensor_list = self.robot.sensors_names[effort.type]
            for sensor_name in sensor_list:
                sensor_idx = self.robot.get_sensor(effort.type, sensor_name).idx
                motor_idx = self.robot.get_sensor(effort.type, sensor_name).motor_idx
                sensor_space_raw[effort.type]['min'][0, sensor_idx] = action_low[motor_idx]
                sensor_space_raw[effort.type]['max'][0, sensor_idx] = action_high[motor_idx]

        # Replace inf bounds by the appropriate universal bound for the Force sensors
        if force.type in sensor_data.keys():
            sensor_space_raw[force.type]['min'][:, :] = -SENSOR_FORCE_UNIVERSAL_MAX
            sensor_space_raw[force.type]['max'][:, :] = +SENSOR_FORCE_UNIVERSAL_MAX

        # Replace inf bounds by the appropriate universal bound for the IMU sensors
        if imu.type in sensor_data.keys():
            quat_imu_indices = ['Quat' in field for field in imu.fieldnames]
            sensor_space_raw[imu.type]['min'][quat_imu_indices, :] = -1.0
            sensor_space_raw[imu.type]['max'][quat_imu_indices, :] = 1.0

            gyro_imu_indices = ['Gyro' in field for field in imu.fieldnames]
            sensor_space_raw[imu.type]['min'][gyro_imu_indices, :] = -SENSOR_GYRO_UNIVERSAL_MAX
            sensor_space_raw[imu.type]['max'][gyro_imu_indices, :] = +SENSOR_GYRO_UNIVERSAL_MAX

            accel_imu_indices = ['Accel' in field for field in imu.fieldnames]
            sensor_space_raw[imu.type]['min'][accel_imu_indices, :] = -SENSOR_ACCEL_UNIVERSAL_MAX
            sensor_space_raw[imu.type]['max'][accel_imu_indices, :] = +SENSOR_ACCEL_UNIVERSAL_MAX

        sensor_space = spaces.Dict(
            {key: spaces.Box(low=value["min"], high=value["max"], dtype=np.float64)
             for key, value in sensor_space_raw.items()})

        ## Observation space
        state_limit_lower = np.concatenate((position_limit_lower, -velocity_limit))
        state_limit_upper = np.concatenate((position_limit_upper, +velocity_limit))

        self.observation_space = spaces.Dict(
            state = spaces.Box(low=state_limit_lower, high=state_limit_upper, dtype=np.float64),
            sensor = sensor_space)


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
        self.sensor_data = self.engine_py.sensor_data
        return [seed]

    def _sample_state(self):
        """
        @brief      Returns a random valid initial state.

        @details    The default implementation only return the neural configuration,
                    with offsets on the freeflyer to enforce that the frame associated
                    with the first contact point is aligned with the canonical frame
                    and touches the ground.
        """
        q0 = neutral(self.robot.nq)
        if self.robot.has_freeflyer:
            compute_freeflyer_state_from_fixed_body(
                self.robot, self.robot.contact_frames_names[0], q0,
                use_theoretical_model=False)
            groundFct = self.engine_py._engine.get_options()['world']['groundProfile']
            q0[2] += groundFct(np.zeros(3))[0]
        v0 = np.zeros(self.robot.nv)
        x0 = np.concatenate((q0, v0))
        return x0

    def reset(self):
        """
        @brief      Reset the simulation.

        @details    The initial state is randomly sampled.

        @return     Initial state of the simulation
        """
        self.state = self._sample_state()
        self.sensor_data = None
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

        @details    By default, it corresponds to the state and the sensors' data.
        """
        return (self.state, self.sensor_data)


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
