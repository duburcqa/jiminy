## @file

"""
@package    gym_jiminy

@brief      Package containing python-native helper methods for Gym Jiminy Open Source.
"""

import time
import numpy as np

from gym import core, spaces, logger
from gym.utils import seeding

from pinocchio import neutral
from jiminy_py.core import EncoderSensor as enc, \
                           EffortSensor as effort, \
                           ForceSensor as force, \
                           ImuSensor as imu
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.engine_asynchronous import EngineAsynchronous
from jiminy_py.viewer import sleep

from .render_out_mock import RenderOutMock
from .play import loop_interactive


# Define universal bounds for the observation space
FREEFLYER_POS_TRANS_UNIVERSAL_MAX = 1000.0
FREEFLYER_VEL_LIN_UNIVERSAL_MAX = 1000.0
FREEFLYER_VEL_ANG_UNIVERSAL_MAX = 10000.0
JOINT_POS_UNIVERSAL_MAX = 10000.0
JOINT_VEL_UNIVERSAL_MAX = 100.0
FLEX_VEL_ANG_UNIVERSAL_MAX = 10000.0
MOTOR_EFFORT_UNIVERSAL_MAX = 1000.0
SENSOR_FORCE_UNIVERSAL_MAX = 100000.0
SENSOR_GYRO_UNIVERSAL_MAX = 100.0
SENSOR_ACCEL_UNIVERSAL_MAX = 10000.0
T_UNIVERSAL_MAX = 10000.0


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

    def __init__(self,
                 robot_name: str,
                 engine_py: EngineAsynchronous,
                 dt: float,
                 debug: bool = False):
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
        self.rg = np.random.RandomState()
        self._seed = None
        self.dt = dt
        self.debug = debug

        ## Configure the action and observation spaces
        self.action_space = None
        self.observation_space = None

        ## Current observation of the robot
        self.is_running = False
        self.observation = None
        self.action_prev = None

        ## Information about the learning process
        self.learning_info = {'is_success': False}

        ## Number of simulation steps performed after having met the stopping criterion
        self._steps_beyond_done = None

        # ############################# Initialize the engine ############################

        ## Set the seed of the simulation and reset the simulation
        self.seed()
        self.reset()

    def _setup_environment(self):
        # Enforce some options by default for the robot and the engine

        robot_options = self.robot.get_options()
        engine_options = self.engine_py.get_engine_options()

        ### Disable completely the telemetry in non debug mode to speed up the simulation
        for field in robot_options["telemetry"].keys():
            robot_options["telemetry"][field] = self.debug
        for field in engine_options["telemetry"].keys():
            if field[:6] == 'enable':
                engine_options["telemetry"][field] = self.debug

        ### Set the position and velocity bounds of the robot
        robot_options["model"]["joints"]["enablePositionLimit"] = True
        robot_options["model"]["joints"]["enableVelocityLimit"] = True

        ### Set the effort limits of the motors
        for motor_name in robot_options["motors"].keys():
            robot_options["motors"][motor_name]["enableEffortLimit"] = True

        ### Configure the stepper update period, and disable max number of iterations and timeout
        engine_options["stepper"]["iterMax"] = -1
        engine_options["stepper"]["timeout"] = -1
        engine_options["stepper"]["sensorsUpdatePeriod"] = self.dt
        engine_options["stepper"]["controllerUpdatePeriod"] = self.dt
        engine_options["stepper"]["logInternalStepperSteps"] = self.debug

        ### Set the seed
        engine_options["stepper"]["randomSeed"] = self._seed

        self.robot.set_options(robot_options)
        self.engine_py.set_engine_options(engine_options)

    def _refresh_learning_spaces(self):
        ## Define some proxies for convenience
        sensors_data = self.engine_py.sensors_data
        model_options = self.robot.get_model_options()

        ## Extract some proxies
        joints_position_idx = self.robot.rigid_joints_position_idx
        joints_velocity_idx = self.robot.rigid_joints_velocity_idx
        motors_velocity_idx = self.robot.motors_velocity_idx
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
            velocity_limit[jointVelIdx + np.arange(3)] = \
                FLEX_VEL_ANG_UNIVERSAL_MAX

        if not model_options['joints']['enablePositionLimit']:
            position_limit_lower[joints_position_idx] = \
                -JOINT_POS_UNIVERSAL_MAX
            position_limit_upper[joints_position_idx] = \
                +JOINT_POS_UNIVERSAL_MAX
        else:
            # Joint bounds are not hard bounds, so margins need to be added
            position_limit_lower[joints_position_idx] -= 5.0e-2
            position_limit_upper[joints_position_idx] += 5.0e-2

        if not model_options['joints']['enableVelocityLimit']:
            velocity_limit[joints_velocity_idx] = JOINT_VEL_UNIVERSAL_MAX
        else:
            velocity_limit[joints_velocity_idx] += 2.0e-0

        # Replace inf bounds by the appropriate universal bound for the action space
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_limit[motor.joint_velocity_idx] = \
                    MOTOR_EFFORT_UNIVERSAL_MAX

        ## Action space
        action_low  = -effort_limit[motors_velocity_idx]
        action_high = +effort_limit[motors_velocity_idx]

        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float64)

        ## Sensor space
        sensor_space_raw = {
            key: {'min': np.full(value.shape, -np.inf),
                  'max': np.full(value.shape, np.inf)}
            for key, value in self.engine_py.sensors_data.items()
        }

        # Replace inf bounds by the appropriate universal bound for the Encoder sensors
        if enc.type in sensors_data.keys():
            sensor_list = self.robot.sensors_names[enc.type]
            for sensor_name in sensor_list:
                sensor = self.robot.get_sensor(enc.type, sensor_name)
                sensor_idx = sensor.idx
                pos_idx = sensor.joint_position_idx
                sensor_space_raw[enc.type]['min'][0, sensor_idx] = \
                    position_limit_lower[pos_idx]
                sensor_space_raw[enc.type]['max'][0, sensor_idx] = \
                    position_limit_upper[pos_idx]
                vel_idx = sensor.joint_velocity_idx
                sensor_space_raw[enc.type]['min'][1, sensor_idx] = \
                    - velocity_limit[vel_idx]
                sensor_space_raw[enc.type]['max'][1, sensor_idx] = \
                    velocity_limit[vel_idx]

        # Replace inf bounds by the appropriate universal bound for the Effort sensors
        if effort.type in sensors_data.keys():
            sensor_list = self.robot.sensors_names[effort.type]
            for sensor_name in sensor_list:
                sensor = self.robot.get_sensor(effort.type, sensor_name)
                sensor_idx = sensor.idx
                motor_idx = sensor.motor_idx
                sensor_space_raw[effort.type]['min'][0, sensor_idx] = \
                    action_low[motor_idx]
                sensor_space_raw[effort.type]['max'][0, sensor_idx] = \
                    action_high[motor_idx]

        # Replace inf bounds by the appropriate universal bound for the Force sensors
        if force.type in sensors_data.keys():
            sensor_space_raw[force.type]['min'][:,:] = \
                -SENSOR_FORCE_UNIVERSAL_MAX
            sensor_space_raw[force.type]['max'][:,:] = \
                +SENSOR_FORCE_UNIVERSAL_MAX

        # Replace inf bounds by the appropriate universal bound for the IMU sensors
        if imu.type in sensors_data.keys():
            quat_imu_idx = ['Quat' in field for field in imu.fieldnames]
            sensor_space_raw[imu.type]['min'][quat_imu_idx,:] = -1.0
            sensor_space_raw[imu.type]['max'][quat_imu_idx,:] = 1.0

            gyro_imu_idx = ['Gyro' in field for field in imu.fieldnames]
            sensor_space_raw[imu.type]['min'][gyro_imu_idx,:] = \
                -SENSOR_GYRO_UNIVERSAL_MAX
            sensor_space_raw[imu.type]['max'][gyro_imu_idx,:] = \
                +SENSOR_GYRO_UNIVERSAL_MAX

            accel_imu_idx = ['Accel' in field for field in imu.fieldnames]
            sensor_space_raw[imu.type]['min'][accel_imu_idx,:] = \
                -SENSOR_ACCEL_UNIVERSAL_MAX
            sensor_space_raw[imu.type]['max'][accel_imu_idx,:] = \
                +SENSOR_ACCEL_UNIVERSAL_MAX

        sensor_space = spaces.Dict({
            key: spaces.Box(
                low=value["min"], high=value["max"], dtype=np.float64)
            for key, value in sensor_space_raw.items()
        })

        ## Observation space
        state_limit_lower = np.concatenate(
            (position_limit_lower, -velocity_limit))
        state_limit_upper = np.concatenate(
            (position_limit_upper, velocity_limit))

        self.observation_space = spaces.Dict(
            t = spaces.Box(
                low=0.0,
                high=T_UNIVERSAL_MAX,
                shape=(1,), dtype=np.float64),
            state = spaces.Box(
                low=state_limit_lower,
                high=state_limit_upper,
                dtype=np.float64),
            sensors = sensor_space
        )

        self.observation = {'t': None, 'state': None, 'sensors': None}

    def _sample_state(self):
        """
        @brief      Returns a random valid initial state.

        @details    The default implementation only return the neural configuration,
                    with offsets on the freeflyer to ensure no contact points are
                    going through the ground and a single one is touching it.
        """
        q0 = neutral(self.robot.pinocchio_model)
        if self.robot.has_freeflyer:
            ground_fun = self.engine.get_options()['world']['groundProfile']
            compute_freeflyer_state_from_fixed_body(
                self.robot, q0, ground_profile=ground_fun,
                use_theoretical_model=False)
        v0 = np.zeros(self.robot.nv)
        x0 = np.concatenate((q0, v0))
        return x0

    def _update_observation(self, obs):
        """
        @brief      Update the observation based on the current state of the robot.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.
        """
        obs['t'] = self.engine_py.t
        obs['state'] = self.engine_py.state
        obs['sensors'] = {
            sensor_type: self.engine_py.sensors_data[sensor_type]
            for sensor_type in self.engine_py.sensors_data.keys()
        }

    def _is_done(self):
        """
        @brief      Determine whether the episode is over

        @details    By default, it always returns False.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     Boolean flag
        """
        return False

    def _compute_reward(self):
        """
        @brief      Compute the reward at the current episode state.

        @details    By default it always return 0.0.

        @return     The computed reward.
        """
        return 0.0

    def seed(self, seed=None):
        """
        @brief      Specify the seed of the environment.

        @details    Note that it also resets the low-level Jiminy engine.
                    One must call the `reset` method manually afterward.

        @param[in]  seed    non-negative integer.
                            Optional: A strongly random seed will be generated by gym if omitted.

        @return     Updated seed of the environment
        """
        # Generate a 8 bytes (uint64) seed using gym utils
        self.rg, self._seed = seeding.np_random(seed)

        # Convert it into a 4 bytes uint32 seed.
        # Note that hashing is used to get rid off possible
        # correlation in the presence of concurrency.
        self._seed = np.uint32(
            seeding._int_list_from_bigint(seeding.hash_seed(self._seed))[0])

        # Reset the seed of Jiminy Engine, if available
        if self.engine_py is not None:
            self.engine_py.seed(self._seed)

        return [self._seed]

    def reset(self):
        """
        @brief      Reset the environment.

        @details    The initial state is randomly sampled.

        @return     Initial state of the episode
        """
        # Make sure the environment is properly setup
        self._setup_environment()

        # Reset the low-level engine
        self.engine_py.reset(self._sample_state())

        # Refresh the observation and action spaces
        self._refresh_learning_spaces()

        # Reset some internal buffers
        self.is_running = False
        self._steps_beyond_done = None
        self.action_prev = None
        self._update_observation(self.observation)

        return self.observation

    def step(self, action):
        """
        @brief      Run a simulation step for a given action.

        @param[in]  action   The action to perform in the action space
                             Set to None to NOT update the action.

        @return     The next observation, the reward, the status of the episode
                    (done or not), and a dictionary of extra information
        """
        self.engine_py.step(action_next=action, dt_desired=self.dt)
        self.is_running = True
        self.action_prev = action

        # Extract information about the current simulation state
        self._update_observation(self.observation)
        done = self._is_done()
        self.learning_info = {'is_success': done}

        reward = self._compute_reward()

        # Make sure the simulation is not already over
        if done:
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already \
                                 returned done = True. You should always call 'reset()' once you \
                                 receive 'done = True' -- any further steps are undefined behavior.")
                self._steps_beyond_done += 1

        return self.observation, reward, done, self.learning_info

    def render(self, mode=None, **kwargs):
        """
        @brief      Render the current state of the robot.

        @details    Do not suport Multi-Rendering RGB output because it is not
                    possible to create window in new tabs programmatically.

        @param[in]  mode    Unused. Defined for compatibility with Gym OpenAI.

        @return     Fake output for compatibility with Gym OpenAI.
        """

        self.engine_py.render(return_rgb_array=False, **kwargs)
        return RenderOutMock()

    @staticmethod
    def _key_to_action(key):
        raise NotImplementedError()

    @loop_interactive()
    def play_interactive(self, key=None):
        t_init = time.time()
        if key is not None:
            action = self._key_to_action(key)
        else:
            action = None
        _, _, done, _ = self.step(action)
        self.render()
        sleep(self.dt - (time.time() - t_init))
        return done

    def close(self):
        """
        @brief      Terminate the Python Jiminy engine. Mostly defined for
                    compatibility with Gym OpenAI.
        """
        self.engine_py.close()

    @property
    def robot(self):
        return self.engine_py.robot

    @property
    def engine(self):
        return self.engine_py.engine


class RobotJiminyGoalEnv(RobotJiminyEnv, core.GoalEnv):
    """
    @brief      Base class to train a robot in Gym OpenAI using a user-specified
                Jiminy Engine for physics computations.

                It creates an Gym environment wrapping Jiminy Engine and behaves
                like any other Gym goal-environment.

    @details    The Jiminy Engine must be completely initialized beforehand, which
                means that the Jiminy Robot and Controller are already setup.
    """
    def __init__(self,
                 robot_name: str,
                 engine_py: EngineAsynchronous,
                 dt: float):
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

        ## Sample a new goal
        self.goal = self._sample_goal()

    def _refresh_learning_spaces(self):
        super()._refresh_learning_spaces()

        ## Append default desired and achieved goal spaces to the observation space
        self.observation_space = spaces.Dict(
            desired_goal=spaces.Box(
                -np.inf, np.inf, shape=self.goal.shape, dtype=np.float64),
            achieved_goal=spaces.Box(
                -np.inf, np.inf, shape=self.goal.shape, dtype=np.float64),
            observation=self.observation_space)

        ## Current observation of the robot
        self.observation = {'observation': self.observation,
                            'achieved_goal': None,
                            'desired_goal': None}

    def _sample_goal(self):
        """
        @brief      Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _get_achieved_goal(self):
        """
        @brief      Compute the achieved goal based on the current state of the robot.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     The currently achieved goal
        """
        raise NotImplementedError()

    def _update_observation(self, obs):
        # @copydoc RobotJiminyEnv::_update_observation
        super()._update_observation(obs['observation'])
        obs['achieved_goal'] = self._get_achieved_goal(),
        obs['desired_goal'] = self.goal.copy()

    def _is_done(self, achieved_goal, desired_goal):
        """
        @brief      Determine whether a desired goal has been achieved.

        @param[in]  achieved_goal   Achieved goal
        @param[in]  desired_goal    Desired goal

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.

        @return     Boolean flag
        """
        raise NotImplementedError()

    def _compute_reward(self):
        # @copydoc RobotJiminyEnv::_compute_reward
        return self.compute_reward(self.observation['achieved_goal'],
                                   self.observation['desired_goal'],
                                   self.learning_info)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        @brief      Compute the reward for any given episode state.

        @param[in]  achieved_goal   Achieved goal
        @param[in]  desired_goal    Desired goal
        @param[in]  info            Dictionary of extra information
                                    (must NOT be used, since not available using HER)

        @return     The computed reward.
        """
        # Must NOT use info, since it is not available while using HER (Experience Replay)
        raise NotImplementedError()

    def reset(self):
        # @copydoc RobotJiminyEnv::reset
        self.goal = self._sample_goal()
        return super().reset()
