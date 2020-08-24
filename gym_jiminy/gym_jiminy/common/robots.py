## @file

"""
@package    gym_jiminy

@brief      Package containing python-native helper methods for Gym Jiminy Open Source.
"""
import time
import tempfile
import numpy as np
from typing import Optional

import gym
from gym import logger
from gym.utils import seeding

from pinocchio import neutral
from jiminy_py.core import EncoderSensor as enc, \
                           EffortSensor as effort, \
                           ForceSensor as force, \
                           ImuSensor as imu
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.engine_asynchronous import EngineAsynchronous
from jiminy_py.viewer import sleep, play_logfiles

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


class RobotJiminyEnv(gym.core.Env):
    """
    @brief      Base class to train a robot in Gym OpenAI using a
                user-specified Python Jiminy engine for physics computations.

                It creates an Gym environment wrapping Jiminy Engine and
                behaves like any other Gym environment.
    """

    def __init__(self,
                 engine_py: Optional[EngineAsynchronous],
                 dt: float,
                 debug: bool = False):
        """
        @brief      Constructor

        @param[in]  engine_py   Python Jiminy engine used for physics
                                computations. For now, the only engine
                                available is `EngineAsynchronous`. Not
                                required provided that '_setup_environment'
                                has been overwritten such that 'self.engine_py'
                                is a valid and completely initialized engine.
        @param[in]  dt          Desired update period of the simulation
        @param[in]  debug       Whether or not the debug mode must be enabled.
                                Doing it enables telemetry recording.

        @return     Instance of the environment.
        """

        # ###################### Configure the learning environment ######################

        ## Jiminy engine associated with the robot (used for physics computations)
        self.engine_py = engine_py
        self.rg = np.random.RandomState()
        self._seed = None
        self.dt = dt
        self.debug = debug
        self._log_data = None
        if self.debug is not None:
            self._log_file = tempfile.NamedTemporaryFile(
                prefix="log_", suffix=".data", delete=(not debug))
        else:
            self._log_file = None

        ## Set the metadata of the environment. Those information are
        #  used by some gym wrappers such as VideoRecorder.
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        ## Configure the action and observation spaces
        self.action_space = None
        self.observation_space = None

        ## Current observation of the robot
        self._observation = None

        ## Information about the learning process
        self._info = {'is_success': False}
        self._enable_reward_terminal = self._compute_reward_terminal.__func__ \
            is not RobotJiminyEnv._compute_reward_terminal

        ## Number of simulation steps performed after episode termination
        self._steps_beyond_done = None

        # ############################# Initialize the engine ############################

        ## Set the seed of the simulation and reset the simulation
        self.seed()
        self.reset()

    @property
    def robot(self):
        return self.engine_py.robot

    @property
    def engine(self):
        return self.engine_py.engine

    @property
    def log_path(self):
        if self.debug is not None:
            return self._log_file.name

    # methods to override:
    # ----------------------------

    def _setup_environment(self):
        """
        @brief      Configure the environment. It must guarantee that its
                    internal state is valid after calling this method.

        @details    By default, it enforces some options of the engine.

        @remark     This method is called internally by 'reset' method at the
                    very beginning. This method can be overwritten to postpone
                    the engine and robot creation at 'reset'. One have to
                    delegate the creation and initialization of the engine to
                    this method, so that it alleviates the requirement to
                    specify a valid the engine at environment instantiation.
        """
        # Extract some proxies
        robot_options = self.robot.get_options()
        engine_options = self.engine_py.get_engine_options()

        # Disable part of the telemetry in non debug mode, to speed up
        # the simulation. Only the required data for log replay are enabled.
        # It is up to the user to overload this method if logging more data
        # is necessary for terminal reward computation.
        for field in robot_options["telemetry"].keys():
            robot_options["telemetry"][field] = self.debug
        for field in engine_options["telemetry"].keys():
            if field[:6] == 'enable':
                engine_options["telemetry"][field] = self.debug
        engine_options['telemetry']['enableConfiguration'] = True

        # Enable the position and velocity bounds of the robot
        robot_options["model"]["joints"]["enablePositionLimit"] = True
        robot_options["model"]["joints"]["enableVelocityLimit"] = True

        # Enable the effort limits of the motors
        for motor_name in robot_options["motors"].keys():
            robot_options["motors"][motor_name]["enableEffortLimit"] = True

        # Configure the stepper update period, and disable max number of
        # iterations and timeout.
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
        """
        @brief      Configure the observation and action space of the
                    environment.

        @details    By default, the observation is a dictionary gathering the
                    current simulation time, the real robot state, and the
                    sensors data.

        @remark     This method is called internally by 'reset' method at the
                    very end, just before computing and returning the initial
                    observation.  This method, alongside '_update_obs', must be
                    overwritten in order to use a custom observation space.
        """
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

        if not model_options['joints']['enableVelocityLimit']:
            velocity_limit[joints_velocity_idx] = JOINT_VEL_UNIVERSAL_MAX

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

        self.action_space = gym.spaces.Box(
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

        sensor_space = gym.spaces.Dict({
            key: gym.spaces.Box(
                low=value["min"], high=value["max"], dtype=np.float64)
            for key, value in sensor_space_raw.items()
        })

        ## Observation space
        state_limit_lower = np.concatenate(
            (position_limit_lower, -velocity_limit))
        state_limit_upper = np.concatenate(
            (position_limit_upper, velocity_limit))

        self.observation_space = gym.spaces.Dict(
            t = gym.spaces.Box(
                low=0.0,
                high=T_UNIVERSAL_MAX,
                shape=(1,), dtype=np.float64),
            state = gym.spaces.Box(
                low=state_limit_lower,
                high=state_limit_upper,
                dtype=np.float64),
            sensors = sensor_space
        )

        self._observation = {'t': None, 'state': None, 'sensors': None}

    def _sample_state(self):
        """
        @brief      Returns a random valid configuration and velocity for the
                    robot.

        @details    The default implementation only return the neural
                    configuration, with offsets on the freeflyer to ensure no
                    contact points are going through the ground and a single
                    one is touching it.

        @remark     This method is called internally by 'reset' to generate the
                    initial state. It can be overloaded to act as a random
                    state generator.
        """
        qpos = neutral(self.robot.pinocchio_model)
        if self.robot.has_freeflyer:
            ground_fun = self.engine.get_options()['world']['groundProfile']
            compute_freeflyer_state_from_fixed_body(
                self.robot, qpos, ground_profile=ground_fun,
                use_theoretical_model=False)
        qvel = np.zeros(self.robot.nv)
        return qpos, qvel

    def _update_obs(self, obs):
        """
        @brief      Update the observation based on the current state of the
                    robot.

        @details    By default, no filtering is applied on the raw data extracted
                    from the engine.

        @remark     This method, alongside '_refresh_learning_spaces', must be
                    overwritten in order to use a custom observation space.
        """
        obs['t'] = self.engine_py.t
        obs['state'] = self.engine_py.state
        obs['sensors'] = {
            sensor_type: self.engine_py.sensors_data[sensor_type]
            for sensor_type in self.engine_py.sensors_data.keys()
        }

    def _get_obs(self):
        """
        @brief      Post-processed observation.

        @details    The default implementation clamps the observation to make
                    sure it does not violate the lower and upper bounds.
        """
        def _clamp(space, x):
            if isinstance(space, gym.spaces.Dict):
                return {
                    k: _clamp(subspace, x[k])
                    for k, subspace in space.spaces.items()
                }
            else:
                return np.clip(x, space.low, space.high)

        return _clamp(self.observation_space, self._observation)

    def _is_done(self):
        """
        @brief      Determine whether the episode is over

        @details    By default, it returns True if the observation reaches or
                    exceeds the lower or upper limit.

        @return     Boolean flag
        """
        return not self.observation_space.contains(self._observation)

    def _compute_reward(self):
        """
        @brief      Compute reward at current episode state.

        @details    By default it always return 'nan', without extra info.

        @return     The computed reward, and any extra info useful for
                    monitoring as a dictionary.
        """
        return float('nan'), {}

    def _compute_reward_terminal(self):
        """
        @brief      Compute terminal reward at current episode final state.

        @details    Implementation is optional. Not computing terminal reward
                    if not overloaded by the user.

        @return     The computed terminal reward, and any extra info useful for
                    monitoring as a dictionary.
        """
        raise NotImplementedError

    # -----------------------------

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

    def set_state(self, qpos, qvel):
        """
        @brief      Reset the simulation and specify the initial state of the robot.
        """
        # Reset the simulator and set the initial state
        self.engine_py.reset(np.concatenate((qpos, qvel)))

        # Reset some internal buffers
        self._steps_beyond_done = None
        self._log_data = None

        # Clear the log file
        if self.debug is not None:
            self._log_file.truncate(0)

    def reset(self):
        """
        @brief      Reset the environment.

        @details    The initial state is obtained by calling '_sample_state'.

        @return     Initial state of the episode
        """
        # Make sure the environment is properly setup
        self._setup_environment()

        # Reset the low-level engine
        self.set_state(*self._sample_state())

        # Refresh the observation and action spaces
        self._refresh_learning_spaces()
        self._update_obs(self._observation)

        return self._get_obs()

    def step(self, action):
        """
        @brief      Run a simulation step for a given action.

        @param[in]  action   The action to perform in the action space
                             Set to None to NOT update the action.

        @return     The next observation, the reward, the status of the episode
                    (done or not), and a dictionary of extra information
        """
        # Try to perform a single simulation step
        try:
            self.engine_py.step(action_next=action, dt_desired=self.dt)
        except RuntimeError as e:
            logger.error("Unrecoverable Jiminy engine exception:\n" + str(e))
            self._update_obs(self._observation)
            return self._get_obs(), 0.0, True, {'is_success': False}
        self._update_obs(self._observation)

        # Check if the simulation is over and if not already the case
        done = self._is_done()
        if done:
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn(
                        "Calling 'step' even though this environment has "\
                        "already returned done = True whereas debug mode or "\
                        "terminal reward is enabled. You must call 'reset' "\
                        "to avoid further undefined behavior.")
                self._steps_beyond_done += 1

        # Compute reward and extra information
        self._info = {'is_success': done}
        reward, reward_info = self._compute_reward()
        if reward_info is not None:
            self._info['reward'] = reward_info

        # Finalize the episode is the simulation is over
        if done:
            if self._steps_beyond_done is None:
                # Write log file if simulation is over (debug mode only)
                if self.debug:
                    self.engine.write_log(self.log_path)

                if self._steps_beyond_done == 0 and \
                        self._enable_reward_terminal:
                    # Extract log data from the simulation, which
                    # could be used for computing terminal reward.
                    self._log_data, _ = self.engine.get_log()

                    # Compute the terminal reward
                    reward_final, reward_final_info = \
                        self._compute_reward_terminal()
                    reward += reward_final
                    if reward_final_info is not None:
                        self._info.setdefault('reward', {}).update(
                            reward_final_info)

        return self._get_obs(), reward, done, self._info

    def render(self, mode='human', **kwargs):
        """
        @brief      Render the current state of the robot.

        @details    Do not suport Multi-Rendering RGB output because it is not
                    possible to create window in new tabs programmatically.

        @param[in]  mode     Unused. Defined for compatibility with Gym OpenAI.
        @param[in]  kwargs   Extra keyword arguments for 'Viewer' delegation.

        @return     Fake output for compatibility with Gym OpenAI.
        """
        if mode == 'human':
            return_rgb_array = False
        elif mode == 'rgb_array':
            return_rgb_array = True
        else:
            raise ValueError(f"Rendering mode {mode} not supported.")
        return self.engine_py.render(return_rgb_array, **kwargs)

    def replay(self, **kwargs):
        """
        @brief      Replay the current episode until now.

        @param[in]  kwargs   Extra keyword arguments for 'play_logfiles' delegation.
        """
        if self._log_data is not None:
            log_data = self._log_data
        else:
            log_data, _ = self.engine.get_log()
        self.engine_py._viewer = play_logfiles(
            [self.robot], [log_data], viewers=[self.engine_py._viewer],
            close_backend=False, verbose=True, **kwargs
        )[0]

    def close(self):
        """
        @brief      Terminate the Python Jiminy engine. Mostly defined for
                    compatibility with Gym OpenAI.
        """
        self.engine_py.close()

    @staticmethod
    def _key_to_action(key):
        raise NotImplementedError

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


class RobotJiminyGoalEnv(RobotJiminyEnv, gym.core.GoalEnv):
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
        self.observation_space = gym.spaces.Dict(
            desired_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self.goal.shape, dtype=np.float64),
            achieved_goal=gym.spaces.Box(
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
        raise NotImplementedError

    def _get_achieved_goal(self):
        """
        @brief      Compute the achieved goal based on the current state of the robot.

        @return     The currently achieved goal
        """
        raise NotImplementedError

    def _update_obs(self, obs):
        # @copydoc RobotJiminyEnv::_update_obs
        super()._update_obs(obs['observation'])
        obs['achieved_goal'] = self._get_achieved_goal(),
        obs['desired_goal'] = self.goal.copy()

    def _is_done(self, achieved_goal, desired_goal):
        """
        @brief      Determine whether a desired goal has been achieved.

        @param[in]  achieved_goal   Achieved goal
        @param[in]  desired_goal    Desired goal

        @details    By default, it returns True if the observation reaches or
                    exceeds the lower or upper limit.

        @return     Boolean flag
        """
        return not self.observation_space.spaces['observation'].contains(
            self._observation['observation'])

    def _compute_reward(self):
        # @copydoc RobotJiminyEnv::_compute_reward
        return self.compute_reward(self._observation['achieved_goal'],
                                   self._observation['desired_goal'],
                                   self._info)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        @brief      Compute the reward for any given episode state.

        @param[in]  achieved_goal   Achieved goal
        @param[in]  desired_goal    Desired goal
        @param[in]  info            Dictionary of extra information
                                    (must NOT be used, since not available using HER)

        @return     The computed reward, and any extra info useful for
                    monitoring as a dictionary.
        """
        # Must NOT use info, since it is not available while using HER (Experience Replay)
        raise NotImplementedError

    def reset(self):
        # @copydoc RobotJiminyEnv::reset
        self.goal = self._sample_goal()
        return super().reset()
