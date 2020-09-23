## @file
"""
@package    gym_jiminy

@brief      Package containing python-native helper methods for Gym Jiminy Open Source.
"""
import time
import tempfile
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

import gym
from gym import logger
from gym.utils import seeding

import jiminy_py.core as jiminy
from jiminy_py.core import (EncoderSensor as enc,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.engine import BaseJiminyEngine
from jiminy_py.viewer import sleep, play_logfiles

from pinocchio import neutral

from .wrappers import SpaceDictRecursive
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
SENSOR_MOMENT_UNIVERSAL_MAX = 10000.0
SENSOR_GYRO_UNIVERSAL_MAX = 100.0
SENSOR_ACCEL_UNIVERSAL_MAX = 10000.0
T_UNIVERSAL_MAX = 10000.0


class BaseJiminyEnv(gym.core.Env):
    """
    @brief Base class to train a robot in Gym OpenAI using a user-specified
           Python Jiminy engine for physics computations.

           It creates an Gym environment wrapping Jiminy Engine and behaves
           like any other Gym environment.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 engine_py: Optional[BaseJiminyEngine],
                 dt: float,
                 debug: bool = False,
                 **kwargs):
        """
        @brief Constructor

        @param engine_py  Python Jiminy engine used for physics computations.
                          Can be `None` if `_setup_environment` has been
                          overwritten such that 'self.engine_py' is a valid
                          and completely initialized engine.
        @param dt  Desired update period of the simulation
        @param debug  Whether or not the debug mode must be enabled. Doing it
                      enables telemetry recording.
        """
        # ################# Configure the learning environment ################

        ## Jiminy engine used for physics computations
        self.engine_py = engine_py
        self.rg = np.random.RandomState()
        self._seed = None
        self.dt = dt
        self.debug = debug
        self._log_data = None
        self._log_file = None

        ## Use instance-specific action and observation spaces instead of the
        #  class-wide ones provided by `gym.core.Env`.
        self.action_space = None
        self.observation_space = None

        ## Current observation of the robot
        self._observation = None

        ## Information about the learning process
        self._info = {'is_success': False}
        self._enable_reward_terminal = self._compute_reward_terminal.__func__ \
            is not BaseJiminyEnv._compute_reward_terminal

        ## Number of simulation steps performed after episode termination
        self._steps_beyond_done = None

        # ####################### Initialize the engine #######################

        ## Set the seed of the simulation and reset the simulation
        self.seed()
        self.reset()

    @property
    def robot(self) -> jiminy.Robot:
        return self.engine_py.robot

    @property
    def log_path(self) -> Optional[str]:
        if self.debug is not None:
            return self._log_file.name
        return None

    # methods to override:
    # ----------------------------

    def _setup_environment(self) -> None:
        """
        @brief Configure the environment. It must guarantee that its internal
               state is valid after calling this method.

        @details By default, it enforces some options of the engine.

        @remark This method is called internally by 'reset' method at the very
                beginning. This method can be overwritten to postpone the
                engine and robot creation at 'reset'. One have to delegate the
                creation and initialization of the engine to this method, so
                that it alleviates the requirement to specify a valid the
                engine at environment instantiation.
        """
        # Extract some proxies
        robot_options = self.robot.get_options()
        engine_options = self.engine_py.get_engine_options()

        # Disable part of the telemetry in non debug mode, to speed up the
        # simulation. Only the required data for log replay are enabled. It is
        # up to the user to overload this method if logging more data is
        # necessary for terminal reward computation.
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

    def _refresh_observation_space(self) -> None:
        """
        @brief Configure the observation of the environment.

        @details By default, the observation is a dictionary gathering the
                 current simulation time, the real robot state, and the sensors
                 data.

        @remark This method is called internally by 'reset' method at the very
                end, just before computing and returning the initial
                observation. This method, alongside '_update_obs', must be
                overwritten in order to use a custom observation space.
        """
        ## Define some proxies for convenience
        sensors_data = self.engine_py.sensors_data
        model_options = self.robot.get_model_options()

        ## Extract some proxies
        joints_position_idx = self.robot.rigid_joints_position_idx
        joints_velocity_idx = self.robot.rigid_joints_velocity_idx
        position_limit_upper = self.robot.position_limit_upper
        position_limit_lower = self.robot.position_limit_lower
        velocity_limit = self.robot.velocity_limit
        effort_limit = self.robot.effort_limit

        # Replace inf bounds of the state space
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

        # Replace inf bounds of the action space
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_limit[motor.joint_velocity_idx] = \
                    MOTOR_EFFORT_UNIVERSAL_MAX

        ## Sensor space
        sensor_space_raw = {
            key: {'min': np.full(value.shape, -np.inf),
                  'max': np.full(value.shape, np.inf)}
            for key, value in self.engine_py.sensors_data.items()
        }

        # Replace inf bounds of the encoder sensor space
        if enc.type in sensors_data.keys():
            sensor_list = self.robot.sensors_names[enc.type]
            for sensor_name in sensor_list:
                # Get the position and velocity bounds of the sensor.
                # Note that for rotary unbounded encoders, the sensor bounds
                # cannot be extracted from the configuration vector limits
                # since the representation is different: cos/sin for the
                # configuration, and principal value of the angle for the
                # sensor.
                sensor = self.robot.get_sensor(enc.type, sensor_name)
                sensor_idx = sensor.idx
                joint = self.robot.pinocchio_model.joints[sensor.joint_idx]
                if sensor.joint_type == jiminy.joint_t.ROTARY_UNBOUNDED:
                    sensor_position_lower = -np.pi
                    sensor_position_upper = np.pi
                else:
                    sensor_position_lower = position_limit_lower[joint.idx_q]
                    sensor_position_upper = position_limit_upper[joint.idx_q]
                sensor_velocity_limit = velocity_limit[joint.idx_v]

                # Update the bounds accordingly
                sensor_space_raw[enc.type]['min'][0, sensor_idx] = \
                    sensor_position_lower
                sensor_space_raw[enc.type]['max'][0, sensor_idx] = \
                    sensor_position_upper
                sensor_space_raw[enc.type]['min'][1, sensor_idx] = \
                    - sensor_velocity_limit
                sensor_space_raw[enc.type]['max'][1, sensor_idx] = \
                    sensor_velocity_limit

        # Replace inf bounds of the effort sensor space
        if effort.type in sensors_data.keys():
            sensor_list = self.robot.sensors_names[effort.type]
            for sensor_name in sensor_list:
                sensor = self.robot.get_sensor(effort.type, sensor_name)
                sensor_idx = sensor.idx
                motor_idx = self.robot.motors_velocity_idx[sensor.motor_idx]
                sensor_space_raw[effort.type]['min'][0, sensor_idx] = \
                    -effort_limit[motor_idx]
                sensor_space_raw[effort.type]['max'][0, sensor_idx] = \
                    +effort_limit[motor_idx]

        # Replace inf bounds of the contact sensor space
        if contact.type in sensors_data.keys():
            sensor_space_raw[contact.type]['min'][:,:] = \
                -SENSOR_FORCE_UNIVERSAL_MAX
            sensor_space_raw[contact.type]['max'][:,:] = \
                +SENSOR_FORCE_UNIVERSAL_MAX

        # Replace inf bounds of the force sensor space
        if force.type in sensors_data.keys():
            sensor_space_raw[force.type]['min'][:3,:] = \
                -SENSOR_FORCE_UNIVERSAL_MAX
            sensor_space_raw[force.type]['max'][:3,:] = \
                +SENSOR_FORCE_UNIVERSAL_MAX
            sensor_space_raw[force.type]['min'][3:,:] = \
                -SENSOR_MOMENT_UNIVERSAL_MAX
            sensor_space_raw[force.type]['max'][3:,:] = \
                +SENSOR_MOMENT_UNIVERSAL_MAX

        # Replace inf bounds of the imu sensor space
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

    def _refresh_action_space(self) -> None:
        """
        @brief Configure the action space of the environment.

        @details By default, the action is a vector gathering the torques of
                 the actuator of the robot.

        @remark This method is called internally by 'reset' method.
        """
        # Replace inf bounds of the effort limit
        effort_limit = self.robot.effort_limit
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_limit[motor.joint_velocity_idx] = \
                    MOTOR_EFFORT_UNIVERSAL_MAX

        # Set the action space
        self.action_space = gym.spaces.Box(
            low=-effort_limit[self.robot.motors_velocity_idx],
            high=effort_limit[self.robot.motors_velocity_idx],
            dtype=np.float64)

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief Returns a random valid configuration and velocity for the robot.

        @details The default implementation only return the neural
                 configuration, with offsets on the freeflyer to ensure no
                 contact points are going through the ground and a single one
                 is touching it.

        @remark This method is called internally by 'reset' to generate the
                initial state. It can be overloaded to act as a random state
                generator.
        """
        qpos = neutral(self.robot.pinocchio_model)
        if self.robot.has_freeflyer:
            ground_fun = self.engine_py.get_options()['world']['groundProfile']
            compute_freeflyer_state_from_fixed_body(
                self.robot, qpos, ground_profile=ground_fun,
                use_theoretical_model=False)
        qvel = np.zeros(self.robot.nv)
        return qpos, qvel

    def _update_obs(self, obs: SpaceDictRecursive) -> None:
        """
        @brief Update the observation based on the current state of the robot.

        @details By default, no filtering is applied on the raw data extracted
                 from the engine.

        @remark This method, alongside '_refresh_observation_space', must be
                overwritten in order to use a custom observation space.
        """
        obs['t'] = self.engine_py.t
        obs['state'] = self.engine_py.state
        obs['sensors'] = {
            sensor_type: self.engine_py.sensors_data[sensor_type]
            for sensor_type in self.engine_py.sensors_data.keys()
        }

    def _get_obs(self) -> SpaceDictRecursive:
        """
        @brief Post-processed observation.

        @details The default implementation clamps the observation to make sure
                 it does not violate the lower and upper bounds.
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

    def _is_done(self) -> bool:
        """
        @brief Determine whether the episode is over.

        @details By default, it returns True if the observation reaches or
                 exceeds the lower or upper limit.
        """
        return not self.observation_space.contains(self._observation)

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """
        @brief Compute reward at current episode state.

        @details By default it always return 'nan', without extra info.

        @return The computed reward, and any extra info useful for monitoring
                as a dictionary.
        """
        return float('nan'), {}

    def _compute_reward_terminal(self) -> Tuple[float, Dict[str, Any]]:
        """
        @brief Compute terminal reward at current episode final state.

        @details Implementation is optional. Not computing terminal reward if
                 not overloaded by the user.

        @return The computed terminal reward, and any extra info useful for
                monitoring as a dictionary.
        """
        raise NotImplementedError

    # -----------------------------

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        @brief Specify the seed of the environment.

        @details Note that it also resets the low-level Jiminy engine. One must
                 call the `reset` method manually afterward.

        @param seed  Random seed, as a positive integer.
                     Optional: A strongly random seed will be generated by gym
                     if omitted.

        @return Updated seed of the environment
        """
        # Generate a 8 bytes (uint64) seed using gym utils
        self.rg, self._seed = seeding.np_random(seed)

        # Convert it into a 4 bytes uint32 seed.
        # Note that hashing is used to get rid off possible correlation in the
        # presence of concurrency.
        self._seed = np.uint32(
            seeding._int_list_from_bigint(seeding.hash_seed(self._seed))[0])

        # Reset the seed of Jiminy Engine, if available
        if self.engine_py is not None:
            self.engine_py.seed(self._seed)

        return [self._seed]

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """
        @brief Reset the simulation and specify the initial state of the robot.
        """
        # Reset the simulator and set the initial state
        self.engine_py.reset(np.concatenate((qpos, qvel)))

        # Reset some internal buffers
        self._steps_beyond_done = None
        self._log_data = None

        # Create a new log file
        if self.debug is not None:
            if self._log_file is not None:
                self._log_file.close()
            self._log_file = tempfile.NamedTemporaryFile(
                prefix="log_", suffix=".data", delete=(not self.debug))

    def reset(self) -> SpaceDictRecursive:
        """
        @brief Reset the environment.

        @details The initial state is obtained by calling '_sample_state'.

        @return Initial state of the episode.
        """
        # Make sure the environment is properly setup
        self._setup_environment()

        # Reset the low-level engine
        self.set_state(*self._sample_state())

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._update_obs(self._observation)
        self._refresh_action_space()

        return self._get_obs()

    def step(self, action: np.ndarray
            ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """
        @brief Run a simulation step for a given action.

        @param action  The action to perform in the action space. `None` to NOT
                       update the action.

        @return The next observation, the reward, the status of the episode
                (done or not), and a dictionary of extra information
        """
        # Try to perform a single simulation step
        is_step_failed = True
        try:
            self.engine_py.step(action_next=action, dt_desired=self.dt)
            is_step_failed = False
        except RuntimeError as e:
            logger.error("Unrecoverable Jiminy engine exception:\n" + str(e))
        self._update_obs(self._observation)

        # Check if the simulation is over
        done = is_step_failed or self._is_done()
        self._info = {'is_success': done}

        # Check if stepping after done and if it is an undefined behavior
        if self._steps_beyond_done is None:
            if done:
                self._steps_beyond_done = 0
        else:
            if self._enable_reward_terminal and self._steps_beyond_done == 0:
                logger.error(
                    "Calling 'step' even though this environment has "
                    "already returned done = True whereas terminal "
                    "reward is enabled. You must call 'reset' "
                    "to avoid further undefined behavior.")
            self._steps_beyond_done += 1

        # Early return in case of low-level engine integration failure
        if is_step_failed:
            return self._get_obs(), 0.0, done, self._info

        # Compute reward and extra information
        reward, reward_info = self._compute_reward()
        if reward_info is not None:
            self._info['reward'] = reward_info

        # Finalize the episode is the simulation is over
        if done and self._steps_beyond_done == 0:
            # Write log file if simulation is over (debug mode only)
            if self.debug:
                self.engine_py.write_log(self.log_path)

            # Extract log data from the simulation, which could be used
            # for computing terminal reward.
            self._log_data, _ = self.engine_py.get_log()

            # Compute the terminal reward
            reward_final, reward_final_info = \
                self._compute_reward_terminal()
            reward += reward_final
            if reward_final_info is not None:
                self._info.setdefault('reward', {}).update(
                    reward_final_info)

        return self._get_obs(), reward, done, self._info

    def render(self, mode: str = 'human', **kwargs) -> Optional[np.ndarray]:
        """
        @brief Render the current state of the robot.

        @details Do not suport Multi-Rendering RGB output because it is not
                 possible to create window in new tabs programmatically.

        @param mode  Unused. Defined for compatibility with Gym OpenAI.
        @param kwargs  Extra keyword arguments for 'Viewer' delegation.

        @return Fake output for compatibility with Gym OpenAI.
        """
        if mode == 'human':
            return_rgb_array = False
        elif mode == 'rgb_array':
            return_rgb_array = True
        else:
            raise ValueError(f"Rendering mode {mode} not supported.")
        return self.engine_py.render(return_rgb_array, **kwargs)

    def replay(self, **kwargs) -> None:
        """
        @brief Replay the current episode until now.

        @param kwargs  Extra keyword arguments for 'play_logfiles' delegation.
        """
        if self._log_data is not None:
            log_data = self._log_data
        else:
            log_data, _ = self.engine_py.get_log()
        self.engine_py._viewer = play_logfiles(
            [self.robot], [log_data], viewers=[self.engine_py._viewer],
            close_backend=False, verbose=True, **kwargs
        )[0]

    def close(self) -> None:
        """
        @brief Terminate the Python Jiminy engine. Mostly defined for
               compatibility with Gym OpenAI.
        """
        self.engine_py.close()

    @staticmethod
    def _key_to_action(key: str) -> np.ndarray:
        raise NotImplementedError

    @loop_interactive()
    def play_interactive(self, key: str = None) -> bool:
        t_init = time.time()
        if key is not None:
            action = self._key_to_action(key)
        else:
            action = None
        _, _, done, _ = self.step(action)
        self.render()
        sleep(self.dt - (time.time() - t_init))
        return done


class BaseJiminyGoalEnv(BaseJiminyEnv, gym.core.GoalEnv):
    """
    @brief Base class to train a robot in Gym OpenAI using a user-specified
           Jiminy Engine for physics computations.

           It creates an Gym environment wrapping Jiminy Engine and behaves
           like any other Gym goal-environment.

    @details The Jiminy Engine must be completely initialized beforehand, which
             means that the Jiminy Robot and Controller are already setup.
    """
    def __init__(self,
                 engine_py: Optional[BaseJiminyEngine],
                 dt: float,
                 debug: bool = False):
        super().__init__(engine_py, dt, debug)

        ## Sample a new goal
        self.goal = self._sample_goal()

    def _refresh_observation_space(self) -> None:
        # Initialize the original observation space first
        super()._refresh_observation_space()

        # Append default desired and achieved goal spaces to observation space
        self.observation_space = gym.spaces.Dict(
            desired_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self.goal.shape, dtype=np.float64),
            achieved_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self.goal.shape, dtype=np.float64),
            observation=self.observation_space)

        # Current observation of the robot
        self.observation = {'observation': self.observation,
                            'achieved_goal': None,
                            'desired_goal': None}

    def _sample_goal(self) -> np.ndarray:
        """
        @brief      Samples a new goal and returns it.
        """
        raise NotImplementedError

    def _get_achieved_goal(self) -> np.ndarray:
        """
        @brief Compute the achieved goal based on current state of the robot.

        @return The currently achieved goal.
        """
        raise NotImplementedError

    def _update_obs(self, obs: SpaceDictRecursive) -> None:
        # @copydoc BaseJiminyEnv::_update_obs
        super()._update_obs(obs['observation'])
        obs['achieved_goal'] = self._get_achieved_goal(),
        obs['desired_goal'] = self.goal.copy()

    def _is_done(self,
                 achieved_goal: np.ndarray,
                 desired_goal: np.ndarray) -> bool:
        """
        @brief Determine whether a desired goal has been achieved.

        @param achieved_goal  Achieved goal.
        @param desired_goal  Desired goal.

        @details By default, it returns True if the observation reaches or
                 exceeds the lower or upper limit.
        """
        return not self.observation_space.spaces['observation'].contains(
            self._observation['observation'])

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        # @copydoc BaseJiminyEnv::_compute_reward
        return self.compute_reward(self._observation['achieved_goal'],
                                   self._observation['desired_goal'],
                                   self._info)

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        @brief Compute the reward for any given episode state.

        @param achieved_goal  Achieved goal.
        @param desired_goal  Desired goal.
        @param info  Dictionary of extra information

        @return The computed reward, and any extra info useful for monitoring
                as a dictionary.
        """
        raise NotImplementedError

    def reset(self) -> SpaceDictRecursive:
        # @copydoc BaseJiminyEnv::reset
        self.goal = self._sample_goal()
        return super().reset()
