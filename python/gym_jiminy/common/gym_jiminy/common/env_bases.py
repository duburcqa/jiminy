import time
import tempfile
import numpy as np
from collections import OrderedDict
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
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import sleep, play_logfiles
from jiminy_py.dynamics import update_quantities

from pinocchio import neutral

from .wrappers import SpaceDictRecursive
from .play import loop_interactive


# Define universal bounds for the observation space
FREEFLYER_POS_TRANS_MAX = 1000.0
FREEFLYER_VEL_LIN_MAX = 1000.0
FREEFLYER_VEL_ANG_MAX = 10000.0
JOINT_POS_MAX = 10000.0
JOINT_VEL_MAX = 100.0
FLEX_VEL_ANG_MAX = 10000.0
MOTOR_EFFORT_MAX = 1000.0
SENSOR_FORCE_MAX = 100000.0
SENSOR_MOMENT_MAX = 10000.0
SENSOR_GYRO_MAX = 100.0
SENSOR_ACCEL_MAX = 10000.0


class BaseJiminyEnv(gym.core.Env):
    """Base class to train a robot in Gym OpenAI using a user-specified Python
    Jiminy engine for physics computations.

    It creates an Gym environment wrapping Jiminy Engine and behaves like any
    other Gym environment.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 simulator: Optional[Simulator],
                 dt: float,
                 debug: bool = False,
                 **kwargs):
        """
        :param simulator: Jiminy Python simulator used for physics
                          computations. Can be `None` if `_setup_environment`
                          has been overwritten such that 'self.simulator' is
                          a valid and completely initialized engine.
        :param dt: Desired update period of the simulation
        :param debug: Whether or not the debug mode must be enabled. Doing it
                      enables telemetry recording.
        """
        # Backup some user arguments
        self.simulator = simulator
        self.dt = dt
        self.debug = debug

        # Jiminy engine used for physics computations
        self.rg = np.random.RandomState()
        self._is_ready = False
        self._seed = None
        self._log_data = None
        self._log_file = None

        # Use instance-specific action and observation spaces instead of the
        # class-wide ones provided by `gym.core.Env`.
        self.action_space = None
        self.observation_space = None

        # Current observation and action of the robot
        self._state = None
        self._sensors_data = None
        self._observation = None
        self._action = None

        # Information about the learning process
        self._info = {}
        self._enable_reward_terminal = self._compute_reward_terminal.__func__ \
            is not BaseJiminyEnv._compute_reward_terminal

        # Number of simulation steps performed
        self.num_steps = -1
        self.max_steps = None
        self._num_steps_beyond_done = None

        # Set the seed of the simulation and reset the simulation
        self.seed()
        self.reset()

    @property
    def robot(self) -> jiminy.Robot:
        if self.simulator is not None:
            return self.simulator.robot
        else:
            return None

    @property
    def log_path(self) -> Optional[str]:
        if self.debug is not None:
            return self._log_file.name
        return None

    def _send_command(self,
                      t: float,
                      q: np.ndarray,
                      v: np.ndarray,
                      sensors_data: jiminy.sensorsData,
                      u_command: np.ndarray) -> None:
        """This method implement the callback function required by Jiminy
        Controller to get the command. In practice, it only updates a variable
        shared between C++ and Python to the internal value stored by this
        class.

        .. note::
            This is a hidden function that is not listed as part of the member
            methods of the class. It is not intended to be called manually.
        """
        u_command[:] = self._action

    def _get_time_space(self) -> None:
        """Get time space.
        """
        return gym.spaces.Box(
            low=0.0, high=self.simulator.simulation_duration_max, shape=(1,),
            dtype=np.float64)

    def _get_state_space(self,
                         use_theoretical_model: Optional[bool] = None,
                         enforce_bounded: bool = True) -> None:
        r"""Get state space.

        :param use_theoretical_model: Whether to compute the state space
                                      corresponding to the theoretical model to
                                      the actual one. `None` to use internal
                                      value 'simulator.use_theoretical_model'.
                                      Optional: `None` by default.
        :param enforce_bounded: Whether or not to enforce finite bounds. If so,
                                then '\*_MAX' are used whenever it is
                                necessary. Note that whose bounds are very
                                spread to make sure it is suitable for the vast
                                majority of systems.
        """
        # Handling of default argument
        if use_theoretical_model is None:
            use_theoretical_model = self.simulator.use_theoretical_model

        # Define some proxies for convenience
        model_options = self.robot.get_model_options()
        joints_position_idx = self.robot.rigid_joints_position_idx
        joints_velocity_idx = self.robot.rigid_joints_velocity_idx
        position_limit_upper = self.robot.position_limit_upper
        position_limit_lower = self.robot.position_limit_lower
        velocity_limit = self.robot.velocity_limit

        # Replace inf bounds of the state space if requested
        if enforce_bounded:
            if self.robot.has_freeflyer:
                position_limit_lower[:3] = -FREEFLYER_POS_TRANS_MAX
                position_limit_upper[:3] = +FREEFLYER_POS_TRANS_MAX
                velocity_limit[:3] = FREEFLYER_VEL_LIN_MAX
                velocity_limit[3:6] = FREEFLYER_VEL_ANG_MAX

            for jointIdx in self.robot.flexible_joints_idx:
                jointVelIdx = self.robot.pinocchio_model.joints[jointIdx].idx_v
                velocity_limit[jointVelIdx + np.arange(3)] = FLEX_VEL_ANG_MAX

            if not model_options['joints']['enablePositionLimit']:
                position_limit_lower[joints_position_idx] = -JOINT_POS_MAX
                position_limit_upper[joints_position_idx] = JOINT_POS_MAX

            if not model_options['joints']['enableVelocityLimit']:
                velocity_limit[joints_velocity_idx] = JOINT_VEL_MAX

        # Define bounds of the state space
        if use_theoretical_model:
            state_limit_lower = np.concatenate((
                position_limit_lower[joints_position_idx],
                -velocity_limit[joints_velocity_idx]))
            state_limit_upper = np.concatenate((
                position_limit_upper[joints_position_idx],
                velocity_limit[joints_velocity_idx]))
        else:
            state_limit_lower = np.concatenate((
                position_limit_lower, -velocity_limit))
            state_limit_upper = np.concatenate((
                position_limit_upper, velocity_limit))

        return gym.spaces.Box(
            low=state_limit_lower, high=state_limit_upper, dtype=np.float64)

    def _get_sensors_space(self, enforce_bounded: bool = True) -> None:
        """   TODO
        """
        # Define some proxies for convenience
        sensors_data = self.robot.sensors_data
        effort_limit = self.robot.effort_limit

        state_space = self._get_state_space(
            use_theoretical_model=False, enforce_bounded=enforce_bounded)

        # Replace inf bounds of the action space
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_limit[motor.joint_velocity_idx] = MOTOR_EFFORT_MAX

        # Initialize the bounds of the sensor space
        sensor_space_lower = OrderedDict(
            (key, np.full(value.shape, -np.inf))
            for key, value in sensors_data.items())
        sensor_space_upper = OrderedDict(
            (key, np.full(value.shape, np.inf))
            for key, value in sensors_data.items())

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
                    sensor_position_lower = state_space.low[joint.idx_q]
                    sensor_position_upper = state_space.high[joint.idx_q]
                sensor_velocity_limit = state_space.high[
                    self.robot.nq + joint.idx_v]

                # Update the bounds accordingly
                sensor_space_lower[enc.type][0, sensor_idx] = \
                    sensor_position_lower
                sensor_space_upper[enc.type][0, sensor_idx] = \
                    sensor_position_upper
                sensor_space_lower[enc.type][1, sensor_idx] = \
                    - sensor_velocity_limit
                sensor_space_upper[enc.type][1, sensor_idx] = \
                    sensor_velocity_limit

        # Replace inf bounds of the effort sensor space
        if effort.type in sensors_data.keys():
            sensor_list = self.robot.sensors_names[effort.type]
            for sensor_name in sensor_list:
                sensor = self.robot.get_sensor(effort.type, sensor_name)
                sensor_idx = sensor.idx
                motor_idx = self.robot.motors_velocity_idx[sensor.motor_idx]
                sensor_space_lower[effort.type][0, sensor_idx] = \
                    -effort_limit[motor_idx]
                sensor_space_upper[effort.type][0, sensor_idx] = \
                    +effort_limit[motor_idx]

        # Replace inf bounds of the imu sensor space
        if imu.type in sensors_data.keys():
            quat_imu_idx = [
                field.startswith('Quat') for field in imu.fieldnames]
            sensor_space_lower[imu.type][quat_imu_idx, :] = -1.0 - 1e-12
            sensor_space_upper[imu.type][quat_imu_idx, :] = 1.0 + 1e-12

        if enforce_bounded:
            # Replace inf bounds of the contact sensor space
            if contact.type in sensors_data.keys():
                sensor_space_lower[contact.type][:, :] = -SENSOR_FORCE_MAX
                sensor_space_upper[contact.type][:, :] = SENSOR_FORCE_MAX

            # Replace inf bounds of the force sensor space
            if force.type in sensors_data.keys():
                sensor_space_lower[force.type][:3, :] = -SENSOR_FORCE_MAX
                sensor_space_upper[force.type][:3, :] = SENSOR_FORCE_MAX
                sensor_space_lower[force.type][3:, :] = -SENSOR_MOMENT_MAX
                sensor_space_upper[force.type][3:, :] = SENSOR_MOMENT_MAX

            # Replace inf bounds of the imu sensor space
            if imu.type in sensors_data.keys():
                gyro_imu_idx = [
                    field.startswith('Gyro') for field in imu.fieldnames]
                sensor_space_lower[imu.type][gyro_imu_idx, :] = \
                    -SENSOR_GYRO_MAX
                sensor_space_upper[imu.type][gyro_imu_idx, :] = \
                    SENSOR_GYRO_MAX

                accel_imu_idx = [
                    field.startswith('Accel') for field in imu.fieldnames]
                sensor_space_lower[imu.type][accel_imu_idx, :] = \
                    -SENSOR_ACCEL_MAX
                sensor_space_upper[imu.type][accel_imu_idx, :] = \
                    SENSOR_ACCEL_MAX

        return gym.spaces.Dict(OrderedDict(
            (key, gym.spaces.Box(low=min_val, high=max_val, dtype=np.float64))
            for (key, min_val), max_val in zip(
                sensor_space_lower.items(), sensor_space_upper.values())))

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Specify the seed of the environment.

        .. warning::
            It also resets the low-level jiminy Engine. Therefore one must call
            the `reset` method manually afterward.

        :param seed: Random seed, as a positive integer.
                     Optional: A strongly random seed will be generated by gym
                     if omitted.

        :returns: Updated seed of the environment
        """
        # Generate a 8 bytes (uint64) seed using gym utils
        self.rg, self._seed = seeding.np_random(seed)

        # Convert it into a 4 bytes uint32 seed.
        # Note that hashing is used to get rid off possible correlation in the
        # presence of concurrency.
        self._seed = np.uint32(
            seeding._int_list_from_bigint(seeding.hash_seed(self._seed))[0])

        # Reset the seed of Jiminy Engine, if available
        if self.simulator is not None:
            self.simulator.seed(self._seed)

        return [self._seed]

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Reset the simulation and specify the initial state of the robot.

        This method is also in charge of setting the initial action (at the
        beginning) and observation (at the end).

        .. warning::
            It does NOT start the simulation immediately but rather wait for
            the first 'step' call. Note that once the simulations starts, it
            is no longer possible to changed the robot (included options).

        :param qpos: Configuration of the robot.
        :param qvel: Velocity vector of the robot.
        """
        # Reset the simulator
        self.simulator.reset()

        # Set default action. It will be used for doing the initial step.
        self._action = np.zeros(self.robot.nmotors)

        # Start the engine, in order to initialize the sensors data
        hresult = self.simulator.start(
            qpos, qvel, self.simulator.use_theoretical_model)
        if (hresult != jiminy.hresult_t.SUCCESS):
            raise RuntimeError("Invalid initial state.")

        # Backup sensors data
        self._sensors_data = OrderedDict(self.robot.sensors_data)  # copy

        # Initialize some internal buffers
        self._is_ready = True
        self.num_steps = 0
        self.max_steps = int(self.simulator.simulation_duration_max / self.dt)

        # Stop the engine, to avoid locking the robot and the telemetry too
        # early, so that it remains possible to register external forces,
        # register log variables, change the options...etc.
        self.simulator.reset()

        # Restore the initial internal pinocchio data
        update_quantities(
            self.robot, qpos, qvel,
            update_physics=True, update_com=True, update_energy=True,
            use_theoretical_model=self.simulator.use_theoretical_model)

        # Reset some internal buffers
        self._num_steps_beyond_done = None
        self._log_data = None

        # Create a new log file
        if self.debug is not None:
            if self._log_file is not None:
                self._log_file.close()
            self._log_file = tempfile.NamedTemporaryFile(
                prefix="log_", suffix=".data", delete=(not self.debug))

        # Update the observation
        self._state = (qpos, qvel)
        self._observation = self._fetch_obs()

    def reset(self) -> SpaceDictRecursive:
        """Reset the environment.

        The initial state is obtained by calling '_sample_state'.

        :returns: Initial state of the episode.
        """
        # Stop simulator if still running
        if self.simulator is not None:
            self.simulator.stop()

        # Make sure the environment is properly setup
        self._setup_environment()

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._refresh_action_space()

        # Initialize the observation buffer with a random observation
        self._observation = self.observation_space.sample()

        # Enforce the low-level controller
        controller = jiminy.ControllerFunctor(
            compute_command=self._send_command)
        controller.initialize(self.robot)
        self.simulator.set_controller(controller)

        # Sample the initial state and reset the low-level engine
        qpos, qvel = self._sample_state()
        if not jiminy.is_position_valid(
                self.simulator.pinocchio_model, qpos):
            raise RuntimeError(
                "The initial state provided by `_sample_state` is "
                "inconsistent with the dimension or types of joints of the "
                "model.")
        self.set_state(qpos, qvel)

        # Make sure the state is valid, otherwise there `_fetch_obs` and
        # `_refresh_observation_space` are inconsistent.
        try:
            is_obs_valid = self.observation_space.contains(self._observation)
        except Exception:
            is_obs_valid = False
        if not is_obs_valid:
            raise RuntimeError(
                "The observation returned by `_fetch_obs` is inconsistent "
                "with the observation space defined by "
                "`_refresh_observation_space`.")

        if self._is_done():
            raise RuntimeError(
                "The simulation is already done at `reset`. "
                "Check the implementation of `_is_done` if overloaded.")

        return self.get_obs()

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Action to perform in the action space. `None` to NOT
                       update the action.

        :returns: Next observation, the reward, the status of the episode
                  (done or not), and a dictionary of extra information
        """
        # Try to perform a single simulation step
        is_step_failed = True
        try:
            # Set the desired action
            if action is not None:
                self._action = action

            # Start the simulation if it is not already the case
            if not self.simulator.is_simulation_running:
                if not self._is_ready:
                    raise RuntimeError(
                        "Simulation not initialized. Please call `reset` "
                        "once before calling 'step'.")
                hresult = self.simulator.start(
                    *self._state, self.simulator.use_theoretical_model)
                if (hresult != jiminy.hresult_t.SUCCESS):
                    raise RuntimeError("Failed to start the simulation.")
                self._is_ready = False

            # Perform a single inetgration step
            return_code = self.simulator.step(self.dt)
            if (return_code != jiminy.hresult_t.SUCCESS):
                raise RuntimeError("Failed to perform the simulation step.")

            # Update some internal buffers
            self.num_steps += 1
            is_step_failed = False
        except RuntimeError as e:
            logger.error("Unrecoverable Jiminy engine exception:\n" + str(e))

        # Fetch the new observation
        self._sensors_data = OrderedDict(self.robot.sensors_data)  # copy
        self._state = self.simulator.state
        self._observation = self._fetch_obs()

        # Check if the simulation is over.
        # Note that 'done' is always True if the integration failed or if the
        # maximum number of steps will be exceeded next step.
        done = is_step_failed or (self.num_steps + 1 > self.max_steps) or \
            self._is_done()
        self._info = {}

        # Check if stepping after done and if it is an undefined behavior
        if self._num_steps_beyond_done is None:
            if done:
                self._num_steps_beyond_done = 0
        else:
            if self._enable_reward_terminal and \
                    self._num_steps_beyond_done == 0:
                logger.error(
                    "Calling 'step' even though this environment has "
                    "already returned done = True whereas terminal "
                    "reward is enabled. You must call `reset` "
                    "to avoid further undefined behavior.")
            self._num_steps_beyond_done += 1

        # Early return in case of low-level engine integration failure
        if is_step_failed:
            return self.get_obs(), 0.0, done, self._info

        # Compute reward and extra information
        reward, reward_info = self._compute_reward()
        if reward_info is not None:
            self._info['reward'] = reward_info

        # Finalize the episode is the simulation is over
        if done and self._num_steps_beyond_done == 0:
            # Write log file if simulation is over (debug mode only)
            if self.debug:
                self.simulator.write_log(self.log_path)

            # Extract log data from the simulation, which could be used
            # for computing terminal reward.
            self._log_data, _ = self.simulator.get_log()

            # Compute the terminal reward, if any
            if self._enable_reward_terminal:
                reward_final, reward_final_info = \
                    self._compute_reward_terminal()
                reward += reward_final
                if reward_final_info is not None:
                    self._info.setdefault('reward', {}).update(
                        reward_final_info)

        return self.get_obs(), reward, done, self._info

    def render(self, mode: str = 'human', **kwargs) -> Optional[np.ndarray]:
        """Render the current state of the robot.

        .. note::
            Do not suport Multi-Rendering RGB output for now.

        :param mode: Rendering mode. It can be either 'human' to display the
                     current simulation state, or 'rgb_array' to return
                     instead a snapshot of it as an RGB array without showing
                     it on the screen.
        :param kwargs: Extra keyword arguments to forward to
                       `jiminy_py.simulator.Simulator.render` method.

        :returns: RGB array if 'mode' is 'rgb_array', None otherwise.
        """
        if mode == 'human':
            return_rgb_array = False
        elif mode == 'rgb_array':
            return_rgb_array = True
        else:
            raise ValueError(f"Rendering mode {mode} not supported.")
        return self.simulator.render(return_rgb_array, **kwargs)

    def replay(self, **kwargs) -> None:
        """Replay the current episode until now.

        :param kwargs: Extra keyword arguments for `play_logfiles` delegation.
        """
        if self._log_data is not None:
            log_data = self._log_data
        else:
            log_data, _ = self.simulator.get_log()
        self.simulator._viewer = play_logfiles(
            [self.robot], [log_data], viewers=[self.simulator._viewer],
            close_backend=False, verbose=True, **kwargs
        )[0]

    def close(self) -> None:
        """Terminate the Python Jiminy engine. Mostly defined for
               compatibility with Gym OpenAI.
        """
        self.simulator.close()

    # methods to override:
    # ----------------------------

    def _setup_environment(self) -> None:
        """Configure the environment. It must guarantee that its internal state
        is valid after calling this method.

        By default, it enforces some options of the engine.

        .. note::
            This method is called internally by `reset` method at the very
            beginning. This method can be overwritten to postpone the engine
            and robot creation at `reset`. One have to delegate the creation
            and initialization of the engine to this method, so that it
            alleviates the requirement to specify a valid the engine during
            the instantiation of the environment.
        """
        # Extract some proxies
        robot_options = self.robot.get_options()
        engine_options = self.simulator.engine.get_options()

        # Disable part of the telemetry in non debug mode, to speed up the
        # simulation. Only the required data for log replay are enabled. It is
        # up to the user to overload this method if logging more data is
        # necessary for terminal reward computation.
        for field in robot_options["telemetry"].keys():
            robot_options["telemetry"][field] = self.debug
        for field in engine_options["telemetry"].keys():
            if field.startswith('enable'):
                engine_options["telemetry"][field] = self.debug
        engine_options['telemetry']['enableConfiguration'] = True

        # Enable the position and velocity bounds of the robot
        robot_options["model"]["joints"]["enablePositionLimit"] = True
        robot_options["model"]["joints"]["enableVelocityLimit"] = True

        # Enable the friction model and effort limits of the motors
        for motor_name in robot_options["motors"].keys():
            robot_options["motors"][motor_name]["enableFriction"] = True
            robot_options["motors"][motor_name]["enableEffortLimit"] = True

        # Configure the stepper
        engine_options["stepper"]["iterMax"] = -1
        engine_options["stepper"]["timeout"] = -1
        engine_options["stepper"]["logInternalStepperSteps"] = self.debug
        engine_options["stepper"]["randomSeed"] = self._seed

        # Set the options
        self.robot.set_options(robot_options)
        self.simulator.engine.set_options(engine_options)

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the environment.

        By default, the observation is a dictionary gathering the current
        simulation time, the real robot state, and the sensors data.

        .. note::
            This method is called internally by `reset` method at the very end,
            just before computing and returning the initial observation. This
            method, alongside '_fetch_obs', must be overwritten in order to use
            a custom observation space.
        """
        self.observation_space = gym.spaces.Dict(
            t=self._get_time_space(),
            state=self._get_state_space(),
            sensors=self._get_sensors_space())

    def _refresh_action_space(self) -> None:
        """Configure the action space of the environment.

        By default, the action is a vector gathering the torques of the
        actuator of the robot.

        .. note::
            This method is called internally by `reset` method.
        """
        # Replace inf bounds of the effort limit
        effort_limit = self.robot.effort_limit
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableEffortLimit"]:
                effort_limit[motor.joint_velocity_idx] = \
                    MOTOR_EFFORT_MAX

        # Set the action space
        self.action_space = gym.spaces.Box(
            low=-effort_limit[self.robot.motors_velocity_idx],
            high=effort_limit[self.robot.motors_velocity_idx],
            dtype=np.float64)

    def _neutral(self) -> np.ndarray:
        """Returns a neutral valid configuration for the robot.

        The default implementation returns the neutral configuration if valid,
        the "mean" configuration otherwise (right in the middle of the position
        lower and upper bounds).

        .. warning::
            Beware there is no guarantee for this configuration to be
            statically stable.

        .. note::
            This method is called internally by '_sample_state' to generate the
            initial state. It can be overloaded to ensure static stability of
            the configuration.
        """
        # Get the neutral configuration of the actual model
        qpos = neutral(self.robot.pinocchio_model)

        # Make sure it is not out-of-bounds
        if np.any(self.robot.position_limit_upper < qpos) or \
                np.any(qpos < self.robot.position_limit_lower):
            mask = np.isfinite(self.robot.position_limit_upper)
            qpos[mask] = 0.5 * (
                self.robot.position_limit_upper[mask] +
                self.robot.position_limit_lower[mask])

        # Return the desired configuration
        if self.simulator.use_theoretical_model:
            return qpos[self.robot.rigid_joints_position_idx]
        else:
            return qpos

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a valid configuration and velocity for the robot.

        The default implementation returns the neutral configuration and zero
        velocity.

        Offsets are applied on the freeflyer to ensure no contact points are
        going through the ground and up to three are in contact.

        .. note::
            This method is called internally by `reset` to generate the
            initial state. It can be overloaded to act as a random state
            generator.
        """
        # Get the neutral configuration
        qpos = self._neutral()

        # Make sure the robot impacts the ground
        if self.robot.has_freeflyer:
            engine_options = self.simulator.engine.get_options()
            ground_fun = engine_options['world']['groundProfile']
            compute_freeflyer_state_from_fixed_body(
                self.robot, qpos, ground_profile=ground_fun)

        # Zero velocity
        qvel = np.zeros(self.simulator.pinocchio_model.nv)

        return qpos, qvel

    def _fetch_obs(self) -> SpaceDictRecursive:
        """Fetch the observation based on the current state of the robot.

        By default, no filtering is applied on the raw data extracted from the
        engine.

        .. note::
            This method is called right after updating the internal buffer
            `_state`. This method, alongside `_refresh_observation_space`, must
            be overwritten in order to use a custom observation space.
        """
        obs = OrderedDict()
        obs['t'] = np.array([self.simulator.stepper_state.t])
        obs['state'] = np.concatenate(self._state)
        obs['sensors'] = self._sensors_data
        return obs

    def get_obs(self) -> SpaceDictRecursive:
        """Post-processed observation.

        The default implementation clamps the observation to make sure it does
        not violate the lower and upper bounds. This method is not supposed to
        be overloaded.
        """
        def _clamp(space, x):
            if isinstance(space, gym.spaces.Dict):
                return OrderedDict(
                    (k, _clamp(subspace, x[k]))
                    for k, subspace in space.spaces.items())
            else:
                return np.clip(x, space.low, space.high)

        return _clamp(self.observation_space, self._observation)

    def _is_done(self) -> bool:
        """Determine whether the episode is over.

        By default, it returns True if the observation reaches or exceeds the
        lower or upper limit.

        .. note::
            This method is called right after calling `_fetch_obs`, so that the
            internal buffer '_observation' is up-to-date. It can be overloaded
            to implement a custom termination condition for the simulation.
        """
        if not self.observation_space.contains(self._observation):
            return True
        return False

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """Compute reward at current episode state.

        By default it always return 'nan', without extra info.

        .. note::
            This method is called after updating the internal buffer
            '_num_steps_beyond_done', which is None if the simulation is not
            done, 0 right after, and so on. It must be overloaded to implement
            a proper reward function.

        :returns: [0] Total reward.
                  [1] Any extra info useful for monitoring as a dictionary.
        """
        return float('nan'), {}

    def _compute_reward_terminal(self) -> Tuple[float, Dict[str, Any]]:
        """Compute terminal reward at current episode final state.

        .. note::
            Implementation is optional. Not computing terminal reward if not
            overloaded by the user.

        :returns: Terminal reward, and any extra info useful for monitoring as
                  a dictionary.
        """
        raise NotImplementedError

    @staticmethod
    def _key_to_action(key: str) -> np.ndarray:
        """   TODO
        """
        raise NotImplementedError

    @loop_interactive()
    def play_interactive(self, key: str = None) -> bool:
        """   TODO
        """
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
    """Base class to train a robot in Gym OpenAI using a user-specified Jiminy
    Engine for physics computations.

    It creates an Gym environment wrapping Jiminy Engine and behaves like any
    other Gym goal-environment.
    """
    def __init__(self,
                 simulator: Optional[Simulator],
                 dt: float,
                 debug: bool = False):
        super().__init__(simulator, dt, debug)

        # Sample a new goal
        self._desired_goal = self._sample_goal()

    def _refresh_observation_space(self) -> None:
        # Initialize the original observation space first
        super()._refresh_observation_space()

        # Append default desired and achieved goal spaces to observation space
        self.observation_space = gym.spaces.Dict(
            observation=self.observation_space,
            desired_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self._desired_goal.shape,
                dtype=np.float64),
            achieved_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self._desired_goal.shape,
                dtype=np.float64))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal randomly.

        .. note::
            This method is called internally by `reset` to sample the new
            desired goal that the agent will have to achieve. It is called
            BEFORE `super().reset` so non goal-env-specific internal buffers
            are NOT up-to-date. This method must be overloaded while
            implementing a goal environment.
        """
        raise NotImplementedError

    def _get_achieved_goal(self) -> np.ndarray:
        """Compute the achieved goal based on current state of the robot.

        .. note::
            This method is called internally by `_fetch_obs` to get the
            currently achieved goal. This method must be overloaded while
            implementing a goal environment.

        :returns: Currently achieved goal.
        """
        raise NotImplementedError

    def _fetch_obs(self) -> SpaceDictRecursive:
        obs = OrderedDict()
        obs['observation'] = super()._fetch_obs()
        obs['achieved_goal'] = self._get_achieved_goal(),
        obs['desired_goal'] = self._desired_goal.copy()
        return obs

    def _is_done(self,
                 achieved_goal: Optional[np.ndarray] = None,
                 desired_goal: Optional[np.ndarray] = None) -> bool:
        """Determine whether a desired goal has been achieved.

        By default, it uses the termination condition inherited from normal
        environment.

        .. note::
            This method is called right after calling `_fetch_obs`, so that the
            internal buffer '_observation' is up-to-date. This method can be
            overloaded while implementing a goal environment.

        :param achieved_goal: Achieved goal. If set to None, one is supposed
                              to call `_get_achieved_goal` instead.
                              Optional: None by default.
        :param desired_goal: Desired goal. If set to None, one is supposed to
                             use the internal buffer '_desired_goal' instead.
                             Optional: None by default.
        """
        return super()._is_done()

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """Compute the reward for any given episode state.

        .. note:
            This method is not supposed to be overloaded in the case of goal
            environment. It is just a proxy method calling `compute_reward`
            without specifying any achieved and desired goals, for
            compatibility with the API of normal environments.

        :returns: [0] Total reward.
                  [1] Any extra info useful for monitoring as a dictionary.
        """
        return self.compute_reward(None, None, self._info), {}

    def compute_reward(self,
                       achieved_goal: Optional[np.ndarray],
                       desired_goal: Optional[np.ndarray],
                       info: Dict[str, Any]) -> float:
        """Compute the reward for any given episode state.

        .. note::
            This method is part of the standard OpenAI Gym GoalEnv API. It must
            be overloaded to implement the reward function.

        :param achieved_goal: Achieved goal. Must be set to None to evalute the
                              reward for currently achieved goal.
        :param desired_goal: Desired goal. Must be set to None to evalute the
                             reward for currently desired goal.
        :param info: Dictionary of extra information.
                     Optional: None by default

        :returns: Total reward
        """
        raise NotImplementedError

    def reset(self) -> SpaceDictRecursive:
        self._desired_goal = self._sample_goal()
        return super().reset()
