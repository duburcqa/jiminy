""" TODO: Write documentation.
"""
import os
import time
import tempfile
from collections import OrderedDict
from typing import Optional, Tuple, Sequence, Dict, Any, Callable

import numpy as np
import gym
from gym import logger
from gym.utils import seeding

import jiminy_py.core as jiminy
from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import sleep, play_logfiles
from jiminy_py.controller import (
    ObserverHandleType, ControllerHandleType, BaseJiminyObserverController)

from pinocchio import neutral

from ..utils import _clamp, zeros, fill, set_value, SpaceDictNested
from ..bases import ObserverControllerInterface
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


class BaseJiminyEnv(ObserverControllerInterface, gym.Env):
    """Base class to train a robot in Gym OpenAI using a user-specified Python
    Jiminy engine for physics computations.

    It creates an Gym environment wrapping Jiminy Engine and behaves like any
    other Gym environment.

    The observation space is a dictionary gathering the current simulation
    time, the real robot state, and the sensors data. The action is a vector
    gathering the torques of the actuator of the robot.

    There is no reward by default. It is up to the user to overload this class
    to implement one. It has been designed to be highly flexible and easy to
    customize by overloading it to fit the vast majority of users' needs.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 simulator: Simulator,
                 step_dt: float,
                 enforce_bounded: Optional[bool] = False,
                 debug: bool = False,
                 **kwargs: Any) -> None:
        r"""
        :param simulator: Jiminy Python simulator used for physics
                          computations.
        :param step_dt: Simulation timestep for learning. Note that it is
                        independent from the controller and observation update
                        periods. The latter are configured via
                        `engine.set_options`.
        :param enforce_bounded: Whether or not to enforce finite bounds for the
                                observation and action spaces. If so, then
                                '\*_MAX' are used whenever it is necessary.
                                Note that whose bounds are very spread to make
                                sure it is suitable for the vast majority of
                                systems.
        :param debug: Whether or not the debug mode must be enabled. Doing it
                      enables telemetry recording.
        :param kwargs: Extra keyword arguments that may be useful for derived
                       environments with multiple inheritance, and to allow
                       automatic pipeline wrapper generation.
        """
        # pylint: disable=unused-argument

        # Backup some user arguments
        self.simulator = simulator
        self.step_dt = step_dt
        self.enforce_bounded = enforce_bounded
        self.debug = debug

        # Initialize the interfaces through multiple inheritance
        super().__init__()  # Do not forward extra arguments, if any

        # Internal buffers for physics computations
        self.rg = np.random.RandomState()
        self._seed: Optional[np.uint32] = None
        self._log_data: Optional[Dict[str, np.ndarray]] = None
        self.log_path: Optional[str] = None

        # Initialize sensors data shared memory
        self._sensors_data = OrderedDict(self.robot.sensors_data)

        # Information about the learning process
        self._info: Dict[str, Any] = {}

        # Number of simulation steps performed
        self.num_steps = -1
        self.max_steps = int(
            self.simulator.simulation_duration_max // self.step_dt)
        self._num_steps_beyond_done: Optional[int] = None

        # Initialize the seed of the environment
        self.seed()

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._refresh_action_space()

        # Assertion(s) for type checker
        assert (self.observation_space is not None and
                self.action_space is not None)

        # Initialize some internal buffers
        self._action = zeros(self.action_space)
        self._state = (np.zeros(self.robot.nq), np.zeros(self.robot.nv))
        self._observation = zeros(self.observation_space)

    @property
    def robot(self) -> jiminy.Robot:
        """ Get robot.
        """
        return self.simulator.robot

    def _get_time_space(self) -> gym.Space:
        """Get time space.
        """
        return gym.spaces.Box(
            low=0.0, high=self.simulator.simulation_duration_max, shape=(1,),
            dtype=np.float64)

    def _get_state_space(self,
                         use_theoretical_model: Optional[bool] = None
                         ) -> gym.Space:
        """Get state space.

        This method is not meant to be overloaded in general since the
        definition of the state space is mostly consensual. One must rather
        overload `_refresh_observation_space` to customize the observation
        space as a whole.

        :param use_theoretical_model: Whether to compute the state space
                                      corresponding to the theoretical model to
                                      the actual one. `None` to use internal
                                      value 'simulator.use_theoretical_model'.
                                      Optional: `None` by default.
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
        if self.enforce_bounded:
            if self.robot.has_freeflyer:
                position_limit_lower[:3] = -FREEFLYER_POS_TRANS_MAX
                position_limit_upper[:3] = +FREEFLYER_POS_TRANS_MAX
                velocity_limit[:3] = FREEFLYER_VEL_LIN_MAX
                velocity_limit[3:6] = FREEFLYER_VEL_ANG_MAX

            for joint_idx in self.robot.flexible_joints_idx:
                joint_vel_idx = \
                    self.robot.pinocchio_model.joints[joint_idx].idx_v
                velocity_limit[joint_vel_idx + np.arange(3)] = FLEX_VEL_ANG_MAX

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

    def _get_sensors_space(self) -> gym.Space:
        """Get sensor space.

        It gathers the sensors data in a dictionary. It maps each available
        type of sensor to the associated data matrix. Rows correspond to the
        sensor type's fields, and columns correspond to each individual sensor.

        .. note:
            The mapping between row `i` of data matrix and associated sensor
            type's field is given by:

            .. code-block:: python

                field = getattr(jiminy_py.core, key).fieldnames[i]

            The mapping between column `j` of data matrix and associated sensor
            name and object are given by:

            .. code-block:: python

                sensor_name = env.robot.sensors_names[key][j]
                sensor = env.robot.get_sensor(key, sensor_name)

        .. warning:
            This method is not meant to be overloaded in general since the
            definition of the sensor space is mostly consensual. One must
            rather overload `_refresh_observation_space` to customize the
            observation space as a whole.
        """
        # Define some proxies for convenience
        sensors_data = self.robot.sensors_data
        effort_limit = self.robot.effort_limit

        state_space = self._get_state_space(use_theoretical_model=False)

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
        if encoder.type in sensors_data.keys():
            sensor_list = self.robot.sensors_names[encoder.type]
            for sensor_name in sensor_list:
                # Get the position and velocity bounds of the sensor.
                # Note that for rotary unbounded encoders, the sensor bounds
                # cannot be extracted from the configuration vector limits
                # since the representation is different: cos/sin for the
                # configuration, and principal value of the angle for the
                # sensor.
                sensor = self.robot.get_sensor(encoder.type, sensor_name)
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
                sensor_space_lower[encoder.type][0, sensor_idx] = \
                    sensor_position_lower
                sensor_space_upper[encoder.type][0, sensor_idx] = \
                    sensor_position_upper
                sensor_space_lower[encoder.type][1, sensor_idx] = \
                    - sensor_velocity_limit
                sensor_space_upper[encoder.type][1, sensor_idx] = \
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

        if self.enforce_bounded:
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

    def _refresh_action_space(self) -> None:
        """Configure the action space of the environment.

        The action is a vector gathering the torques of the actuator of the
        robot.

        .. warning::
            This method is called internally by `reset` method. It is not
            meant to be overloaded since the actual action space of the
            robot is uniquely defined.
        """
        # Get effort limit
        effort_limit = self.robot.effort_limit

        # Replace inf bounds of the effort limit if requested
        if self.enforce_bounded:
            for motor_name in self.robot.motors_names:
                motor = self.robot.get_motor(motor_name)
                motor_options = motor.get_options()
                if not motor_options["enableEffortLimit"]:
                    effort_limit[motor.joint_velocity_idx] = \
                        MOTOR_EFFORT_MAX

        # Set the action space
        motors_velocity_idx = self.robot.motors_velocity_idx
        self.action_space = gym.spaces.Box(
            low=-effort_limit[motors_velocity_idx],
            high=effort_limit[motors_velocity_idx],
            dtype=np.float64)

    def reset(self,
              controller_hook: Optional[Callable[[], Optional[Tuple[
                  Optional[ObserverHandleType],
                  Optional[ControllerHandleType]]]]] = None
              ) -> SpaceDictNested:
        """Reset the environment.

        In practice, it resets the backend simulator and set the initial state
        of the robot. The initial state is obtained by calling '_sample_state'.
        This method is also in charge of setting the initial action (at the
        beginning) and observation (at the end).

        .. warning::
            It starts the simulation immediately. As a result, it is not
            possible to change the robot (included options), nor to register
            log variable. The only way to do so is via 'controller_hook'.

        :param controller_hook: Custom controller hook. It will be executed
                                right after initialization of the environment,
                                and just before actually starting the
                                simulation. It is a callable that optionally
                                returns observer and/or controller handles. If
                                defined, it will be used to initialize the
                                low-level jiminy controller. It is useful to
                                override partially the configuration of the
                                low-level engine, set a custom low-level
                                observer/controller handle, or to register
                                custom variables to the telemetry. Set to
                                `None` if unused.
                                Optional: Disable by default.

        :returns: Initial observation of the episode.
        """
        # pylint: disable=arguments-differ

        # Assertion(s) for type checker
        assert self.observation_space is not None

        # Reset the simulator
        self.simulator.reset()

        # Make sure the environment is properly setup
        self._setup()

        # Re-initialize sensors data shared memory.
        # It must be done because the robot may have changed.
        self._sensors_data = OrderedDict(self.robot.sensors_data)

        # Set default action.
        # It will be used for the initial step.
        fill(self._action, 0.0)

        # Reset some internal buffers
        self.num_steps = 0
        self._num_steps_beyond_done = None
        self._log_data = None

        # Create a new log file
        if self.debug:
            fd, self.log_path = tempfile.mkstemp(prefix="log_", suffix=".data")
            os.close(fd)

        # Extract the observer/controller update period.
        # There is no actual observer by default, apart from the robot's state
        # and raw sensors data. Similarly, there is no actual controller by
        # default, apart from forwarding the command torque to the motors.
        engine_options = self.simulator.engine.get_options()
        self.control_dt = self.observe_dt = \
            float(engine_options['stepper']['controllerUpdatePeriod'])

        # Enforce the low-level controller.
        # The backend robot may have changed, for example if it is randomly
        # generated based on different URDF files. As a result, it is necessary
        # to instantiate a new low-level controller.
        # Note that `BaseJiminyObserverController` is used by default in place
        # of `jiminy.ControllerFunctor`. Although it is less efficient because
        # it adds an extra layer of indirection, it makes it possible to update
        # the controller handle without instantiating a new controller, which
        # is necessary to allow registering telemetry variables before knowing
        # the controller handle in advance.
        controller = BaseJiminyObserverController()
        controller.initialize(self.robot)
        self.simulator.set_controller(controller)

        # Run controller hook and set the observer and controller handles
        observer_handle, controller_handle = None, None
        if controller_hook is not None:
            handles = controller_hook()
            if handles is not None:
                observer_handle, controller_handle = handles
        if observer_handle is None:
            observer_handle = self._observer_handle
        self.simulator.controller.set_observer_handle(observer_handle)
        if controller_handle is None:
            controller_handle = self._controller_handle
        self.simulator.controller.set_controller_handle(controller_handle)

        # Sample the initial state and reset the low-level engine
        qpos, qvel = self._sample_state()
        if not jiminy.is_position_valid(
                self.simulator.pinocchio_model, qpos):
            raise RuntimeError(
                "The initial state provided by `_sample_state` is "
                "inconsistent with the dimension or types of joints of the "
                "model.")

        # Start the engine
        self.simulator.start(qpos, qvel, self.simulator.use_theoretical_model)

        # Update the observation
        self._observation = self.compute_observation()

        # Make sure the state is valid, otherwise there `compute_observation`
        # and s`_refresh_observation_space` are probably inconsistent.
        try:
            is_obs_valid = self.observation_space.contains(self._observation)
        except AttributeError:
            is_obs_valid = False
        if not is_obs_valid:
            raise RuntimeError(
                "The observation returned by `compute_observation` is "
                "inconsistent with the observation space defined by "
                "`_refresh_observation_space`.")

        if self.is_done():
            raise RuntimeError(
                "The simulation is already done at `reset`. "
                "Check the implementation of `is_done` if overloaded.")

        return self.get_observation()

    def seed(self, seed: Optional[int] = None) -> Sequence[np.uint32]:
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

        # Reset the seed of Jiminy Engine
        self.simulator.seed(self._seed)

        return [self._seed]

    def close(self) -> None:
        """Terminate the Python Jiminy engine. Mostly defined for
           compatibility with Gym OpenAI.
        """
        self.simulator.close()

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictNested, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Action to perform. `None` to not update the action.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information
        """
        # Make sure a simulation is already running
        if not self.simulator.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please call `reset` before `step`.")

        # Update the action to perform if necessary
        if action is not None:
            np.copyto(self._action, action)

        # Try to perform a single simulation step
        is_step_failed = True
        try:
            # Perform a single integration step
            self.simulator.step(self.step_dt)

            # Update some internal buffers
            self.num_steps += 1
            is_step_failed = False
        except RuntimeError as e:
            logger.error("Unrecoverable Jiminy engine exception:\n" + str(e))

        # Check if the simulation is over.
        # Note that 'done' is always True if the integration failed or if the
        # maximum number of steps will be exceeded next step.
        done = is_step_failed or (self.num_steps + 1 > self.max_steps) or \
            self.is_done()
        self._info = {}

        # Check if stepping after done and if it is an undefined behavior
        if self._num_steps_beyond_done is None:
            if done:
                self._num_steps_beyond_done = 0
        else:
            if self.enable_reward_terminal and \
                    self._num_steps_beyond_done == 0:
                logger.error(
                    "Calling 'step' even though this environment has already "
                    "returned done = True whereas terminal reward is enabled. "
                    "Please call `reset` to avoid further undefined behavior.")
            self._num_steps_beyond_done += 1

        # Early return in case of low-level engine integration failure.
        # In such a case, it always returns reward = 0.0 and done = True.
        if is_step_failed:
            return self.get_observation(), 0.0, True, self._info

        # Compute reward and extra information
        reward = self.compute_reward(info=self._info)

        # Finalize the episode is the simulation is over
        if done and self._num_steps_beyond_done == 0:
            # Write log file if simulation is over (debug mode only)
            if self.debug:
                self.simulator.write_log(self.log_path, format="data")

            # Extract log data from the simulation, which could be used
            # for computing terminal reward.
            self._log_data, _ = self.get_log()

            # Compute the terminal reward, if any
            if self.enable_reward_terminal:
                reward += self.compute_reward_terminal(info=self._info)

        return self.get_observation(), reward, done, self._info

    def get_log(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Get log of recorded variable since the beginning of the episode.
        """
        if not self.simulator.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please call `reset` at least one "
                "before getting log.")
        return self.simulator.get_log()

    def render(self,
               mode: str = 'human',
               **kwargs: Any) -> Optional[np.ndarray]:
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
            # Only Meshcat backend can return rgb array without poping window
            return_rgb_array = True
            self.simulator.viewer_backend = 'meshcat'
        else:
            raise ValueError(f"Rendering mode {mode} not supported.")
        return self.simulator.render(return_rgb_array, **kwargs)

    def plot(self) -> None:
        """Display common simulation data over time.
        """
        self.simulator.plot()

    def replay(self, **kwargs: Any) -> None:
        """Replay the current episode until now.

        :param kwargs: Extra keyword arguments for delegation to
                       `viewer.play_trajectories` method.
        """
        if self._log_data is not None:
            log_data = self._log_data
        else:
            log_data, _ = self.get_log()
        self.simulator._viewer = play_logfiles(
            [self.robot], [log_data], viewers=[self.simulator._viewer],
            close_backend=False, verbose=True, **kwargs)[0]

    @loop_interactive()
    def play_interactive(self, key: Optional[str] = None) -> bool:
        """Activate interact mode enabling to control the robot using keyboard.

        It stops automatically as soon as 'done' flag is True. One has to press
        a key to start the interaction. If no key is pressed, the action is
        not updated and the previous one keeps being sent to the robot.

        .. warning::
            This method requires `_key_to_action` method to be implemented by
            the user by overloading it. Otherwise, calling it will raise an
            exception.

        :param key: Key to press to start the interaction.
        """
        t_init = time.time()
        if key is not None:
            action = self._key_to_action(key)
        else:
            action = None
        _, _, done, _ = self.step(action)
        self.render()
        sleep(self.step_dt - (time.time() - t_init))
        return done

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        """Configure the environment. It must guarantee that its internal state
        is valid after calling this method.

        By default, it enforces some options of the engine.

        .. note::
            This method is called internally by `reset` methods.
        """
        # Extract some proxies
        robot_options = self.robot.get_options()
        engine_options = self.simulator.engine.get_options()

        # Disable part of the telemetry in non debug mode, to speed up the
        # simulation. Only the required data for log replay are enabled. It is
        # up to the user to overload this method if logging more data is
        # necessary for computating the terminal reward.
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
            method, alongside `compute_observation`, must be overwritten in
            order to define a custom observation space.
        """
        self.observation_space = gym.spaces.Dict(OrderedDict(
            t=self._get_time_space(),
            state=self._get_state_space(),
            sensors=self._get_sensors_space()))

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
        return qpos

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a valid configuration and velocity for the robot.

        The default implementation returns the neutral configuration and zero
        velocity.

        Offsets are applied on the freeflyer to ensure no contact points are
        going through the ground and up to three are in contact.

        .. note::
            This method is called internally by `reset` to generate the initial
            state. It can be overloaded to act as a random state generator.
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

    def compute_observation(self) -> SpaceDictNested:  # type: ignore[override]
        """Compute the observation based on the current state of the robot.
        """
        # pylint: disable=arguments-differ

        # Update some internal buffers
        if self.simulator.is_simulation_running:
            self._state = self.simulator.state

        return OrderedDict(
            t=np.array([self.simulator.stepper_state.t]),
            state=np.concatenate(self._state),
            sensors=self._sensors_data)

    def compute_command(self,
                        measure: SpaceDictNested,
                        action: np.ndarray
                        ) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        By default, it just clamps the target to make sure it does not violate
        the lower and upper bounds. There is no further processing whatsoever
        since the target is the command by default.

        :param measure: Observation of the environment.
        :param action: Desired motors efforts.
        """
        set_value(self._action, action)
        return _clamp(self.action_space, action)

    def is_done(self, *args: Any, **kwargs: Any) -> bool:
        """Determine whether the episode is over.

        By default, it returns True if the observation reaches or exceeds the
        lower or upper limit.

        .. note::
            This method is called right after calling `compute_observation`, so
            that the internal buffer '_observation' is up-to-date. It can be
            overloaded to implement a custom termination condition for the
            simulation.

        :param args: Extra arguments that may be useful for derived
                     environments, for example `Gym.GoalEnv`.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # pylint: disable=unused-argument

        # Assertion(s) for type checker
        assert self.observation_space is not None

        if not self.observation_space.contains(self._observation):
            return True
        return False

    @staticmethod
    def _key_to_action(key: str) -> np.ndarray:
        """Mapping between keyword keys and actions to send to the robot.

        .. warning::
            Overloading this method is required for using `play_interactive`.

        :param key: Key pressed by the user as a string.

        :returns: Action to send to the robot.
        """
        raise NotImplementedError


BaseJiminyEnv.compute_reward.__doc__ = \
    """Compute reward at current episode state.

    See `ControllerInterface.compute_reward` for details.

    .. note::
        This method is called after updating the internal buffer
        '_num_steps_beyond_done', which is None if the simulation is not done,
        0 right after, and so on.

    :param args: Extra arguments that may be useful for derived environments,
                 for example `Gym.GoalEnv`.
    :param info: Dictionary of extra information for monitoring.
    :param kwargs: Extra keyword arguments. See 'args'.

    :returns: Total reward.
    """


class BaseJiminyGoalEnv(BaseJiminyEnv, gym.core.GoalEnv):  # Don't change order
    """Base class to train a robot in Gym OpenAI using a user-specified Jiminy
    Engine for physics computations.

    It creates an Gym environment wrapping Jiminy Engine and behaves like any
    other Gym goal-environment.
    """
    def __init__(self,
                 simulator: Optional[Simulator],
                 step_dt: float,
                 debug: bool = False) -> None:
        # Define some internal buffers
        self._desired_goal: Optional[np.ndarray] = None

        # Initialize base class
        super().__init__(simulator, step_dt, debug)

    def _refresh_observation_space(self) -> None:
        # Assertion(s) for type checker
        assert isinstance(self._desired_goal, np.ndarray), (
            "`BaseJiminyGoalEnv` only supports np.ndarray goal space for now.")

        # Initialize the original observation space first
        super()._refresh_observation_space()

        # Append default desired and achieved goal spaces to observation space
        self.observation_space = gym.spaces.Dict(OrderedDict(
            observation=self.observation_space,
            desired_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self._desired_goal.shape,
                dtype=np.float64),
            achieved_goal=gym.spaces.Box(
                -np.inf, np.inf, shape=self._desired_goal.shape,
                dtype=np.float64)))

    def compute_observation(self) -> SpaceDictNested:  # type: ignore[override]
        # Assertion(s) for type checker
        assert self._desired_goal is not None

        return OrderedDict(
            observation=super().compute_observation(),
            achieved_goal=self._get_achieved_goal(),
            desired_goal=self._desired_goal.copy()
        )

    def reset(self,
              controller_hook: Optional[Callable[[], Optional[Tuple[
                  Optional[ObserverHandleType],
                  Optional[ControllerHandleType]]]]] = None
              ) -> SpaceDictNested:
        self._desired_goal = self._sample_goal()
        return super().reset(controller_hook)

    # methods to override:
    # ----------------------------

    def _sample_goal(self) -> SpaceDictNested:
        """Sample a goal randomly.

        .. note::
            This method is called internally by `reset` to sample the new
            desired goal that the agent will have to achieve. It is called
            BEFORE `super().reset` so non goal-env-specific internal buffers
            are NOT up-to-date. This method must be overloaded while
            implementing a goal environment.
        """
        raise NotImplementedError

    def _get_achieved_goal(self) -> SpaceDictNested:
        """Compute the achieved goal based on current state of the robot.

        .. note::
            This method can be called by `compute_observation` to get the
            currently achieved goal. This method must be overloaded while
            implementing a goal environment.

        :returns: Currently achieved goal.
        """
        raise NotImplementedError

    def is_done(self,  # type: ignore[override]
                achieved_goal: Optional[SpaceDictNested] = None,
                desired_goal: Optional[SpaceDictNested] = None) -> bool:
        """Determine whether a desired goal has been achieved.

        By default, it uses the termination condition inherited from normal
        environment.

        .. note::
            This method is called right after calling `compute_observation`, so
            that the internal buffer '_observation' is up-to-date. This method
            can be overloaded while implementing a goal environment.

        :param achieved_goal: Achieved goal. If set to None, one is supposed to
                              call `_get_achieved_goal` instead.
                              Optional: None by default.
        :param desired_goal: Desired goal. If set to None, one is supposed to
                             use the internal buffer '_desired_goal' instead.
                             Optional: None by default.
        """
        # pylint: disable=arguments-differ

        raise NotImplementedError

    def compute_reward(self,  # type: ignore[override]
                       achieved_goal: Optional[SpaceDictNested] = None,
                       desired_goal: Optional[SpaceDictNested] = None,
                       *, info: Dict[str, Any]) -> float:
        """Compute the reward for any given episode state.

        :param achieved_goal: Achieved goal. `None` to evalute the reward for
                              currently achieved goal.
        :param desired_goal: Desired goal. `None` to evalute the reward for
                             currently desired goal.
        :param info: Dictionary of extra information for monitoring.

        :returns: Total reward.
        """
        # pylint: disable=arguments-differ

        raise NotImplementedError
