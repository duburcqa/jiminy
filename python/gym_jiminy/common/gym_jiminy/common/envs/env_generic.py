""" TODO: Write documentation.
"""
import os
import time
import tempfile
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Tuple, Sequence, Dict, Any, Callable, List

import numpy as np
import gym
from gym import logger, spaces
from gym.utils import seeding

import jiminy_py.core as jiminy
from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.dynamics import (update_quantities,
                                compute_freeflyer_state_from_fixed_body)
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import sleep

from pinocchio import neutral, normalize

from ..utils import (zeros,
                     fill,
                     set_value,
                     clip,
                     get_fieldnames,
                     register_variables,
                     FieldDictNested,
                     SpaceDictNested)
from ..bases import ObserverControllerInterface

from .internal import (ObserverHandleType,
                       ControllerHandleType,
                       BaseJiminyObserverController,
                       loop_interactive)


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

    observation_space: spaces.Space
    action_space: spaces.Space

    def __init__(self,
                 simulator: Simulator,
                 step_dt: float,
                 enforce_bounded_spaces: Optional[bool] = False,
                 debug: bool = False,
                 **kwargs: Any) -> None:
        r"""
        :param simulator: Jiminy Python simulator used for physics
                          computations. It must be fully initialized.
        :param step_dt: Simulation timestep for learning. Note that it is
                        independent from the controller and observation update
                        periods. The latter are configured via
                        `engine.set_options`.
        :param enforce_bounded_spaces:
            Whether or not to enforce finite bounds for the observation and
            action spaces. If so, then '\*_MAX' are used whenever it is
            necessary. Note that whose bounds are very spread to make sure it
            is suitable for the vast majority of systems.
        :param debug: Whether or not the debug mode must be enabled. Doing it
                      enables telemetry recording.
        :param kwargs: Extra keyword arguments that may be useful for derived
                       environments with multiple inheritance, and to allow
                       automatic pipeline wrapper generation.
        """
        # pylint: disable=unused-argument

        # Initialize the interfaces through multiple inheritance
        super().__init__()  # Do not forward extra arguments, if any

        # Backup some user arguments
        self.simulator: Simulator = simulator
        self.step_dt = step_dt
        self.enforce_bounded_spaces = enforce_bounded_spaces
        self.debug = debug

        # Define some proxies for fast access
        self.engine: jiminy.EngineMultiRobot = self.simulator.engine
        self.stepper_state: jiminy.StepperState = self.engine.stepper_state
        self.system_state: jiminy.SystemState = self.engine.system_state
        self.sensors_data: jiminy.sensorsData = dict(self.robot.sensors_data)

        # Internal buffers for physics computations
        self.rg = np.random.RandomState()
        self._seed: Optional[np.uint32] = None
        self.log_path: Optional[str] = None
        self.logfile_action_headers: Optional[FieldDictNested] = None

        # Information about the learning process
        self._info: Dict[str, Any] = {}

        # Number of simulation steps performed
        self.num_steps = -1
        self.max_steps = int(
            self.simulator.simulation_duration_max // self.step_dt)
        self._num_steps_beyond_done: Optional[int] = None

        # Initialize the seed of the environment.
        # Note that reseting the seed also reset robot internal state.
        self.seed()

        # Set robot in neutral configuration for rendering
        qpos = self._neutral()
        update_quantities(self.robot, qpos, use_theoretical_model=False)

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._refresh_action_space()

        # Assertion(s) for type checker
        assert (isinstance(self.observation_space, spaces.Space) and
                isinstance(self.action_space, spaces.Space))

        # Initialize some internal buffers.
        # Note that float64 dtype must be enforced for the action, otherwise
        # it would be impossible to register action to controller's telemetry.
        self._action = zeros(self.action_space, dtype=np.float64)
        self._observation = zeros(self.observation_space)

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Simulator directly, without having to do it through `simulator`.

        .. note::
            This method is not meant to be called manually.
        """
        return getattr(self.__getattribute__('simulator'), name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self.simulator.__dir__()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:   # pylint: disable=broad-except
            # This method must not fail under any circumstances
            pass

    def _controller_handle(self,
                           t: float,
                           q: np.ndarray,
                           v: np.ndarray,
                           sensors_data: jiminy.sensorsData,
                           command: np.ndarray) -> None:
        command[:] = self.compute_command(
            self.get_observation(), deepcopy(self._action))

    def _get_time_space(self) -> gym.Space:
        """Get time space.
        """
        return spaces.Box(
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
        if self.enforce_bounded_spaces:
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
            position_limit_lower = position_limit_lower[joints_position_idx]
            position_limit_upper = position_limit_upper[joints_position_idx]
            velocity_limit = velocity_limit[joints_velocity_idx]

        return spaces.Dict(Q=spaces.Box(low=position_limit_lower,
                                        high=position_limit_upper,
                                        dtype=np.float64),
                           V=spaces.Box(low=-velocity_limit,
                                        high=velocity_limit,
                                        dtype=np.float64))

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
        command_limit = self.robot.command_limit

        state_space = self._get_state_space(use_theoretical_model=False)

        # Replace inf bounds of the action space
        for motor_name in self.robot.motors_names:
            motor = self.robot.get_motor(motor_name)
            motor_options = motor.get_options()
            if not motor_options["enableCommandLimit"]:
                command_limit[motor.joint_velocity_idx] = MOTOR_EFFORT_MAX

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
                    sensor_position_lower = state_space['Q'].low[joint.idx_q]
                    sensor_position_upper = state_space['Q'].high[joint.idx_q]
                sensor_velocity_limit = state_space['V'].high[joint.idx_v]

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
                    -command_limit[motor_idx]
                sensor_space_upper[effort.type][0, sensor_idx] = \
                    +command_limit[motor_idx]

        # Replace inf bounds of the imu sensor space
        if imu.type in sensors_data.keys():
            quat_imu_idx = [
                field.startswith('Quat') for field in imu.fieldnames]
            sensor_space_lower[imu.type][quat_imu_idx, :] = -1.0 - 1e-12
            sensor_space_upper[imu.type][quat_imu_idx, :] = 1.0 + 1e-12

        if self.enforce_bounded_spaces:
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

        return spaces.Dict(OrderedDict(
            (key, spaces.Box(low=min_val, high=max_val, dtype=np.float64))
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
        command_limit = self.robot.command_limit

        # Replace inf bounds of the effort limit if requested
        if self.enforce_bounded_spaces:
            for motor_name in self.robot.motors_names:
                motor = self.robot.get_motor(motor_name)
                motor_options = motor.get_options()
                if not motor_options["enableCommandLimit"]:
                    command_limit[motor.joint_velocity_idx] = \
                        MOTOR_EFFORT_MAX

        # Set the action space.
        # Note that float32 is used instead of float64, because otherwise it
        # would requires the neural network to perform float64 computations
        # or cast the output for no really advantage since the action is
        # directly forwarded to the motors, without intermediary computations.
        action_scale = command_limit[self.robot.motors_velocity_idx]
        self.action_space = spaces.Box(low=-action_scale.astype(np.float32),
                                       high=action_scale.astype(np.float32),
                                       dtype=np.float32)

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

        # Stop the simulator
        self.simulator.stop()

        # Remove external forces, if any
        self.simulator.remove_all_forces()

        # Make sure the environment is properly setup
        self._setup()

        # Make sure the low-level engine has not changed,
        # otherwise some proxies would be corrupted.
        if self.engine is not self.simulator.engine:
            raise RuntimeError(
                "The memory address of the low-level has changed.")

        # Enforce the low-level controller.
        # The robot may have changed, for example if it is randomly generated
        # based on different URDF files. As a result, it is necessary to
        # instantiate a new low-level controller.
        # Note that `BaseJiminyObserverController` is used in place of
        # `jiminy.BaseControllerFunctor`. Although it is less efficient because
        # it adds an extra layer of indirection, it makes it possible to update
        # the controller handle without instantiating a new controller, which
        # is necessary to allow registering telemetry variables before knowing
        # the controller handle in advance.
        controller = BaseJiminyObserverController()
        controller.initialize(self.robot)
        self.simulator.set_controller(controller)

        # Reset the simulator.
        # Note that the controller must be set BEFORE calling 'reset', because
        # otherwise the controller would be corrupted if the robot has changed.
        self.simulator.reset()

        # Re-initialize some shared memories.
        # It must be done because the robot may have changed.
        self.sensors_data = dict(self.robot.sensors_data)

        # Set default action.
        # It will be used for the initial step.
        fill(self._action, 0.0)

        # Reset some internal buffers
        self.num_steps = 0
        self._num_steps_beyond_done = None

        # Create a new log file
        if self.debug:
            fd, self.log_path = tempfile.mkstemp(prefix="log_", suffix=".data")
            os.close(fd)

        # Extract the observer/controller update period.
        # There is no actual observer by default, apart from the robot's state
        # and raw sensors data. Similarly, there is no actual controller by
        # default, apart from forwarding the command torque to the motors.
        engine_options = self.simulator.engine.get_options()
        self.control_dt = self.observe_dt = float(
            engine_options['stepper']['controllerUpdatePeriod'])

        # Run controller hook and set the observer and controller handles
        observer_handle, controller_handle = None, None
        if controller_hook is not None:
            handles = controller_hook()
            if handles is not None:
                observer_handle, controller_handle = handles
        if observer_handle is None:
            observer_handle = self._observer_handle
        self.simulator.controller.set_observer_handle(
            observer_handle, unsafe=True)
        if controller_handle is None:
            controller_handle = self._controller_handle
        self.simulator.controller.set_controller_handle(
            controller_handle, unsafe=True)

        # Register the action to the telemetry
        if isinstance(self._action, np.ndarray) and (
                self._action.size == self.robot.nmotors):
            # Default case: assuming there is one scalar action per motor
            self.logfile_action_headers = [
                ".".join(("action", e)) for e in self.robot.motors_names]
        else:
            # Fallback: Get generic fieldnames otherwise
            self.logfile_action_headers = get_fieldnames(
                self.action_space, "action")
        is_success = register_variables(self.simulator.controller,
                                        self.logfile_action_headers,
                                        self._action)
        if not is_success:
            self.logfile_action_headers = None
            logger.warn(
                "Action must have dtype np.float64 to be registered to the "
                "telemetry.")

        # Sample the initial state and reset the low-level engine
        qpos, qvel = self._sample_state()
        if not jiminy.is_position_valid(
                self.simulator.pinocchio_model, qpos):
            raise RuntimeError(
                "The initial state provided by `_sample_state` is "
                "inconsistent with the dimension or types of joints of the "
                "model.")

        # Start the engine
        self.simulator.start(
            qpos, qvel, None, self.simulator.use_theoretical_model)

        # Initialize the observer.
        # Note that it is responsible of refreshing the environment's
        # observation before anything else, so no need to do it twice.
        self.engine.controller.refresh_observation(
            self.stepper_state.t,
            self.system_state.q,
            self.system_state.v,
            self.sensors_data)

        # Make sure the state is valid, otherwise there `refresh_observation`
        # and `_refresh_observation_space` are probably inconsistent.
        try:
            obs = clip(self.observation_space, self.get_observation())
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "The observation computed by `refresh_observation` is "
                "inconsistent with the observation space defined by "
                "`_refresh_observation_space` at initialization.") from e

        if self.is_done():
            raise RuntimeError(
                "The simulation is already done at `reset`. Check the "
                "implementation of `is_done` if overloaded.")

        # Note that the viewer must be reset if available, otherwise it would
        # keep using the old robot model for display, which must be avoided.
        if self.simulator.is_viewer_available:
            self.simulator.viewer._setup(self.robot)
            self.render(mode='rgb_array')

        return obs

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
        """Terminate the Python Jiminy engine.
        """
        self.simulator.close()

    def step(self,
             action: Optional[SpaceDictNested] = None
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
            set_value(self._action, action)

        # Trying to perform a single simulation step
        is_step_failed = True
        try:
            # Do a single step
            self.simulator.step(self.step_dt)

            # Update the observer at the end of the step. Indeed, internally,
            # it is called at the beginning of the every integration steps,
            # during the controller update.
            self.engine.controller.refresh_observation(
                self.stepper_state.t,
                self.system_state.q,
                self.system_state.v,
                self.sensors_data)

            # Update some internal buffers
            is_step_failed = False
        except RuntimeError as e:
            logger.error(f"Unrecoverable Jiminy engine exception:\n{str(e)}")

        # Get clipped observation
        obs = clip(self.observation_space, self.get_observation())

        # Check if the simulation is over.
        # Note that 'done' is always True if the integration failed or if the
        # maximum number of steps will be exceeded next step.
        done = is_step_failed or (self.num_steps + 1 > self.max_steps) or \
            not self.is_simulation_running or self.is_done()
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
            return obs, 0.0, True, deepcopy(self._info)

        # Compute reward and extra information
        reward = self.compute_reward(info=self._info)

        # Finalize the episode is the simulation is over
        if done and self._num_steps_beyond_done == 0:
            # Write log file if simulation is over (debug mode only)
            if self.debug:
                self.simulator.write_log(self.log_path, format="binary")

            # Compute terminal reward if any
            if self.enable_reward_terminal:
                # Add terminal reward to current reward
                reward += self.compute_reward_terminal(info=self._info)

        # Check if the observation is out-of-bounds, in debug mode only
        if not done and self.debug and \
                not self.observation_space.contains(obs):
            logger.warn("The observation is out-of-bounds.")

        # Update number of (successful) steps
        self.num_steps += 1

        return obs, reward, done, deepcopy(self._info)

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
            return_rgb_array = True
        else:
            raise ValueError(f"Rendering mode {mode} not supported.")
        return self.simulator.render(**{
            'return_rgb_array': return_rgb_array, **kwargs})

    def plot(self, **kwargs: Any) -> None:
        """Display common simulation data and action over time.

        .. Note:
            It adds "Action" tab on top of original `Simulator.plot`.

        :param kwargs: Extra keyword arguments to forward to `simulator.plot`.
        """
        # Call base implementation
        self.simulator.plot(**kwargs)

        # Extract action.
        # If telemetry action fieldnames is a dictionary, it cannot be nested.
        # In such a case, keys corresponds to subplots, and values are
        # individual scalar data over time to be displayed to the same subplot.
        log_data = self.simulator.log_data
        t = log_data["Global.Time"]
        tab_data = {}
        if self.logfile_action_headers is None:
            # It was impossible to register the action to the telemetry, likely
            # because of incompatible dtype. Early return without adding tab.
            return
        if isinstance(self.logfile_action_headers, dict):
            for field, subfields in self.logfile_action_headers.items():
                if not isinstance(subfields, list):
                    logger.error("Action space not supported.")
                    return
                tab_data[field] = {
                    field.split(".", 1)[1]: log_data[
                        ".".join(("HighLevelController", field))]
                    for field in subfields}
        elif isinstance(self.logfile_action_headers, list):
            tab_data.update({
                field.split(".", 1)[1]: log_data[
                    ".".join(("HighLevelController", field))]
                for field in self.logfile_action_headers})

        # Add action tab
        self.figure.add_tab("Action", t, tab_data)

    def replay(self, enable_travelling: bool = True, **kwargs: Any) -> None:
        """Replay the current episode until now.

        :param enable_travelling: Whether or not enable travelling, following
                                  the motion of the root frame of the model.
                                  This parameter is ignored if the model has no
                                  freeflyer.
                                  Optional: True by default.
        :param kwargs: Extra keyword arguments for delegation to
                       `replay.play_trajectories` method.
        """
        # Do not open graphical window automatically if recording requested.
        # Note that backend is closed automatically is there is no viewer
        # backend available at this point, to reduce memory pressure, but it
        # will take time to restart it systematically for every recordings.
        if kwargs.get('record_video_path', None) is not None:
            kwargs['mode'] = 'rgb_array'
            kwargs['close_backend'] = not self.simulator.is_viewer_available

        # Call render before replay in order to take into account custom
        # backend viewer instantiation options, such as initial camera pose.
        self.render(**kwargs)

        if enable_travelling and self.robot.has_freeflyer:
            # It is worth noting that the first and second frames are
            # respectively "universe" and "root_joint", no matter if the robot
            # has a freeflyer or not.
            kwargs['travelling_frame'] = \
                self.robot.pinocchio_model.frames[2].name

        self.simulator.replay(**{'verbose': False, **kwargs})

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
        # Get options
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

        # Enable the friction model
        for motor_name in robot_options["motors"].keys():
            robot_options["motors"][motor_name]["enableFriction"] = True

        # Configure the stepper
        engine_options["stepper"]["iterMax"] = -1
        engine_options["stepper"]["timeout"] = -1
        engine_options["stepper"]["logInternalStepperSteps"] = False
        engine_options["stepper"]["randomSeed"] = self._seed

        # Set options
        self.robot.set_options(robot_options)
        self.simulator.engine.set_options(engine_options)

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the environment.

        By default, the observation is a dictionary gathering the current
        simulation time, the real robot state, and the sensors data.

        .. note::
            This method is called internally by `reset` method at the very end,
            just before computing and returning the initial observation. This
            method, alongside `refresh_observation`, must be overwritten in
            order to define a custom observation space.
        """
        observation_spaces = OrderedDict()
        observation_spaces['t'] = self._get_time_space()
        observation_spaces['state'] = self._get_state_space()
        if self.sensors_data:
            observation_spaces['sensors'] = self._get_sensors_space()
        self.observation_space = spaces.Dict(observation_spaces)

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
        for i in range(len(qpos)):  # pylint: disable=consider-using-enumerate
            lo = self.robot.position_limit_lower[i]
            hi = self.robot.position_limit_upper[i]
            if hi < qpos[i] or qpos[i] < lo:
                qpos[i] = np.mean([lo, hi])

        # Make sure the configuration is valid
        qpos = normalize(self.robot.pinocchio_model, qpos)

        # Return rigid/flexible configuration
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

    def refresh_observation(self) -> None:  # type: ignore[override]
        """Compute the observation based on the current state of the robot.

        .. note::
            There is no way in the current implementation to discriminate the
            initialization of the observation buffer from the next one. A
            workaround is to check if the simulation already stated. Even
            though it is not the same rigorousely speaking, it does the job of
            preserving efficiency.

        .. warning::
            In practice, it updates the internal buffer directly for the sake
            of efficiency.

        :param full_refresh: Whether or not to do a full refresh. This is
                             usually done once, when calling `reset` method.
        """
        # pylint: disable=arguments-differ

        # Assertion(s) for type checker
        assert isinstance(self._observation, dict)

        self._observation['t'][0] = self.stepper_state.t
        if not self.simulator.is_simulation_running:
            (self._observation['state']['Q'],
             self._observation['state']['V']) = self.simulator.state
            if self.sensors_data:
                self._observation['sensors'] = self.sensors_data
        else:
            if self.simulator.use_theoretical_model and self.robot.is_flexible:
                position, velocity = self.simulator.state
                self._observation['state']['Q'][:] = position
                self._observation['state']['V'][:] = velocity

    def compute_command(self,
                        measure: SpaceDictNested,
                        action: np.ndarray
                        ) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        By default, it does not perform any processing. One is responsible of
        overloading this method to clip the action if necessary to make sure it
        does not violate the lower and upper bounds.

        :param measure: Observation of the environment.
        :param action: Desired motors efforts.
        """
        # Check if the action is out-of-bounds, in debug mode only
        if self.debug and not self.action_space.contains(action):
            logger.warn("The action is out-of-bounds.")

        return action

    def is_done(self, *args: Any, **kwargs: Any) -> bool:
        """Determine whether the episode is over.

        By default, it returns True if the observation reaches or exceeds the
        lower or upper limit.

        .. note::
            This method is called right after calling `refresh_observation`, so
            that the internal buffer '_observation' is up-to-date. It can be
            overloaded to implement a custom termination condition for the
            simulation. Moreover, as it is called before `compute_reward`, it
            can be used to update some share intermediary computations to avoid
            redundant calculus and thus improve efficiency.

        :param args: Extra arguments that may be useful for derived
                     environments, for example `Gym.GoalEnv`.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # pylint: disable=unused-argument

        # Assertion(s) for type checker
        assert self.observation_space is not None

        return not self.observation_space.contains(self._observation)

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
        # Initialize base class
        super().__init__(simulator, step_dt, debug)

        # Append default desired and achieved goal spaces to observation space
        goal_space = self._get_goal_space()
        self.observation_space = spaces.Dict(OrderedDict(
            observation=self.observation_space,
            desired_goal=goal_space,
            achieved_goal=goal_space))

        # Define some internal buffers
        self._desired_goal = zeros(goal_space)

    def get_observation(self) -> SpaceDictNested:
        """ TODO: Write documentation.
        """
        return OrderedDict(
            observation=super().get_observation(),
            achieved_goal=self._get_achieved_goal(),
            desired_goal=self._desired_goal)

    def reset(self,
              controller_hook: Optional[Callable[[], Optional[Tuple[
                  Optional[ObserverHandleType],
                  Optional[ControllerHandleType]]]]] = None
              ) -> SpaceDictNested:
        self._desired_goal = self._sample_goal()
        return super().reset(controller_hook)

    # methods to override:
    # ----------------------------

    def _get_goal_space(self) -> gym.Space:
        """Get goal space.

        .. note::
            This method is called internally at init to define the observation
            space. It is called BEFORE `super().reset` so non goal-env-specific
            internal buffers are NOT up-to-date. This method must be overloaded
            while implementing a goal environment.
        """
        raise NotImplementedError

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
            This method can be called by `refresh_observation` to get the
            currently achieved goal. This method must be overloaded while
            implementing a goal environment.

        :returns: Currently achieved goal.
        """
        raise NotImplementedError

    def is_done(self,  # type: ignore[override]
                achieved_goal: Optional[SpaceDictNested] = None,
                desired_goal: Optional[SpaceDictNested] = None) -> bool:
        """Determine whether a termination condition has been reached.

        By default, it uses the termination condition inherited from normal
        environment.

        .. note::
            This method is called right after calling `refresh_observation`, so
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
