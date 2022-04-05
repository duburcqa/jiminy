"""Generic gym environment specifically tailored to work with Jiminy Simulator
as backend physics engine, and Jiminy Viewer as 3D visualizer. It implements
the official OpenAI Gym API and extended it to add more functionalities.
"""
import os
import tempfile
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Mapping
from typing import (
    Optional, Tuple, Dict, Any, Callable, List, Union, Iterator,
    Mapping as MappingT, MutableMapping as MutableMappingT)

import tree
import numpy as np
import gym
from gym import logger, spaces

import jiminy_py.core as jiminy
from jiminy_py.core import (EncoderSensor as encoder,
                            EffortSensor as effort,
                            ContactSensor as contact,
                            ForceSensor as force,
                            ImuSensor as imu)
from jiminy_py.viewer.viewer import (
    DEFAULT_CAMERA_XYZRPY_REL, check_display_available, Viewer)
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.simulator import Simulator
from jiminy_py.log import extract_data_from_log

from pinocchio import neutral, normalize, framesForwardKinematics

from ..utils import (zeros,
                     fill,
                     set_value,
                     clip,
                     get_fieldnames,
                     register_variables,
                     FieldNested,
                     DataNested)
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


class _LazyDictItemFilter(Mapping):
    def __init__(self,
                 dict_packed: MappingT[str, Tuple[Any, ...]],
                 item_index: int) -> None:
        self.dict_packed = dict_packed
        self.item_index = item_index

    def __getitem__(self, name: str) -> Any:
        return self.dict_packed[name][self.item_index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.dict_packed)

    def __len__(self) -> int:
        return len(self.dict_packed)


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

        # Set the available rendering modes
        self.metadata['render.modes'] = ['rgb_array']
        if check_display_available():
            self.metadata['render.modes'].append('human')

        # Define some proxies for fast access
        self.engine: jiminy.EngineMultiRobot = self.simulator.engine
        self.stepper_state: jiminy.StepperState = self.engine.stepper_state
        self.system_state: jiminy.SystemState = self.engine.system_state
        self.sensors_data: jiminy.sensorsData = dict(self.robot.sensors_data)

        # Store references to the variables to register to the telemetry
        self._registered_variables: MutableMappingT[
            str, Tuple[FieldNested, DataNested]] = {}
        self.log_headers: MappingT[str, FieldNested] = _LazyDictItemFilter(
            self._registered_variables, 0)

        # Internal buffers for physics computations
        self._seed: List[np.uint32] = []
        self.rg = np.random.Generator(np.random.SFC64())
        self.log_path: Optional[str] = None

        # Whether evaluation mode is active
        self.is_training = True

        # Whether play interactive mode is active
        self._is_interactive = False

        # Information about the learning process
        self._info: Dict[str, Any] = {}

        # Keep track of cumulative reward
        self.total_reward = 0.0

        # Number of simulation steps performed
        self.num_steps = -1
        self.max_steps = 0
        self._num_steps_beyond_done: Optional[int] = None

        # Initialize the seed of the environment.
        # Note that resetting the seed also reset robot internal state.
        self.seed()

        # Set robot in neutral configuration
        qpos = self._neutral()
        framesForwardKinematics(
            self.robot.pinocchio_model, self.robot.pinocchio_data, qpos)

        # Refresh the observation and action spaces.
        # Note that it is necessary to refresh the action space before the
        # observation one, since it may be useful to observe the action.
        self._initialize_action_space()
        self._initialize_observation_space()

        # Assertion(s) for type checker
        assert (isinstance(self.observation_space, spaces.Space) and
                isinstance(self.action_space, spaces.Space))

        # Initialize some internal buffers.
        # Note that float64 dtype must be enforced for the action, otherwise
        # it would be impossible to register action to controller's telemetry.
        self._action = zeros(self.action_space, dtype=np.float64)
        self._observation = zeros(self.observation_space)

        # Register the action to the telemetry automatically iif there is
        # exactly one scalar action per motor.
        if isinstance(self._action, np.ndarray):
            action_size = self._action.size
            if action_size > 0 and action_size == self.robot.nmotors:
                action_headers = [
                    ".".join(("action", e)) for e in self.robot.motors_names]
                self.register_variable("action", self._action, action_headers)

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
        """Thin wrapper around user-specified `compute_command` method.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
        """
        assert self._action is not None
        command[:] = self.compute_command(
            self.get_observation(), self._action)

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
        overload `_initialize_observation_space` to customize the observation
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

        return spaces.Dict(OrderedDict(
            Q=spaces.Box(low=position_limit_lower,
                         high=position_limit_upper,
                         dtype=np.float64),
            V=spaces.Box(low=-velocity_limit,
                         high=velocity_limit,
                         dtype=np.float64)))

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
            rather overload `_initialize_observation_space` to customize the
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

    def _initialize_action_space(self) -> None:
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

        # Set the action space
        action_scale = command_limit[self.robot.motors_velocity_idx]
        self.action_space = spaces.Box(
            low=-action_scale, high=action_scale, dtype=np.float64)

    def register_variable(self,
                          name: str,
                          value: DataNested,
                          fieldnames: Optional[
                              Union[str, FieldNested]] = None,
                          namespace: Optional[str] = None) -> None:
        """ TODO: Write documentation.
        """
        # Create default fieldnames if not specified
        if fieldnames is None:
            fieldnames = get_fieldnames(value, name)

        # Store string in a list
        if isinstance(fieldnames, str):
            fieldnames = [fieldnames]

        # Prepend with namespace if requested
        if namespace:
            fieldnames = tree.map_structure(
                lambda key: ".".join(filter(None, (namespace, key))),
                fieldnames)

        # Early return with a warning is fieldnames is empty
        if not fieldnames:
            logger.warn("'value' or 'fieldnames' cannot be empty.")
            return

        # Check if variable can be registered successfully to the telemetry.
        # Note that a dummy controller must be created to avoid using the
        # actual one to keep control of when registering will take place.
        try:
            is_success = register_variables(
                jiminy.BaseController(), fieldnames, value)
        except ValueError as e:
            raise ValueError(
                f"'fieldnames' ({fieldnames})' if not consistent with the "
                f"'value' ({value})") from e

        # Combine namespace and variable name if provided
        name = ".".join(filter(None, (namespace, name)))

        # Store the header and a reference to the variable if successful
        if is_success:
            self._registered_variables[name] = (fieldnames, value)

    def reset(self,
              controller_hook: Optional[Callable[[], Optional[Tuple[
                  Optional[ObserverHandleType],
                  Optional[ControllerHandleType]]]]] = None
              ) -> DataNested:
        """Reset the environment.

        In practice, it resets the backend simulator and set the initial state
        of the robot. The initial state is obtained by calling '_sample_state'.
        This method is also in charge of setting the initial action (at the
        beginning) and observation (at the end).

        .. warning::
            It starts the simulation immediately. As a result, it is not
            possible to change the robot (included options), nor to register
            log variable. The only way to do so is via 'controller_hook'.

        :param controller_hook: Used internally for chaining multiple
                                `BasePipelineWrapper`. It is not meant to be
                                defined manually.
                                Optional: None by default.

        :returns: Initial observation of the episode.
        """
        # pylint: disable=arguments-differ

        # Assertion(s) for type checker
        assert self.observation_space is not None
        assert self._action is not None

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

        # Re-initialize some shared memories.
        # It must be done because the robot may have changed.
        self.sensors_data = dict(self.robot.sensors_data)

        # Enforce the low-level controller.
        # The robot may have changed, for example it could be randomly
        # generated, which would corrupt the old controller. As a result, it is
        # necessary to either instantiate a new low-level controller and to
        # re-initialize the existing one by calling `controller.initialize`
        # method BEFORE calling `reset` method because otherwise it would
        # cause a segfault. In practice, `BaseJiminyObserverController` must be
        # used because it enables to define observer and controller handles
        # seperately, while dealing with all the logics internally. This extra
        # layer of indirection makes it computionally less efficient than
        # `jiminy.BaseControllerFunctor` but it is a small price to pay.
        controller = BaseJiminyObserverController()
        controller.initialize(self.robot)
        self.simulator.set_controller(controller)

        # Reset the simulator.
        # Do NOT remove all forces since it has already been done before, and
        # because it would make it impossible to register forces in  `_setup`.
        self.simulator.reset(remove_all_forces=False)

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
        # The controller update period is used by default for the observer if
        # it was not specify by the user in `_setup`.
        engine_options = self.simulator.engine.get_options()
        self.control_dt = float(
            engine_options['stepper']['controllerUpdatePeriod'])
        if self.observe_dt < 0.0:
            self.observe_dt = self.control_dt

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

        # Configure the maximum number of steps
        self.max_steps = int(
            self.simulator.simulation_duration_max / self.step_dt)

        # Register user-specified variables to the telemetry
        for header, value in self._registered_variables.values():
            register_variables(self.simulator.controller, header, value)

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

        # Initialize shared buffers
        self._initialize_buffers()

        # Update shared buffers
        self._refresh_buffers()

        # Initialize the observer.
        # Note that it is responsible of refreshing the environment's
        # observation before anything else, so no need to do it twice.
        self.engine.controller.refresh_observation(
            self.stepper_state.t,
            self.system_state.q,
            self.system_state.v,
            self.sensors_data)

        # Make sure the state is valid, otherwise there `refresh_observation`
        # and `_initialize_observation_space` are probably inconsistent.
        try:
            obs = clip(self.observation_space, self.get_observation())
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "The observation computed by `refresh_observation` is "
                "inconsistent with the observation space defined by "
                "`_initialize_observation_space` at initialization.") from e

        # Make sure there is no 'nan' value in observation
        for value in tree.flatten(obs):
            if np.isnan(value).any():
                raise RuntimeError(
                    f"'nan' value found in observation ({obs}). Something "
                    "went wrong with `refresh_observation` method.")

        # The simulation cannot be done before doing a single step.
        if self.is_done():
            raise RuntimeError(
                "The simulation is already done at `reset`. Check the "
                "implementation of `is_done` if overloaded.")

        # Reset cumulative reward
        self.total_reward = 0.0

        # Note that the viewer must be reset if available, otherwise it would
        # keep using the old robot model for display, which must be avoided.
        if self.simulator.is_viewer_available:
            self.simulator.viewer._setup(self.robot)
            self.render(mode='rgb_array')

        return obs

    def seed(self, seed: Optional[int] = None) -> List[np.uint32]:
        """Specify the seed of the environment.

        .. warning::
            It also resets the low-level jiminy Engine. Therefore one must call
            the `reset` method manually afterward.

        :param seed: Random seed, as a positive integer.
                     Optional: A strongly random seed will be generated by gym
                     if omitted.

        :returns: Updated seed of the environment
        """
        # Generate a sequence of 3 bytes uint32 seeds
        self._seed = list(np.random.SeedSequence(seed).generate_state(3))

        # Instantiate a new random number generator based on the provided seed
        self.rg = np.random.Generator(np.random.SFC64(self._seed))

        # Reset the seed of Jiminy Engine
        self.simulator.seed(self._seed[0])

        return self._seed

    def close(self) -> None:
        """Terminate the Python Jiminy engine.
        """
        self.simulator.close()

    def step(self,
             action: Optional[DataNested] = None
             ) -> Tuple[DataNested, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Action to perform. `None` to not update the action.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information
        """
        # Assertion(s) for type checker
        assert self._action is not None

        # Make sure a simulation is already running
        if not self.simulator.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please call `reset` before `step`.")

        # Update of the action to perform if provided
        if action is not None:
            # Make sure the action is valid
            for value in tree.flatten(action):
                if np.isnan(value).any():
                    raise RuntimeError(
                        f"'nan' value found in action ({action}).")

            # Update the action
            set_value(self._action, action)

        # Perform a single simulation step
        self.simulator.step(self.step_dt)

        # Update shared buffers
        self._refresh_buffers()

        # Update the observer at the end of the step. Indeed, internally,
        # it is called at the beginning of the every integration steps,
        # during the controller update.
        self.engine.controller.refresh_observation(
            self.stepper_state.t,
            self.system_state.q,
            self.system_state.v,
            self.sensors_data)

        # Get clipped observation
        obs = clip(self.observation_space, self.get_observation())

        # Make sure there is no 'nan' value in observation
        if np.isnan(self.system_state.a).any():
            raise RuntimeError(
                "The acceleration of the system is 'nan'. Something went "
                "wrong with jiminy engine.")

        # Reset the extra information buffer
        self._info = {}

        # Check if the simulation is over.
        # Note that 'done' is always True if the integration failed or if the
        # maximum number of steps will be exceeded next step.
        done = not self.simulator.is_simulation_running or \
            self.num_steps >= self.max_steps or self.is_done()

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

        # Make sure the reward is not 'nan'
        if np.isnan(reward):
            raise RuntimeError(
                "The reward is 'nan'. Something went wrong with "
                "`compute_reward` or `compute_reward_terminal` methods.")

        # Update cumulative reward
        self.total_reward += reward

        # Update number of (successful) steps
        self.num_steps += 1

        return obs, reward, done, deepcopy(self._info)

    def render(self,
               mode: Optional[str] = None,
               **kwargs: Any) -> Optional[np.ndarray]:
        """Render the world.

        :param mode: Rendering mode. It can be either 'human' to display the
                     current simulation state, or 'rgb_array' to return a
                     snapshot as an RGB array without showing it on the screen.
                     Optional: 'human' by default if available, 'rgb_array'
                     otherwise.
        :param kwargs: Extra keyword arguments to forward to
                       `jiminy_py.simulator.Simulator.render` method.

        :returns: RGB array if 'mode' is 'rgb_array', None otherwise.
        """
        # Handling of default rendering mode
        if mode is None:
            if 'human' in self.metadata['render.modes']:
                mode = 'human'
            else:
                mode = 'rgb_array'

        # Make sure the rendering mode is valid.
        # Note that it is not possible to raise an exception, because the
        # default is overwritten by gym wrappers by mistake to 'human'.
        if mode not in self.metadata['render.modes']:
            mode = 'rgb_array'

        # Call base implementation
        return self.simulator.render(
            return_rgb_array=(mode == 'rgb_array'), **kwargs)

    def plot(self, **kwargs: Any) -> None:
        """Display common simulation data and action over time.

        .. Note:
            It adds "Action" tab on top of original `Simulator.plot`.

        :param kwargs: Extra keyword arguments to forward to `simulator.plot`.
        """
        # Call base implementation
        self.simulator.plot(**kwargs)

        # Extract log data
        log_data = self.simulator.log_data
        if not log_data:
            raise RuntimeError(
                "Nothing to plot. Please run a simulation before calling "
                "`plot` method.")

        # Extract action.
        # If telemetry action fieldnames is a dictionary, it cannot be nested.
        # In such a case, keys corresponds to subplots, and values are
        # individual scalar data over time to be displayed to the same subplot.
        t = log_data["Global.Time"]
        tab_data = {}
        action_headers = self.log_headers.get("action", None)
        if action_headers is None:
            # It was impossible to register the action to the telemetry, likely
            # because of incompatible dtype. Early return without adding tab.
            return
        if isinstance(action_headers, dict):
            for group, fieldnames in action_headers.items():
                if not isinstance(fieldnames, list):
                    logger.error("Action space not supported by this method.")
                    return
                tab_data[group] = {
                    ".".join(key.split(".")[1:]): value
                    for key, value in extract_data_from_log(
                        log_data, fieldnames, as_dict=True).items()}
        elif isinstance(action_headers, list):
            tab_data.update({
                ".".join(key.split(".")[1:]): value
                for key, value in extract_data_from_log(
                    log_data, action_headers, as_dict=True).items()})

        # Add action tab
        self.simulator.figure.add_tab("Action", t, tab_data)

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

        # Stop any running simulation before replay if `is_done` is True
        if self.simulator.is_simulation_running and self.is_done():
            self.simulator.stop()

        # Set default camera pose if viewer not already available
        if not self.simulator.is_viewer_available and \
                self.robot.has_freeflyer and not Viewer.has_gui():
            # Get root frame name.
            # The first and second frames are respectively "universe" no matter
            # if the robot has a freeflyer or not, and the second one is the
            # freeflyer joint "root_joint" if any.
            root_name = self.robot.pinocchio_model.frames[2].name

            # Set default camera pose options.
            # Note that the actual signature is hacked to set relative pose.
            kwargs["camera_xyzrpy"] = (*DEFAULT_CAMERA_XYZRPY_REL, root_name)

        # Call render before replay in order to take into account custom
        # backend viewer instantiation options, such as initial camera pose,
        # and to update the ground profile.
        self.render(update_ground_profile=True, **kwargs)

        # Set default travelling options
        if enable_travelling and self.robot.has_freeflyer:
            kwargs['travelling_frame'] = \
                self.robot.pinocchio_model.frames[2].name

        self.simulator.replay(**{'verbose': False, **kwargs})

    @staticmethod
    def play_interactive(env: Union["BaseJiminyEnv", gym.Wrapper],
                         enable_travelling: Optional[bool] = None,
                         start_paused: bool = True,
                         enable_is_done: bool = True,
                         verbose: bool = True,
                         **kwargs: Any) -> None:
        """Activate interact mode enabling to control the robot using keyboard.

        It stops automatically as soon as 'done' flag is True. One has to press
        a key to start the interaction. If no key is pressed, the action is
        not updated and the previous one keeps being sent to the robot.

        .. warning::
            This method requires `_key_to_action` method to be implemented by
            the user by overloading it, otherwise it raises an exception.

        :param env: `BaseJiminyEnv` environment instance to play with,
                    eventually wrapped by composition, typically using
                    `gym.Wrapper`.
        :param enable_travelling: Whether or not enable travelling, following
                                  the motion of the root frame of the model.
                                  This parameter is ignored if the model has no
                                  freeflyer.
                                  Optional: Enabled by default iif 'panda3d'
                                  viewer backend is used.
        :param start_paused: Whether or not to start in pause.
                             Optional: Enabled by default.
        :param verbose: Whether or not to display status messages.
        :param kwargs: Extra keyword arguments to forward to `_key_to_action`
                       method.
        """
        # Get unwrapped environment
        if isinstance(env, gym.Wrapper):
            self = env.unwrapped
        else:
            self = env

        # Make sure the unwrapped environment derive from this class
        assert isinstance(self, BaseJiminyEnv), (
            "Unwrapped environment must derived from `BaseJiminyEnv`.")

        # Enable play interactive flag and make sure training flag is disabled
        is_training = self.is_training
        self._is_interactive = True
        self.is_training = False

        # Make sure viewer gui is open, so that the viewer will shared external
        # forces with the robot automatically.
        if not (self.simulator.is_viewer_available and
                self.simulator.viewer.has_gui()):
            env.render(update_ground_profile=False)

        # Reset the environnement
        obs = env.reset()
        reward = None

        # Refresh the ground profile
        env.render(update_ground_profile=True)

        # Enable travelling
        if enable_travelling is None:
            enable_travelling = \
                self.simulator.viewer.backend.startswith('panda3d')
        enable_travelling = enable_travelling and self.robot.has_freeflyer
        if enable_travelling:
            tracked_frame = self.robot.pinocchio_model.frames[2].name
            self.simulator.viewer.attach_camera(tracked_frame)

        # Refresh the scene once again to update camera placement
        env.render()

        # Define interactive loop
        def _interact(key: Optional[str] = None) -> bool:
            nonlocal obs, reward, enable_is_done
            action = self._key_to_action(
                key, obs, reward, **{"verbose": verbose, **kwargs})
            obs, reward, done, _ = env.step(action)
            env.render()
            if not enable_is_done and env.robot.has_freeflyer:
                return env.system_state.q[2] < 0.0
            return done

        # Run interactive loop
        loop_interactive(max_rate=self.step_dt,
                         start_paused=start_paused,
                         verbose=verbose)(_interact)()

        # Disable travelling if it enabled
        if enable_travelling:
            self.simulator.viewer.detach_camera()

        # Stop the simulation to unlock the robot.
        # It will enable to display contact forces for replay.
        if self.simulator.is_simulation_running:
            self.simulator.stop()

        # Disable play interactive mode flag and restore training flag
        self._is_interactive = False
        self.is_training = is_training

    @staticmethod
    def evaluate(env: gym.Env,
                 policy_fn: Callable[
                     [DataNested, Optional[float]], DataNested],
                 seed: Optional[int] = None,
                 horizon: Optional[int] = None,
                 enable_stats: bool = True,
                 enable_replay: bool = True,
                 **kwargs: Any) -> List[Dict[str, Any]]:
        r"""Evaluate a policy on the environment over a complete episode.

        :param env: `BaseJiminyEnv` environment instance to play with,
                    eventually wrapped by composition, typically using
                    `gym.Wrapper`.
        :param policy_fn:
            .. raw:: html

                Policy to evaluate as a callback function. It must have the
                following signature (**rew** = None at reset):

            | policy_fn\(**obs**: DataNested,
            |            **reward**: Optional[float]
            |            \) -> DataNested  # **action**
        :param seed: Seed of the environment to be used for the evaluation of
                     the policy.
                     Optional: Random seed if not provided.
        :param horizon: Horizon of the simulation, namely maximum number of
                        steps before termination. `None` to disable.
                        Optional: Disabled by default.
        :param enable_stats: Whether or not to print high-level statistics
                             after simulation.
                             Optional: Enabled by default.
        :param enable_replay: Whether or not to enable replay of the
                              simulation, and eventually recording if the extra
                              keyword argument `record_video_path` is provided.
                              Optional: Enabled by default.
        :param kwargs: Extra keyword arguments to forward to the `replay`
                       method if replay is requested.
        """
        # Initialize frame stack

        # Make sure evaluation mode is enabled
        is_training = env.is_training
        if is_training:
            env.eval()

        # Reset the seed of the environment
        env.seed(seed)

        # Initialize the simulation
        obs = env.reset()
        reward = None

        # Run the simulation
        try:
            info_episode = []
            done = False
            while not done:
                action = policy_fn(obs, reward)
                obs, reward, done, info = env.step(action)
                info_episode.append(info)
                if done or (horizon is not None and env.num_steps > horizon):
                    break
        except KeyboardInterrupt:
            pass

        # Restore training mode if it was enabled
        if is_training:
            env.train()

        # Display some statistic if requested
        if enable_stats:
            print("env.num_steps:", env.num_steps)
            print("cumulative reward:", env.total_reward)

        # Replay the result if requested
        if enable_replay:
            try:
                env.replay(**kwargs)
            except Exception as e:  # pylint: disable=broad-except
                # Do not fail because of replay/recording exception
                logger.warn(str(e))

        return info_episode

    def train(self) -> None:
        """Sets the environment in training mode.

        .. note::
            This mode is enabled by default.
        """
        self.is_training = True

    def eval(self) -> None:
        """Sets the environment in evaluation mode.

        This has any effect only on certain environment. See documentations of
        particular environment for details of their behaviors in training and
        evaluation modes, if they are affected. It can be used to activate
        clipping or some filtering of the action specifical at evaluation time.
        """
        self.is_training = False

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        """Configure the environment. It must guarantee that its internal state
        is valid after calling this method.

        By default, it enforces some options of the engine.

        .. warning::
            Beware this method is called BEFORE `observe_dt` and
            `controller_dt` are properly set, so one cannot rely on it at this
            point. Yet, `step_dt` is available and should always be. One can
            still access the low-level controller update period through
            `engine_options['stepper']['controllerUpdatePeriod']`.

        .. note::
            The user must overload this method to enforce custom observer
            update period, otherwise it will be the same of the controller.

        .. note::
            This method is called internally by `reset` methods.
        """
        # Call base implementation
        super()._setup()

        # Configure the low-level integrator
        engine_options = self.simulator.engine.get_options()
        engine_options["stepper"]["iterMax"] = 0
        engine_options["stepper"]["timeout"] = 0.0
        engine_options["stepper"]["logInternalStepperSteps"] = False
        self.simulator.engine.set_options(engine_options)

        # Set robot in neutral configuration
        qpos = self._neutral()
        framesForwardKinematics(
            self.robot.pinocchio_model, self.robot.pinocchio_data, qpos)

    def _initialize_observation_space(self) -> None:
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
        position_limit_lower = self.robot.position_limit_lower
        position_limit_upper = self.robot.position_limit_upper
        for idx, val in enumerate(qpos):
            lo, hi = position_limit_lower[idx], position_limit_upper[idx]
            if hi < val or val < lo:
                qpos[idx] = 0.5 * (lo + hi)

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

        # Make sure the configuration is not out-of-bound
        pinocchio_model = self.robot.pinocchio_model
        position_limit_lower = pinocchio_model.lowerPositionLimit
        position_limit_upper = pinocchio_model.upperPositionLimit
        qpos = np.minimum(np.maximum(
            qpos, position_limit_lower), position_limit_upper)

        # Make sure the configuration is normalized
        qpos = normalize(pinocchio_model, qpos)

        # Make sure the robot impacts the ground
        if self.robot.has_freeflyer:
            engine_options = self.simulator.engine.get_options()
            ground_fun = engine_options['world']['groundProfile']
            compute_freeflyer_state_from_fixed_body(
                self.robot, qpos, ground_profile=ground_fun)

        # Zero velocity
        qvel = np.zeros(self.simulator.pinocchio_model.nv)

        return qpos, qvel

    def _initialize_buffers(self) -> None:
        """Initialize internal buffers for fast access to shared memory or to
        avoid redundant computations.

        .. note::
            This method is called at `reset`, right after
            `self.simulator.start`. At this point, the simulation is running
            but `refresh_observation` has never been called, so that it can be
            used to initialize buffers involving the engine state but required
            to refresh the observation. Note that it is not appropriate to
            initialize buffers that would be used by `compute_command`.

        .. note::
            Buffers requiring manual update must be refreshed using
            `_refresh_buffers` method.
        """

    def _refresh_buffers(self) -> None:
        """Refresh internal buffers that must be updated manually.

        .. note::
            This method is called right after every internal `engine.step`, so
            it is the right place to update shared data between `is_done` and
            `compute_reward`.

        .. note::
            `_initialize_buffers` method can be used to initialize buffers that
            may requires special care.

        .. warning::
            Be careful when using it to update buffers that are used by
            `refresh_observation`. The later is called at `self.observe_dt`
            update period, while the others are called at `self.step_dt` update
            period. `self.observe_dt` is likely to be different from
            `self.step_dt`, unless configured manually when overloading
            `_setup` method.
        """

    def refresh_observation(self) -> None:  # type: ignore[override]
        """Compute the observation based on the current state of the robot.

        .. note::
            This method is called and the end of every low-level `Engine.step`.

        .. note::
            Note that `np.nan` values will be automatically clipped to 0.0 by
            `get_observation` method before return it, so it is valid.

        .. warning::
            In practice, it updates the internal buffer directly for the sake
            of efficiency.

            As a side note, there is no way in the current implementation to
            discriminate the initialization of the observation buffer from the
            next one. The workaround is to check if the simulation already
            started. Even though it is not the same rigorously speaking, it
            does the job here since it is only about preserving efficiency.
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
                        measure: DataNested,
                        action: DataNested
                        ) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        By default, it does not perform any processing. One is responsible of
        overloading this method to clip the action if necessary to make sure it
        does not violate the lower and upper bounds.

        .. warning::
            There is not good place to initialize buffers that are necessary to
            compute the command. The only solution for now is to define
            initialization inside this method itself, using the safeguard
            `if not self.simulator.is_simulation_running:`.

        :param measure: Observation of the environment.
        :param action: Desired motors efforts.
        """
        # Assertion(s) for type checker
        assert self.action_space is not None

        # Check if the action is out-of-bounds, in debug mode only
        if self.debug and not self.action_space.contains(action):
            logger.warn("The action is out-of-bounds.")

        if not isinstance(action, np.ndarray):
            raise RuntimeError(
                "`BaseJiminyEnv.compute_command` must be overloaded unless "
                "the action space has type `gym.spaces.Box`.")

        return action

    def is_done(self, *args: Any, **kwargs: Any) -> bool:
        """Determine whether the episode is over.

        By default, it returns True if the observation reaches or exceeds the
        lower or upper limit. It must be overloaded to implement a custom
        termination condition for the simulation.

        .. note::
            This method is called after `refresh_observation`, so that the
            internal buffer '_observation' is up-to-date.

        :param args: Extra arguments that may be useful for derived
                     environments, for example `Gym.GoalEnv`.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # pylint: disable=unused-argument

        # Assertion(s) for type checker
        assert self.observation_space is not None

        return not self.observation_space.contains(self._observation)

    def _key_to_action(self,
                       key: Optional[str],
                       obs: DataNested,
                       reward: Optional[float],
                       **kwargs: Any) -> DataNested:
        """Mapping from input keyboard keys to actions.

        .. note::
            This method is called before `step` method systematically, even if
            not key has been pressed, or reward is not defined. In such a case,
            the value is `None`.

        .. note::
            The mapping can be state dependent, and the key can be used for
            something different than computing the action directly. For
            instance, one can provide as extra argument to this method a
            custom policy taking user parameters mapped to keyboard in input.

        .. warning::
            Overloading this method is required for calling `play_interactive`
            method.

        :param key: Key pressed by the user as a string. `None` if no key has
                    been pressed since the last step of the environment.
        :param obs: Previous observation from last step of the environment.
                    It is always available, included right after `reset`.
        :param reward: Previous reward from last step of the environment.
                       Not available before first step right after `reset`.
        :param kwargs: Extra keyword argument provided by the user when calling
                       `play_interactive` method.

        :returns: Action to forward to the environment.
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


class BaseJiminyGoalEnv(BaseJiminyEnv):
    """A goal-based environment. It functions just as any regular OpenAI Gym
    environment but it imposes a required structure on the observation_space.
    More concretely, the observation space is required to contain at least
    three elements, namely `observation`, `desired_goal`, and `achieved_goal`.
    Here, `desired_goal` specifies the goal that the agent should attempt to
    achieve. `achieved_goal` is the goal that it currently achieved instead.
    `observation` contains the actual observations of the environment as per
    usual.
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

    def get_observation(self) -> DataNested:
        """Get post-processed observation.

        It gathers the original observation from the environment with the
        currently achieved and desired goal, as a dictionary. See
        `ObserverInterface.get_observation` documentation for details.
        """
        return OrderedDict(
            observation=super().get_observation(),
            achieved_goal=self._get_achieved_goal(),
            desired_goal=self._desired_goal)

    def reset(self,
              controller_hook: Optional[Callable[[], Optional[Tuple[
                  Optional[ObserverHandleType],
                  Optional[ControllerHandleType]]]]] = None
              ) -> DataNested:
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

    def _sample_goal(self) -> DataNested:
        """Sample a goal randomly.

        .. note::
            This method is called internally by `reset` to sample the new
            desired goal that the agent will have to achieve. It is called
            BEFORE `super().reset` so non goal-env-specific internal buffers
            are NOT up-to-date. This method must be overloaded while
            implementing a goal environment.
        """
        raise NotImplementedError

    def _get_achieved_goal(self) -> DataNested:
        """Compute the achieved goal based on current state of the robot.

        .. note::
            This method can be called by `refresh_observation` to get the
            currently achieved goal. This method must be overloaded while
            implementing a goal environment.

        :returns: Currently achieved goal.
        """
        raise NotImplementedError

    def is_done(self,  # type: ignore[override]
                achieved_goal: Optional[DataNested] = None,
                desired_goal: Optional[DataNested] = None) -> bool:
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
                       achieved_goal: Optional[DataNested] = None,
                       desired_goal: Optional[DataNested] = None,
                       *, info: Dict[str, Any]) -> float:
        """Compute the step reward. This externalizes the reward function and
        makes it dependent on a desired goal and the one that was achieved. If
        you wish to include additional rewards that are independent of the
        goal, you can include the necessary values to derive it in 'info' and
        compute it accordingly.

        :param achieved_goal: Achieved goal. `None` to evalute the reward for
                              currently achieved goal.
        :param desired_goal: Desired goal. `None` to evalute the reward for
                             currently desired goal.
        :param info: Dictionary of extra information for monitoring.

        Returns:
            The reward that corresponds to the provided achieved goal wrt to
            the desired goal. Note that the following should always hold true:
            ```
            obs, reward, done, info = env.step()
            assert reward == env.compute_reward(
                obs['achieved_goal'], obs['desired_goal'], info=info)
            ```
        """
        # pylint: disable=arguments-differ

        raise NotImplementedError
