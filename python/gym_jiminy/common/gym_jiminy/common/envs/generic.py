"""Generic gym environment specifically tailored to work with Jiminy Simulator
as backend physics engine, and Jiminy Viewer as 3D visualizer. It implements
the official OpenAI Gym API and extended it to add more functionalities.
"""
import os
import math
import weakref
import logging
import tempfile
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
from typing import (
    Dict, Any, List, cast, no_type_check, Optional, Tuple, Callable, Union,
    SupportsFloat, Iterator, Generic, Sequence as SequenceT,
    Mapping as MappingT, MutableMapping as MutableMappingT)

import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

import jiminy_py.core as jiminy
from jiminy_py import tree
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)
from jiminy_py.dynamics import compute_freeflyer_state_from_fixed_body
from jiminy_py.log import extract_variables_from_log
from jiminy_py.simulator import Simulator, TabbedFigure
from jiminy_py.viewer.viewer import (DEFAULT_CAMERA_XYZRPY_REL,
                                     interactive_mode,
                                     get_default_backend,
                                     Viewer)
from jiminy_py.viewer.replay import viewer_lock

import pinocchio as pin

from ..utils import (FieldNested,
                     DataNested,
                     zeros,
                     is_nan,
                     fill,
                     build_clip,
                     build_copyto,
                     build_contains,
                     get_fieldnames,
                     get_robot_state_space,
                     get_robot_measurements_space,
                     register_variables)
from ..bases import (DT_EPS,
                     Obs,
                     Act,
                     InfoType,
                     SensorMeasurementStackMap,
                     EngineObsType,
                     InterfaceJiminyEnv)
from ..quantities import QuantityManager

from .internal import loop_interactive


# Maximum realtime slowdown of simulation steps before triggering timeout error
TIMEOUT_RATIO = 15

# Absolute tolerance when checking that observations are valid
OBS_CONTAINS_TOL = 0.01


PolicyCallbackFun = Callable[
    [Obs, Optional[Act], Optional[float], bool, bool, InfoType], Act]


LOGGER = logging.getLogger(__name__)


class _LazyDictItemFilter(Mapping):
    def __init__(self,
                 dict_packed: MappingT[str, SequenceT[Any]],
                 item_index: int) -> None:
        self.dict_packed = dict_packed
        self.item_index = item_index

    def __getitem__(self, name: str) -> Any:
        return self.dict_packed[name][self.item_index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.dict_packed)

    def __len__(self) -> int:
        return len(self.dict_packed)


class BaseJiminyEnv(InterfaceJiminyEnv[Obs, Act],
                    Generic[Obs, Act]):
    """Base class to train an agent in Gym OpenAI using Jiminy simulator for
    physics computations.

    It creates an Gym environment wrapping an already instantiated Jiminy
    simulator and behaves like any standard Gym environment.

    The observation space is a dictionary gathering the current simulation
    time, the real robot state, and the sensors data. The action is a vector
    gathering the torques of the actuator of the robot.

    There is no reward by default. It is up to the user to overload this class
    to implement one. It has been designed to be highly flexible and easy to
    customize by overloading it to fit the vast majority of users' needs.
    """

    derived: InterfaceJiminyEnv
    """Top-most block from which this environment is part of when leveraging
    modular pipeline design capability.
    """

    def __init__(self,
                 simulator: Simulator,
                 step_dt: float,
                 simulation_duration_max: float = 86400.0,
                 debug: bool = False,
                 render_mode: Optional[str] = None,
                 **kwargs: Any) -> None:
        r"""
        :param simulator: Jiminy Python simulator used for physics
                          computations. It must be fully initialized.
        :param step_dt: Environment timestep for learning. Note that it is
                        independent from the controller and observation update
                        periods. The latter are configured via
                        `engine.set_options`.
        :param simulation_duration_max:
            Maximum duration of a simulation. If the current simulation time
            exceeds this threshold, then it will triggers `is_truncated=True`.
            It cannot exceed the maximum possible duration before telemetry
            log time overflow which is extremely large (about 30 years). Beware
            that log data are stored in RAM, which may cause out-of-memory
            error if the episode is lasting for too long without reset.
            Optional: About 4GB of log data assuming 5ms control update period
            and telemetry disabled for everything but the robot configuration.
        :param render_mode: Desired rendering mode, ie "human" or "rgb_array".
                            If "human" is specified, calling `render` will open
                            a graphical window for visualization, otherwise a
                            rgb image is returned, as a 3D numpy array whose
                            first dimension are the 3 red, green, blue channels
                            and the two subsequent dimensions are the pixel
                            height and weight respectively. `None` to select
                            automatically the most appropriate mode based on
                            the user-specified rendering backend if any, or the
                            machine environment. Note that "rgb_array" does not
                            require a graphical window manager.
                            Optional: None by default.
        :param debug: Whether the debug mode must be enabled. Doing it enables
                      telemetry recording.
        :param kwargs: Extra keyword arguments that may be useful for derived
                       environments with multiple inheritance, and to allow
                       automatic pipeline wrapper generation.
        """
        # Make sure that the simulator is single-robot
        if len(simulator.robots) > 1:
            raise NotImplementedError(
                "Multi-robot simulation is not supported for now.")

        # Handling of default rendering mode
        viewer_backend = (
            (simulator.viewer or Viewer).backend or
            simulator.viewer_kwargs.get('backend'))
        if render_mode is None:
            # 'rgb_array' by default if the backend is or will be
            # 'panda3d-sync', otherwise 'human' if available.
            backend = viewer_backend or get_default_backend()
            if backend == "panda3d-sync":
                render_mode = 'rgb_array'
            elif 'human' in self.metadata['render_modes']:
                render_mode = 'human'
            else:
                render_mode = 'rgb_array'

        # Force backend if none is specified and rendering mode is RGB array
        if ("backend" not in simulator.viewer_kwargs and
                render_mode == 'rgb_array'):
            simulator.viewer_kwargs['backend'] = "panda3d-sync"

        # Make sure that the robot name is unique
        simulator.viewer_kwargs['robot_name'] = None

        # Make sure that rendering mode is valid
        assert render_mode in self.metadata['render_modes']

        # Backup some user arguments
        self.simulator = simulator
        self.simulation_duration_max = simulation_duration_max
        self._step_dt = step_dt
        self.render_mode = render_mode
        self.debug = debug

        # Define some proxies for fast access.
        # Note that some of them will be initialized in `_setup` method.
        self.engine: jiminy.Engine = self.simulator.engine
        self.stepper_state = self.simulator.stepper_state
        self.is_simulation_running = self.simulator.is_simulation_running
        self.robot = self.simulator.robot
        self.robot_state = self.simulator.robot_state
        self._robot_state_q = np.array([])
        self._robot_state_v = np.array([])
        self._robot_state_a = np.array([])
        self.sensor_measurements: SensorMeasurementStackMap = OrderedDict(
            self.robot.sensor_measurements)

        # Top-most block of the pipeline is the environment itself by default
        self.derived = self

        # Store references to the variables to register to the telemetry
        self._registered_variables: MutableMappingT[
            str, Tuple[FieldNested, DataNested]] = {}
        self.log_fieldnames: MappingT[str, FieldNested] = _LazyDictItemFilter(
            self._registered_variables, 0)

        # Random number generator.
        # This is used for generating random observations and actions, sampling
        # the initial state of the robot, and domain randomization.
        self.np_random = np.random.Generator(np.random.SFC64())

        # Log of the "previous" simulation in debug and evaluation mode
        self.log_path: Optional[str] = None

        # Original simulation options of the ongoing episode before partially
        # overwriting it.
        self._simu_options_orig: Optional[Dict[str, Any]] = None

        # Whether training mode is active, as opposed to evaluation mode
        self._is_training = True

        # Whether play interactive mode is active
        self._is_interactive = False

        # Information about the learning process
        self._info: InfoType = {}

        # Keep track of cumulative reward
        self.total_reward = 0.0

        # Number of simulation steps performed
        self.num_steps = np.array(-1, dtype=np.int64)
        self._num_steps_beyond_terminate: Optional[int] = None

        # Initialize a quantity manager for later use
        self.quantities = QuantityManager(self)

        # Initialize the interfaces through multiple inheritance
        super().__init__()  # Do not forward extra arguments, if any

        # Initialize the seed of the environment
        self._initialize_seed()

        # Initialize the observation and action buffers
        self.observation: Obs = zeros(self.observation_space)
        self.action: Act = zeros(self.action_space)

        # Check that the action and observation spaces are consistent with
        # 'compute_command' and 'refresh_observation' respectively.
        cls = type(self)
        is_action_space_custom = (
            BaseJiminyEnv._initialize_action_space is not
            cls._initialize_action_space)
        if (BaseJiminyEnv.compute_command is cls.compute_command and
                is_action_space_custom):
            raise NotImplementedError(
                "`BaseJiminyEnv.compute_command` must be overloaded when "
                "defining a custom action space.")
        is_observation_space_custom = (
            BaseJiminyEnv._initialize_observation_space is not
            cls._initialize_observation_space)
        if (BaseJiminyEnv.refresh_observation is cls.refresh_observation and
                is_observation_space_custom):
            raise NotImplementedError(
                "`BaseJiminyEnv.refresh_observation` must be overloaded when "
                "defining a custom observation space.")

        # Define flattened default observation for efficiency if not overloaded
        self._observation_flat: Sequence[np.ndarray] = ()
        if not is_observation_space_custom:
            assert isinstance(self.observation, dict)
            self._observation_flat = (
                self.observation['t'],
                self.observation['states']['agent']['q'],
                self.observation['states']['agent']['v'],
                *self.observation['measurements'].values())

        # Define specialized operators for efficiency.
        # Note that a partial view of observation corresponding to measurement
        # must be extracted since only this one must be updated during refresh.
        self._copyto_action = build_copyto(self.action)
        self._contains_action = build_contains(self.action, self.action_space)
        reduced_obs_space: spaces.Space[DataNested] = self.observation_space
        if not is_observation_space_custom:
            # Note that time is removed from the observation space because it
            # will be checked independently.
            assert isinstance(reduced_obs_space, spaces.Dict)
            reduced_obs_space = spaces.Dict(OrderedDict(
                (key, value)
                for key, value in reduced_obs_space.items()
                if key != 't'))
        self._contains_observation = build_contains(
            self.observation, reduced_obs_space, tol_rel=OBS_CONTAINS_TOL)
        self._get_clipped_env_observation: Callable[[], DataNested] = (
            OrderedDict)

        # Set robot in neutral configuration
        q = self._neutral()
        pin.framesForwardKinematics(
            self.robot.pinocchio_model, self.robot.pinocchio_data, q)

        # Configure the default camera pose if not already done
        if "camera_pose" not in self.simulator.viewer_kwargs:
            if self.robot.has_freeflyer:
                # Get root frame name.
                # The first and second frames are respectively "universe" no
                # matter if the robot has a freeflyer or not, and the second
                # one is the freeflyer joint "root_joint" if any.
                root_name = self.robot.pinocchio_model.names[1]

                # Relative camera pose wrt the root frame by default
                self.simulator.viewer_kwargs["camera_pose"] = (
                    *DEFAULT_CAMERA_XYZRPY_REL, root_name)
            else:
                # Absolute camera pose by default
                self.simulator.viewer_kwargs["camera_pose"] = (
                    (0.0, 7.0, 0.0), (np.pi/2, 0.0, np.pi), None)

        # Register the action to the telemetry automatically iif there is
        # exactly one scalar action per motor.
        if isinstance(self.action, np.ndarray):
            action_size = self.action.size
            if action_size > 0 and action_size == self.robot.nmotors:
                action_fieldnames = [
                    ".".join(('action', motor.name))
                    for motor in self.robot.motors]
                self.register_variable(
                    'action', self.action, action_fieldnames)

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
        return [*super().__dir__(), *dir(self.simulator)]

    def __del__(self) -> None:
        try:
            if self.log_path is not None:
                try:
                    os.remove(self.log_path)
                except FileNotFoundError:
                    pass
                self.log_path = None
            self.close()
        except Exception:   # pylint: disable=broad-except
            # This method must not fail under any circumstances
            pass

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
        command_limit = np.array([
            motor.effort_limit for motor in self.robot.motors])

        # Set the action space
        self.action_space = cast(spaces.Space[Act], spaces.Box(
            low=-command_limit, high=command_limit, dtype=np.float64))

    def _initialize_seed(self, seed: Optional[int] = None) -> None:
        """Specify the seed of the environment.

        .. note::
            This method is not meant to be called manually.

        .. warning::
            It also resets the low-level jiminy Engine. Therefore one must call
            the `reset` method afterward.

        :param seed: Random seed, as a positive integer.
                     Optional: A strongly random seed will be generated by gym
                     if omitted.

        :returns: Updated seed of the environment
        """
        # Generate distinct sequences of 3 bytes uint32 seeds for the engine
        # and environment.
        engine_seed = np.random.SeedSequence(seed).generate_state(3)
        np_seed = np.random.SeedSequence(engine_seed).generate_state(3)

        # Re-initialize the low-level bit generator based on the provided seed
        self.np_random.bit_generator.state = np.random.SFC64(np_seed).state

        # Reset the seed of the action and observation spaces
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        # Reset the seed of Jiminy Engine
        self.simulator.seed(engine_seed)

    def register_variable(self,
                          name: str,
                          value: DataNested,
                          fieldnames: Optional[
                              Union[str, FieldNested]] = None,
                          namespace: Optional[str] = None) -> None:
        """Register variable to the telemetry.

        .. warning::
            Variables are registered by reference. Consequently, the user is
            responsible to manage the lifetime of the data to prevent it from
            being garbage collected.

        .. seealso::
            See `gym_jiminy.common.utils.register_variables` for details.

        :param name: Base name of the variable. It will be used to prepend
                     fields, using '.' delimiter.
        :param value: Variable to register. It supports any nested data
                      structure whose leaves have type `np.ndarray` and either
                      dtype `np.float64` or `np.int64`.
        :param fieldnames: Nested fieldnames with the exact same data structure
                           as the variable to register 'value'. Individual
                           elements of each leaf array must have its own
                           fieldname, all gathered in a nested tuple with the
                           same shape of the array.
                           Optional: Generic fieldnames will be generated
                           automatically.
        :param namespace: Namespace used to prepend the base name 'name', using
                          '.' delimiter. Empty string to disable.
                          Optional: Disabled by default.
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
            LOGGER.warning("'value' or 'fieldnames' cannot be empty.")
            return

        # Check if variable can be registered successfully to the telemetry.
        # Note that a dummy controller must be created to avoid using the
        # actual one to keep control of when registration will take place.
        register_variables(jiminy.BaseController(), fieldnames, value)

        # Combine namespace and variable name if provided
        name = ".".join(filter(None, (namespace, name)))

        # Store the header and a reference to the variable if successful
        self._registered_variables[name] = (fieldnames, value)

    @property
    def step_dt(self) -> float:
        return self._step_dt

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def unwrapped(self) -> "BaseJiminyEnv":
        return self

    def train(self) -> None:
        if self.is_simulation_running:
            raise RuntimeError(
                "Switching between training and evaluation modes is forbidden "
                "if a simulation is already running. Please call `stop` "
                "method beforehand.")
        self._is_training = True

    def eval(self) -> None:
        if self.is_simulation_running:
            raise RuntimeError(
                "Switching between training and evaluation modes is forbidden "
                "if a simulation is already running. Please call `stop` "
                "method beforehand.")
        self._is_training = False

    def update_pipeline(self, derived: Optional[InterfaceJiminyEnv]) -> None:
        if self.derived is not self:
            derived_old = self.derived
            self.derived = self
            derived_old.update_pipeline(None)
        self.derived = derived or self

    def reset(self,  # type: ignore[override]
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None,
              ) -> Tuple[DataNested, InfoType]:
        """Reset the environment.

        In practice, it resets the backend simulator and set the initial state
        of the robot. The initial state is obtained by calling '_sample_state'.
        This method is also in charge of setting the initial action (at the
        beginning) and observation (at the end).

        .. warning::
            It starts the simulation immediately. As a result, it is not
            possible to change the robot (included options), nor to register
            log variable.

        :param seed: Random seed, as a positive integer.
                     Optional: `None` by default. If `None`, then the internal
                     random generator of the environment will be kept as-is,
                     without updating its seed.
        :param options: Additional information to specify how the environment
                        is reset. The field 'reset_hook' is reserved for
                        chaining multiple `BasePipelineWrapper`. It is not
                        meant to be defined manually.
                        Optional: None by default.

        :returns: Initial observation of the episode and some auxiliary
                  information for debugging or monitoring purpose.
        """
        # Stop the episode if one is still running
        self.stop()

        # Reset the seed if requested
        if seed is not None:
            self._initialize_seed(seed)

        # Remove external forces, if any
        self.simulator.remove_all_forces()

        # Make sure the environment is properly setup.
        # Note that the robot is not allowed to change after this point.
        self._setup()

        # Make sure the low-level engine has not changed,
        # otherwise some proxies would be corrupted.
        if self.engine is not self.simulator.engine:
            raise RuntimeError(
                "Changing the memory address of the low-level jiminy engine "
                "is an undefined behavior.")

        # Re-initialize some shared memories.
        # It is necessary because the robot may have changed.
        self.robot = self.simulator.robot
        self.robot_state = self.simulator.robot_state
        self.sensor_measurements = OrderedDict(self.robot.sensor_measurements)

        # Reset action
        fill(self.action, 0)

        # Enforce the low-level controller.
        # The robot may have changed, for example it could be randomly
        # generated, which would corrupt the old controller. As a result, it is
        # necessary to either instantiate a new low-level controller and to
        # re-initialize the existing one by calling `controller.initialize`
        # method BEFORE calling `reset` method because doing otherwise would
        # cause a segfault.
        self.robot.controller = None

        # Reset the simulator.
        # Do NOT remove all forces since it has already been done before, and
        # because it would make it impossible to register forces in  `_setup`.
        self.simulator.reset(remove_all_forces=False)

        # Reset some internal buffers
        self.num_steps[()] = 0
        self._num_steps_beyond_terminate = None

        # Extract the observer/controller update period.
        # The controller update period is used by default for the observer if
        # it was not specify by the user in `_setup`.
        engine_options = self.simulator.get_options()
        self.control_dt = float(
            engine_options['stepper']['controllerUpdatePeriod'])
        if self.observe_dt < 0.0:
            self.observe_dt = self.control_dt

        # Make sure that both the observer and the controller are running
        # faster than the environment to which it is attached for the action to
        # take effect and be observable. Moreover, their respective update
        # period must be a divisor of the environment step for both
        # computational efficiency and avoid breaking markovian assumption due
        # to previous action having a direct effect on the next step.
        control_update_ratio, observe_update_ratio = 0.0, 0.0
        if self.observe_dt > 0.0:
            observe_update_ratio = round(self.step_dt / self.observe_dt, 10)
            assert round(observe_update_ratio) == observe_update_ratio, (
                "Observer update period must be a divisor of environment "
                "simulation timestep")
        if self.control_dt > 0.0:
            control_update_ratio = round(self.step_dt / self.control_dt, 10)
            assert round(control_update_ratio) == control_update_ratio, (
                "Controller update period must be a divisor of environment "
                "simulation timestep")

        # Sample the initial state and reset the low-level engine
        q_init, v_init = self._sample_state()
        if not jiminy.is_position_valid(self.robot.pinocchio_model, q_init):
            raise RuntimeError(
                "The initial state provided by `_sample_state` is "
                "inconsistent with the dimension or types of joints of the "
                "model.")

        # Set robot in initial configuration
        pin.framesForwardKinematics(
            self.robot.pinocchio_model, self.robot.pinocchio_data, q_init)

        # Initialize sensor measurements that are zero-ed at this point. This
        # may be necessary for pre-compiling blocks before actually starting
        # the simulation to avoid triggering timeout error. Indeed, some
        # computations may require valid sensor data, such as normalized
        # quaternion or non-zero linear acceleration.
        a_init, u_motor = (np.zeros(self.robot.nv),) * 2
        f_external = [pin.Force.Zero(),] * self.robot.pinocchio_model.njoints
        self.robot.compute_sensor_measurements(
            0.0, q_init, v_init, a_init, u_motor, f_external)

        # Run the reset hook if any.
        # Note that the reset hook must be called after `_setup` because it
        # expects that the robot is not going to change anymore at this point.
        # Similarly, the observer and controller update periods must be set.
        reset_hook: Optional[Callable[[], InterfaceJiminyEnv]] = (
            options or {}).get("reset_hook")
        env: InterfaceJiminyEnv = self
        if reset_hook is not None:
            assert callable(reset_hook)
            env_derived = reset_hook() or env
            assert env_derived.unwrapped is self
            env = env_derived

        # Update the environment pipeline if necessary
        if env is not self.derived:
            env.update_pipeline(env)

        # Re-initialize the quantity manager.
        # Note that computation graph tracking is never reset automatically.
        # It is the responsibility of the practitioner implementing a derived
        # environment whenever it makes sense for its specific use-case.
        self.quantities.reset(reset_tracking=False)

        # Instantiate the actual controller.
        # Note that a weak reference must be used to avoid circular reference.
        self.robot.controller = jiminy.FunctionalController(
            partial(type(env)._controller_handle, weakref.proxy(env)))

        # Register user-specified variables to the telemetry
        for header, value in self._registered_variables.values():
            register_variables(self.robot.controller, header, value)

        # Start the simulation
        self.simulator.start(q_init, v_init)

        # Refresh robot_state proxies. It must be done here because memory is
        # only allocated by the engine when starting a simulation.
        self._robot_state_q = self.robot_state.q
        self._robot_state_v = self.robot_state.v
        self._robot_state_a = self.robot_state.a

        # Initialize shared buffers
        self._initialize_buffers()

        # Update shared buffers
        self._refresh_buffers()

        # Clear cache and auto-refresh managed quantities
        self.quantities.clear()

        # Initialize the observation
        env._observer_handle(
            self.stepper_state.t,
            self._robot_state_q,
            self._robot_state_v,
            self.robot.sensor_measurements)

        # Initialize specialized most-derived observation clipping operator
        self._get_clipped_env_observation = build_clip(
            env.observation, env.observation_space)

        # Make sure the state is valid, otherwise there `refresh_observation`
        # and `_initialize_observation_space` are probably inconsistent.
        try:
            obs = self._get_clipped_env_observation()
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "The observation computed by `refresh_observation` is "
                "inconsistent with the observation space defined by "
                "`_initialize_observation_space` at initialization.") from e

        # Make sure there is no 'nan' value in observation
        for value in tree.flatten(obs):
            if is_nan(value):
                raise RuntimeError(
                    f"'nan' value found in observation ({obs}). Something "
                    "went wrong with `refresh_observation` method.")

        # Reset the extra information buffer
        self._info.clear()

        # The simulation cannot be done before doing a single step.
        if any(self.derived.has_terminated(self._info)):
            raise RuntimeError(
                "The simulation has already terminated at `reset`. Check the "
                "implementation of `has_terminated` if overloaded.")

        # Reset cumulative reward
        self.total_reward = 0.0

        # Note that the viewer must be reset if available, otherwise it would
        # keep using the old robot model for display, which must be avoided.
        if self.simulator.is_viewer_available:
            viewer = self.simulator.viewer
            assert viewer is not None  # Assert(s) for type checker
            viewer._setup(self.robot)  # type: ignore[attr-defined]
            if viewer.has_gui():
                viewer.refresh()

        return obs, deepcopy(self._info)

    def close(self) -> None:
        """Clean up the environment after the user has finished using it.

        It terminates the Python Jiminy engine.

        .. warning::
            Calling `reset` or `step` afterward is an undefined behavior.
        """
        self.simulator.close()

    def step(self,  # type: ignore[override]
             action: Act
             ) -> Tuple[DataNested, SupportsFloat, bool, bool, InfoType]:
        """Integration timestep of the environment's dynamics under prescribed
        agent's action over a single timestep, i.e. collect one transition step
        of the underlying Markov Decision Process of the learning problem.

        .. warning::
            When reaching the end of an episode (``terminated or truncated``),
            it is necessary to reset the environment for starting a new episode
            before being able to do steps once again.

        :param action: Action performed by the agent during the ongoing step.

        :returns:
            * observation (ObsType) - The next observation of the environment's
              as a result of taking agent's action.
            * reward (float): the reward associated with the transition step.
            * terminated (bool): Whether the agent reaches the terminal state,
              which means success or failure depending of the MDP of the task.
            * truncated (bool): Whether some truncation condition outside the
              scope of the MDP is satisfied. This can be used to end an episode
              prematurely before a terminal state is reached, for instance if
              the agent's state is going out of bounds.
            * info (dict): Contains auxiliary information that may be helpful
              for debugging and monitoring. This might, for instance, contain:
              metrics that describe the agent's performance state, variables
              that are hidden from observations, or individual reward terms
              from which the total reward is derived.
        """
        # Make sure a simulation is already running
        if not self.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please call `reset` before `step`.")

        # Update of the action to perform if relevant
        if action is not self.action:
            # Make sure the action is valid in debug mode
            if self.debug:
                for value in tree.flatten(action):
                    if is_nan(value):
                        raise RuntimeError(
                            f"'nan' value found in action ({action}).")

            # Update the action
            self._copyto_action(action)

        # Try performing a single simulation step
        try:
            self.simulator.step(self.step_dt)
        except Exception:
            # Stop the simulation before raising the exception
            self.stop()
            raise

        # Make sure there is no 'nan' value in observation
        if is_nan(self._robot_state_a):
            raise RuntimeError(
                "The acceleration of the system is 'nan'. Something went "
                "wrong with jiminy engine.")

        # Update number of (successful) steps
        self.num_steps += 1

        # Update shared buffers
        self._refresh_buffers()

        # Clear cache and auto-refresh managed quantities
        self.quantities.clear()

        # Update the observer at the end of the step.
        # This is necessary because, internally, it is called at the beginning
        # of the every integration steps, during the controller update.
        self.derived._observer_handle(
            self.stepper_state.t,
            self._robot_state_q,
            self._robot_state_v,
            self.robot.sensor_measurements)

        # Reset the extra information buffer
        self._info.clear()

        # Check if the simulation is over.
        # Note that 'truncated' is forced to True if the integration failed or
        # if the maximum number of steps will be exceeded next step.
        terminated, truncated = self.derived.has_terminated(self._info)
        truncated = (
            truncated or not self.is_simulation_running or
            self.stepper_state.t + DT_EPS > self.simulation_duration_max)

        # Check if stepping after done and if it is an undefined behavior
        if self._num_steps_beyond_terminate is None:
            if terminated:
                self._num_steps_beyond_terminate = 0
        else:
            if self.is_training and self._num_steps_beyond_terminate == 0:
                LOGGER.error(
                    "Calling `step` after termination is an undefined "
                    "behavior, and as such, is strongly discouraged in "
                    "training mode. The reward will be forced to 'nan' from "
                    "now on. Please call `reset`.")
            self._num_steps_beyond_terminate += 1

        # Compute reward if not beyond termination
        if self._num_steps_beyond_terminate:
            reward = float('nan')
        else:
            # Compute reward and update extra information
            reward = self.compute_reward(terminated, self._info)

            # Make sure the reward is not 'nan'
            if math.isnan(reward):
                raise RuntimeError(
                    "The reward is 'nan'. Something went wrong with "
                    "`compute_reward` implementation.")

            # Update cumulative reward
            self.total_reward += reward

        # Clip (and copy) the most derived observation before returning it
        obs = self._get_clipped_env_observation()

        return obs, reward, terminated, truncated, deepcopy(self._info)

    def stop(self) -> None:
        # Check whether it is worth saving log file
        has_simulation_data = self.is_simulation_running and self.num_steps > 0

        # Stop the engine.
        # This must be done BEFORE writing log, otherwise the final simulation
        # state will be missing as it gets flushed after stopping.
        self.simulator.stop()

        # Write log of previous simulation before starting a new one if not
        # already done.
        # This would be the case if the previous episode was never terminated
        # nor truncated, or because it was wrapped with a non-jiminy-specific
        # layer such as `TimeLimit`.
        if has_simulation_data:
            if self.log_path is not None:
                os.remove(self.log_path)
                self.log_path = None
            if self.debug or not self.is_training:
                fd, self.log_path = tempfile.mkstemp(suffix=".data")
                os.close(fd)
                self.simulator.write_log(self.log_path, format="binary")

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Render the agent in its environment.

        .. versionchanged::
            This method does not take input arguments anymore due to changes of
            the official `gym.Wrapper` API. A workaround is to set
            `simulator.viewer_kwargs` beforehand. Alternatively, it is possible
            to call the low-level implementation `simulator.render` directly to
            avoid API restrictions.

        :returns: RGB array if 'render_mode' is 'rgb_array', None otherwise.
        """
        # Set the available rendering modes
        viewer_backend = (self.simulator.viewer or Viewer).backend
        if self.render_mode == 'human' and viewer_backend == "panda3d-sync":
            Viewer.close()

        # Call base implementation
        return self.simulator.render(  # type: ignore[return-value]
            return_rgb_array=self.render_mode == 'rgb_array')

    def plot(self,
             enable_block_states: bool = False,
             **kwargs: Any) -> TabbedFigure:
        """Display common simulation data and action over time.

        .. Note:
            It adds tabs for the base environment action plus all blocks
            ((state, action) for controllers and (state, features) for
            observers) on top of original `Simulator.plot`.

        :param enable_block_states: Whether to display the internal state of
                                    all blocks.
        :param kwargs: Extra keyword arguments to forward to `simulator.plot`.
        """
        # Call base implementation
        figure = self.simulator.plot(**kwargs)
        assert not isinstance(figure, Sequence)

        # Extract log data
        log_vars = self.simulator.log_data.get("variables", {})
        if not log_vars:
            raise RuntimeError(
                "Nothing to plot. Please run a simulation before calling "
                "`plot` method.")

        # Plot all registered variables
        for key, fieldnames in self.log_fieldnames.items():
            # Filter state if requested
            if not enable_block_states and key.endswith(".state"):
                continue

            # Store fieldnames in dict systematically to avoid code duplication
            if not isinstance(fieldnames, dict):
                fieldnames = {"": fieldnames}

            # Extract hierarchical time series.
            # Fieldnames stored in a dictionary cannot be nested. In such a
            # case, keys corresponds to subplots, and values are individual
            # scalar data over time to be displayed to the same subplot.
            t = log_vars["Global.Time"]
            base_name = key.replace(".", " ")
            for group, subfieldnames in fieldnames.items():
                if not isinstance(subfieldnames, (list, tuple)):
                    LOGGER.error(
                        "Action space not supported by this method.")
                    return figure

                tab_name = " ".join(filter(None, (base_name, group)))
                value_map = extract_variables_from_log(
                    log_vars, subfieldnames, "controller", as_dict=True)
                tab_data = {key.split(".", 2)[2]: value
                            for key, value in value_map.items()}

                grid_spec: Tuple[Optional[int], Optional[int]] = (None, None)
                nrows = len(subfieldnames)
                if nrows and isinstance(subfieldnames[0], (list, tuple)):
                    ncols_all = set(map(len, subfieldnames))
                    if len(ncols_all) == 1:
                        grid_spec = (nrows, next(iter(ncols_all)))

                try:
                    figure.add_tab(tab_name,
                                   t,
                                   tab_data,  # type: ignore[arg-type]
                                   nrows=grid_spec[0],
                                   ncols=grid_spec[1])
                except ValueError:
                    LOGGER.error("Invalid plot spec for variable %s. Moving "
                                 "to the next one", key)

        # Return figure for convenience and consistency with Matplotlib
        return figure

    def replay(self, **kwargs: Any) -> None:
        """Replay the current episode until now.

        :param kwargs: Extra keyword arguments for delegation to
                       `replay.play_trajectories` method.
        """
        # Do not open graphical window automatically if recording requested.
        # Note that backend is closed automatically is there is no viewer
        # backend available at this point, to reduce memory pressure, but it
        # will take time to restart it systematically for every recordings.
        if kwargs.get('record_video_path') is not None:
            kwargs['close_backend'] = not self.simulator.is_viewer_available

        # Stop any running simulation before replay if `has_terminated` is True
        if self.is_simulation_running and any(self.derived.has_terminated({})):
            self.stop()

        with viewer_lock:
            # Call render before replay in order to take into account custom
            # backend viewer instantiation options, eg the initial camera pose,
            # and to update the ground profile.
            self.simulator.render(
                update_ground_profile=True,
                return_rgb_array="record_video_path" in kwargs.keys(),
                **kwargs)

            viewer_kwargs: Dict[str, Any] = {
                'verbose': False,
                'enable_travelling': self.robot.has_freeflyer,
                **kwargs}
            self.simulator.replay(**viewer_kwargs)

    def play_interactive(self,
                         enable_travelling: Optional[bool] = None,
                         start_paused: bool = True,
                         enable_is_done: bool = True,
                         verbose: bool = True,
                         **kwargs: Any) -> None:
        """Activate interact mode enabling to control the robot using keyboard.

        It stops automatically as soon as `terminated or truncated` is True.
        One has to press a key to start the interaction. If no key is pressed,
        the action is not updated and the previous one keeps being sent to the
        robot.

        .. warning::
            It ignores any external `gym.Wrapper` that may be used for training
            but are not considered part of the environment pipeline.

        .. warning::
            This method requires `_key_to_action` method to be implemented by
            the user by overloading it, otherwise it raises an exception.

        :param enable_travelling: Whether enable travelling, following the
                                  motion of the root frame of the model. This
                                  parameter is ignored if the model has no
                                  freeflyer.
                                  Optional: Enabled by default iif 'panda3d'
                                  viewer backend is used.
        :param start_paused: Whether to start in pause.
                             Optional: Enabled by default.
        :param verbose: Whether to display status messages.
        :param kwargs: Extra keyword arguments to forward to `_key_to_action`
                       method.
        """
        # Stop the episode if one is still running
        self.stop()

        # Enable play interactive flag and make sure training flag is disabled
        is_training = self.is_training
        self._is_interactive = True
        if is_training:
            self.eval()

        # Make sure viewer gui is open, so that the viewer will shared external
        # forces with the robot automatically.
        viewer = self.simulator.viewer
        if viewer is None or not viewer.has_gui():
            self.simulator.render(update_ground_profile=False)

        # Initialize the simulation
        obs, _ = self.derived.reset()
        reward = None

        # Refresh the ground profile
        self.simulator.render(update_ground_profile=True)
        viewer = self.simulator.viewer
        assert viewer is not None  # Assert(s) for type checker

        # Enable travelling
        if enable_travelling is None:
            backend = viewer.backend
            assert backend is not None  # Assert(s) for type checker
            enable_travelling = backend.startswith('panda3d')
        enable_travelling = enable_travelling and self.robot.has_freeflyer
        if enable_travelling:
            tracked_frame = self.robot.pinocchio_model.frames[2].name
            viewer.attach_camera(tracked_frame)

        # Refresh the scene once again to update camera placement
        self.render()

        # Define interactive loop
        def _interact(key: Optional[str] = None) -> bool:
            nonlocal obs, reward, enable_is_done
            if key is not None:
                action = self._key_to_action(
                    key, obs, reward, **{"verbose": verbose, **kwargs})
            if action is None:
                action = self.action
            _, reward, terminated, truncated, _ = self.step(action)
            obs = self.observation
            self.render()
            if not enable_is_done and self.robot.has_freeflyer:
                return self._robot_state_q[2] < 0.0
            return terminated or truncated

        # Run interactive loop
        loop_interactive(max_rate=self.step_dt,
                         start_paused=start_paused,
                         verbose=verbose)(_interact)()

        # Disable travelling if it enabled
        if enable_travelling:
            viewer.detach_camera()

        # Stop the simulation to unlock the robot.
        # It will enable to display contact forces for replay.
        self.stop()

        # Disable play interactive mode flag and restore training flag
        self._is_interactive = False
        if is_training:
            self.train()

    def evaluate(self,
                 policy_fn: PolicyCallbackFun,
                 seed: Optional[int] = None,
                 horizon: Optional[int] = None,
                 enable_stats: bool = True,
                 enable_replay: Optional[bool] = None,
                 **kwargs: Any) -> List[InfoType]:
        r"""Evaluate a policy on the environment over a complete episode.

        .. warning::
            It ignores any external `gym.Wrapper` that may be used for training
            but are not considered part of the environment pipeline.

        :param policy_fn:
            .. raw:: html

                Policy to evaluate as a callback function. It must have the
                following signature (**rew** = None at reset):

            | policy_fn\(**obs**: Obs,
            |            **action_prev**: Optional[Act],
            |            **reward**: Optional[float],
            |            **terminated**: bool,
            |            **truncated**: bool,
            |            **info**: InfoType
            |            \) -> Act  # **action**
        :param seed: Seed of the environment to be used for the evaluation of
                     the policy.
                     Optional: `None` by default. If not specified, then a
                     strongly random seed will be generated by gym.
        :param horizon: Horizon of the simulation, namely maximum number of
                        env steps before termination. `None` to disable.
                        Optional: Disabled by default.
        :param enable_stats: Whether to print high-level statistics after the
                             simulation.
                             Optional: Enabled by default.
        :param enable_replay: Whether to enable replay of the simulation, and
                              eventually recording if the extra
                              keyword argument `record_video_path` is provided.
                              Optional: Enabled by default if display is
                              available, disabled otherwise.
        :param kwargs: Extra keyword arguments to forward to the `replay`
                       method if replay is requested.
        """
        # Handling of default arguments
        if enable_replay is None:
            enable_replay = (
                (Viewer.backend or get_default_backend()) != "panda3d-sync" or
                interactive_mode() >= 2)

        # Stop the episode if one is still running
        self.stop()

        # Make sure evaluation mode is enabled
        is_training = self.is_training
        if is_training:
            self.eval()

        # Set the seed without forcing full reset of the environment
        self._initialize_seed(seed)

        # Initialize the simulation
        env = self.derived
        obs, info = env.reset()
        action, reward, terminated, truncated = None, None, False, False

        # Run the simulation
        info_episode = [info]
        try:
            while horizon is None or self.num_steps < horizon:
                action = policy_fn(
                    obs, action, reward, terminated, truncated, info)
                if terminated or truncated:
                    # Break AFTER calling the policy callback if the episode is
                    # terminated or truncated, which gives the policy the
                    # opportunity to observe and record the final state.
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                info_episode.append(info)
            self.stop()
        except KeyboardInterrupt:
            pass

        # Restore training mode if it was enabled
        if is_training:
            self.train()

        # Display some statistic if requested
        if enable_stats:
            print("env.num_steps:", self.num_steps)
            print("cumulative reward:", self.total_reward)

        # Replay the result if requested
        if enable_replay:
            try:
                self.replay(**kwargs)
            except Exception as e:  # pylint: disable=broad-except
                # Do not fail because of replay/recording exception
                LOGGER.warning("%s", e)

        return info_episode

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
        # Restore the original simulation options
        if self._simu_options_orig is not None:
            self.simulator.set_simulation_options(self._simu_options_orig)

        # Call base implementation
        super()._setup()

        # Backup simulation options
        self._simu_options_orig = self.simulator.get_simulation_options()

        # Configure the low-level integrator
        engine_options = self.simulator.get_options()
        engine_options["stepper"]["iterMax"] = 0
        if self.debug:
            engine_options["stepper"]["verbose"] = True
        if self.debug or not self.is_training:
            engine_options["stepper"]["logInternalStepperSteps"] = True

        # Set maximum computation time for single internal integration steps
        engine_options["stepper"]["timeout"] = self.step_dt * TIMEOUT_RATIO
        if self.debug:
            engine_options["stepper"]["timeout"] = 0.0

        # Enable full logging in debug and evaluation mode
        if self.debug or not self.is_training:
            # Enable all telemetry data at engine-level
            telemetry_options = engine_options["telemetry"]
            for key in telemetry_options.keys():
                if key.startswith("enable"):
                    telemetry_options[key] = True

            # Enable telemetry persistence.
            # The visual and collision meshes will be stored in log file, so
            # that the robot can be loaded on any machine with access to the
            # original URDF and mesh files.
            engine_options["telemetry"]["isPersistent"] = True

            # Enable all telemetry data at robot-level
            robot_options = self.robot.get_options()
            robot_telemetry_options = robot_options["telemetry"]
            for key in robot_telemetry_options.keys():
                robot_telemetry_options[key] = True
            self.robot.set_options(robot_options)

        # Update engine options
        self.simulator.set_options(engine_options)

    def _initialize_observation_space(self) -> None:
        """Configure the observation of the environment.

        By default, the observation is a dictionary gathering the current
        simulation time, the real agent state, and the sensors data.

        .. note::
            This method is called internally by `reset` method at the very end,
            just before computing and returning the initial observation. This
            method, alongside `refresh_observation`, must be overwritten in
            order to define a custom observation space.
        """
        observation_spaces: Dict[str, spaces.Space] = OrderedDict()
        observation_spaces['t'] = spaces.Box(
            low=0.0,
            high=self.simulation_duration_max,
            shape=(),
            dtype=np.float64)
        observation_spaces['states'] = (
            spaces.Dict(agent=get_robot_state_space(self.robot)))
        observation_spaces['measurements'] = (
            get_robot_measurements_space(self.robot))
        self.observation_space = cast(
            spaces.Space[Obs], spaces.Dict(observation_spaces))

    def _neutral(self) -> np.ndarray:
        """Returns a neutral valid configuration for the agent.

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
        q = pin.neutral(self.robot.pinocchio_model)

        # Make sure it is not out-of-bounds before returning
        position_limit_lower = self.robot.pinocchio_model.lowerPositionLimit
        position_limit_upper = self.robot.pinocchio_model.upperPositionLimit
        for idx, val in enumerate(q):
            lo, hi = position_limit_lower[idx], position_limit_upper[idx]
            if hi < val or val < lo:
                q[idx] = 0.5 * (lo + hi)
        return q

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a randomized yet valid configuration and velocity for the
        robot.

        The default implementation returns the neutral configuration and zero
        velocity.

        Offsets are applied on the freeflyer to ensure no contact points are
        going through the ground and up to three are in contact.

        .. note::
            This method is called internally by `reset` to generate the initial
            state. It can be overloaded to act as a random state generator.
        """
        # Get the neutral configuration
        q = self._neutral()

        # Make sure the configuration is not out-of-bound
        q.clip(self.robot.pinocchio_model.lowerPositionLimit,
               self.robot.pinocchio_model.upperPositionLimit,
               out=q)

        # Make sure the configuration is normalized
        q = pin.normalize(self.robot.pinocchio_model, q)

        # Make sure the robot impacts the ground
        if self.robot.has_freeflyer:
            engine_options = self.simulator.get_options()
            ground_fun = engine_options['world']['groundProfile']
            compute_freeflyer_state_from_fixed_body(
                self.robot, q, ground_profile=ground_fun)

        # Zero velocity
        v = np.zeros(self.robot.pinocchio_model.nv)

        return q, v

    def _initialize_buffers(self) -> None:
        """Initialize internal buffers for fast access to shared memory or to
        avoid redundant computations.

        .. note::
            This method is called at every `reset`, right after
            `self.simulator.start`. At this point, the simulation is running
            but `refresh_observation` has never been called, so that it can be
            used to initialize buffers involving the engine state but required
            to refresh the observation.

        .. note::
            Buffers requiring manual update must be refreshed using
            `_refresh_buffers` method.

        .. warning::
            This method is not appropriate for initializing buffers involved in
            `compute_command`. At the time being, there is no better way that
            taking advantage of the flag `self.is_simulation_running` in the
            method `compute_command` itself.
        """

    def _refresh_buffers(self) -> None:
        """Refresh internal buffers that must be updated manually.

        .. note::
            This method is called after every internal `engine.step` and before
            refreshing the observation one last time. As such, it is the right
            place to update shared data between `has_terminated` and
            `compute_reward`. However, it is not appropriate for quantities
            involved in `refresh_observation` not `compute_command`, which may
            be called more often than once per step.

        .. note::
            `_initialize_buffers` method can be used to initialize buffers that
            may requires special care.

        .. warning::
            Be careful when using this method to update buffers involved in
            `refresh_observation`. The latter is called at `self.observe_dt`
            update period, while this method is called at `self.step_dt` update
            period. `self.observe_dt` is likely to be different from
            `self.step_dt`, unless configured manually when overloading
            `_setup` method.
        """

    @no_type_check
    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute the observation based on the current state of the robot.

        In practice, it updates the internal buffer directly for the sake of
        efficiency.

        By default, it sets the observation to the value of the measurement,
        which would not work unless `Obs` corresponds to `EngineObsType`.

        .. note::
            This method is called and the end of every low-level `Engine.step`.

        .. warning::
            This method may be called without any simulation running, either
            to perform basic consistency checking or allocate and initialize
            buffers. There is no way at the time being to distinguish the
            initialization stage in particular. A workaround consists in
            checking whether the simulation already started. It is not exactly
            the same but it does the job regarding preserving efficiency.

        .. warning::
            One must only rely on `measurement` to get the state of the robot,
            as anything else is not reliable for this. More specifically,
            `self.robot_state` would not be valid if an adaptive stepper is
            being used for physics integration.
        """
        # Manually flatten measurement
        agent_state_in = measurement['states']['agent']
        measurement_flat = (
            measurement['t'],
            agent_state_in['q'],
            agent_state_in['v'],
            *measurement['measurements'].values())

        # Copy all arrays at once
        multi_array_copyto(self._observation_flat, measurement_flat)

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        By default, all it does is forwarding the input action as is, without
        performing any processing. One is responsible of overloading this
        method if the action space has been customized, or just to clip the
        action to make sure it is never out-of-bounds if necessary.

        .. warning::
            There is not good place to initialize buffers that are necessary to
            compute the command. The only solution for now is to define
            initialization inside this method itself, using the safeguard
            `if not self.is_simulation_running:`.

        :param action: High-level target to achieve by means of the command.
        """
        # Check if the action is out-of-bounds, in debug mode only
        if self.debug and not self._contains_action():
            LOGGER.warning("The action is out-of-bounds.")

        assert isinstance(action, np.ndarray)
        array_copyto(command, action)

    def has_terminated(self, info: InfoType) -> Tuple[bool, bool]:
        """Determine whether the episode is over, because a terminal state of
        the underlying MDP has been reached or an aborting condition outside
        the scope of the MDP has been triggered.

        By default, it always returns `terminated=False`, and `truncated=True`
        iif the observation is out-of-bounds. One can overload this method to
        implement custom termination conditions for the environment at hand.

        .. warning::
            No matter what, truncation will happen when reaching the maximum
            simulation duration, i.e. 'self.simulation_duration_max'.

        .. note::
            This method is called after `refresh_observation`, so that the
            internal buffer 'observation' is up-to-date.

        :param info: Dictionary of extra information for monitoring.

        :returns: terminated and truncated flags.
        """
        # Make sure that a simulation is running
        if not self.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please start one before calling this "
                "method.")

        # Check if the observation is out-of-bounds in debug mode only
        truncated = not self._contains_observation()

        return False, truncated

    def _key_to_action(self,
                       key: str,
                       obs: Obs,
                       reward: Optional[float],
                       **kwargs: Any) -> Optional[Act]:
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
