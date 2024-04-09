# mypy: disable-error-code="attr-defined, name-defined"
"""This module implements a basic wrapper on top of the lower-level Jiminy
Engine that simplifies the user API for the very common single-robot scenario
while extending its capability by integrating native support of 3D scene
rendering and figure plotting of telemetry log data.
"""
import os
import re
import atexit
import logging
import pathlib
import tempfile
import warnings
from copy import deepcopy
from functools import partial
from itertools import cycle, islice
from typing import Any, List, Dict, Optional, Union, Sequence, Callable, Tuple

import toml
import numpy as np

from . import core as jiminy
from .robot import BaseJiminyRobot, generate_default_hardware_description_file
from .dynamics import TrajectoryDataType
from .log import read_log, build_robot_from_log
from .viewer import (CameraPoseType,
                     interactive_mode,
                     get_default_backend,
                     extract_replay_data_from_log,
                     play_trajectories,
                     Viewer,
                     COLORS)

if interactive_mode() >= 2:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm  # type: ignore[assignment]

try:
    from .plot import TabbedFigure
except ImportError:
    TabbedFigure = type(None)  # type: ignore[misc,assignment]


LOGGER = logging.getLogger(__name__)


DEFAULT_UPDATE_PERIOD = 1.0e-3  # 0.0 for time continuous update
DEFAULT_GROUND_STIFFNESS = 4.0e6
DEFAULT_GROUND_DAMPING = 2.0e3

ROBOTS_COLORS = list(islice(cycle(COLORS.values()), len(COLORS)))

ProfileForceFunc = Callable[[float, np.ndarray, np.ndarray, np.ndarray], None]


def _build_robot_from_urdf(name: str,
                           urdf_path: str,
                           hardware_path: Optional[str] = None,
                           mesh_path_dir: Optional[str] = None,
                           has_freeflyer: bool = True,
                           avoid_instable_collisions: bool = True,
                           debug: bool = False) -> jiminy.Robot:
    """
    Create and initialize a new robot from scratch, based on configuration
    files only. Create a temporary harware file if needed.

    :param urdf_path: Path of the urdf model to be used for the simulation.
    :param hardware_path: Path of Jiminy hardware description toml file.
                            Optional: Looking for '\*_hardware.toml' file in
                            the same folder and with the same name.
    :param mesh_path_dir: Path to the folder containing all the meshes.
                            Optional: Env variable 'JIMINY_DATA_PATH' will be
                            used if available.
    :param has_freeflyer: Whether the robot is fixed-based wrt its root
                            link, or can move freely in the world.
                            Optional: True by default.
    :param config_path: Configuration toml file to import. It will be
                        imported AFTER loading the hardware description
                        file. It can be automatically generated from an
                        instance by calling `export_config_file` method.
                        Optional: Looking for '\*_options.toml' file in the
                        same folder and with the same name. If not found,
                        using default configuration.
    :param avoid_instable_collisions: Prevent numerical instabilities by
                                        replacing collision mesh by vertices
                                        of associated minimal volume bounding
                                        box, and replacing primitive box by
                                        its vertices.
    :param debug: Whether the debug mode must be activated. Doing it
                    enables temporary files automatic deletion.
    """
    # Check if robot name is valid
    if name != re.sub('[^A-Za-z0-9_]', '_', name):
        raise Exception("The name of the robot should be case-insensitive "
                        "ASCII alphanumeric characters plus underscore.")

    # Generate a temporary Hardware Description File if necessary
    if hardware_path is None:
        hardware_path = str(pathlib.Path(
            urdf_path).with_suffix('')) + '_hardware.toml'
        if not os.path.exists(hardware_path):
            # Create a file that will be closed (thus deleted) at exit
            urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
            fd, hardware_path = tempfile.mkstemp(
                prefix=f"{urdf_name}_", suffix="_hardware.toml")
            os.close(fd)

            if not debug:
                def remove_file_at_exit(file_path: str) -> None:
                    try:
                        os.remove(file_path)
                    except (PermissionError, FileNotFoundError):
                        pass

                atexit.register(partial(
                    remove_file_at_exit, hardware_path))

            # Generate default Hardware Description File
            generate_default_hardware_description_file(
                urdf_path, hardware_path, verbose=debug)

    # Build the robot
    robot = BaseJiminyRobot(name)
    robot.initialize(
        urdf_path, hardware_path, mesh_path_dir, (), has_freeflyer,
        avoid_instable_collisions, load_visual_meshes=debug, verbose=debug)

    return robot

class Simulator:
    """Simulation wrapper providing a unified user-API on top of
    the low-level jiminy C++ core library and Python-native modules for 3D
    rendering and log data visualization. The simulation can now be multi-robot
    but it has been design to remain as easy of use as possible for 
    single-robot simulation.
    """
    def __init__(self,  # pylint: disable=unused-argument
                 robot: jiminy.Robot,
                 viewer_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs: Any) -> None:
        """
        :param robot: Jiminy robot already initialized.
        :param viewer_kwargs: Keyword arguments to override default arguments
                              whenever a viewer must be instantiated, eg when
                              `render` method is first called. Specifically,
                              `backend` is ignored if one is already available.
                              Optional: Empty by default.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup the user arguments
        self.viewer_kwargs = deepcopy(viewer_kwargs or {})

        # Check if robot name is valid
        if robot.name != re.sub('[^A-Za-z0-9_]', '_', robot.name):
            raise Exception("The name of the robot should be case-insensitive "
                            "ASCII alphanumeric characters plus underscore.")

        # Instantiate the low-level Jiminy engine, then initialize it
        self.engine = jiminy.Engine()
        self.engine.add_robot(robot)

        # Create shared memories and python-native attribute for fast access
        self.stepper_state = self.engine.stepper_state
        self.is_simulation_running = self.engine.is_simulation_running

        # Viewer management
        self._viewers: Sequence[Viewer] = []

        # Internal buffer for progress bar management
        self._pbar: Optional[tqdm] = None

        # Figure holder
        self._figure: Optional[TabbedFigure] = None

        # Reset the low-level jiminy engine
        self.reset()

    @classmethod
    def build(cls,
              urdf_path: str,
              hardware_path: Optional[str] = None,
              mesh_path_dir: Optional[str] = None,
              has_freeflyer: bool = True,
              config_path: Optional[str] = None,
              avoid_instable_collisions: bool = True,
              debug: bool = False,
              name : str = "",
              **kwargs: Any) -> 'Simulator':
        """Create a new simulator instance from scratch, based on
        configuration files only.

        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '\*_hardware.toml' file in
                              the same folder and with the same name.
        :param mesh_path_dir: Path to the folder containing all the meshes.
                              Optional: Env variable 'JIMINY_DATA_PATH' will be
                              used if available.
        :param has_freeflyer: Whether the robot is fixed-based wrt its root
                              link, or can move freely in the world.
                              Optional: True by default.
        :param config_path: Configuration toml file to import. It will be
                            imported AFTER loading the hardware description
                            file. It can be automatically generated from an
                            instance by calling `export_config_file` method.
                            Optional: Looking for '\*_options.toml' file in the
                            same folder and with the same name. If not found,
                            using default configuration.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, and replacing primitive box by
                                          its vertices.
        :param debug: Whether the debug mode must be activated. Doing it
                      enables temporary files automatic deletion.
        :param kwargs: Keyword arguments to forward to class constructor.
        """       
        # Instantiate and initialize the robot
        robot = _build_robot_from_urdf(name, urdf_path, hardware_path,
                                       mesh_path_dir, has_freeflyer,
                                       avoid_instable_collisions, debug)

        # Instantiate and initialize the engine
        simulator = Simulator.__new__(cls)
        Simulator.__init__(
            simulator, robot, engine_class=jiminy.Engine, **kwargs)

        # Get engine options
        engine_options = simulator.engine.get_options()

        # Update controller/sensors update period, based on extra toml info
        control_period = robot.extra_info.pop('controllerUpdatePeriod', None)
        sensors_period = robot.extra_info.pop('sensorsUpdatePeriod', None)
        if control_period is None and sensors_period is None:
            control_period = DEFAULT_UPDATE_PERIOD
            sensors_period = DEFAULT_UPDATE_PERIOD
        elif control_period is None:
            control_period = sensors_period
        else:
            sensors_period = control_period
        engine_options['stepper']['controllerUpdatePeriod'] = control_period
        engine_options['stepper']['sensorsUpdatePeriod'] = sensors_period

        # Handling of ground model parameters, based on extra toml info
        engine_options['contacts']['stiffness'] = \
            robot.extra_info.pop('groundStiffness', DEFAULT_GROUND_STIFFNESS)
        engine_options['contacts']['damping'] = \
            robot.extra_info.pop('groundDamping', DEFAULT_GROUND_DAMPING)

        simulator.engine.set_options(engine_options)

        # Override the default options by the one in the configuration file
        if config_path != "":
            simulator.import_options(config_path)

        return simulator

    def add_robot(self,
                  name: str,
                  urdf_path: str,
                  hardware_path: Optional[str] = None,
                  mesh_path_dir: Optional[str] = None,
                  has_freeflyer: bool = True,
                  avoid_instable_collisions: bool = True,
                  debug: bool = False) -> None:
        """Create and add a new robot to the simulator from scratch, based on
        configuration files only.

        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '\*_hardware.toml' file in
                              the same folder and with the same name.
        :param mesh_path_dir: Path to the folder containing all the meshes.
                              Optional: Env variable 'JIMINY_DATA_PATH' will be
                              used if available.
        :param has_freeflyer: Whether the robot is fixed-based wrt its root
                              link, or can move freely in the world.
                              Optional: True by default.
        :param config_path: Configuration toml file to import. It will be
                            imported AFTER loading the hardware description
                            file. It can be automatically generated from an
                            instance by calling `export_config_file` method.
                            Optional: Looking for '\*_options.toml' file in the
                            same folder and with the same name. If not found,
                            using default configuration.
        :param avoid_instable_collisions: Prevent numerical instabilities by
                                          replacing collision mesh by vertices
                                          of associated minimal volume bounding
                                          box, and replacing primitive box by
                                          its vertices.
        :param debug: Whether the debug mode must be activated. Doing it
                      enables temporary files automatic deletion.
        """
        robot = _build_robot_from_urdf(name, urdf_path, hardware_path,
                                       mesh_path_dir, has_freeflyer,
                                       avoid_instable_collisions, debug)

        # Get engine options
        engine_options = self.engine.get_options()

        for hardware_path, option_nested_path in (
            ('groundStiffness', ('contacts', 'stiffness')), 
            ('groundDamping', ('contacts', 'damping')),
            ('controllerUpdatePeriod', ('stepper', 'controllerUpdatePeriod')),
            ('sensorsUpdatePeriod', ('stepper', 'sensorsUpdatePeriod'))):
            if hardware_path in robot.extra_info.keys():
                option = engine_options
                for option_path in option_nested_path:
                    option = option[option_path]
                if robot.extra_info[hardware_path] != option:
                    warnings.warn(
                        f"You have speficied a different {hardware_path} then "
                        f"the one of the engine, the simulation will run with "
                        f"{option}.")

        # Add the new robot to the engine
        self.engine.add_robot(robot)

    def __del__(self) -> None:
        """Custom deleter to make sure the close is properly closed at exit.
        """
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Convenient fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Jiminy engine directly, without having to do it through `engine`.

        .. note::
            This method is not meant to be called manually.
        """
        return getattr(self.__getattribute__('engine'), name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return [*super().__dir__(), *dir(self.engine)]
    
    def _create_viewer(self,
                       robot: jiminy.Robot,
                       robot_state: jiminy.RobotState,
                       robot_color: Dict[str, Tuple[float]],
                       **viewer_kwargs: Dict[str, Any]) -> None:
        """Helper method to create a viewer associated to robot i.

        :param robot: The robot of which the viewer will be created.
        :param robot_state: The state associated to the robot.
        :param color: the color the robot will be in the viewer.
        :param viewer_kwargs: Extra keyword arguments to forward at `Viewer`
                              initialization.
        """

        if len(self.robots) == 1:
            robot_color = None

        self._viewers.append(Viewer(
            robot,
            robot_name=re.sub('[^A-Za-z0-9_]', '_', robot.name),
            use_theoretical_model=False,
            open_gui_if_parent=False,
            robot_color=robot_color,
            **viewer_kwargs))

        assert self._viewers[-1].backend is not None

        # Share the external force buffer of the viewer with the engine
        if self.is_simulation_running:
            self._viewers[-1].f_external = [*robot_state.f_external][1:]
        if self._viewers[-1].backend.startswith('panda3d'):
            # Enable display of COM, DCM and contact markers by default if
            # the robot has freeflyer.
            if robot.has_freeflyer:
                if "display_com" not in viewer_kwargs:
                    self._viewers[-1].display_center_of_mass(True)
                if "display_dcm" not in viewer_kwargs:
                    self._viewers[-1].display_capture_point(True)
                if "display_contacts" not in viewer_kwargs:
                    self._viewers[-1].display_contact_forces(True)

            # Enable display of external forces by default only for
            # the joints having an external force registered to it.
            if "display_f_external" not in viewer_kwargs:
                profile_forces, *_ = self.engine.profile_forces.values()
                force_frames = set(
                    robot.pinocchio_model.frames[f.frame_index].parent
                    for f in profile_forces)
                impulse_forces, *_ = self.engine.impulse_forces.values()
                force_frames |= set(
                    robot.pinocchio_model.frames[f.frame_index].parent
                    for f in impulse_forces)
                visibility = self._viewers[-1]._display_f_external
                assert isinstance(visibility, list)
                for i in force_frames:
                    visibility[i - 1] = True
                self._viewers[-1].display_external_forces(visibility)
    
    @property
    def viewer(self) -> Viewer:
        """Convenience proxy to get the viewer in single-robot simulations.

        .. warning::
            Method only supported for single-robot simulations. Call
            `self._viewers` in multi-robot simulations.
        """
        if len(self.robots) != 1:
            raise NotImplementedError(
                "Method only supported for single-robot simulations. Call"
                "`self._viewers` in multi-robot simulations.")
        if len(self._viewers) == 0:
            return None
        
        return self._viewers[0]

    @property
    def robot(self) -> jiminy.Robot:
        """Convenience proxy to get the robot in single-robot simulations.

        Internally, all it does is returning `self.engine.robots[0]` without
        any additional processing.

        .. warning::
            Method only supported for single-robot simulations. Call
        `self.engine.robots` in multi-robot simulations.
        """
        # A property is used in place of an instance attribute to keep pointing
        # to the right robot if the latter has been replaced by the user. Doing
        # so is dodgy and rarely necessary. For this reason, this capability is
        # not advertised but supported nonetheless.
        robot, = self.engine.robots
        return robot

    @property
    def robot_state(self) -> jiminy.RobotState:
        """Convenience proxy to get the state of the robot in single-robot
        simulations. 

        Internally, all it does is returning `self.engine.robot_states[0]`
        without any additional processing.

        .. warning::
            Method only supported for single-robot simulations. Call
            `self.engine.robot_states` in multi-robot simulations.
        """
        # property is used in place of attribute for extra safety
        return self.engine.robot_states[0]

    @property
    def is_viewer_available(self) -> bool:
        """Returns whether a viewer instance associated with the ongoing 
        single-robot simulation.

        .. warning::
            Method only supported for single-robot simulations.
        """
        return (self.viewer is not None and
                self.viewer.is_open())  # type: ignore[misc]
    
    @property
    def are_viewers_available(self) -> bool:
        """Returns whether there are viewer instances associated with the
        ongoing multi-robot simulation.
        """
        return (len(self._viewers) > 0 and
                all([viewer.is_open() for viewer in self._viewers]))

    def register_profile_force(self,
                               frame_name: str,
                               force_func: ProfileForceFunc,
                               update_period: float = 0.0) -> None:
        r"""Apply an external force profile on a given frame.

        The force can be time- and state-dependent, and may be time-continuous
        or updated periodically (Zero-Order Hold).

        :param frame_name: Name of the frame at which to apply the force.
        :param force_func:
            .. raw:: html

                Force profile as a callable with signature:

            | force_func\(
            |    **t**: float,
            |    **q**: np.ndarray,
            |    **v**: np.ndarray,
            |    **force**: np.ndarray
            |    \) -> None

            where `force` corresponds the spatial force in local world aligned
            frame, ie its origin is located at application frame but its basis
            is aligned with world frame. It is represented as a `np.ndarray`
            (Fx, Fy, Fz, Mx, My, Mz) that must be updated in-place.
        :param update_period: Update period of the force. It must be set to 0.0
                              for time-continuous. Discrete update is strongly
                              recommended for native Python callables because
                              evaluating them is so slow that it would slowdown
                              the whole simulation. There is no issue for C++
                              bindings such as `jiminy.RandomPerlinProcess`.
        """
        if len(self.engine.robots) > 1:
            raise NotImplementedError(
                "To register a force in multirobot simulation, you should use "
                "`simulation.engine.register_profile_force` and specify the "
                "name of the robot you want the force applied to.")
        return self.engine.register_profile_force(
            "", frame_name, force_func, update_period)

    def register_impulse_force(self,
                               frame_name: str,
                               t: float,
                               dt: float,
                               force: np.ndarray) -> None:
        r"""Apply an external impulse force on a given frame.

        The force starts at the fixed point in time and lasts a given duration.
        In the meantime, its profile is square-shaped, ie the force remains
        constant.

        :param frame_name: Name of the frame at which to apply the force.
        :param t: Time at which to start applying the external force.
        :param dt: Duration of the force.
        :param force_func: Spatial force in local world aligned frame, ie its
                           origin is located at application frame but its basis
                           is aligned with world frame. It is represented as a
                           `np.ndarray` (Fx, Fy, Fz, Mx, My, Mz).
        """
        if len(self.engine.robots) > 1:
            raise NotImplementedError(
                "To register a force in multirobot simulation, you should use "
                "`simulation.engine.register_profile_force` and specify the "
                "name of the robot you want the force applied to.")
        return self.engine.register_impulse_force("", frame_name, t, dt, force)

    def seed(self, seed: Union[np.uint32, np.ndarray]) -> None:
        """Set the seed of the simulation and reset the simulation.

        .. warning::
            It also resets the low-level jiminy Engine. Therefore one must call
            the `reset` method manually afterward.

        :param seed: Desired seed as a sequence of unsigned integers 32 bits.
        """
        assert seed.dtype == np.uint32, "'seed' must have dtype np.uint32."

        # Make sure no simulation is running before setting the seed
        self.engine.stop()

        # Set the seed in engine options
        engine_options = self.engine.get_options()
        engine_options["stepper"]["randomSeedSeq"] = np.asarray(seed)
        self.engine.set_options(engine_options)

        # It is expected by OpenAI Gym API to reset env after setting the seed
        self.reset()

    def reset(self, remove_all_forces: bool = False) -> None:
        """Reset the simulator.

        It resets the simulation time to zero, and generate a new random model
        of the robot. If one wants to get exactly the same result as before,
        either set the randomness of the model and sensors to zero, or set the
        seed once again to reinitialize the random number generator.

        :param remove_all_forces:
            Whether to remove already registered external forces. Note that it
            can also be done separately by calling `remove_all_forces` method.
            Optional: Do not remove by default.
        """
        # Reset the backend engine
        self.engine.reset(False, remove_all_forces)

    def start(self,
              q_init: Union[np.ndarray, Dict[str, np.ndarray]],
              v_init: Union[np.ndarray, Dict[str, np.ndarray]],
              a_init: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
              is_state_theoretical: Optional[bool] = None) -> None:
        """Initialize a simulation, starting from (q_init, v_init) at t=0.

        :param q_init: Initial configuration (by robot if it is a dictionnary).
        :param v_init: Initial velocity (by robot if it is a dictionnary).
        :param a_init: Initial acceleration (by robot if it is a dictionnary).
                       It is only used by acceleration dependent sensors and
                       controllers, such as IMU and force sensors.
        :param is_state_theoretical: In single robot simulations, whether the initial state is associated
                                     with the actual or theoretical model of
                                     the robot.
        """
        # Call base implementation
        # Single-robot simulations with `np.ndarray`, `is_state_theoretical` is
        # supported.
        if isinstance(q_init, np.ndarray):
            if is_state_theoretical is None:
                is_state_theoretical = False
            self.engine.start(q_init, v_init, a_init, is_state_theoretical)
        # Multi-robot simulations or single-robot simulations with
        # dictionnaries, `is_state_theoretical` is not supported.
        else:
            if is_state_theoretical is not None:
                raise Exception(
                    "`is_state_theoretical` is only supported for single-"
                    "robot simulations with `np.ndarray` initial configurations.")
            self.engine.start(q_init, v_init, a_init)

        # Share the external force buffer of the viewer with the engine.
        # Note that the force vector must be converted to pain list to avoid
        # copy with external sub-vector.
        for robot_state, viewer in zip(self.robot_states, self._viewers):
            viewer.f_external = [*robot_state.f_external][1:]

    def simulate(self,
                 t_end: float,
                 q_init: Union[np.ndarray, Dict[str, np.ndarray]],
                 v_init: Union[np.ndarray, Dict[str, np.ndarray]],
                 a_init: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                 is_state_theoretical: Optional[bool] = None,
                 callback: Optional[Callable[[], bool]] = None,
                 log_path: Optional[str] = None,
                 show_progress_bar: bool = True) -> None:
        """Run a simulation, starting from x0=(q0,v0) at t=0 up to tf.

        .. note::
            Optionally, log the result of the simulation.

        :param t_end: Simulation duration.
        :param q_init: Initial configuration (by robot if it is a dictionnary).
        :param v_init: Initial velocity (by robot if it is a dictionnary).
        :param a_init: Initial acceleration (by robot if it is a dictionnary).
                       It is only used by acceleration dependent sensors and
                       controllers, such as IMU and force sensors.
        :param is_state_theoretical: In single robot simulations, whether the
                                     initial state is associated with the
                                     actual or theoretical model of the robot.
        :param callback: Callable that can be specified to abort simulation. It
                         will be evaluated after every simulation step. Abort
                         if false is returned.
                         Optional: None by default.
        :param log_path: Save log data to this location. Disable if None.
                         Note that the format extension '.data' is enforced.
                         Optional, disable by default.
        :param show_progress_bar: Whether to display a progress bar during the
                                  simulation. None to enable only if available.
                                  Optional: None by default.
        """
        # Handling of progress bar if requested
        if show_progress_bar:
            # Initialize the progress bar
            self._pbar = tqdm(total=t_end, bar_format=(
                "{percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} "
                "[{elapsed}<{remaining}]"))

            # Define callable responsible for updating the progress bar
            def update_progress_bar() -> bool:
                """Update progress bar based on current simulation time.
                """
                nonlocal self
                if self._pbar is not None:
                    t = self.engine.stepper_state.t
                    self._pbar.update(t - self._pbar.n)
                return True

            # Hijack simulation callback to also update the progress bar
            if callback is None:
                callback = update_progress_bar
            else:
                def callback_wrapper(callback: Callable[[], bool]) -> bool:
                    """Update progress bar based on current simulation time,
                    then call a given callback function.
                    """
                    nonlocal update_progress_bar
                    return update_progress_bar() and callback()

                callback = partial(callback_wrapper, self, callback)

        # Run the simulation
        err = None
        try:
            # Call base implementation
            # Single-robot simulations with `np.ndarray`, `is_state_theoretical` is
            # supported.
            if isinstance(q_init, np.ndarray):
                if is_state_theoretical is None:
                    is_state_theoretical = False
                self.engine.simulate(
                    t_end, q_init, v_init, a_init, is_state_theoretical, callback)
            # Multi-robot simulations or single-robot simulations with
            # dictionnaries, `is_state_theoretical` is not supported.
            else:
                if is_state_theoretical is not None:
                    raise Exception(
                        "`is_state_theoretical` is only supported for single-"
                        "robot simulations with `np.ndarray` initial configurations.")
                self.engine.simulate(t_end, q_init, v_init, a_init, callback)

        except Exception as e:  # pylint: disable=broad-exception-caught
            err = e

        # Make sure that the progress bar is properly closed
        if show_progress_bar:
            assert self._pbar is not None
            self._pbar.close()
            self._pbar = None

        # Re-throw exception if not successful
        if err is not None:
            raise err

        # Write log
        if log_path is not None and self.engine.stepper_state.q:
            # Log data would be available as long as the stepper state is
            # initialized. It may be the case no matter if a simulation is
            # actually running, since data are cleared at reset not at stop.
            log_suffix = pathlib.Path(log_path).suffix[1:]
            if log_suffix not in ("data", "hdf5"):
                raise ValueError(
                    "Log format must be either '.data' or '.hdf5'.")
            log_format = log_suffix if log_suffix != 'data' else 'binary'
            self.engine.write_log(log_path, format=log_format)

    def render(self,
               return_rgb_array: bool = False,
               width: Optional[int] = None,
               height: Optional[int] = None,
               camera_pose: Optional[CameraPoseType] = None,
               update_ground_profile: Optional[bool] = None,
               **kwargs: Any) -> Optional[np.ndarray]:
        """Render the current state of the simulation. One can display it
               or return an RGB array instead.

        :param return_rgb_array: Whether to return the current frame as an rgb
                                 array or render it directly.
        :param width: Width of the returned RGB frame, if enabled.
        :param height: Height of the returned RGB frame, if enabled.
        :param camera_pose: Tuple position [X, Y, Z], rotation [Roll, Pitch,
                            Yaw], frame name/index specifying the absolute or
                            relative pose of the camera. `None` to disable.
                            Optional: `None` by default.
        :param update_ground_profile: Whether to update the ground profile. It
                                      must be called manually only if necessary
                                      because it is costly.
                                      Optional: True by default if no viewer
                                      available, False otherwise.
        :param kwargs: Extra keyword arguments to forward at `Viewer`
                       initialization.

        :returns: Rendering as an RGB array (3D numpy array), if enabled, None
                  otherwise.
        """
        # Handle default arguments
        if update_ground_profile is None:
            update_ground_profile = not self.are_viewers_available

        # Close the current viewer backend if not suitable
        if kwargs.get("backend", Viewer.backend) != Viewer.backend:
            Viewer.close()

        if self.are_viewers_available:
            backend = self._viewers[0].backend
        else:
            backend = Viewer.backend

        # Update viewer_kwargs with provided kwargs
        viewer_kwargs: Dict[str, Any] = {**dict(
            backend=backend,
            delete_robot_on_close=True),
            **self.viewer_kwargs,
            **kwargs}

        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        for i, viewer in enumerate(self._viewers):
            if not viewer.is_open():
                viewer.close()
                self._viewers.pop(i)

        camera_pose_need_initialization = True if len(self._viewers) == 0 else False
        
        # Cycle the colors if needed
        COLORS = ROBOTS_COLORS * (len(self.robots) // len(ROBOTS_COLORS) + 1)

        for robot, robot_state, color in zip(
            self.robots, self.engine.robot_states, COLORS):

            # Create new viewer instance if needed or use the existing one.
            if robot.name not in Viewer._backend_robot_names:
                self._create_viewer(robot, robot_state, color, **viewer_kwargs)
        
        if camera_pose_need_initialization:
            # Initialize camera pose
            if self._viewers[-1].is_backend_parent and camera_pose is None:
                camera_pose = viewer_kwargs.get("camera_pose", (
                    (9.0, 0.0, 2e-5), (np.pi/2, 0.0, np.pi/2), None))

        # Enable the ground profile is requested and available
        assert self._viewers and Viewer.backend is not None
        if update_ground_profile:
            engine_options = self.engine.get_options()
            ground_profile = engine_options["world"]["groundProfile"]
            Viewer.update_floor(ground_profile, show_meshes=False)

        # Set the camera pose if requested
        if camera_pose is not None:
            self._viewers[0].set_camera_transform(*camera_pose)

        # Make sure the graphical window is open if required
        if not return_rgb_array and Viewer.backend != "panda3d-sync":
            Viewer.open_gui()

        # Try refreshing the viewer
        for viewer in self._viewers:
            viewer.refresh()

        # Compute and return rgb array if needed
        if return_rgb_array:
            return Viewer.capture_frame(
                width or viewer_kwargs.get("width"),
                height or viewer_kwargs.get("height"))
        return None

    def replay(self,
               extra_logs_files: Sequence[Dict[str, np.ndarray]] = (),
               extra_trajectories: Sequence[TrajectoryDataType] = (),
               **kwargs: Any) -> None:
        """Replay the current episode until now.

        :param kwargs: Extra keyword arguments for delegation to
                       `replay.play_trajectories` method.
        """
        if len(self.robots) > 1:
            raise NotImplementedError("Simulator.replay() is only supported "
                                      "for single-robot simulations.")

        # Close extra viewer instances if any
        for viewer in self._viewers[1:]:
            viewer.delete_robot_on_close = True
            viewer.close()

        # Extract log data and robot from extra log files
        robots = [self.robot]
        logs_data = [self.log_data]
        for log_file in extra_logs_files:
            log_data = read_log(log_file)
            robot = build_robot_from_log(
                log_data, mesh_package_dirs=self.robot.mesh_package_dirs)
            robots.append(robot)
            logs_data.append(log_data)

        # Extract trajectory data from pairs (robot, log)
        trajectories: List[TrajectoryDataType] = []
        update_hooks: List[
            Optional[Callable[[float, np.ndarray, np.ndarray], None]]] = []
        extra_kwargs: Dict[str, Any] = {}
        for robot, log_data in zip(robots, logs_data):
            if log_data:
                traj, update_hook, _kwargs = \
                    extract_replay_data_from_log(log_data, robot)
                trajectories.append(traj)
                update_hooks.append(update_hook)
                extra_kwargs.update(_kwargs)
        trajectories += list(extra_trajectories)
        update_hooks += [None for _ in extra_trajectories]

        # Make sure there is something to replay
        if not trajectories:
            raise RuntimeError(
                "Nothing to replay. Please run a simulation before calling "
                "`replay` method, or provided data manually.")

        # Make sure the viewer is instantiated before replaying
        backend = (kwargs.get('backend', (self.viewer or Viewer).backend) or
                   get_default_backend())
        must_not_open_gui = (
            backend.startswith("panda3d") or
            kwargs.get('record_video_path') is not None)
        self.render(**{
            'return_rgb_array': must_not_open_gui,
            'update_floor': True,
            **kwargs})

        # Define sequence of viewer instances
        viewers = [self.viewer, *[None for _ in trajectories[:-1]]]

        # Replay the trajectories
        self._viewers = play_trajectories(
            trajectories,
            update_hooks,
            viewers=viewers,
            **{'verbose': True,
               **self.viewer_kwargs,
               **extra_kwargs,
               'display_f_external': None,
               **kwargs})

    def close(self) -> None:
        """Close the connection with the renderer.
        """
        if hasattr(self, "_viewers") and len(self._viewers) > 0:
            for viewer in self._viewers:
                viewer.close()
            self._viewers.clear()
        if hasattr(self, "figure") and self._figure is not None:
            self._figure.close()
            self._figure = None

    def plot(self,
             enable_flexiblity_data: bool = False,
             block: Optional[bool] = None,
             **kwargs: Any) -> TabbedFigure:
        """Display common simulation data over time.

        The figure features several tabs:

          - Subplots with robot configuration
          - Subplots with robot velocity
          - Subplots with robot acceleration
          - Subplots with motors torques
          - Subplots with raw sensor data (one tab for each type of sensor)

        :param enable_flexiblity_data:
            Enable display of flexibility joints in robot's configuration,
            velocity and acceleration subplots.
            Optional: False by default.
        :param block: Whether to wait for the figure to be closed before
                      returning.
                      Optional: False in interactive mode, True otherwise.
        :param kwargs: Extra keyword arguments to forward to `TabbedFigure`.
        """
        # Make sure plot submodule is available
        try:
            # pylint: disable=import-outside-toplevel
            from .plot import plot_log
        except ImportError as e:
            raise ImportError(
                "Method not available. Please install 'jiminy_py[plot]'."
                ) from e

        # Create figure, without closing the existing one
        self._figure = plot_log(
            self.log_data, self.robot, enable_flexiblity_data, block, **kwargs)

        return self._figure

    def get_options(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get the options of the engine plus the robot (including controller).
        """
        return {'engine': self.engine.get_options(),
                'robot': self.robot.get_options()}

    def set_options(self,
                    options: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Set the options of robot (including controller), and engine.
        """
        self.engine.set_options(options['engine'])
        self.robot.set_options(options['robot'])

    def export_options(self,
                       config_path: Optional[Union[str, os.PathLike]] = None
                       ) -> None:
        """Export the full configuration, ie the options of the robot (
        including controller), and the engine.

        .. note::
            the configuration can be imported thereafter using `import_options`
            method.
        """
        if config_path is None:
            if isinstance(self.robot, BaseJiminyRobot):
                urdf_path = self.robot._urdf_path_orig
            else:
                urdf_path = self.robot.urdf_path
            if not urdf_path:
                raise ValueError(
                    "'config_path' must be provided if the robot is not "
                    "associated with any URDF.")
            config_path = str(pathlib.Path(
                urdf_path).with_suffix('')) + '_options.toml'
        with open(config_path, 'w') as f:
            toml.dump(
                self.get_options(), f, encoder=toml.TomlNumpyEncoder())

    def import_options(self,
                       config_path: Optional[Union[str, os.PathLike]] = None
                       ) -> None:
        """Import the full configuration, ie the options of the robot (
        including controller), and the engine.

        .. note::
            Configuration can be exported beforehand using `export_options`
            method.
        """
        def deep_update(original: Dict[str, Any],
                        new_dict: Dict[str, Any],
                        *, _key_root: str = "") -> Dict[str, Any]:
            """Updates `original` dict with values from `new_dict` recursively.
            If a new key should be introduced, then an error is thrown instead.

            .. warning::
                Modify `original` in place.

            :param original: Dictionary with default values.
            :param new_dict: Dictionary with values to be updated.
            :param _key_root: Internal variable keeping track of current depth
                              within nested dict hierarchy.
            :returns: Update dictionary.
            """
            for key, value in new_dict.items():
                key_root = "/".join((_key_root, key))
                if key not in original:
                    raise ValueError(f"Key '{key_root}' not found")
                if isinstance(value, dict):
                    deep_update(original[key], value, _key_root=key_root)
                else:
                    original[key] = new_dict[key]
            return original

        if config_path is None:
            if isinstance(self.robot, BaseJiminyRobot):
                urdf_path = self.robot._urdf_path_orig
            else:
                urdf_path = self.robot.urdf_path
            if not urdf_path:
                raise ValueError(
                    "'config_path' must be provided if the robot is not "
                    "associated with any URDF.")
            config_path = str(pathlib.Path(
                urdf_path).with_suffix('')) + '_options.toml'
            if not os.path.exists(config_path):
                return

        options = deep_update(self.get_options(), toml.load(str(config_path)))
        self.set_options(options)
