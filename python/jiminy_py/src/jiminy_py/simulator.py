# mypy: disable-error-code="attr-defined, name-defined"
""" TODO: Write documentation.
"""
import os
import re
import atexit
import logging
import pathlib
import tempfile
from weakref import ref
from copy import deepcopy
from itertools import chain
from functools import partial
from collections import OrderedDict
from typing import (
    Any, List, Dict, Optional, Union, Type, Sequence, Iterable, Callable)

import toml
import numpy as np

import pinocchio as pin
from . import core as jiminy
from .robot import (generate_default_hardware_description_file,
                    BaseJiminyRobot)
from .dynamics import TrajectoryDataType
from .log import read_log, build_robot_from_log
from .viewer import (CameraPoseType,
                     interactive_mode,
                     get_default_backend,
                     extract_replay_data_from_log,
                     play_trajectories,
                     Viewer)

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


class Simulator:
    """This class wraps the different submodules of Jiminy, namely the robot,
    controller, engine, and viewer, as a single simulation environment. The
    user only as to create a robot and associated controller if any, and
    give high-level instructions to the simulator.
    """
    def __init__(self,  # pylint: disable=unused-argument
                 robot: jiminy.Robot,
                 controller: Optional[jiminy.AbstractController] = None,
                 engine_class: Type[jiminy.Engine] = jiminy.Engine,
                 use_theoretical_model: bool = False,
                 viewer_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs: Any) -> None:
        """
        :param robot: Jiminy robot already initialized.
        :param controller: Jiminy (observer-)controller already initialized.
                           Optional: None by default.
        :param engine_class: Class of engine to use.
                             Optional: jiminy_py.core.Engine by default.
        :param use_theoretical_model: Whether the state corresponds to the
                                      theoretical model when updating and
                                      fetching the robot's state.
        :param viewer_kwargs: Keyword arguments to override default arguments
                              whenever a viewer must be instantiated, eg when
                              `render` method is first called. Specifically,
                              `backend` is ignored if one is already available.
                              Optional: Empty by default.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup the user arguments
        self.use_theoretical_model = use_theoretical_model
        self.viewer_kwargs = deepcopy(viewer_kwargs or {})

        # Handling of default argument(s)
        if "robot_name" not in self.viewer_kwargs:
            base_name = re.sub('[^A-Za-z0-9_]', '_', robot.name)
            robot_name = f"{base_name}_{next(tempfile._get_candidate_names())}"
            self.viewer_kwargs["robot_name"] = robot_name

        # Wrap callback in nested function to hide update of progress bar
        # Note that a weak reference must be used to avoid circular reference
        # resulting in uncollectable object and hence memory leak.
        simulator_ref = ref(self)

        def callback_wrapper(t: float,
                             *args: Any,
                             **kwargs: Any) -> None:
            nonlocal simulator_ref
            simulator = simulator_ref()
            assert simulator is not None
            if simulator.__pbar is not None:
                simulator.__pbar.update(t - simulator.__pbar.n)
            simulator._callback(t, *args, **kwargs)

        # Instantiate the low-level Jiminy engine, then initialize it
        self.engine = engine_class()
        hresult = self.engine.initialize(robot, controller, callback_wrapper)
        if hresult != jiminy.hresult_t.SUCCESS:
            raise RuntimeError(
                "Invalid robot or controller. Make sure they are both "
                "initialized.")

        # Create shared memories and python-native attribute for fast access
        self.stepper_state = self.engine.stepper_state
        self.is_simulation_running = self.engine.is_simulation_running

        # Viewer management
        self.viewer: Optional[Viewer] = None
        self._viewers: Sequence[Viewer] = []

        # Internal buffer for progress bar management
        self.__pbar: Optional[tqdm] = None

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
              **kwargs: Any) -> 'Simulator':
        r"""Create a new simulator instance from scratch, based on
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

        # Instantiate and initialize the robot
        robot = BaseJiminyRobot()
        robot.initialize(
            urdf_path, hardware_path, mesh_path_dir, (), has_freeflyer,
            avoid_instable_collisions, load_visual_meshes=debug, verbose=debug)

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

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), dir(self.engine))

    @property
    def pinocchio_model(self) -> pin.Model:
        """Getter of the pinocchio model, depending on the value of
           'use_theoretical_model'.
        """
        if self.use_theoretical_model and self.robot.is_flexible:
            return self.robot.pinocchio_model_th
        return self.robot.pinocchio_model

    @property
    def pinocchio_data(self) -> pin.Data:
        """Getter of the pinocchio data, depending on the value of
           'use_theoretical_model'.
        """
        if self.use_theoretical_model and self.robot.is_flexible:
            return self.robot.pinocchio_data_th
        return self.robot.pinocchio_data

    @property
    def is_viewer_available(self) -> bool:
        """Returns whether a viewer instance associated with the robot
        is available.
        """
        return (self.viewer is not None and
                self.viewer.is_open())  # type: ignore[misc]

    def _callback(self,
                  t: float,  # pylint: disable=unused-argument
                  q: np.ndarray,  # pylint: disable=unused-argument
                  v: np.ndarray,  # pylint: disable=unused-argument
                  out: np.ndarray) -> None:
        """Callback method for the simulation.
        """
        out[()] = True

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
              q_init: np.ndarray,
              v_init: np.ndarray,
              a_init: Optional[np.ndarray] = None,
              is_state_theoretical: bool = False) -> None:
        """Initialize a simulation, starting from (q_init, v_init) at t=0.

        :param q_init: Initial configuration.
        :param v_init: Initial velocity.
        :param a_init: Initial acceleration. It is only used by acceleration
                       dependent sensors and controllers, such as IMU and force
                       sensors.
        :param is_state_theoretical: Whether the initial state is associated
                                     with the actual or theoretical model of
                                     the robot.
        """
        # Call base implementation
        hresult = self.engine.start(
            q_init, v_init, a_init, is_state_theoretical)
        if hresult != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("Failed to start the simulation.")

        # Share the external force buffer of the viewer with the engine.
        # Note that the force vector must be converted to pain list to avoid
        # copy with external sub-vector.
        if self.viewer is not None:
            self.viewer.f_external = [*self.system_state.f_external][1:]

    def step(self, step_dt: float = -1) -> None:
        """Integrate system dynamics from current state for a given duration.

        :param step_dt: Duration for which to integrate. -1 to use default
                        duration, namely until the next breakpoint if any,
                        or 'engine_options["stepper"]["dtMax"]'.
        """
        # Perform a single integration step
        if not self.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please call `start` before `step`.")
        return_code = self.engine.step(step_dt)
        if return_code != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("Failed to perform the simulation step.")

    def simulate(self,
                 t_end: float,
                 q_init: np.ndarray,
                 v_init: np.ndarray,
                 a_init: Optional[np.ndarray] = None,
                 is_state_theoretical: bool = True,
                 log_path: Optional[str] = None,
                 show_progress_bar: bool = True) -> None:
        """Run a simulation, starting from x0=(q0,v0) at t=0 up to tf.

        .. note::
            Optionally, log the result of the simulation.

        :param t_end: Simulation end time.
        :param q_init: Initial configuration.
        :param v_init: Initial velocity.
        :param a_init: Initial acceleration.
        :param is_state_theoretical: Whether the initial state is associated
                                     with the actual or theoretical model of
                                     the robot.
        :param log_path: Save log data to this location. Disable if None.
                         Note that the format extension '.data' is enforced.
                         Optional, disable by default.
        :param show_progress_bar: Whether to display a progress bar during the
                                  simulation. None to enable only if available.
                                  Optional: None by default.
        """
        # Show progress bar if requested
        if show_progress_bar:
            self.__pbar = tqdm(total=t_end, bar_format=(
                "{percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} "
                "[{elapsed}<{remaining}]"))

        # Run the simulation
        try:
            return_code = self.engine.simulate(
                t_end, q_init, v_init, a_init, is_state_theoretical)
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOGGER.warning(
                "The simulation failed due to Python exception:\n %s", e)
            return_code = jiminy.hresult_t.ERROR_GENERIC
        finally:  # Make sure that the progress bar is properly closed
            if show_progress_bar:
                assert self.__pbar is not None
                self.__pbar.close()
                self.__pbar = None

        # Throw exception if not successful
        if return_code != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("The simulation failed internally.")

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
            update_ground_profile = not self.is_viewer_available

        # Close the current viewer backend if not suitable
        if kwargs.get("backend", Viewer.backend) != Viewer.backend:
            Viewer.close()

        # Update viewer_kwargs with provided kwargs
        viewer_kwargs: Dict[str, Any] = {**dict(
            backend=(self.viewer or Viewer).backend,
            delete_robot_on_close=True),
            **self.viewer_kwargs,
            **kwargs}

        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        if not self.is_viewer_available:
            # Make sure that the current viewer is properly closed if any
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

            # Create new viewer instance
            self.viewer = Viewer(
                self.robot,
                use_theoretical_model=False,
                open_gui_if_parent=False,
                **viewer_kwargs)
            assert self.viewer is not None and self.viewer.backend is not None

            # Share the external force buffer of the viewer with the engine
            if self.is_simulation_running:
                self.viewer.f_external = [*self.system_state.f_external][1:]

            if self.viewer.backend.startswith('panda3d'):
                # Enable display of COM, DCM and contact markers by default if
                # the robot has freeflyer.
                if self.robot.has_freeflyer:
                    if "display_com" not in viewer_kwargs:
                        self.viewer.display_center_of_mass(True)
                    if "display_dcm" not in viewer_kwargs:
                        self.viewer.display_capture_point(True)
                    if "display_contacts" not in viewer_kwargs:
                        self.viewer.display_contact_forces(True)

                # Enable display of external forces by default only for
                # the joints having an external force registered to it.
                if "display_f_external" not in viewer_kwargs:
                    force_frames = set(
                        self.robot.pinocchio_model.frames[f_i.frame_idx].parent
                        for f_i in self.engine.forces_profile)
                    force_frames |= set(
                        self.robot.pinocchio_model.frames[f_i.frame_idx].parent
                        for f_i in self.engine.forces_impulse)
                    visibility = self.viewer._display_f_external
                    assert isinstance(visibility, list)
                    for i in force_frames:
                        visibility[i - 1] = True
                    self.viewer.display_external_forces(visibility)

            # Initialize camera pose
            if self.viewer.is_backend_parent and camera_pose is None:
                camera_pose = viewer_kwargs.get("camera_pose", (
                    (9.0, 0.0, 2e-5), (np.pi/2, 0.0, np.pi/2), None))

        # Enable the ground profile is requested and available
        assert self.viewer is not None and self.viewer.backend is not None
        if update_ground_profile:
            engine_options = self.engine.get_options()
            ground_profile = engine_options["world"]["groundProfile"]
            Viewer.update_floor(ground_profile, show_meshes=False)

        # Set the camera pose if requested
        if camera_pose is not None:
            self.viewer.set_camera_transform(*camera_pose)

        # Make sure the graphical window is open if required
        if not return_rgb_array:
            Viewer.open_gui()

        # Try refreshing the viewer
        self.viewer.refresh()

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
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
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
            Enable display of flexible joints in robot's configuration,
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

    def get_controller_options(self) -> dict:
        """Getter of the options of Jiminy Controller.
        """
        return self.engine.controller.get_options()

    def set_controller_options(self, options: dict) -> None:
        """Setter of the options of Jiminy Controller.
        """
        self.engine.controller.set_options(options)

    def get_options(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get the options of robot (including controller), and engine.
        """
        options: Dict[str, Dict[str, Dict[str, Any]]] = OrderedDict(
            system=OrderedDict(robot=OrderedDict(), controller=OrderedDict()),
            engine=OrderedDict())
        robot_options = options['system']['robot']
        robot_options_copy = self.robot.get_options()
        robot_options['model'] = robot_options_copy['model']
        robot_options['motors'] = robot_options_copy['motors']
        robot_options['sensors'] = robot_options_copy['sensors']
        robot_options['telemetry'] = robot_options_copy['telemetry']
        options['system']['controller'] = self.get_controller_options()
        engine_options = options['engine']
        engine_options_copy = self.engine.get_options()
        engine_options['stepper'] = engine_options_copy['stepper']
        engine_options['world'] = engine_options_copy['world']
        engine_options['joints'] = engine_options_copy['joints']
        engine_options['constraints'] = engine_options_copy['constraints']
        engine_options['contacts'] = engine_options_copy['contacts']
        engine_options['telemetry'] = engine_options_copy['telemetry']
        return options

    def set_options(self,
                    options: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Set the options of robot (including controller), and engine.
        """
        controller_options = options['system']['controller']
        self.robot.set_options(options['system']['robot'])
        self.set_controller_options(controller_options)
        self.engine.set_options(options['engine'])

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
