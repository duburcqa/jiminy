import os
import toml
import atexit
import logging
import pathlib
import tempfile
from collections import OrderedDict
from typing import Optional, Union, Type, Dict, Tuple, Sequence, List, Any

import numpy as np

from . import core as jiminy
from .core import (EncoderSensor as encoder,
                   EffortSensor as effort,
                   ContactSensor as contact,
                   ForceSensor as force,
                   ImuSensor as imu)
from .robot import (generate_default_hardware_description_file,
                    BaseJiminyRobot)
from .dynamics import TrajectoryDataType
from .log import (read_log,
                  build_robots_from_log)
from .viewer import (interactive_mode,
                     get_default_backend,
                     extract_replay_data_from_log,
                     play_trajectories,
                     Viewer)

if interactive_mode() >= 2:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger(__name__)


SENSORS_FIELDS = {
    encoder: encoder.fieldnames,
    effort: effort.fieldnames,
    contact: contact.fieldnames,
    force: {
        k: [e[len(k):] for e in force.fieldnames if e.startswith(k)]
        for k in ['F', 'M']
    },
    imu: {
        k: [e[len(k):] for e in imu.fieldnames if e.startswith(k)]
        for k in ['Quat', 'Gyro', 'Accel']
    }
}

DEFAULT_UPDATE_PERIOD = 1.0e-3  # 0.0 for time continuous update
DEFAULT_GROUND_STIFFNESS = 4.0e6
DEFAULT_GROUND_DAMPING = 2.0e3


def _sanitize_state(engine: jiminy.EngineMultiRobot,
                    q_init: Optional[Union[np.ndarray, Dict[
                        str, np.ndarray]]] = None,
                    v_init: Optional[Union[np.ndarray, Dict[
                        str, np.ndarray]]] = None,
                    a_init: Optional[Union[np.ndarray, Dict[
                        str, np.ndarray]]] = None,
                    is_state_theoretical: Optional[Union[bool, Dict[
                        str, bool]]] = None):
    """Adapt the jiminy.Robot kinematic vectors to dictionnaries of vectors for
    the EngineMultiRobot simulator.

    :param engine: Engine used in the simulation.
    :param q_init: Array of positions for a single robot or a dictionnary of
                   positions for EngineMultiRobot.
                   Optional: None by default.
    :param v_init: Array of velocities for a single robot or a dictionnary of
                   velocities for EngineMultiRobot.
                   Optional: None by default.
    :param a_init: Array of accelerations for a single robot or a dictionnary
                   of accelerations for EngineMultiRobot.
                   Optional: None by default.
    :param q_init: Boolean for a single robot or a dictionnary of booleans for
                   EngineMultiRobot.
                   Optional: None by default.

    :returns: Tuple of kinematic vectors and theoretical states adapted for the
              EngineMultiRobot simualtor.
    """
    if len(engine.systems) == 1:
        if isinstance(q_init, np.ndarray):
            q_init = {engine.systems_names[0]: q_init}
        if isinstance(v_init, np.ndarray):
            v_init = {engine.systems_names[0]: v_init}
        if isinstance(a_init, np.ndarray):
            a_init = {engine.systems_names[0]: a_init}

    if isinstance(is_state_theoretical, bool):
        is_state_theoretical = {
            name: is_state_theoretical
            for name in engine.systems_names}

    for system in engine.systems:
        name = system.name
        if is_state_theoretical[name]:
            q_init[name] = \
                system.robot.get_flexible_configuration_from_rigid(
                    q_init[name])
            v_init[name] = \
                system.robot.get_flexible_velocity_from_rigid(v_init[name])
            if a_init:
                a_init[name] = \
                    system.robot.get_flexible_velocity_from_rigid(a_init[name])

    return q_init, v_init, a_init, is_state_theoretical


def unwrap_log_vars(systems_names, log_vars):
    if len(systems_names) == 1 and systems_names[0] == "":
        logs_vars = {systems_names[0]: log_vars}
    else:
        logs_vars = {
            system_name: {'Global.Time': log_vars['Global.Time']}
            for system_name in systems_names}
        for key in log_vars.keys():
            for system_name in systems_names:
                if system_name in key:
                    logs_vars[system_name][key.replace(
                        f'{system_name}.', '')] = log_vars[key]
    return logs_vars


def unwrap_log_constants(systems_names, log_constants):
    if len(systems_names) == 1 and systems_names[0] == "":
        logs_vars = {systems_names[0]: log_constants}
    else:
        logs_vars = {system_name: {}
                     for system_name in systems_names}
        for key in log_constants.keys():
            for system_name in systems_names:
                if system_name in key:
                    logs_vars[system_name][key.replace(
                        f'{system_name}.', '')] = log_constants[key]
    return logs_vars


class Simulator:
    """
    This class wraps the different submodules of Jiminy, namely the robot(s),
    controller(s), engine, and viewer, as a single simulation environment. The
    user only as to create (a) robot(s) and associated controller(s) if any,
    and give high-level instructions to the simulator.
    """
    def __new__(cls, *args: Any, **kwargs: Any) -> "Simulator":
        # Instantiate base class
        self = super().__new__(cls)

        # Viewer management
        self.viewers = []
        self._viewers_all = []
        Viewer.close()

        # Internal buffer for progress bar management
        self.__pbar: Optional[tqdm] = None

        # Figure holder
        self.figures = []

        return self

    def __init__(self,
                 robot: jiminy.Robot,
                 controller: Optional[jiminy.AbstractController] = None,
                 engine_class: Type[
                     jiminy.EngineMultiRobot] = jiminy.EngineMultiRobot,
                 viewer_backend: Optional[str] = None,
                 system_name: str = "",
                 **kwargs: Any) -> None:
        """
        :param robot: Jiminy robot already initialized.
        :param controller: Jiminy (observer-)controller already initialized.
                           Optional: None by default.
        :param engine_class: Class of engine to use.
                             Optional: jiminy_py.core.Engine by default.
        :param viewer_backend: Backend of the viewer, e.g. panda3d or meshcat.
                               Optional: It is setup-dependent. See `Viewer`
                               documentation for details about it.
        :param system_name: Name of the first system registered during the
                            initialization.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup the user arguments
        self.viewer_backend = viewer_backend
        # Wrap callback in nested function to hide update of progress bar

        def callback_wrapper(t: float,
                             *args: Any,
                             **kwargs: Any) -> None:
            nonlocal self
            if self.__pbar is not None:
                self.__pbar.update(t - self.__pbar.n)
            self._callback(t, *args, **kwargs)

        # Instantiate the low-level Jiminy engine, then initialize it
        self.engine = engine_class()

        hresult = self.engine.add_system(
            system_name, robot, controller, callback_wrapper)

        if hresult != jiminy.hresult_t.SUCCESS:
            raise RuntimeError(
                "Invalid robot or controller. Make sure they are both "
                "initialized.")

        # Create shared memories and python-native attribute for fast access
        self.is_simulation_running = self.engine.is_simulation_running
        self.stepper_state = self.engine.stepper_state
        self.system_name = system_name

        # Reset the low-level jiminy engine
        self.reset()

    @classmethod
    def build(cls,
              urdf_path: str,
              hardware_path: Optional[str] = None,
              mesh_path: Optional[str] = None,
              has_freeflyer: bool = True,
              config_path: Optional[str] = None,
              avoid_instable_collisions: bool = True,
              debug: bool = False,
              **kwargs) -> 'Simulator':
        r"""Create a new simulator instance from scratch, based on
        configuration files only.

        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '\*_hardware.toml' file in
                              the same folder and with the same name.
        :param mesh_path: Path to the folder containing the model meshes.
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
        :param debug: Whether or not the debug mode must be activated.
                      Doing it enables temporary files automatic deletion.
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
                    def remove_file_at_exit(file_path=hardware_path):
                        try:
                            os.remove(file_path)
                        except (PermissionError, FileNotFoundError):
                            pass

                    atexit.register(remove_file_at_exit)

                # Generate default Hardware Description File
                generate_default_hardware_description_file(
                    urdf_path, hardware_path, verbose=debug)

        # Instantiate and initialize the robot
        robot = BaseJiminyRobot()
        robot.initialize(
            urdf_path, hardware_path, mesh_path, has_freeflyer,
            avoid_instable_collisions, load_visual_meshes=debug, verbose=debug)

        # Instantiate and initialize the engine
        simulator = Simulator.__new__(cls)
        Simulator.__init__(
            simulator, robot, engine_class=jiminy.EngineMultiRobot, **kwargs)

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
        """Fallback attribute getter.

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
        return super().__dir__() + self.engine.__dir__()

    @property
    def system_state(self):
        """Getter of system_state in single robot.
        """
        try:
            return self.engine.systems_states[""]
        except KeyError as e:
            raise RuntimeError(
                "This proxy is only available for single-robot simulation"
                ) from e

    @property
    def system(self):
        """Getter of system in single robot.

        .. warning:
            This proxy is supposed to be used in single-robot simulation,
            otherwise it returns the first system that was added to the
            engine for efficiency.
        """
        return self.engine.systems[0]

    @property
    def log_data(self) -> Dict[str, str]:
        """Getter of the telemetry constants.
        """
        return self.engine.log_data

    @property
    def log_vars(self) -> Dict[str, np.ndarray]:
        """Getter of the telemetry variables.
        """
        return self.engine.log_data["variables"]

    @property
    def log_constants(self) -> Dict[str, str]:
        """Getter of the telemetry constants.
        """
        return self.engine.log_data["constants"]

    @property
    def is_viewer_available(self) -> bool:
        """Returns whether or not a viewer instance associated with the robot
        is available.
        """
        return self.viewers and self.viewers[0].is_open()

    def _callback(self,
                  t: float,
                  q: np.ndarray,
                  v: np.ndarray,
                  out: np.ndarray) -> None:
        """Callback method for the simulation.
        """
        out[()] = True

    def seed(self, seed: np.uint32) -> None:
        """Set the seed of the simulation and reset the simulation.

        .. warning::
            It also resets the low-level jiminy Engine. Therefore one must call
            the `reset` method manually afterward.

        :param seed: Desired seed (Unsigned integer 32 bits).
        """
        assert isinstance(seed, np.uint32), "'seed' must have type np.uint32."

        # Make sure no simulation is running before setting the seed
        self.engine.stop()

        # Force to reset the seed of the low-level engine
        jiminy.reset_random_generator(seed)

        # Set the seed in engine options to keep track of the seed and log it
        # automatically in the telemetry as constant.
        engine_options = self.engine.get_options()
        engine_options["stepper"]["randomSeed"] = \
            np.array(seed, dtype=np.dtype('uint32'))
        self.engine.set_options(engine_options)

        # It is expected by OpenAI Gym API to reset env after setting the seed
        self.reset()

    def reset(self, remove_all_forces: bool = False):
        """Reset the simulator.

        It resets the simulation time to zero, and generate a new random model
        of the robot. If one wants to get exactly the same result as before,
        either set the randomness of the model and sensors to zero, or set the
        seed once again to reinitialize the random number generator.

        :param remove_all_forces:
            Whether or not to remove already registered external forces. Note
            that it can also be done separately by calling `remove_all_forces`
            method.
            Optional: Do not remove by default.
        """
        # Reset the backend engine
        self.engine.reset(False, remove_all_forces)

    def start(self,
              q_init: Union[np.ndarray, Dict[str, np.ndarray]],
              v_init: Union[np.ndarray, Dict[str, np.ndarray]],
              a_init: Optional[
                  Union[np.ndarray, Dict[str, np.ndarray]]] = None,
              is_state_theoretical: Union[bool, Dict[str, bool]] = False
              ) -> None:
        """Initialize a simulation, starting from (q_init, v_init) at t=0.

        :param q_init: Initial configuration.
        :param v_init: Initial velocity.
        :param a_init: Initial acceleration. It is only used by acceleration
                       dependent sensors and controllers, such as IMU and force
                       sensors.
        :param is_state_theoretical: Whether or not the initial state is
                                     associated with the actual or theoretical
                                     model of the robot.
        """

        q_init, v_init, a_init, is_state_theoretical = _sanitize_state(
            self.engine, q_init, v_init, a_init, is_state_theoretical)

        for robot in self.engine.systems:
            if is_state_theoretical[robot.name]:
                q_init[robot.name] = \
                    robot.get_flexible_configuration_from_rigid(
                        q_init[robot.name])
                v_init[robot.name] = robot.get_flexible_velocity_from_rigid(
                        v_init[robot.name])
                if a_init[robot.name]:
                    a_init[robot.name] = \
                        robot.get_flexible_velocity_from_rigid(
                        a_init[robot.name])
        # Call base implementation
        hresult = self.engine.start(
            q_init, v_init, a_init)
        if hresult != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("Failed to start the simulation.")

        # Share the external force buffer of the viewer with the engine.
        # Note that the force vector must be converted to pain list to avoid
        # copy with external sub-vector.
        for viewer, system_state in zip(self.viewers, self.systems_states):
            if viewer is not None:
                viewer.f_external = [*system_state.f_external][1:]

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
                 q_init: Union[np.ndarray, Dict[str, np.ndarray]],
                 v_init: Union[np.ndarray, Dict[str, np.ndarray]],
                 a_init: Optional[
                    Union[np.ndarray, Dict[str, np.ndarray]]] = None,
                 is_state_theoretical: Union[bool, Dict[str, bool]] = False,
                 log_path: Optional[str] = None,
                 show_progress_bar: bool = True) -> None:
        """Run a simulation, starting from x0=(q0,v0) at t=0 up to tf.

        .. note::
            Optionally, log the result of the simulation.

        :param t_end: Simulation end time.
        :param q_init: Initial configuration.
        :param v_init: Initial velocity.
        :param a_init: Initial acceleration.
        :param is_state_theoretical: Whether or not the initial state is
                                     associated with the actual or theoretical
                                     model of the robot.
        :param log_path: Save log data to this location. Disable if None.
                         Note that the format extension '.data' is enforced.
                         Optional, disable by default.
        :param show_progress_bar: Whether or not to display a progress bar
                                  during the simulation. None to enable only
                                  if available.
                                  Optional: None by default.
        """
        q_init, v_init, a_init, is_state_theoretical = _sanitize_state(
            self.engine, q_init, v_init, a_init, is_state_theoretical)

        # Show progress bar if requested
        if show_progress_bar:
            self.__pbar = tqdm(total=t_end, bar_format=(
                "{percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} "
                "[{elapsed}<{remaining}]"))
        # Run the simulation
        try:
            return_code = self.engine.simulate(
                t_end, q_init, v_init, a_init)
        except Exception as e:
            logger.warning(
                "The simulation failed due to Python exception:\n", str(e))
            return_code = jiminy.hresult_t.ERROR_GENERIC
        finally:  # Make sure that the progress bar is properly closed
            if show_progress_bar:
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
            if log_suffix not in ("data", "csv", "hdf5"):
                raise ValueError(
                    "Log format not recognized. It must be either '.data', "
                    "'.csv', or '.hdf5'.")
            log_format = log_suffix if log_suffix != 'data' else 'binary'
            self.engine.write_log(log_path, format=log_format)

    def render(self,
               return_rgb_array: bool = False,
               width: Optional[int] = None,
               height: Optional[int] = None,
               camera_xyzrpy: Optional[Tuple[
                   Union[Tuple[float, float, float], np.ndarray],
                   Union[Tuple[float, float, float], np.ndarray]]] = None,
               update_ground_profile: Optional[bool] = None,
               **kwargs: Any) -> Optional[np.ndarray]:
        """Render the current state of the simulation. One can display it
               or return an RGB array instead.

        :param return_rgb_array: Whether or not to return the current frame as
                                 an rgb array.
        :param width: Width of the returned RGB frame, if enabled.
        :param height: Height of the returned RGB frame, if enabled.
        :param camera_xyzrpy: Tuple position [X, Y, Z], rotation [Roll, Pitch,
                              Yaw] corresponding to the absolute pose of the
                              camera. None to disable.
                              Optional: None by default.
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
        # Consider no viewer is available if the backend is the wrong one
        if kwargs.get("backend", self.viewer_backend) != self.viewer_backend:
            for viewer in self.viewers:
                self._viewers_all.pop(viewer)
                viewer.close()
            self.viewers = []

        # Handle default arguments
        if update_ground_profile is None:
            update_ground_profile = not self.is_viewer_available

        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        if not self.is_viewer_available:
            # Generate a new unique identifier if necessary
            if self.viewers:
                robots_names = [viewer.robot_name for viewer in self.viewers]
                scene_name = self.viewers[0].scene_name
            else:
                scene_name = "current simulation"
                robots_names = [f"{system.name}" for system in self.systems]

            # Create new viewer instance
            viewer_backend = self.viewer_backend or Viewer.backend
            self.viewers = [
                Viewer(system.robot,
                       use_theoretical_model=False,
                       open_gui_if_parent=False,
                       **{'scene_name': scene_name,
                          'robot_name': robot_name,
                          'backend': viewer_backend,
                          'delete_robot_on_close': True,
                          **kwargs})
                for robot_name, system in zip(robots_names, self.systems)]

            # Backup current backend
            self.viewer_backend = self.viewer_backend or Viewer.backend

            # Share the external force buffer of the viewer with the engine
            if self.is_simulation_running:
                for i, system in enumerate(self.systems):
                    viewer = self.viewers[i]
                    viewer.f_external = system.f_external[1:]

            if self.viewer_backend.startswith('panda3d'):
                # Enable display of COM, DCM and contact markers by default if
                # the robot has freeflyer.
                for i, system in enumerate(self.systems):
                    viewer = self.viewers[i]
                    if system.robot.has_freeflyer:
                        if "display_com" not in kwargs:
                            viewer.display_center_of_mass(True)
                        if "display_dcm" not in kwargs:
                            viewer.display_capture_point(True)
                        if "display_contacts" not in kwargs:
                            viewer.display_contact_forces(True)

                # Enable display of external forces by default only for
                # the joints having an external force registered to it.
                if "display_f_external" not in kwargs:
                    for i, system in enumerate(self.systems):
                        viewer = self.viewers[i]
                        force_frames = set([
                            system.robot.pinocchio_model.frames[
                                f_i.frame_idx].parent
                            for f_i in self.engine.forces_profile[
                                system.name]])
                        force_frames |= set([
                            system.robot.pinocchio_model.frames[
                                f_i.frame_idx].parent
                            for f_i in self.engine.forces_impulse[
                                system.name]])
                        visibility = viewer._display_f_external
                        for i in force_frames:
                            visibility[i - 1] = True
                        viewer.display_external_forces(visibility)

            # Initialize camera pose
            if camera_xyzrpy is None and any(
                    viewer.is_backend_parent for viewer in self.viewers):
                camera_xyzrpy = [(9.0, 0.0, 2e-5), (np.pi/2, 0.0, np.pi/2)]

        # Enable the ground profile is requested and available
        if self.viewer_backend.startswith('panda3d') and update_ground_profile:
            engine_options = self.engine.get_options()
            ground_profile = engine_options["world"]["groundProfile"]
            Viewer.update_floor(ground_profile, show_meshes=False)

        # Set the camera pose if requested
        if camera_xyzrpy is not None:
            Viewer.set_camera_transform(None, *camera_xyzrpy)

        # Make sure the graphical window is open if required
        if not return_rgb_array:
            Viewer.open_gui()

        # Try refreshing the viewer
        for viewer in self.viewers:
            viewer.refresh()

        # Compute rgb array if needed
        if return_rgb_array:
            return Viewer.capture_frame(width, height)

    def replay(self,
               extra_logs_files: Sequence[Dict[str, np.ndarray]] = (),
               extra_trajectories: Sequence[TrajectoryDataType] = (),
               **kwargs: Any) -> None:
        """Replay the current episode until now.

        :param kwargs: Extra keyword arguments for delegation to
                       `replay.play_trajectories` method.
        """
        legend = []

        # Close extra viewer instances if any
        for viewer in self._viewers_all[len(self.viewers):]:
            viewer.delete_robot_on_close = True
            self._viewers_all.pop(viewer)
            viewer.close()
            legend.append(viewer.robot_name)
        # Extract log data and robot from extra log files
        robots = {system.name: system.robot for system in self.engine.systems}

        logs_vars = unwrap_log_vars(self.engine.systems_names, self.log_vars)
        logs_constants = unwrap_log_constants(
            self.engine.systems_names, self.log_constants)

        for log_file in extra_logs_files:
            log_data = read_log(log_file)
            log_vars = log_data["variables"]
            log_constants = log_data["constants"]
            mesh_package_dirs = []
            for system in self.systems:
                mesh_package_dirs.append(system.robot.mesh_package_dirs[0])
            new_robots = build_robots_from_log(
                log_constants, mesh_package_dirs)
            systems_names = new_robots.keys()
            unwrapped_log_vars = unwrap_log_vars(systems_names, log_vars)
            unwrapped_log_constants = unwrap_log_constants(
                systems_names, log_constants)
            for name, robot in new_robots.items():
                if name in robots.keys():
                    if '/' in log_file:
                        scene_name = log_file[log_file.rfind('/')+1:-5]
                    else:
                        scene_name = log_file[:-5]
                    robots[f"{name} {scene_name}"] = robot
                    logs_vars[f"{name} {scene_name}"] = unwrapped_log_vars[
                        name]
                    logs_constants[f"{name} {scene_name}"] = \
                        unwrapped_log_constants[name]
                else:
                    robots[name] = robot
                    logs_vars[name] = unwrapped_log_vars[name]
                    logs_constants[name] = unwrapped_log_constants[name]

        # Extract trajectory data from pairs (robot, log)
        trajectories, update_hooks, extra_kwargs = [], [], {}
        for system_name in robots.keys():
            robot = robots[system_name]
            log_vars = logs_vars[system_name]
            log_constants = logs_constants[system_name]
            log_data = {"variables": log_vars, "constants": log_constants}
            if log_vars:
                traj, update_hook, _kwargs = \
                    extract_replay_data_from_log(log_vars, robot=robot)
                trajectories.append(traj)
                update_hooks.append(update_hook)
                extra_kwargs.update(_kwargs)
                legend.append(system_name)
        trajectories += list(extra_trajectories)
        update_hooks += [None for _ in extra_trajectories]

        # Make sure there is something to replay
        if not trajectories:
            raise RuntimeError(
                "Nothing to replay. Please run a simulation before calling "
                "`replay` method, or provided data manually.")

        # Make sure the viewer is instantiated before replaying
        backend = (kwargs.get('backend', self.viewer_backend) or
                   get_default_backend())
        must_not_open_gui = (
            backend.startswith("panda3d") or
            kwargs.get('record_video_path', None) is not None)
        self.render(**{
            'return_rgb_array': must_not_open_gui,
            'update_floor': True,
            **kwargs})

        # Define sequence of viewer instances
        viewers = [*self.viewers, *[
            None for _ in trajectories[:-len(self.viewers)]]]

        # Replay the trajectories
        self._viewers_all = play_trajectories(
            trajectories,
            update_hooks,
            viewers=viewers,
            legend=legend,
            **{'verbose': True,
               'backend': self.viewer_backend,
               **extra_kwargs,
               'display_f_external': None,
               **kwargs})

    def close(self) -> None:
        """Close the connection with the renderer.
        """
        Viewer.close()
        self.viewers = []
        self._viewers_all = []
        for figure in self.figures:
            figure.close()
        self.figures = []

    def get_options(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get the options of robot (including controller), and engine.
        """
        options = OrderedDict(
            system=OrderedDict(robot=OrderedDict(), controller=OrderedDict()),
            engine=OrderedDict())

        for system in self.engine.systems:
            if system.name == "":
                name = "system"
            else:
                name = f"{system.name}.system"
            robot_options = OrderedDict()
            robot_options_copy = system.robot.get_options()
            robot_options['model'] = robot_options_copy['model']
            robot_options['motors'] = robot_options_copy['motors']
            robot_options['sensors'] = robot_options_copy['sensors']
            robot_options['telemetry'] = robot_options_copy['telemetry']
            options[name]['robot'] = robot_options

            controller_options = OrderedDict(system.controller.get_options())
            options[name]['controller'] = controller_options
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
        for system in self.systems:
            if system.name == "":
                name = "system"
            else:
                name = f"{system.name}.system"
            controller_options = options[name]['controller']
            system.robot.set_options(options[name]['robot'])
            system.controller.set_options(controller_options)
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
            for system in self.systems:
                if isinstance(system.robot, BaseJiminyRobot):
                    urdf_path = system.robot._urdf_path_orig
                else:
                    urdf_path = system.robot.urdf_path
                if not urdf_path:
                    raise ValueError(
                        "'config_path' must be provided if the robot is not "
                        "associated with any URDF.")
                config_path = str(pathlib.Path(
                    urdf_path).with_suffix('')) + '_options.toml'
        with open(config_path, 'w') as f:
            toml.dump(self.get_options(), f, encoder=toml.TomlNumpyEncoder())

    def import_options(self,
                       config_path: Optional[Union[str, os.PathLike]] = None
                       ) -> None:
        """Import the full configuration, ie the options of the robot (
        including controller), and the engine.

        .. note::
            Configuration can be exported beforehand using `export_options`
            method.
        """
        def deep_update(source, overrides):
            """
            Update a nested dictionary or similar mapping.
            Modify ``source`` in place.
            """
            for key, value in overrides.items():
                if isinstance(value, dict) and value:
                    source[key] = deep_update(source[key], value)
                else:
                    source[key] = overrides[key]
            return source

        if config_path is None:
            for system in self.systems:
                if isinstance(system.robot, BaseJiminyRobot):
                    urdf_path = system.robot._urdf_path_orig
                else:
                    urdf_path = system.robot.urdf_path
                if not urdf_path:
                    raise ValueError(
                        "'config_path' must be provided if the robot is not "
                        "associated with any URDF.")
                config_path = str(pathlib.Path(
                    urdf_path).with_suffix('')) + '_options.toml'
                if not os.path.exists(config_path):
                    return
            options = deep_update(self.get_options(), toml.load(config_path))
            self.set_options(options)
