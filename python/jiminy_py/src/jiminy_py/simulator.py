import os
import json
import toml
import atexit
import pathlib
import tempfile
from collections import OrderedDict
from typing import Optional, Union, Type, Dict, Tuple, Sequence, List, Any

import numpy as np
from matplotlib.figure import Figure

import pinocchio as pin
from . import core as jiminy
from .core import (EncoderSensor as encoder,
                   EffortSensor as effort,
                   ContactSensor as contact,
                   ForceSensor as force,
                   ImuSensor as imu)
from .robot import (generate_default_hardware_description_file,
                    BaseJiminyRobot)
from .plot import TabbedFigure
from .log import read_log, build_robot_from_log, extract_data_from_log
from .viewer import (TrajectoryDataType,
                     interactive_mode,
                     extract_replay_data_from_log_data,
                     play_trajectories,
                     Viewer)

if interactive_mode():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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


class Simulator:
    """This class wraps the different submodules of Jiminy, namely the robot,
    controller, engine, and viewer, as a single simulation environment. The
    user only as to create a robot and associated controller if any, and
    give high-level instructions to the simulator.
    """
    def __init__(self,
                 robot: jiminy.Robot,
                 controller: Optional[jiminy.AbstractController] = None,
                 engine_class: Type[jiminy.Engine] = jiminy.Engine,
                 use_theoretical_model: bool = False,
                 viewer_backend: Optional[str] = None,
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
        :param viewer_backend: Backend of the viewer, e.g. panda3d or meshcat.
                               Optional: It is setup-dependent. See `Viewer`
                               documentation for details about it.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup the user arguments
        self.use_theoretical_model = use_theoretical_model
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
        hresult = self.engine.initialize(robot, controller, callback_wrapper)
        if hresult != jiminy.hresult_t.SUCCESS:
            raise RuntimeError(
                "Invalid robot or controller. Make sure they are both "
                "initialized.")

        # Create shared memories and python-native attribute for fast access
        self.is_simulation_running = self.engine.is_simulation_running
        self.stepper_state = self.engine.stepper_state
        self.system_state = self.engine.system_state
        self.sensors_data = self.robot.sensors_data

        # Viewer management
        self.viewer = None
        self._viewers = []

        # Internal buffer for progress bar management
        self.__pbar: Optional[tqdm] = None

        # Figure holder
        self.figure: Optional[Figure] = None

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
        """Create a new simulator instance from scratch, based on configuration
        files only.

        :param urdf_path: Path of the urdf model to be used for the simulation.
        :param hardware_path: Path of Jiminy hardware description toml file.
                              Optional: Looking for '*_hardware.toml' file in
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
                            Optional: Looking for '*_options.toml' file in the
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
        robot.initialize(urdf_path, hardware_path, mesh_path, has_freeflyer,
                         avoid_instable_collisions, verbose=debug)

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
    def pinocchio_model(self) -> pin.Model:
        """Getter of the pinocchio model, depending on the value of
           'use_theoretical_model'.
        """
        if self.use_theoretical_model and self.robot.is_flexible:
            return self.robot.pinocchio_model_th
        else:
            return self.robot.pinocchio_model

    @property
    def pinocchio_data(self) -> pin.Data:
        """Getter of the pinocchio data, depending on the value of
           'use_theoretical_model'.
        """
        if self.use_theoretical_model and self.robot.is_flexible:
            return self.robot.pinocchio_data_th
        else:
            return self.robot.pinocchio_data

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Getter of the current state of the robot.

        .. warning::
            Return a reference whenever it is possible, which is
            computationally efficient but unsafe.
        """
        q = self.system_state.q
        v = self.system_state.v
        if self.use_theoretical_model and self.robot.is_flexible:
            q = self.robot.get_rigid_configuration_from_flexible(q)
            v = self.robot.get_rigid_velocity_from_flexible(v)
        return q, v

    @property
    def log_data(self) -> Dict[str, np.ndarray]:
        """Getter of the telemetry variables.
        """
        return self.engine.get_log()[0]

    @property
    def log_constants(self) -> Dict[str, str]:
        """Getter of the telemetry constants.
        """
        return self.engine.get_log()[1]

    @property
    def is_viewer_available(self) -> bool:
        """Returns whether or not a viewer instance associated with the robot
        is available.
        """
        return self.viewer is not None and self.viewer.is_open()

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
        :param is_state_theoretical: Whether or not the initial state is
                                     associated with the actual or theoretical
                                     model of the robot.
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
        # Show progress bar if requested
        if show_progress_bar:
            self.__pbar = tqdm(total=t_end, bar_format=(
                "{percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} "
                "[{elapsed}<{remaining}]"))

        # Run the simulation
        try:
            return_code = self.engine.simulate(
                t_end, q_init, v_init, a_init, is_state_theoretical)
        finally:  # Make sure that the progress bar is properly closed
            if show_progress_bar:
                self.__pbar.close()
                self.__pbar = None

        # Throw exception if not successful
        if return_code != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("The simulation failed.")

        # Write log
        if log_path is not None:
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
        # Handle default arguments
        if update_ground_profile is None:
            update_ground_profile = not self.is_viewer_available

        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        if not self.is_viewer_available:
            # Generate a new unique identifier if necessary
            if self.viewer is None:
                uniq_id = next(tempfile._get_candidate_names())
                robot_name = f"{uniq_id}_robot"
                scene_name = f"{uniq_id}_scene"
            else:
                robot_name = self.viewer.robot_name
                scene_name = self.viewer.scene_name

            # Create new viewer instance
            viewer_backend = self.viewer_backend or Viewer.backend
            self.viewer = Viewer(self.robot,
                                 use_theoretical_model=False,
                                 open_gui_if_parent=False,
                                 **{'scene_name': scene_name,
                                    'robot_name': robot_name,
                                    'backend': viewer_backend,
                                    'delete_robot_on_close': True,
                                    **kwargs})

            # Backup current backend
            self.viewer_backend = self.viewer_backend or Viewer.backend

            # Share the external force buffer of the viewer with the engine
            if self.is_simulation_running:
                self.viewer.f_external = self.system_state.f_external[1:]

            if self.viewer_backend.startswith('panda3d'):
                # Enable display of COM, DCM and contact markers by default if
                # the robot has freeflyer.
                if self.robot.has_freeflyer:
                    if "display_com" not in kwargs:
                        self.viewer.display_center_of_mass(True)
                    if "display_dcm" not in kwargs:
                        self.viewer.display_capture_point(True)
                    if "display_contacts" not in kwargs:
                        self.viewer.display_contact_forces(True)

                # Enable display of external forces by default only for
                # the joints having an external force registered to it.
                if "display_f_external" not in kwargs:
                    force_frames = set([
                        self.robot.pinocchio_model.frames[f_i.frame_idx].parent
                        for f_i in self.engine.forces_profile])
                    force_frames |= set([
                        self.robot.pinocchio_model.frames[f_i.frame_idx].parent
                        for f_i in self.engine.forces_impulse])
                    visibility = self.viewer._display_f_external
                    for i in force_frames:
                        visibility[i - 1] = True
                    self.viewer.display_external_forces(visibility)

            # Initialize camera pose
            if self.viewer.is_backend_parent and camera_xyzrpy is None:
                camera_xyzrpy = [(9.0, 0.0, 2e-5), (np.pi/2, 0.0, np.pi/2)]

        # Enable the ground profile is requested and available
        if self.viewer_backend.startswith('panda3d') and update_ground_profile:
            engine_options = self.engine.get_options()
            ground_profile = engine_options["world"]["groundProfile"]
            self.viewer.update_floor(ground_profile, show_meshes=False)

        # Set the camera pose if requested
        if camera_xyzrpy is not None:
            self.viewer.set_camera_transform(*camera_xyzrpy)

        # Make sure the graphical window is open if required
        if not return_rgb_array:
            self.viewer.open_gui()

        # Try refreshing the viewer
        self.viewer.refresh()

        # Compute rgb array if needed
        if return_rgb_array:
            return self.viewer.capture_frame(width, height)

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
            log_data, log_constants = read_log(log_file)
            robot = build_robot_from_log(
                log_constants, self.robot.mesh_package_dirs)
            robots.append(robot)
            logs_data.append(log_data)

        # Extract trajectory data from pairs (robot, log)
        trajectories, update_hooks, extra_kwargs = [], [], {}
        for robot, log_data in zip(robots, logs_data):
            if log_data:
                traj, update_hook, _kwargs = \
                    extract_replay_data_from_log_data(robot, log_data)
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

        # Make sure the viewer is instantiated
        self.render(**{
            'return_rgb_array': kwargs.get(
                'record_video_path', None) is not None,
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
               'backend': self.viewer_backend,
               **extra_kwargs,
               'display_f_external': None,
               **kwargs})

    def close(self) -> None:
        """Close the connection with the renderer.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.figure is not None:
            self.figure.close()
            self.figure = None

    def plot(self,
             enable_flexiblity_data: bool = False,
             **kwargs: Any) -> None:
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
        :param kwargs: Extra keyword arguments to forward to `TabbedFigure`.
        """
        # Extract log data
        log_data, log_constants = self.log_data, self.log_constants
        if not log_constants:
            return RuntimeError(
                "No data to replay. Please run a simulation first.")

        # Make sure that the log file is compatible with the current model
        is_incompatible = False
        options_old = json.loads(log_constants['HighLevelController.options'])
        model_options_old = options_old["system"]["robot"]["model"]
        dyn_options_old = model_options_old['dynamics']
        model_options = self.robot.get_model_options()
        dyn_options = model_options['dynamics']
        if (dyn_options['enableFlexibleModel'] !=
                dyn_options_old['enableFlexibleModel']):
            is_incompatible = True
        elif dyn_options['enableFlexibleModel']:
            flex_options_old = dyn_options_old["flexibilityConfig"]["value"]
            flex_options = dyn_options["flexibilityConfig"]
            if len(flex_options_old) != len(flex_options):
                is_incompatible = True
            else:
                for flex, flex_old in zip(flex_options, flex_options_old):
                    if flex["frameName"] != flex_old["frameName"]:
                        is_incompatible = True
                        break
        if is_incompatible and enable_flexiblity_data:
            raise RuntimeError("Log data are incompatible with current model.")

        # Figures data structure as a dictionary
        tabs_data = OrderedDict()

        # Get time and robot positions, velocities, and acceleration
        time = log_data["Global.Time"]
        for fields_type in ["Position", "Velocity", "Acceleration"]:
            fieldnames = getattr(
                self.robot, "logfile_" + fields_type.lower() + "_headers")
            if not enable_flexiblity_data:
                # Filter out flexibility data
                fieldnames = list(filter(
                    lambda field: not any(
                        name in field
                        for name in self.robot.flexible_joints_names),
                    fieldnames))
            values = extract_data_from_log(
                log_data, fieldnames, as_dict=True)
            if values is not None:
                tabs_data[' '.join(("State", fields_type))] = OrderedDict(
                    (field[len("current"):].replace(fields_type, ""), elem)
                    for field, elem in values.items())

        # Get motors efforts information
        motor_effort = extract_data_from_log(
            log_data, self.robot.logfile_motor_effort_headers)
        if motor_effort is not None:
            tabs_data['MotorEffort'] = OrderedDict(
                zip(self.robot.motors_names, motor_effort))

        # Get command information
        command = extract_data_from_log(
            log_data, self.robot.logfile_command_headers)
        if command is not None:
            tabs_data['Command'] = OrderedDict(
                zip(self.robot.motors_names, command))

        # Get sensors information
        for sensors_class, sensors_fields in SENSORS_FIELDS.items():
            sensors_type = sensors_class.type
            sensors_names = self.robot.sensors_names.get(sensors_type, [])
            namespace = sensors_type if sensors_class.has_prefix else None
            if isinstance(sensors_fields, dict):
                for fields_prefix, fieldnames in sensors_fields.items():
                    sensors_data = [
                        extract_data_from_log(log_data, [
                            '.'.join((name, fields_prefix + field))
                            for name in sensors_names], namespace)
                        for field in fieldnames]
                    if sensors_data[0] is not None:
                        type_name = ' '.join((sensors_type, fields_prefix))
                        tabs_data[type_name] = OrderedDict(
                            (field, OrderedDict(zip(sensors_names, values)))
                            for field, values in zip(fieldnames, sensors_data))
            else:
                for field in sensors_fields:
                    sensors_data = extract_data_from_log(
                        log_data,
                        ['.'.join((name, field)) for name in sensors_names],
                        namespace)
                    if sensors_data is not None:
                        tabs_data[' '.join((sensors_type, field))] = \
                            OrderedDict(zip(sensors_names, sensors_data))

        # Create figure, without closing the existing one
        self.figure = TabbedFigure.plot(
            time, tabs_data, **{"plot_method": "plot", **kwargs})

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
        options = OrderedDict(
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
        options = deep_update(self.get_options(), toml.load(config_path))
        self.set_options(options)
