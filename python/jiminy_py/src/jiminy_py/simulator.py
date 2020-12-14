import os
import toml
import atexit
import pathlib
import tempfile
import numpy as np
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Optional, Union, Type, Dict, Tuple, Sequence, List, Any

import pinocchio as pin

from . import core as jiminy
from .core import (EncoderSensor as encoder,
                   EffortSensor as effort,
                   ContactSensor as contact,
                   ForceSensor as force,
                   ImuSensor as imu)
from .robot import generate_hardware_description_file, BaseJiminyRobot
from .controller import BaseJiminyObserverController
from .viewer import Viewer, play_logfiles


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
                 controller: Optional[jiminy.ControllerFunctor] = None,
                 engine_class: Type[jiminy.Engine] = jiminy.Engine,
                 use_theoretical_model: bool = False,
                 viewer_backend: Optional[str] = None):
        """
        :param robot: Jiminy robot already initialized.
        :param controller: Jiminy (observer-)controller already initialized.
                           Optional: `BaseJiminyObserverController` doing
                           nothing by default.
        :param engine_class: Class of engine to use.
                             Optional: jiminy_py.core.Engine by default.
        :param use_theoretical_model: Whether the state corresponds to the
                                      theoretical model when updating and
                                      fetching the robot's state.
        :param viewer_backend: Backend of the viewer, eg gepetto-gui or
                               meshcat.
                               Optional: It is setup-dependent. See Viewer
                               documentation for details about it.
        """
        # Backup the user arguments
        self.use_theoretical_model = use_theoretical_model
        self.viewer_backend = viewer_backend

        # Instantiate and initialize a controller doing nothing if necessary
        if controller is None:
            controller = BaseJiminyObserverController()
            controller.initialize(robot)

        # Instantiate the low-level Jiminy engine, then initialize it
        self.engine = engine_class()
        hresult = self.engine.initialize(robot, controller, self._callback)
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
        self._viewer = None
        self._is_viewer_available = False

        # Plot data holder
        self.__plot_data: Optional[Dict[str, Any]] = None

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
                              Optional: Looking for '.hdf' file in the same
                              folder and with the same name.
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
                            Optional: Looking for '.config' file in the same
                            folder and with the same name. If not found,
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
            hardware_path = pathlib.Path(urdf_path).with_suffix('.hdf')
            if not os.path.exists(hardware_path):
                # Create a file that will be closed (thus deleted) at exit
                urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
                fd, hardware_path = tempfile.mkstemp(
                    prefix=(urdf_name + "_hardware_"), suffix=".hdf")
                os.close(fd)

                if not debug:
                    def remove_file_at_exit(file_path=hardware_path):
                        try:
                            os.remove(file_path)
                        except (PermissionError, FileNotFoundError):
                            pass

                    atexit.register(remove_file_at_exit)

                # Generate default Hardware Description File
                generate_hardware_description_file(
                    urdf_path, hardware_path, verbose=debug)

        # Instantiate and initialize the robot
        robot = BaseJiminyRobot()
        robot.initialize(urdf_path, hardware_path, mesh_path, has_freeflyer,
                         avoid_instable_collisions, verbose=debug)

        # Instantiate and initialize the engine
        simulator = cls(robot, engine_class=jiminy.Engine, **kwargs)

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
        simulator.import_options(config_path)

        return simulator

    def __del__(self) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Jiminy engine directly, without having to do it through `engine`.

        .. note::
            This method is not meant to be called manually.
        """
        return getattr(self.engine, name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self.engine.__dir__()

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
        else:
            return q, v

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
    def pinocchio_data(self) -> pin.Model:
        """Getter of the pinocchio data, depending on the value of
           'use_theoretical_model'.
        """
        if self.use_theoretical_model and self.robot.is_flexible:
            return self.robot.pinocchio_data_th
        else:
            return self.robot.pinocchio_data

    def _callback(self,
                  t: float,
                  q: np.ndarray,
                  v: np.ndarray,
                  out: np.ndarray) -> None:
        """Callback method for the simulation.
        """
        out[0] = True

    def seed(self, seed: int) -> None:
        """Set the seed of the simulation and reset the simulation.

        .. warning::
            It also resets the low-level jiminy Engine. Therefore one must call
            the `reset` method manually afterward.

        :param seed: Desired seed (Unsigned integer 32 bits).
        """
        assert isinstance(seed, np.uint32), "'seed' must have type np.uint32."

        engine_options = self.engine.get_options()
        engine_options["stepper"]["randomSeed"] = \
            np.array(seed, dtype=np.dtype('uint32'))
        self.engine.set_options(engine_options)
        self.reset()

    def reset(self, remove_forces: bool = False):
        """Reset the simulator.

        It resets the simulation time to zero, and generate a new random model
        of the robot. If one wants to get exactly the same result as before,
        either set the randomness of the model and sensors to zero, or set the
        seed once again to reinitialize the random number generator.

        :param remove_forces: Whether or not to remove already registered
                              external forces. It can also be done separately.
                              Optional: Do not remove by default.
        """
        # Reset the backend engine
        self.engine.reset(remove_forces)

        # Note that the viewer must only be reset if available, otherwise it
        # will have dangling reference to the old robot model.
        if self._is_viewer_available:
            try:
                self._viewer._setup(self.robot)
            except RuntimeError:
                self._viewer.close()
                self._is_viewer_available = False

    def start(self,
              q0: np.ndarray,
              v0: np.ndarray,
              is_state_theoretical: bool = False) -> None:
        """Initialize a simulation, starting from x0=(q0,v0) at t=0.

        :param q0: Initial configuration.
        :param v0: Initial velocity.
        :param is_state_theoretical: Whether or not the initial state is
                                     associated with the actual or theoretical
                                     model of the robot.
        """
        # Call base implementation
        hresult = self.engine.start(q0, v0, None, is_state_theoretical)
        if hresult != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("Failed to start the simulation.")

        # Update the observer at the end, if suitable
        if isinstance(self.controller, BaseJiminyObserverController):
            self.controller.refresh_observation(
                self.stepper_state.t,
                self.system_state.q,
                self.system_state.v,
                self.sensors_data)

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

        # Update the observer at the end, if suitable
        if isinstance(self.controller, BaseJiminyObserverController):
            self.controller.refresh_observation(
                self.stepper_state.t,
                self.system_state.q,
                self.system_state.v,
                self.sensors_data)

    def simulate(self,
                 t_end: float,
                 q_init: np.ndarray,
                 v_init: np.ndarray,
                 a_init: Optional[np.ndarray] = None,
                 is_state_theoretical: bool = False) -> None:
        return_code = self.engine.simulate(
            t_end, q_init, v_init, a_init, is_state_theoretical)
        if return_code != jiminy.hresult_t.SUCCESS:
            raise RuntimeError("The simulation failed.")

    def run(self,
            tf: float,
            q0: np.ndarray,
            v0: np.ndarray,
            is_state_theoretical: bool = True,
            log_path: Optional[str] = None,
            show_progress_bar: bool = True) -> None:
        """Run a simulation, starting from x0=(q0,v0) at t=0 up to tf.

        .. note::
            Optionally, log the result of the simulation.

        :param tf: Simulation end time.
        :param q0: Initial configuration.
        :param v0: Initial velocity.
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
        # Run the simulation
        if show_progress_bar is not False:
            try:
                self.engine.controller.set_progress_bar(tf)
            except AttributeError as e:
                if show_progress_bar:
                    raise RuntimeError(
                        "'show_progress_bar' can only be used with controller "
                        "inherited from `BaseJiminyObserverController`."
                        ) from e
                show_progress_bar = False
        self.simulate(tf, q0, v0, None, is_state_theoretical)
        if show_progress_bar is not False:
            self.engine.controller.close_progress_bar()

        # Write log
        if log_path is not None:
            log_path = str(pathlib.Path(log_path).with_suffix('.hdf5'))
            self.engine.write_log(log_path, format="hdf5")

    def render(self,
               return_rgb_array: bool = False,
               width: Optional[int] = None,
               height: Optional[int] = None,
               camera_xyzrpy: Optional[Tuple[
                   Union[Tuple[float, float, float], np.ndarray],
                   Union[Tuple[float, float, float], np.ndarray]]] = None
               ) -> Optional[np.ndarray]:
        """Render the current state of the simulation. One can display it
               or return an RGB array instead.

        .. note::
            Gepetto-gui supports parallel rendering, which means that one can
            display multiple simulations at the same time in different tabs.

        :param return_rgb_array: Whether or not to return the current frame as
                                 an rgb array.
        :param width: Width of the returned RGB frame, if enabled.
        :param height: Height of the returned RGB frame, if enabled.
        :param camera_xyzrpy: Tuple position [X, Y, Z], rotation [Roll, Pitch,
                              Yaw] corresponding to the absolute pose of the
                              camera. None to disable.
                              Optional:None by default.

        :returns: Rendering as an RGB array (3D numpy array), if enabled, None
                  otherwise.
        """
        # Make sure that a simulation is running
        if not self.is_simulation_running:
            raise RuntimeError(
                "No simulation running. Please call `start` before `render`.")

        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        if not (self._is_viewer_available and self._viewer.is_alive()):
            # Generate a new unique identifier if necessary
            if self._viewer is None:
                uniq_id = next(tempfile._get_candidate_names())
                robot_name = f"{uniq_id}_robot"
                scene_name = f"{uniq_id}_scene"
                window_name = f"{uniq_id}_window"
            else:
                robot_name = self._viewer.robot_name
                scene_name = self._viewer.scene_name
                window_name = self._viewer.window_name

            # Create a new viewer client
            self._viewer = Viewer(self.robot,
                                  use_theoretical_model=False,
                                  backend=self.viewer_backend,
                                  open_gui_if_parent=(not return_rgb_array),
                                  delete_robot_on_close=True,
                                  robot_name=robot_name,
                                  scene_name=scene_name,
                                  window_name=window_name)
            self.viewer_backend = Viewer.backend
            if self._viewer.is_backend_parent and camera_xyzrpy is None:
                camera_xyzrpy = [(9.0, 0.0, 2e-5), (np.pi/2, 0.0, np.pi/2)]
            self._viewer.wait(require_client=False)  # Wait to finish loading
            self._is_viewer_available = True

        # Set the camera pose if requested
        if camera_xyzrpy is not None:
            self._viewer.set_camera_transform(*camera_xyzrpy)

        # Try refreshing the viewer
        self._viewer.refresh()

        # Compute rgb array if needed
        if return_rgb_array:
            return self._viewer.capture_frame(width, height)

    def replay(self, **kwargs: Any) -> None:
        """Replay the current episode until now.

        :param kwargs: Extra keyword arguments for delegation to
                       `viewer.play_trajectories` method.
        """
        log_data, _ = self.get_log()
        if not log_data:
            raise RuntimeError(
                "Nothing to replay. Please run a simulation before calling "
                "`replay` method.")
        self._viewer = play_logfiles(
            [self.robot], [log_data], viewers=[self._viewer],
            **{'verbose': True, 'backend': self.viewer_backend, **kwargs})[0]
        self._is_viewer_available = True

    def close(self) -> None:
        """Close the connection with the renderer.
        """
        if hasattr(self, 'engine'):
            if self._viewer is not None:
                self._viewer.close()
                self._is_viewer_available = False
                self._viewer = None
            if self.__plot_data is not None:
                plt.close(self.__plot_data['fig'])

    def plot(self) -> None:
        """Display common simulation data over time.

        The figure features several tabs:

          - Subplots with robot configuration
          - Subplots with robot velocity
          - Subplots with robot acceleration
          - Subplots with motors torques
          - Subplots with raw sensor data (one tab for each type of sensor)
        """
        # Define some internal helper functions
        def extract_fields(log_data: Dict[str, np.ndarray],
                           namespace: Optional[str],
                           fieldnames: Sequence[str],
                           ) -> Optional[Sequence[np.ndarray]]:
            """Extract value associated with a set of fieldnames in a specific
            namespace.

            :param log_data: Log file, as returned by `get_log` method.
            :param namespace: Namespace of the fieldnames. None to disable.
            :param fieldnames: Sequence of fieldnames.
            """
            field_values = [log_data.get(
                    '.'.join((filter(None, (namespace, field)))), None)
                for field in fieldnames]
            if not field_values or all(v is None for v in field_values):
                return None
            else:
                return field_values

        # Extract log data
        log_data, _ = self.get_log()
        t = log_data["Global.Time"]

        # Figures data structure as a dictionary
        data = OrderedDict()

        # Get robot positions, velocities, and acceleration
        for fields_type in ["Position", "Velocity", "Acceleration"]:
            fieldnames = getattr(
                self.robot, "logfile_" + fields_type.lower() + "_headers")
            values = extract_fields(
                log_data, 'HighLevelController', fieldnames)
            if values is not None:
                data[' '.join(("State", fields_type))] = OrderedDict(
                    (field[len("current"):].replace(fields_type, ""), val)
                    for field, val in zip(fieldnames, values))

        # Get command information
        u = extract_fields(
            log_data, 'HighLevelController',
            self.robot.logfile_command_headers)
        if u is not None:
            data['Command'] = OrderedDict(
                (field, val) for field, val in zip(self.robot.motors_names, u))

        # Get sensors information
        for sensors_class, sensors_fields in SENSORS_FIELDS.items():
            sensors_type = sensors_class.type
            sensors_names = self.robot.sensors_names.get(sensors_type, [])
            namespace = sensors_type if sensors_class.has_prefix else None
            if isinstance(sensors_fields, dict):
                for fields_prefix, fieldnames in sensors_fields.items():
                    sensors_data = [extract_fields(log_data, namespace, [
                        '.'.join((name, fields_prefix + field))
                        for name in sensors_names]) for field in fieldnames]
                    if sensors_data[0] is not None:
                        type_name = ' '.join((sensors_type, fields_prefix))
                        data[type_name] = OrderedDict(
                            (field, OrderedDict(
                                (name, val) for name, val in zip(
                                    sensors_names, values)))
                            for field, values in zip(fieldnames, sensors_data))
            else:
                for field in sensors_fields:
                    sensors_data = extract_fields(
                        log_data, namespace,
                        ['.'.join((name, field)) for name in sensors_names])
                    if sensors_data is not None:
                        data[' '.join((sensors_type, field))] = OrderedDict(
                            (name, val)
                            for name, val in zip(sensors_names, sensors_data))

        # Plot the data
        fig = plt.figure()
        fig_axes = {}
        tab_active = None
        for fig_name, fig_data in data.items():
            # Compute plot grid arrangement
            n_cols = len(fig_data)
            n_rows = 1
            while n_cols > n_rows + 2:
                n_rows = n_rows + 1
                n_cols = int(np.ceil(len(fig_data) / (1.0 * n_rows)))

            # Initialize axes, and early return if none
            axes = []
            ref_ax = None
            for i, plot_name in enumerate(fig_data.keys()):
                uniq_label = '_'.join((fig_name, plot_name))
                ax = fig.add_subplot(n_rows, n_cols, i+1, label=uniq_label)
                if fig_axes:
                    fig.delaxes(ax)
                if ref_ax is not None:
                    ax.get_shared_x_axes().join(ref_ax, ax)
                else:
                    ref_ax = ax
                axes.append(ax)
            if axes is None:
                continue

            # Update their content
            for (plot_name, plot_data), ax in zip(fig_data.items(), axes):
                if isinstance(plot_data, dict):
                    for line_name, line_data in plot_data.items():
                        ax.plot(t, line_data, label=line_name)
                    ax.legend()
                else:
                    ax.plot(t, plot_data)
                ax.set_title(plot_name, fontsize='medium')
                ax.grid()

            # Add them to tab register
            fig_axes[fig_name] = axes
            if tab_active is None:
                tab_active = fig_name
        fig_nav_stack = {key: [] for key in fig_axes.keys()}
        fig_nav_pos = {key: -1 for key in fig_axes.keys()}

        # Add buttons to show/hide information
        button_axcut = {}
        buttons = {}
        buttons_width = 1.0 / (len(data) + 1)
        for i, fig_name in enumerate(data.keys()):
            button_axcut[fig_name] = plt.axes(
                [buttons_width * (i + 0.5), 0.01, buttons_width, 0.05])
            buttons[fig_name] = Button(
                button_axcut[fig_name], fig_name, color='white')

        def click(event: matplotlib.backend_bases.Event) -> None:
            nonlocal fig, fig_axes, fig_nav_stack, fig_nav_pos, tab_active

            # Update buttons style
            for b in buttons.values():
                if b.ax == event.inaxes:
                    b.ax.set_facecolor('green')
                    b.color = 'green'
                    button_name = b.label.get_text()
                else:
                    b.ax.set_facecolor('white')
                    b.color = 'white'

            # Backup navigation history
            cur_stack = fig.canvas.toolbar._nav_stack
            fig_nav_stack[tab_active] = cur_stack._elements.copy()
            fig_nav_pos[tab_active] = cur_stack._pos

            # Update axes and title
            for ax in fig_axes[tab_active]:
                fig.delaxes(ax)
            for ax in fig_axes[button_name]:
                fig.add_subplot(ax)
            fig.suptitle(button_name)
            tab_active = button_name

            # Restore selected tab navigation history and toolbar state
            cur_stack._elements = fig_nav_stack[button_name]
            cur_stack._pos = fig_nav_pos[button_name]
            fig.canvas.toolbar._actions['forward'].setEnabled(
                fig_nav_pos[button_name] < len(cur_stack) - 1)
            fig.canvas.toolbar._actions['back'].setEnabled(
                fig_nav_pos[button_name] > 0)

            # Refresh figure
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Register buttons events
        for b in buttons.values():
            b.on_clicked(click)

        # Adjust layout and show figure (without blocking)
        fig.subplots_adjust(
            bottom=0.1, top=0.92, left=0.05, right=0.95, wspace=0.15,
            hspace=0.35)
        fig_name = list(fig_axes.keys())[0]
        for ax in fig_axes[fig_name]:
            ax.set_visible(True)
        fig.suptitle(fig_name)
        buttons[fig_name].ax.set_facecolor('green')
        buttons[fig_name].color = 'green'

        # Create plot data holder to avoid garbage collection of widgets,
        # therefore making the buttons unresponsive...
        self.__plot_data = {
            'figure': fig,
            'fig_axes': fig_axes,
            'fig_nav_stack': fig_nav_stack,
            'fig_nav_pos': fig_nav_pos,
            'tab_active': tab_active,
            'buttons': buttons,
        }

        fig.canvas.draw()
        plt.show(block=False)

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
        options = OrderedDict(robot=OrderedDict(), engine=OrderedDict())
        robot_options = options['robot']
        robot_options['model'] = self.robot.get_model_options()
        robot_options['motors'] = self.robot.get_motors_options()
        robot_options['sensors'] = self.robot.get_sensors_options()
        robot_options['telemetry'] = self.robot.get_telemetry_options()
        robot_options['controller'] = self.get_controller_options()
        engine_options = options['engine']
        engine_options_copy = self.engine.get_options()
        engine_options['stepper'] = engine_options_copy['stepper']
        engine_options['world'] = engine_options_copy['world']
        engine_options['joints'] = engine_options_copy['joints']
        engine_options['contacts'] = engine_options_copy['contacts']
        engine_options['telemetry'] = engine_options_copy['telemetry']
        return options

    def set_options(self,
                    options: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Set the options of robot (including controller), and engine.
        """
        controller_options = options['robot'].pop('controller')
        self.robot.set_options(options['robot'])
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
            config_path = pathlib.Path(
                self.robot.urdf_path_orig).with_suffix('.config')
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
        if config_path is None:
            config_path = pathlib.Path(
                self.robot.urdf_path_orig).with_suffix('.config')
            if not os.path.exists(config_path):
                return
        options = toml.load(config_path)
        # TODO: Ground profile import/export is not supported for now
        options['engine']['world']['groundProfile'] = \
            jiminy.HeatMapFunctor(0.0, jiminy.heatMapType_t.CONSTANT)
        self.set_options(options)
