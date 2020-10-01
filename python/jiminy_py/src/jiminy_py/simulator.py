## @file src/jiminy_py/simulator.py
import os
import toml
import atexit
import pathlib
import tempfile
import numpy as np
from collections import OrderedDict
from typing import Optional, Type, Dict, List, Any

import pinocchio as pin

from . import core as jiminy
from .robot import generate_hardware_description_file, BaseJiminyRobot
from .controller import BaseJiminyController
from .viewer import Viewer


DEFAULT_GROUND_STIFFNESS = 4.0e6
DEFAULT_GROUND_DAMPING = 2.0e3


class Simulator:
    """
    @brief This class wraps the different submodules of Jiminy, namely the
           robot, controller, engine, and viewer, as a single simulation
           environment. The user only as to create a robot and associated
           controller if any, and give high-level instructions to the
           simulator.
    """
    def __init__(self,
                 robot: jiminy.Robot,
                 controller: Optional[jiminy.ControllerFunctor] = None,
                 engine_class: Type[jiminy.Engine] = jiminy.Engine,
                 use_theoretical_model: bool = False,
                 viewer_backend: Optional[str] = None):
        """
        @brief Constructor

        @param robot  Jiminy robot already initialized.
        @param controller  Jiminy controller already initialized.
                           Optional: jiminy_py.core.ControllerFunctor doing
                           nothing by default.
        @param engine_class  The class of engine to use.
                             Optional: jiminy_py.core.Engine by default.
        @param use_theoretical_model  Whether the state corresponds to the
                                      theoretical model when updating and
                                      fetching the robot's state.
        @param viewer_backend  Backend of the viewer, eg gepetto-gui or
                               meshcat.
                               Optional: It is setup-dependent. See Viewer
                               documentation for details about it.
        """
        # Backup the user arguments
        self.use_theoretical_model = use_theoretical_model
        self.viewer_backend = viewer_backend

        # Instantiate and initialize a controller doing nothing if necessary
        if controller is None:
            controller = BaseJiminyController()
            controller.initialize(robot)

        # Instantiate the low-level Jiminy engine, then initialize it
        self.engine = engine_class()
        self.engine.initialize(robot, controller, self._callback)

        # Viewer management
        self._viewer = None
        self._is_viewer_available = False

        # Reset the low-level jiminy engine
        self.engine.reset()

    @classmethod
    def build(cls,
              urdf_path: str,
              hardware_path: Optional[str] = None,
              mesh_path: Optional[str] = None,
              has_freeflyer: bool = True,
              config_path: Optional[str] = None,
              debug: bool = False,
              **kwargs) -> 'Simulator':
        """
        @brief Constructor

        @param urdf_path  Path of the urdf model to be used for the simulation.
        @param hardware_path  Path of Jiminy hardware description toml file.
                              Optional: Looking for '.hdf' file in the same
                              folder and with the same name.
        @param mesh_path  Path to the folder containing the model meshes.
                          Optional: Env variable 'JIMINY_DATA_PATH' will be
                          used if available.
        @param has_freeflyer  Whether the robot is fixed-based wrt its root
                              link, or can move freely in the world.
                              Optional: True by default.
        @param config_path  Configuration toml file to import. It will be
                            imported AFTER loading the hardware description
                            file. It can be automatically generated from an
                            instance by calling `export_config_file` method.
                            Optional: Looking for '.config' file in the same
                            folder and with the same name. If not found,
                            using default configuration.
        @param debug  Whether or not the debug mode must be activated.
                      Doing it enables temporary files automatic deletion.
        @param kwargs  Keyword arguments to forward to class constructor.
        """
        # Generate a temporary Hardware Description File if necessary
        if hardware_path is None:
            hardware_path = pathlib.Path(urdf_path).with_suffix('.hdf')
            if not os.path.exists(hardware_path):
                # Create a file that will be closed (thus deleted) at exit
                urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
                hardware_file = tempfile.NamedTemporaryFile(
                    prefix=(urdf_name + "_hardware_"), suffix=".hdf",
                    delete=(not debug))
                def close_file_at_exit(file=hardware_file):
                    file.close()
                atexit.register(close_file_at_exit)

                # Generate default Hardware Description File
                hardware_path = hardware_file.name
                generate_hardware_description_file(urdf_path, hardware_path)

        # Instantiate and initialize the robot
        robot = BaseJiminyRobot()
        robot.initialize(urdf_path, hardware_path, mesh_path, has_freeflyer)

        # Instantiate and initialize the engine
        simulator = cls(robot, engine_class=jiminy.Engine, **kwargs)

        # Set some engine options, based on extra toml information
        engine_options = simulator.engine.get_options()

        ## Handling of controller and sensors update period
        control_period = robot.extra_info.pop('controllerUpdatePeriod', None)
        sensors_period = robot.extra_info.pop('sensorsUpdatePeriod', None)
        if control_period is None and sensors_period is None:
            control_period, sensors_period = 0.0, 0.0
        elif control_period is None:
            control_period = sensors_period
        else:
            sensors_period = control_period
        engine_options['stepper']['controllerUpdatePeriod'] = control_period
        engine_options['stepper']['sensorsUpdatePeriod'] = sensors_period

        ## Handling of ground model parameters
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
        """
        @brief Fallback attribute getter.

        @details Implemented for convenience. It enables to get access to the
                 attribute and methods of the low-level Jiminy engine directly,
                 without having to do it through `engine`.
        """
        if name != 'engine' and hasattr(self, 'engine'):
            return getattr(self.engine, name)
        else:
            return AttributeError(
                f"'{self.__class__}' object has no attribute '{name}'.")

    def __dir__(self) -> List[str]:
        """
        @brief Attribute lookup.

        @details It is used for by autocomplete feature of Ipython. It is
                 overloaded to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self.engine.__dir__()

    @property
    def state(self) -> np.ndarray:
        """
        @brief Getter of the current state of the robot.

        @remark Beware that it returns a copy, which is computationally
                inefficient but intentional.
        """
        x = self.engine.stepper_state.x
        if self.robot.is_flexible and self.use_theoretical_model:
            return self.robot.get_rigid_state_from_flexible(x)
        else:
            return x.copy()

    @property
    def pinocchio_model(self) -> pin.Model:
        """
        @brief Getter of the pinocchio model, depending on the value of
               'use_theoretical_model'.
        """
        if self.robot.is_flexible and self.use_theoretical_model:
            return self.robot.pinocchio_model_th
        else:
            return self.robot.pinocchio_model

    @property
    def pinocchio_data(self) -> pin.Model:
        """
        @brief Getter of the pinocchio data, depending on the value of
               'use_theoretical_model'.
        """
        if self.robot.is_flexible and self.use_theoretical_model:
            return self.robot.pinocchio_data_th
        else:
            return self.robot.pinocchio_data

    def _callback(self,
                  t: float,
                  q: np.ndarray,
                  v: np.ndarray,
                  out: np.ndarray) -> None:
        """
        @brief Callback method for the simulation.
        """
        out[0] = True

    def seed(self, seed: int) -> None:
        """
        @brief Set the seed of the simulation and reset the simulation.

        @details Note that it also resets the low-level jiminy Engine. One must
                 call the `reset` method manually afterward.

        @param seed  Desired seed (Unsigned integer 32 bits).
        """
        assert isinstance(seed, np.uint32), "'seed' must have type np.uint32."

        engine_options = self.engine.get_options()
        engine_options["stepper"]["randomSeed"] = \
            np.array(seed, dtype=np.dtype('uint32'))
        self.engine.set_options(engine_options)
        self.engine.reset()

    def run(self,
            tf: float,
            x0: np.ndarray,
            is_state_theoretical: bool = True,
            log_path: Optional[str] = None,
            show_progress_bar: bool = True) -> None:
        """
        @brief Run a simulation, starting from x0 at t=0 up to tf.

        @remark Optionally, log the result of the simulation.

        @param x0  Initial state.
        @param tf  Simulation end time.
        @param is_state_theoretical  Whether or not the initial state is
                                     associated with the actual or theoretical
                                     model of the robot.
        @param log_path  Save log data to this location. Disable if None.
                         Note that the format extension '.data' is enforced.
                         Optional, disable by default.
        @param show_progress_bar  Whether or not to display a progress bar
                                  during the simulation.
                                  Optional: Enable by default.
        """
        # Run the simulation
        if show_progress_bar:
            try:
                self.engine.controller.set_progress_bar(tf)
            except AttributeError as e:
                raise RuntimeError("'show_progress_bar' can only be used with "
                    "controller inherited from `BaseJiminyController`.") from e
        self.engine.simulate(tf, x0, is_state_theoretical)
        self.engine.controller.close_progress_bar()

        # Write log
        if log_path is not None:
            log_path = str(pathlib.Path(log_path).with_suffix('.data'))
            self.engine.write_log(log_path, True)

    def render(self,
               return_rgb_array: bool = False,
               width: Optional[int] = None,
               height: Optional[int] = None) -> Optional[np.ndarray]:
        """
        @brief Render the current state of the simulation. One can display it
               or return an RGB array instead.

        @remark Note that gepetto-gui supports parallel rendering, which means
                that one can display multiple simulations at the same time in
                different tabs.

        @param return_rgb_array  Whether or not to return the current frame as
                                 an rgb array.
        @param width  Width of the returned RGB frame, if enabled.
        @param height  Height of the returned RGB frame, if enabled.

        @return Rendering as an RGB array (3D numpy array), if enabled, None
                otherwise.
        """
        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        if not (self._is_viewer_available and self._viewer.is_alive()):
            # Reset viewer backend if the existing viewer backend is no
            # longer available for some reason.
            if self._is_viewer_available:
                self._viewer.close()
                self._is_viewer_available = False

            # Generate a new unique identifier if necessary
            if self._viewer is None:
                uniq_id = next(tempfile._get_candidate_names())
                robot_name = "_".join(("robot", uniq_id))
                scene_name = "_".join(("scene", uniq_id))
                window_name = "_".join(("window", uniq_id))
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
            if self._viewer.is_backend_parent:
                self._viewer.set_camera_transform(
                    translation=[9.0, 0.0, 2e-5],
                    rotation=[np.pi/2, 0.0, np.pi/2])
            self._viewer.wait(False)  # Wait for backend to finish loading
            self._is_viewer_available = True

        # Try refreshing the viewer
        self._viewer.refresh()

        # Compute rgb array if needed
        if return_rgb_array:
            return self._viewer.capture_frame(width, height)

    def close(self) -> None:
        """
        @brief Close the connection with the renderer.
        """
        if hasattr(self, '_viewer') and self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def get_controller_options(self) -> dict:
        """
        @brief Getter of the options of Jiminy Controller.
        """
        return self.engine.controller.get_options()

    def set_controller_options(self, options: dict) -> None:
        """
        @brief Setter of the options of Jiminy Controller.
        """
        self.engine.controller.set_options(options)

    def get_options(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        @brief Get the options of robot (including controller), and engine.
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
        """
        @brief Set the options of robot (including controller), and engine.
        """
        controller_options = options['robot'].pop('controller')
        self.robot.set_options(options['robot'])
        self.set_controller_options(controller_options)
        self.engine.set_options(options['engine'])

    def export_options(self, config_path: Optional[str] = None) -> None:
        """
        @brief Export the full configuration, ie the options of the robot (
               including controller), and the engine.

        @remark Configuration can be imported using `import_options` method.
        """
        if config_path is None:
            config_path = pathlib.Path(
                self.robot.urdf_path_orig).with_suffix('.config')
        with open(config_path, 'w') as f:
            toml.dump(self.get_options(), f, encoder=toml.TomlNumpyEncoder())

    def import_options(self, config_path: Optional[str] = None) -> None:
        """
        @brief Import the full configuration, ie the options of the robot (
               including controller), and the engine.

        @remark Configuration can be exported using `export_options` method.
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
