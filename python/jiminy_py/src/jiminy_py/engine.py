## @file src/jiminy_py/engine.py
import os
import pathlib
import tempfile
import numpy as np
from typing import Optional, List, Any

from . import core as jiminy
from .robot import generate_hardware_description_file, BaseJiminyRobot
from .controller import BaseJiminyController
from .viewer import Viewer
from .dynamics import update_quantities


DEFAULT_GROUND_STIFFNESS = 4.0e6
DEFAULT_GROUND_DAMPING = 2.0e3

class EngineAsynchronous:
    """
    @brief Wrapper of Jiminy enabling to update of the command and run
           simulation steps asynchronously. Convenient helper methods are
           available to set the seed of the simulation, reset it, and display
           it.

    @details The method `action` is used to update the command, which is kept
             in memory until `action` is called again. On its side, the method
             `step` without argument is used to run simulation steps. The
             method `step` has an optional argument `dt_desired` to specify the
             number of simulation steps to perform at once.

    @remark This class can be used for synchronous purpose. In such a case, one
            has to call the method `step` specifying the optional argument
            `action_next`.
    """
    def __init__(self,
                 robot: jiminy.Robot,
                 controller_class: jiminy.ControllerFunctor = \
                     jiminy.ControllerFunctor,
                 engine_class: jiminy.Engine = jiminy.Engine,
                 use_theoretical_model: bool = False,
                 viewer_backend: Optional[str] = None):
        """
        @brief Constructor

        @param robot  Jiminy robot instance already initialized.
        @param controller_class  The type of controller to use.
                                 Optional: core.ControllerFunctor without
                                 internal dynamics by default.
        @param engine_class  The class of engine to use.
                             Optional: core.Engine by default.
        @param use_theoretical_model  Whether the state corresponds to the
                                      theoretical model when updating and
                                      fetching the robot's state.
        @param viewer_backend  Backend of the viewer, ie gepetto-gui or
                               meshcat.

        @return Instance of the wrapper.
        """
        # Backup the user arguments
        self.robot = robot
        self.use_theoretical_model = use_theoretical_model
        self.viewer_backend = viewer_backend

        # Initialize some internal buffers
        self._t = -1
        self._state = None
        self._sensors_data = None
        self._action = np.zeros((robot.nmotors,))

        # Instantiate the Jiminy controller if necessary, then initialize it
        self._controller = controller_class(
            compute_command=self._send_command)
        self._controller.initialize(robot)

        # Instantiate the low-level Jiminy engine, then initialize it
        self._engine = engine_class()
        self._engine.initialize(robot, self._controller)

        # Viewer management
        self._viewer = None
        self._is_viewer_available = False

        # Real time rendering management
        self.step_dt_prev = -1
        self.is_ready_to_start = False

        # Reset the low-level jiminy engine
        self._engine.reset()

    def __del__(self) -> None:
        self.close()

    def _send_command(self,
                      t: float,
                      q: np.ndarray,
                      v: np.ndarray,
                      sensors_data: jiminy.sensorsData,
                      u_command: np.ndarray) -> None:
        """
        @brief This method implement the callback function required by
               Jiminy Controller to get the command. In practice, it only
               updates a variable shared between C++ and Python to the
               internal value stored by this class.

        @remark This is a hidden function that is not listed as part of the
                member methods of the class. It is not intended to be called
                manually.
        """
        self._sensors_data = sensors_data  # It is already a snapshot copy of robot.sensors_data
        u_command[:] = self._action

    def __getattr__(self, name: str) -> Any:
        """
        @brief Fallback attribute getter.

        @details Implemented for convenience. It enables to get access to the
                 attribute and methods of the low-level Jiminy engine directly,
                 without having to do it through `_engine`.
        """
        if name != '_engine' and hasattr(self, '_engine'):
            return getattr(self._engine, name)
        else:
            return AttributeError(
                f"'{self.__class__}' object has no attribute '{name}'.")

    def __dir__(self) -> List[str]:
        """
        @brief Attribute lookup.

        @details It is used for by autocomplete feature of Ipython. It is
                 overloaded to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self._engine.__dir__()

    def seed(self, seed: int) -> None:
        """
        @brief Set the seed of the simulation and reset the simulation.

        @details Note that it also resets the low-level jiminy Engine. One must
                 call the `reset` method manually afterward.

        @param seed  Desired seed (Unsigned integer 32 bits).
        """
        assert isinstance(seed, np.uint32), "'seed' must have type np.uint32."

        engine_options = self._engine.get_options()
        engine_options["stepper"]["randomSeed"] = \
            np.array(seed, dtype=np.dtype('uint32'))
        self._engine.set_options(engine_options)
        self._engine.reset()

    def reset(self,
              x0: np.ndarray,
              is_state_theoretical: Optional[bool] = None) -> None:
        """
        @brief Reset the simulation.

        @details It does NOT start the simulation immediately but rather wait
                 for the first 'step' call. At this point, the sensors data are
                 zeroed, until the simulation actually starts.

        @remark Note that once the simulations starts, it is no longer possible
                to changed the robot (included options).

        @param x0  Desired initial state.
        @param is_state_theoretical  Wether the provided initial state is
                                     associated with the theoretical or actual
                                     model.
        """
        # Handling of default argument(s)
        if is_state_theoretical is None:
            is_state_theoretical = self.use_theoretical_model

        # Reset the simulation
        self._engine.reset()

        # Call update_quantities in order to the frame placement for rendering
        if is_state_theoretical:
            x0_rigid = x0
            if self.robot.is_flexible:
                x0 = self.robot.get_flexible_state_from_rigid(x0_rigid)
        else:
            if self.robot.is_flexible and self.use_theoretical_model:
                x0_rigid = self.robot.get_rigid_state_from_flexible(x0)

        # Start the engine, in order to initialize the sensors data
        self._state = x0_rigid if self.use_theoretical_model else x0
        self._engine.start(self._state, self.use_theoretical_model)

        # Backup the sensor data by doing a deep copy manually
        self._sensors_data = jiminy.sensorsData({
            _type: {
                name: self.robot.sensors_data[_type, name].copy()
                for name in self.robot.sensors_data.keys(_type)
            }
            for _type in self.robot.sensors_data.keys()
        })

        # Initialize some internal buffers
        self._t = 0.0
        self.step_dt_prev = -1
        self.is_ready_to_start = True

        # Stop the engine, to avoid locking the robot and the telemetry
        # too early, so that it remains possible to register external
        # forces, register log variables, change the options...etc.
        self._engine.reset()

        # Restore the initial internal pinocchio data
        update_quantities(self.robot,
                          x0[:self.robot.nq],
                          update_physics=True,
                          use_theoretical_model=False)

    def step(self,
             action_next: Optional[np.ndarray] = None,
             dt_desired: float = -1) -> None:
        """
        @brief Run simulation steps.

        @details Even if Jiminy Engine performs several simulation steps
                 internally, this method only output the final state.

        @param action_next  Updated command.
                            Optional: Use the value in the internal buffer by
                            default, namely the previous action.
        @param dt_desired  Time duration of the integration step.
                           Optional: It depends on the configuration of the
                           low-level engine.

        @return Final state of the simulation
        """
        if not self._engine.is_simulation_running:
            if not self.is_ready_to_start:
                raise RuntimeError("Simulation not initialized. "
                    "Please call 'reset' once before calling 'step'.")
            hresult = self._engine.start(self._state, self.use_theoretical_model)
            if (hresult != jiminy.hresult_t.SUCCESS):
                raise RuntimeError("Failed to start the simulation.")
            self.is_ready_to_start = False

        if (action_next is not None):
            self.action = action_next

        return_code = self._engine.step(dt_desired)
        if (return_code != jiminy.hresult_t.SUCCESS):
            raise RuntimeError("Failed to perform the simulation step.")

        if not self._engine.is_simulation_running:
            self.is_ready = False

        self._t = self._engine.stepper_state.t
        self._state = None  # Do not fetch the new current state if not requested to the sake of efficiency
        self.step_dt_prev = self._engine.stepper_state.dt

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

        @return Rendering as an RGB array (3D numpy array), if enabled,
                None otherwise.
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

    @property
    def t(self) -> float:
        """
        @brief Getter of the current time of the simulation.
        """
        return self._t

    @property
    def state(self) -> np.ndarray:
        """
        @brief Getter of the current state of the robot.
        """
        if (self._state is None):
            x = self._engine.stepper_state.x
            if self.robot.is_flexible and self.use_theoretical_model:
                self._state = self.robot.get_rigid_state_from_flexible(x)
            else:
                self._state = x.copy()
        return self._state

    @property
    def sensors_data(self) -> jiminy.sensorsData:
        """
        @brief Getter of the current sensor data of the robot.
        """
        return self._sensors_data

    @property
    def action(self) -> np.ndarray:
        """
        @brief Getter of the current command.
        """
        return self._action

    @action.setter
    def action(self, action_next: np.ndarray) -> None:
        """
        @brief Setter of the command.
        """
        if (not isinstance(action_next, np.ndarray) or \
                action_next.shape[-1] != self.robot.nmotors):
            raise ValueError("The action must be a 1D numpy array \
                              whose length matches the number of motors.")
        if np.any(np.isnan(action_next)):
            raise ValueError("'action_next' cannot contain nan values.")
        self._action[:] = action_next

    def get_controller_options(self) -> dict:
        """
        @brief Getter of the options of Jiminy Controller.
        """
        return self._controller.get_options()

    def set_controller_options(self, options: dict) -> None:
        """
        @brief Setter of the options of Jiminy Controller.
        """
        self._controller.set_options(options)


class BaseJiminyEngine(EngineAsynchronous):
    def __init__(self,
                 urdf_path: str,
                 toml_path: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 has_freeflyer: bool = True,
                 use_theoretical_model: bool = False,
                 viewer_backend: Optional[str] = None,
                 debug: bool = False):
        """
        @brief    TODO
        """
        # Generate a temporary Hardware Description File if necessary
        if toml_path is None:
            toml_path = pathlib.Path(urdf_path).with_suffix('.toml')
            if not os.path.exists(toml_path):
                self._toml_file = tempfile.NamedTemporaryFile(
                    prefix="anymal_hdf_", suffix=".toml", delete=(not debug))
                toml_path = self._toml_file.name
                generate_hardware_description_file(urdf_path, toml_path)

        # Instantiate and initialize the robot
        robot = BaseJiminyRobot()
        robot.initialize(urdf_path, toml_path, mesh_path, has_freeflyer)

        # Instantiate and initialize the engine
        super().__init__(robot,
                         BaseJiminyController,
                         jiminy.Engine,
                         use_theoretical_model,
                         viewer_backend)

        # Set some engine options, based on extra toml information
        engine_options = self.get_options()

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

        self.set_options(engine_options)

    def __del__(self):
        self._toml_file.close()
