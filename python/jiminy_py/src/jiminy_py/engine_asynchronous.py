## @file src/jiminy_py/engine_asynchronous.py

"""
@package    jiminy_py

@brief      Package containing python-native helper methods for Jiminy Open Source.
"""

import tempfile
import numpy as np

from . import core as jiminy
from .viewer import Viewer
from .dynamics import update_quantities


class EngineAsynchronous:
    """
    @brief      Wrapper of Jiminy enabling to update of the command and run simulation
                steps asynchronously. Convenient helper methods are available to set the
                seed of the simulation, reset it, and display it.

    @details    The method `action` is used to update the command, which
                is kept in memory until `action` is called again. On its side, the method
                `step` without argument is used to run simulation steps. The method `step`
                has an optional argument `dt_desired` to specify the number of simulation
                steps to perform at once.

    @remark     This class can be used for synchronous purpose. In such a case, one has
                to call the method `step` specifying the optional argument `action_next`.
    """
    def __init__(self,
                 robot,
                 controller=None,
                 engine=None,
                 use_theoretical_model=False,
                 viewer_backend=None):
        """
        @brief      Constructor

        @param[in]  robot                   Jiminy robot properly setup (eg sensors already added)
        @param[in]  use_theoretical_model   Whether the state corresponds to the theoretical model
        @param[in]  viewer_backend          Backend of the viewer (gepetto-gui or meshcat)

        @return     Instance of the wrapper.
        """
        # Backup the user arguments
        self.robot = robot # For convenience, since the engine will manage its lifetime anyway
        self.use_theoretical_model = use_theoretical_model
        self.viewer_backend = viewer_backend

        # Initialize some internal buffers
        self._t = -1
        self._state = None
        self._sensors_data = None
        self._action = np.zeros((robot.nmotors,))

        # Instantiate the Jiminy controller if necessary, then initialize it
        if controller is None:
            self._controller = jiminy.ControllerFunctor(
                self._send_command, self._internal_dynamics)
        else:
            self._controller = controller
        self._controller.initialize(robot)

        # Instantiate the Jiminy engine
        if controller is None:
            self.engine = jiminy.Engine()
        else:
            self.engine = engine
        self.engine.initialize(robot, self._controller)

        ## Viewer management
        self._viewer = None
        self._is_viewer_available = False

        ## Real time rendering management
        self.step_dt_prev = -1

        # Reset the low-level jiminy engine
        self.engine.reset()

    def __del__(self):
        self.close()

    def _send_command(self, t, q, v, sensors_data, uCommand):
        """
        @brief      This method implement the callback function required by Jiminy
                    Controller to get the command. In practice, it only updates a
                    variable shared between C++ and Python to the internal value
                    stored by this class.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.
        """
        self._sensors_data = sensors_data # It is already a snapshot copy of robot.sensors_data
        uCommand[:] = self._action

    def _internal_dynamics(self, t, q, v, sensors_data, uCommand):
        """
        @brief      This method implement the callback function required by Jiminy
                    Controller to get the internal dynamics. In practice, it does
                    nothing.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.
        """
        pass

    def seed(self, seed):
        """
        @brief      Set the seed of the simulation and reset the simulation.

        @details    Note that it also resets the low-level jiminy Engine.
                    One must call the `reset` method manually afterward.

        @param[in]  seed    Desired seed (Unsigned integer 32 bits)
        """
        assert isinstance(seed, np.uint32),  "'seed' must have type np.uint32."

        engine_options = self.engine.get_options()
        engine_options["stepper"]["randomSeed"] = \
            np.array(seed, dtype=np.dtype('uint32'))
        self.engine.set_options(engine_options)
        self.engine.reset()

    def reset(self, x0, is_state_theoretical=None):
        """
        @brief      Reset the simulation.

        @remark    It does NOT start the simulation immediately but rather wait for the
                   first 'step' call. At this point, the sensors data are zeroed, until
                   the simulation actually starts.
                   Note that once the simulations starts, it is no longer possible to
                   changed the robot (included options).

        @param[in]  x0    Desired initial state
        @param[in]  is_state_theoretical    Wether the provided initial state is associated
                                            with the theoretical or actual model
        """
        # Handling of default argument(s)
        if is_state_theoretical is None:
            is_state_theoretical = self.use_theoretical_model

        # Reset the simulation
        self.engine.reset()

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
        self.engine.start(self._state, self.use_theoretical_model)

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

        # Stop the engine, to avoid locking the robot and the telemetry
        # too early, so that it remains possible to register external
        # forces, register log variables, change the options...etc.
        self.engine.reset()

        # Restore the initial internal pinocchio data
        update_quantities(self.robot,
                          x0[:self.robot.nq],
                          update_physics=True,
                          use_theoretical_model=False)

    def step(self, action_next=None, dt_desired=-1):
        """
        @brief      Run simulation steps.

        @details    Even if Jiminy Engine performs several simulation steps
                    internally, this method only output the final state.

        @param[in]  action_next     Updated command
                                    Optional: Use the value in the internal buffer otherwise
        @param[in]  dt_desired      Simulation time difference between before and after the steps.
                                    Optional: Perform a single integration step otherwise

        @return     Final state of the simulation
        """
        if not self.engine.is_simulation_running:
            flag = self.engine.start(self._state, self.use_theoretical_model)
            if (flag != jiminy.hresult_t.SUCCESS):
                raise RuntimeError("Failed to start the simulation.")

        if (action_next is not None):
            self.action = action_next

        return_code = self.engine.step(dt_desired)
        if (return_code != jiminy.hresult_t.SUCCESS):
            raise RuntimeError("Failed to perform the simulation step.")

        self._t = self.engine.stepper_state.t
        self._state = None # Do not fetch the new current state if not requested to the sake of efficiency
        self.step_dt_prev = self.engine.stepper_state.dt

    def render(self, return_rgb_array=False, width=None, height=None):
        """
        @brief      Render the current state of the simulation. One can display it
                    or return an RGB array instead.

        @remark     Note that it supports parallel rendering, which means that one
                    can display multiple simulations in the same Gepetto-viewer
                    processes at the same time in different tabs.
                    Note that returning an RGB array is not supported by Meshcat in Jupyter.

        @param[in]  return_rgb_array    Whether or not to return the current frame as an rgb array.
                                        Not that this feature is currently not available in Jupyter.
        @param[in]  width               Width of the returned RGB frame if enabled.
        @param[in]  height              Height of the returned RGB frame if enabled.

        @return     Rendering as an RGB array (3D numpy array) if enabled, None otherwise.
        """
        # Instantiate the robot and viewer client if necessary.
        # A new dedicated scene and window will be created.
        if not self._is_viewer_available:
            uniq_id = next(tempfile._get_candidate_names())
            self._viewer = Viewer(self.robot,
                                  use_theoretical_model=False,
                                  backend=self.viewer_backend,
                                  open_gui_if_parent=(not return_rgb_array),
                                  delete_robot_on_close=True,
                                  robot_name="_".join(("robot", uniq_id)),
                                  scene_name="_".join(("scene", uniq_id)),
                                  window_name="_".join(("window", uniq_id)))
            if self._viewer.is_backend_parent:
                self._viewer.set_camera_transform(
                    translation=[9.0, 0.0, 2e-5],
                    rotation=[np.pi/2, 0.0, np.pi/2])
            self._viewer.wait(False)  # Wait for backend to finish loading

        # Try refreshing the viewer
        try:
            self._viewer.refresh()
            self._is_viewer_available = True
        except RuntimeError as e:
            # Check if it failed because viewer backend is no longer available
            if self._is_viewer_available and (Viewer._backend_obj is None or \
                (self._viewer.is_backend_parent and \
                    not self._viewer._backend_proc.is_alive())):
                # Reset viewer backend
                self._viewer.close()
                self._viewer = None
                self._is_viewer_available = False

                # Retry rendering one more time
                return self.render(return_rgb_array, width, height)
            else:
                raise RuntimeError(
                    "Unrecoverable Viewer backend exception.") from e

        # Compute rgb array if needed
        if return_rgb_array:
            return self._viewer.capture_frame(width, height)

    def close(self):
        """
        @brief      Close the connection with the renderer.
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def t(self):
        """
        @brief      Getter of the current time of the simulation.

        @return     Time of the simulation
        """
        return self._t

    @property
    def state(self):
        """
        @brief      Getter of the current state of the robot.

        @return     State of the robot
        """
        if (self._state is None):
            x = self.engine.stepper_state.x
            if self.robot.is_flexible and self.use_theoretical_model:
                self._state = self.robot.get_rigid_state_from_flexible(x)
            else:
                self._state = x
        return self._state

    @property
    def sensors_data(self):
        """
        @brief      Getter of the current sensor data of the robot.

        @return     Sensor data of the robot
        """
        return self._sensors_data

    @property
    def action(self):
        """
        @brief      Getter of the current command.

        @return     Command
        """
        return self._action

    @action.setter
    def action(self, action_next):
        """
        @brief      Setter of the command.

        @param[in]  action_next     Updated command
        """
        if (not isinstance(action_next, np.ndarray) or \
                action_next.shape[-1] != self.robot.nmotors):
            raise ValueError("The action must be a 1D numpy array \
                              whose length matches the number of motors.")
        if np.any(np.isnan(action_next)):
            raise ValueError("'action_next' cannot contain nan values.")
        self._action[:] = action_next

    def get_engine_options(self):
        """
        @brief      Getter of the options of Jiminy Engine.

        @return     Dictionary of options.
        """
        return self.engine.get_options()

    def set_engine_options(self, options):
        """
        @brief      Getter of the options of Jiminy Engine.

        @param[in]  options     Dictionary of options
        """
        self.engine.set_options(options)

    def get_controller_options(self):
        """
        @brief      Getter of the options of Jiminy Controller.

        @return     Dictionary of options.
        """
        return self._controller.get_options()

    def set_controller_options(self, options):
        """
        @brief      Setter of the options of Jiminy Controller.

        @param[in]  options     Dictionary of options
        """
        self._controller.set_options(options)
