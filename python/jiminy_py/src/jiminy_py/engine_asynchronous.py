## @file src/jiminy_py/engine_asynchronous.py

"""
@package    jiminy_py

@brief      Package containing python-native helper methods for Jiminy Open Source.
"""

import os
import tempfile
import time
import numpy as np
from collections import OrderedDict

from pinocchio import neutral

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
    def __init__(self, robot, use_theoretical_model=False, viewer_backend=None):
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

        # Make sure that the sensors have already been added to the robot !
        self._sensors_types = robot.get_sensors_options().keys()
        self._t = -1
        self._state = None
        self._sensor_data = None
        self._action = np.zeros((robot.nmotors,))

        # Instantiate the Jiminy controller
        self._controller = jiminy.ControllerFunctor(self._send_command, self._internal_dynamics)
        self._controller.initialize(robot)

        # Instantiate the Jiminy engine
        self._engine = jiminy.Engine()
        self._engine.initialize(robot, self._controller)

        ## Viewer management
        self._viewer = None
        self._is_viewer_available = False

        ## Real time rendering management
        self.step_dt_prev = -1

        # Initialize the low-level jiminy engine
        q0 = neutral(robot.pinocchio_model)
        v0 = np.zeros(robot.nv)
        self.reset(np.concatenate((q0, v0)), is_state_theoretical=False)

    def _send_command(self, t, q, v, sensor_data, uCommand):
        """
        @brief      This method implement the callback function required by Jiminy
                    Controller to get the command. In practice, it only updates a
                    variable shared between C++ and Python to the internal value
                    stored by this class.

        @remark     This is a hidden function that is not listed as part of the
                    member methods of the class. It is not intended to be called
                    manually.
        """
        self._sensor_data = sensor_data
        uCommand[:] = self._action

    def _internal_dynamics(self, t, q, v, sensor_data, uCommand):
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

        engine_options = self._engine.get_options()
        engine_options["stepper"]["randomSeed"] = np.array(seed, dtype=np.dtype('uint32'))
        self._engine.set_options(engine_options)
        self._engine.reset()

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

        # Reset the simulation. Do NOT start a new one at this point,
        # to avoid locking the robot and the telemetry too early.
        self._engine.reset()

        # Call update_quantities in order to the frame placement for rendering
        if is_state_theoretical:
            x0_rigid = x0
            if self.robot.is_flexible:
                x0 = self.robot.get_rigid_state_from_flexible(x0_rigid)
        else:
            if self.robot.is_flexible and self.use_theoretical_model:
                x0_rigid = self.robot.get_flexible_state_from_rigid(x0)

        # Update the frames placement, for proper rendering
        update_quantities(self.robot,
                          x0[:self.robot.nq],
                          update_physics=True,
                          use_theoretical_model=False)

        # Reset the flags
        self._t = 0.0
        self._state = x0_rigid if self.use_theoretical_model else x0
        self._sensor_data = self.robot.sensors_data
        self._action[:] = 0.0
        self.step_dt_prev = -1

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
        if (not self._engine.is_simulation_running):
            flag = self._engine.start(self._state, self.use_theoretical_model)
            if (flag != jiminy.hresult_t.SUCCESS):
                raise ValueError("Failed to start the simulation.")

        if (action_next is not None):
            self.action = action_next

        return_code = self._engine.step(dt_desired)
        if (return_code != jiminy.hresult_t.SUCCESS):
            raise ValueError("Failed to perform the simulation step.")

        self._t = self._engine.stepper_state.t
        self._state = None # Do not fetch the new current state if not requested to the sake of efficiency
        self.step_dt_prev = self._engine.stepper_state.dt

    def render(self, return_rgb_array=False, lock=None):
        """
        @brief      Render the current state of the simulation. One can display it
                    in Gepetto-viewer or return an RGB array.

        @remark     Note that it supports parallel rendering, which means that one
                    can display multiple simulations in the same Gepetto-viewer
                    processes at the same time in different tabs.
                    Note that returning an RGB array required Gepetto-viewer.

        @param[in]  return_rgb_array    Updated command
                                        Optional: Use the value in the internal buffer otherwise
        @param[in]  lock                Unique threading.Lock for every simulation
                                        Optional: Only required for parallel rendering

        @return     Low-resolution rendering as an RGB array (3D numpy array)
        """
        rgb_array = None

        if (lock is not None):
            lock.acquire()
        try:
            # Instantiate the robot and viewer client if necessary
            if (self._viewer is None):
                scene_name = next(tempfile._get_candidate_names())
                self._viewer = Viewer(self.robot,
                                      backend=self.viewer_backend,
                                      use_theoretical_model=False,
                                      window_name='jiminy', scene_name=scene_name)
                self._viewer.setCameraTransform(translation=[0.0, 9.0, 2e-5],
                                                rotation=[np.pi/2, 0.0, np.pi])

            # Refresh viewer
            self._viewer.refresh()
            self._is_viewer_available = True

            # Compute rgb array if needed
            if return_rgb_array:
                rgb_array = self._viewer.captureFrame()
        except:
            self.close()
            self._viewer = None
            if self._is_viewer_available:
                self._is_viewer_available = False
                return self.render(return_rgb_array, lock)
        finally:
            if (lock is not None):
                lock.release()
            return rgb_array

    def close(self):
        """
        @brief      Close the connection with the renderer, namely Gepetto-viewer.

        @details    Must be called once before the destruction of the engine.
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
            x = self._engine.stepper_state.x
            if self.robot.is_flexible and self.use_theoretical_model:
                self._state = self.robot.get_rigid_state_from_flexible(x)
            else:
                self._state = x
        return self._state

    @property
    def sensor_data(self):
        """
        @brief      Getter of the current sensor data of the robot.

        @return     Sensor data of the robot
        """
        return self._sensor_data

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
        if (not isinstance(action_next, np.ndarray)
                or action_next.shape != (self.robot.nmotors,)):
            raise ValueError("The action must be a 1D numpy array \
                              whose length matches the number of motors.")
        self._action[:] = action_next

    def get_engine_options(self):
        """
        @brief      Getter of the options of Jiminy Engine.

        @return     Dictionary of options.
        """
        return self._engine.get_options()

    def set_engine_options(self, options):
        """
        @brief      Getter of the options of Jiminy Engine.

        @param[in]  options     Dictionary of options
        """
        self._engine.set_options(options)

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
