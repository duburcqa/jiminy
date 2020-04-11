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

from pinocchio import libpinocchio_pywrap as pin
from pinocchio.robot_wrapper import RobotWrapper

from . import core as jiminy
from .viewer import Viewer


class EngineAsynchronous(object):
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
    def __init__(self, robot, viewer_backend=None, viewer_use_theoretical_model=False):
        """
        @brief      Constructor

        @param[in]  robot   Jiminy robot properly setup (eg sensors already added)

        @return     Instance of the wrapper.
        """

        # Make sure that the sensors have already been added to the robot !
        self._sensors_types = robot.get_sensors_options().keys()
        self._state = np.zeros((robot.nx,))
        self._observation = OrderedDict((sensor_type,[]) for sensor_type in self._sensors_types)
        self._action = np.zeros((len(robot.motors_names),))

        # Instantiate the Jiminy controller
        self._controller = jiminy.ControllerFunctor(self._send_command, self._internal_dynamics)
        self._controller.initialize(robot)

        # Instantiate the Jiminy engine (robot and controller are pass-by-reference)
        self._engine = jiminy.Engine()
        self._engine.initialize(robot, self._controller)

        ## Viewer management
        self.viewer_backend = viewer_backend
        self.viewer_use_theoretical_model = viewer_use_theoretical_model
        self._viewer = None
        
        ## Real time rendering management
        self.step_dt_prev = -1
        
        ## Flag to determine if the simulation is running, and if the state is theoretical
        self._is_running = False
        self._is_state_theoretical = False

        self.reset(np.zeros(robot.nx))

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
        for sensor_type in self._sensors_types:
            self._observation[sensor_type] = sensor_data[sensor_type]
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

        @details    The initial state is zero. Execute the method `reset` manually
                    afterward to specify a different initial state.

        @param[in]  seed    Desired seed (Unsigned integer 32 bits)
        """
        engine_options = self._engine.get_options()
        engine_options["stepper"]["randomSeed"] = np.array(seed, dtype=np.dtype('uint32'))
        self._engine.set_options(engine_options)
        self._engine.reset()
        self._is_running = False

    def reset(self, x0, is_state_theoretical=False):
        """
        @brief      Reset the simulation.

        @param[in]  x0    Desired initial state
        """
        self._engine.stop()
        self._is_running = False
        self._state = x0
        self.step_dt_prev = -1
        self._is_state_theoretical = is_state_theoretical

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
        if (not self._is_running):
            flag = self._engine.start(self._state, self._is_state_theoretical)
            if (flag != jiminy.hresult_t.SUCCESS):
                raise ValueError("Failed to start the simulation")
            self._is_running = True

        if (action_next is not None):
            self.action = action_next
        self._state = None
        return_code = self._engine.step(dt_desired)
        self.step_dt_prev = self._engine.stepper_state.dt
        return return_code

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
                self._viewer = Viewer(self._engine.robot,
                                      backend=self.viewer_backend,
                                      use_theoretical_model=self.viewer_use_theoretical_model,
                                      window_name='jiminy', scene_name=scene_name)
                self._viewer.setCameraTransform(translation=[0.0, 9.0, 2e-5],
                                                rotation=[np.pi/2, 0.0, np.pi])
    
            # Refresh viewer
            self._viewer.refresh()

            # Compute rgb array if needed
            if return_rgb_array:
                rgb_array = self._viewer.captureFrame()
        except:
            self._viewer = None
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
    def state(self):
        """
        @brief      Getter of the current state of the robot.

        @return     State of the robot
        """
        if (self._state is None):
            self._state = self._engine.stepper_state.x
            self._is_state_theoretical = False
        else:
            if (self._is_state_theoretical):
                raise RuntimeError("Impossible to get the current state since the initial state \
                                    was theoretical and no steps were performed ever since.")
        return self._state

    @property
    def observation(self):
        """
        @brief      Getter of the current state of the sensors.

        @return     Dictionary whose the keys are the different class of sensors
                    available.
                    (row: data, column: sensor).
        """
        return self._observation

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
        if (not isinstance(action_next, (np.ndarray, np.generic))
                or action_next.size != self._action.size):
            raise ValueError("The action must be a numpy array of the right dimension.")
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
