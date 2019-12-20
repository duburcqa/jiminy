## @file src/jiminy_py/engine_asynchronous.py

"""
@package    jiminy_py

@brief      Package containing python-native helper methods for Jiminy Open Source.
"""

import os
import tempfile
import numpy as np
from PIL import Image
from collections import OrderedDict

from pinocchio import libpinocchio_pywrap as pin
from pinocchio.robot_wrapper import RobotWrapper

import jiminy
import jiminy_py


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
    def __init__(self, model):
        """
        @brief      Constructor

        @param[in]  model   Jiminy model properly setup (eg sensors already added)

        @return     Instance of the wrapper.
        """
        # Make sure that the sensors have already been added to the model !

        self._sensors_types = model.get_sensors_options().keys()
        self._state = np.zeros((model.nx, 1))
        self._observation = OrderedDict((sensor_type,[]) for sensor_type in self._sensors_types)
        self._action = np.zeros((len(model.motors_names), ))

        # Instantiate the Jiminy controller
        self._controller = jiminy.ControllerFunctor(self._send_command, self._internal_dynamics)
        self._controller.initialize(model)

        # Instantiate the Jiminy engine (model and controller are pass-by-reference)
        self._engine = jiminy.Engine()
        self._engine.initialize(model, self._controller)

        ## Flag to determine if Gepetto-viewer is running
        self.is_gepetto_available = False
        self._client = None
        self._window_id = None
        self._viewer_proc = None
        self._id = None
        self._rb = None

        self.reset()


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
        self.reset(x0=None)
        self._engine.set_options(engine_options)


    def reset(self, x0=None):
        """
        @brief      Reset the simulation.

        @details    The initial state is zero. Execute the method `reset` manually
                    afterward to specify a different initial state.

        @param[in]  x0    Desired initial state (2D numpy array: column vector)
        """
        if (x0 is None):
            x0 = np.zeros((self._engine.model.nx, 1))
        if (int(self._engine.reset(x0)) != 1):
            raise ValueError("Reset of engine failed.")
        self._state = x0[:,0]


    def step(self, action_next=None, dt_desired=-1):
        """
        @brief      Run simulation steps.

        @details    Even if Jiminy Engine performs several simulation steps
                    internally, this method only output the final state.

        @param[in]  action_next     Updated command (2D numpy array: column vector)
                                    Optional: Use the value in the internal buffer otherwise
        @param[in]  dt_desired      Simulation time difference between before and after the steps.
                                    Optional: Perform a single integration step otherwise

        @return     Final state of the simulation (2D numpy matrix: column vector ???)
        """
        if (action_next is not None):
            self.action = action_next
        self._state = None
        return self._engine.step(dt_desired)


    def render(self, return_rgb_array=False, lock=None):
        """
        @brief      Render the current state of the simulation. One can display it
                    in Gepetto-viewer or return an RGB array.

        @remark     Note that it supports parallel rendering, which means that one
                    can display multiple simulations in the same Gepetto-viewer
                    processus at the same time in different tabs.
                    Note that it returning an RGB array required Gepetto-viewer.

        @param[in]  return_rgb_array    Updated command (2D numpy array: column vector)
                                        Optional: Use the value in the internal buffer otherwise
        @param[in]  lock                Unique threading.Lock for every simulation
                                        Optional: Only required for parallel rendering

        @return     Low-resolution rendering as an RGB array (3D numpy array)
        """
        rgb_array = None

        if (lock is not None):
            lock.acquire()
        try:
            # Instantiate the Gepetto model and viewer client if necessary
            if (not self.is_gepetto_available):
                self._client, self._viewer_proc = jiminy_py.get_gepetto_client(True)
                self._id = next(tempfile._get_candidate_names())
                self._rb = RobotWrapper()
                collision_model = pin.buildGeomFromUrdf(self._engine.model.pinocchio_model,
                                                        self._engine.model.urdf_path, [],
                                                        pin.GeometryType.COLLISION)
                visual_model = pin.buildGeomFromUrdf(self._engine.model.pinocchio_model,
                                                     self._engine.model.urdf_path, [],
                                                     pin.GeometryType.VISUAL)
                self._rb.__init__(model=self._engine.model.pinocchio_model,
                                  collision_model=collision_model,
                                  visual_model=visual_model)
                self.is_gepetto_available = True

            # Load model in gepetto viewer if needed
            if not self._id in self._client.gui.getSceneList():
                self._client.gui.createSceneWithFloor(self._id)
                self._window_id = self._client.gui.createWindow("jiminy")
                self._client.gui.addSceneToWindow(self._id, self._window_id)
                self._client.gui.createGroup(self._id + '/' + self._id)
                self._client.gui.addLandmark(self._id + '/' + self._id, 0.1)

                self._rb.initViewer(windowName="jiminy", sceneName=self._id, loadModel=False)
                self._rb.loadViewerModel(self._id + '/' + "robot")

                self._client.gui.setCameraTransform(self._window_id,
                                                    [0.0, 9.0, 2e-5, 0.0, 1.0, 1.0, 0.0])

            # Update viewer
            jiminy_py.update_gepetto_viewer(self._rb,
                                            self._engine.model.pinocchio_data,
                                            self._client)

            # return rgb array if needed
            if return_rgb_array:
                png_path = os.path.join("/tmp", self._id + ".png")
                self._client.gui.captureFrame(self._window_id, png_path)
                rgb_array = np.array(Image.open(png_path))[:,:,:-1]
                os.remove(png_path)
        except:
            self.is_gepetto_available = False
            self._client = None
            self._viewer_proc = None
        finally:
            if (lock is not None):
                lock.release()
            return rgb_array


    def close(self):
        """
        @brief      Close the connection with the renderer, namely Gepetto-viewer.

        @details    Must be called once before the destruction of the engine.
        """
        if (self._viewer_proc is not None):
            self._viewer_proc.terminate()
        self.is_gepetto_available = False
        self._client = None
        self._viewer_proc = None


    @property
    def state(self):
        """
        @brief      Getter of the current state of the robot.

        @return     State of the robot (1D numpy array)
        """
        if (self._state is None):
            # Get x by value, then convert the matrix column into an actual 1D array by reference
            self._state = self._engine.stepper_state.x.A1
        return self._state


    @property
    def observation(self):
        """
        @brief      Getter of the current state of the sensors.

        @return     Dictionary whose the keys are the different class of sensors
                    available. The state for a given class is a 2D numpy matrix
                    (row: data, column: sensor).
        """
        return self._observation


    ##
    # @var        action
    # @brief      Current command
    ##
    @property
    def action(self):
        """
        @brief      Getter of the current command.

        @return     Command (1D numpy array).
        """
        return self._action


    @action.setter
    def action(self, action_next):
        """
        @brief      Setter of the command.

        @param[in]  action_next     Updated command (1D numpy array)
        """
        if (not isinstance(action_next, (np.ndarray, np.generic))
        or action_next.size != len(self._action) ):
            raise ValueError("The action must be a numpy array with the right dimension.")
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
