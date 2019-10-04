import os
import tempfile
import numpy as np
from PIL import Image
from collections import OrderedDict

import libpinocchio_pywrap as pin
from pinocchio.robot_wrapper import RobotWrapper

import jiminy
import jiminy_py


class engine_asynchronous(object):
    def __init__(self, model):
        # Make sure that the sensors have already been added to the model !

        self._sensors_types = model.get_sensors_options().keys()
        self._state = np.zeros((model.nx, 1))
        self._observation = OrderedDict((sensor_type,[]) for sensor_type in self._sensors_types)
        self._action = np.zeros((len(model.motors_names), ))

        # Instantiate the Jiminy controller
        self._controller = jiminy.controller_functor(
            self._send_command, self._internal_dynamics, len(self._sensors_types))
        self._controller.initialize(model)

        # Instantiate the Jiminy engine (model and controller are pass-by-reference)
        self._engine = jiminy.engine()
        self._engine.initialize(model, self._controller)

        self.is_gepetto_available = False
        self._client = None
        self._window_id = None
        self._viewer_proc = None
        self._id = None
        self._rb = None

        self.reset()

    def _send_command(self, t, q, v, *args):
        for k, sensor_type in enumerate(self._observation):
            self._observation[sensor_type] = args[k]
        uCommand = args[-1]
        uCommand[:] = self._action

    def _internal_dynamics(self, t, q, v, *args):
        pass

    def seed(self, seed):
        engine_options = self._engine.get_options()
        engine_options["stepper"]["randomSeed"] = np.array(seed, dtype=np.dtype('uint32'))
        self.reset(x0=None, reset_random_generator=True)
        self._engine.set_options(engine_options)

    def reset(self, x0=None, reset_random_generator=False):
        if (x0 is None):
            x0 = np.zeros((self._engine.model.nx, 1))
        if (int(self._engine.reset(x0)) != 1):
            raise ValueError("Reset of engine failed.")
        self._state = x0[:,0]

    def step(self, action_next=None, dt_desired=-1):
        if (action_next is not None):
            self.action = action_next
        self._state = None
        return self._engine.step(dt_desired)

    def render(self, return_rgb_array=False, lock=None):
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

                self._rb.initDisplay("jiminy", self._id, loadModel=False)
                self._rb.loadDisplayModel(self._id + '/' + "robot")

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
        if (self._viewer_proc is not None):
            self._viewer_proc.terminate()
        self.is_gepetto_available = False
        self._client = None
        self._viewer_proc = None

    @property
    def state(self):
        if (self._state is None):
            # Get x by value, then convert the matrix column into an actual 1D array by reference
            self._state = self._engine.stepper_state.x.A1
        return self._state

    @property
    def observation(self):
        return self._observation

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action_next):
        if (not isinstance(action_next, (np.ndarray, np.generic))
        or action_next.size != len(self._action) ):
            raise ValueError("The action must be a numpy array with the right dimension.")
        self._action[:] = action_next

    def get_engine_options(self):
        return self._engine.get_options()

    def set_engine_options(self, options):
        self._engine.set_options(options)

    def get_controller_options(self):
        return self._controller.get_options()

    def set_controller_options(self, options):
        self._controller.set_options(options)
