import os
import numpy as np

from jiminy_py.viewer import Viewer
import jiminy

if Viewer._is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BasicSimulator(object):
    def __init__(self, jiminy_model, jiminy_controller=None):
        assert issubclass(jiminy_model.__class__, jiminy.Model), \
               "'jiminy_model' must inherit from jiminy.Model"
        assert (jiminy_controller is None or issubclass(jiminy_controller.__class__, jiminy.AbstractController)), \
                "'jiminy_controller' must inherit from jiminy.Controller"
        assert jiminy_model.is_initialized, "'jiminy_model' must be initialized."

        # Default arguments

        # Initialize internal state parameters
        self._t_pbar = -1
        self._pbar = None

        # Copy a reference to Jiminy Model
        self.model = jiminy_model

        # User-defined controller handle
        self.controller_handle = lambda *kargs, **kwargs: None
        self._is_controller_handle_init = False

        # Instantiate the controller if necessary and initialize it
        if jiminy_controller is None:
            self.controller = jiminy.ControllerFunctor(self._compute_command_wrapper)
        else:
            self.controller = jiminy_controller
            self.controller.initialize(self.model, self._compute_command_wrapper)

        # Instantiate and initialize the engine
        self.engine = jiminy.Engine()
        self.engine.initialize(self.model, self.controller, self.callback)

        # Configuration the simulation
        self.configure_simulation()

        # Extract some constant
        self.n_motors = len(self.model.motors_names)

    def configure_simulation(self):
        pass

    def set_controller(self, controller_handle):
        try:
            t = 0.0
            y, dy = np.zeros((self.model.nq,)), np.zeros((self.model.nv,))
            sensorsData = self.model.sensors_data
            y_target, dy_target, u_command = np.zeros((self.n_motors,)), np.zeros((self.n_motors,)), np.zeros((self.n_motors,))
            controller_handle(t, y, dy, sensorsData, y_target, dy_target, u_command)
            self.controller_handle = controller_handle
            self._is_controller_handle_init = True
        except:
            raise ValueError("The controller handle has a wrong signature. It is expected \
                              controller_handle(t, y, dy, sensorsData, y_target, dy_target, u_command)")

    @staticmethod
    def callback(t, x, out):
        pass

    def _compute_command_wrapper(self, t, y, dy, sensorsData, y_target, dy_target, u_command):
        if self._pbar is not None:
            self._pbar.update(t - self._t_pbar)
        self.controller_handle(t, y, dy, sensorsData, y_target, dy_target, u_command)
        self._t_pbar = t

    def get_log(self):
        log_info, log_data = self.engine.get_log()
        log_info = list(log_info)
        log_data = np.asarray(log_data)
        log_constants = log_info[1:log_info.index('StartColumns')]
        log_header = log_info[(log_info.index('StartColumns')+1):-1]
        return log_data, log_header, log_constants

    def run(self, x0, tf, log_path=None):
        assert self._is_controller_handle_init, "The controller handle is not initialized. \
                                                 Please call 'set_controller' before running a simulation."

        # Run the simulation
        self._t_pbar = 0.0
        self._pbar = tqdm(total=tf)
        self.engine.simulate(x0, tf)
        self._pbar.update(tf - self._t_pbar)
        self._pbar.close()
        self._pbar = None

        # Write log
        if log_path is not None:
            self.engine.write_log(log_path, True)
