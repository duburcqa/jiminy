import os
import numpy as np

from . import core as jiminy
from .viewer import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BasicSimulator(object):
    """
    @brief   A helper class for launching simulations.

    @details This class handles creation of the engine, the jiminy.controller object, configuration...
              so that the user only has to worry about creating a controller, implemented as a callback
              function, and give high-level instructions to the simulation.
              While this class provides an already functional simulation environment, it is expect that the
              user will develop a child class to customize it to his needs.
    """
    def __init__(self, robot, jiminy_controller=None):
        """
        @brief Constructor

        @param robot jiminy.Robot to use
        @param jiminy_controller Optional, a jiminy controller to use. If None, a ControllerFunctor is created.
        """
        assert issubclass(robot.__class__, jiminy.Robot), \
               "'robot' must inherit from jiminy.Robot"
        assert (jiminy_controller is None or issubclass(jiminy_controller.__class__, jiminy.AbstractController)), \
                "'jiminy_controller' must inherit from jiminy.Controller"
        assert robot.is_initialized, "'robot' must be initialized."

        # Default arguments

        # Initialize internal state parameters
        self._t_pbar = -1
        self._pbar = None

        # Copy a reference to Jiminy Robot
        self.robot = robot

        # User-defined controller handle
        self.controller_handle = lambda *kargs, **kwargs: None
        self.internal_dynamics = lambda *kargs, **kwargs: None
        self._is_controller_handle_init = False

        # Instantiate the controller if necessary and initialize it
        if jiminy_controller is None:
            self.controller = jiminy.ControllerFunctor(self._compute_command_wrapper, self.internal_dynamics)
            self.controller.initialize(self.robot)
        else:
            self.controller = jiminy_controller
            self.controller.initialize(self.robot, self._compute_command_wrapper)

        # Instantiate and initialize the engine
        self.engine = jiminy.Engine()
        self.engine.initialize(self.robot, self.controller, self.callback)

        # Configuration the simulation
        self.configure_simulation()

        # Extract some constant
        self.n_motors = len(self.robot.motors_names)

    def configure_simulation(self):
        """
        @brief User-specific configuration, applied once at the end of the constructor.
        """
        pass

    def set_controller(self, controller_handle):
        """
        @brief Set the controller callback function to be used. It must follow the following signature:
        controller_handle(t, y, dy, sensors_data, u_command).

        @param controller_handle Controller callback to set.
        """
        try:
            t = 0.0
            y, dy = np.zeros(self.robot.nq), np.zeros(self.robot.nv)
            sensors_data = self.robot.sensors_data
            u_command = np.zeros(self.n_motors)
            controller_handle(t, y, dy, sensors_data, u_command)
            self.controller_handle = controller_handle
            self._is_controller_handle_init = True
        except:
            raise RuntimeError("The controller handle has a wrong signature. It is expected " + \
                               "controller_handle(t, y, dy, sensorsData, u_command)")

    @staticmethod
    def callback(t, q, v, out):
        """
        @brief Callback method for the simulation.
        """
        out[:] = True

    def _compute_command_wrapper(self, t, q, v, sensors_data, u):
        """
        @brief Internal controller callback, should not be called directly.
        """
        if self._pbar is not None:
            self._pbar.update(t - self._t_pbar)
        self.controller_handle(t, q, v, sensors_data, u)
        self._t_pbar = t

    def get_log(self):
        """
        @brief Get log data from the engine.
        """
        return self.engine.get_log()

    def run(self, tf, x0, is_state_theoretical=True, log_path=None, show_progress_bar=True):
        """
        @brief Run a simulation, starting from x0 at t=0 up to tf. Optionally, log results in a logfile.

        @param x0 Initial condition
        @param tf Simulation end time
        @param log_path Optional, if set save log data to given file.
        @param show_progress_bar Optional, if set display a progress bar during the simulation
        """
        assert self._is_controller_handle_init, "The controller handle is not initialized." + \
                                                "Please call 'set_controller' before running a simulation."
        # Run the simulation
        self._t_pbar = 0.0
        if show_progress_bar:
            self._pbar = tqdm(total=tf, bar_format="{percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]")
        else:
            self._pbar = None
        self.engine.simulate(tf, x0, is_state_theoretical)
        if show_progress_bar:
            self._pbar.update(tf - self._t_pbar)
            self._pbar.close()
        self._pbar = None

        # Write log
        if log_path is not None:
            self.engine.write_log(log_path, True)
