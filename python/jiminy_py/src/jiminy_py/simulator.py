## @file jiminy_py/simulator.py
import pathlib
import numpy as np
from typing import Optional, Tuple, Dict

from . import core as jiminy
from .viewer import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BasicSimulator:
    """
    @brief Helper class for launching simulations.

    @details This class handles creation of the low-level Jiminy engine and
             controller, along their configuration. The user only has to
             worry about defining the controller entry-point, implemented as
             a callback function, and give high-level instructions to the
             simulator.

    @remark While this class provides an already functional simulation
            environment, it is expected that the user will develop a child
            class to customize it to better fit his needs.
    """
    def __init__(self,
                 robot: jiminy.Robot,
                 controller_class: Optional[jiminy.ControllerFunctor] = \
                     jiminy.ControllerFunctor):
        """
        @brief Constructor

        @param robot  Jiminy robot. It must be already initialized.
        @param controller_class  The type of controller to use.
                                 Optional: core.ControllerFunctor without
                                 internal dynamics by default.
        """

        assert robot.is_initialized, "'robot' must be initialized."

        # Default arguments

        # Initialize internal state parameters
        self._t_pbar = -1
        self._pbar = None

        # Copy a reference to Jiminy Robot
        self.robot = robot

        # User-defined controller handle
        self.controller_handle = lambda *kargs, **kwargs: None
        self._is_controller_handle_initialized = False

        # Instantiate the controller and initialize it
        self.controller = controller_class(
            compute_command=self._compute_command_wrapper)
        self.controller.initialize(self.robot)

        # Instantiate and initialize the engine
        self.engine = jiminy.Engine()
        self.engine.initialize(self.robot, self.controller, self.callback)

        # Configuration the simulation
        self.configure_simulation()

        # Extract some constant
        self.n_motors = len(self.robot.motors_names)

    def configure_simulation(self) -> None:
        """
        @brief User-specific configuration.

        @details It is applied once, at the end of the constructor, before
                 extracting information about the robot, engine...
        """
        pass

    def set_controller(self, controller_handle) -> None:
        """
        @brief Set the controller callback function to be used.

        @details It must have the following signature:
                   controller_handle(t, y, dy, sensors_data, u_command) -> None

        @param controller_handle  Controller callback to set.
        """
        try:
            t = 0.0
            y, dy = np.zeros(self.robot.nq), np.zeros(self.robot.nv)
            sensors_data = self.robot.sensors_data
            u_command = np.zeros(self.n_motors)
            controller_handle(t, y, dy, sensors_data, u_command)
            self.controller_handle = controller_handle
            self._is_controller_handle_initialized = True
        except:
            raise RuntimeError(
                "The controller handle has a wrong signature. It is expected "
                "controller_handle(t, y, dy, sensorsData, u_command) -> None")

    def _callback(self,
                  t: float,
                  q: np.ndarray,
                  v: np.ndarray,
                  out: np.ndarray) -> None:
        """
        @brief Callback method for the simulation.
        """
        out[0] = True

    def _compute_command_wrapper(self,
                                 t: float,
                                 q: np.ndarray,
                                 v: np.ndarray,
                                 sensors_data: jiminy.sensorsData,
                                 u: np.ndarray) -> None:
        """
        @brief Internal controller callback, should not be called directly.
        """
        if self._pbar is not None:
            self._pbar.update(t - self._t_pbar)
        self.controller_handle(t, q, v, sensors_data, u)
        self._t_pbar = t

    def get_log(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        @brief Get log data from the engine.
        """
        return self.engine.get_log()

    def run(self,
            tf: float,
            x0: np.ndarray,
            is_state_theoretical: bool = True,
            log_path: Optional[str] = None,
            show_progress_bar: bool = True) -> None:
        """
        @brief Run a simulation, starting from x0 at t=0 up to tf.
               Optionally, log the result of the simulation.

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
        assert self._is_controller_handle_initialized, (
            "The controller handle is not initialized."
            "Please call 'set_controller' before running a simulation.")
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
            log_path = pathlib.Path(log_path).with_suffix('.data')
            self.engine.write_log(log_path, True)
