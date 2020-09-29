## @file src/jiminy_py/controller.py
import numpy as np
from typing import Callable

from . import core as jiminy
from .robot import BaseJiminyRobot
from .viewer import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseJiminyController(jiminy.ControllerFunctor):
    """
    @brief Base class to instantiate a Jiminy controller based on a callable
           function that can be changed on-the-fly.

    @details This class is primarily intended to be used for prototyping, for
             those who want to experiment various models of internal dynamics
             for a given robot.

    @remark Note that native Python implementation of internal dynamics would
            be extremely slow, so it is recommended to compile this class
            using for instance a JIT compiler such a Numba. Doing it is out of
            scope of this project for now and left to the user.
    """
    def __init__(self):
        """
        @brief Constructor

        @remark Note that since no robot is available at this point, it is
                not possible to specify a callable function directly, since it
                would be impossible to check that it has the right signature
                and is properly defined.
        """
        self.__pbar = None
        self.__robot = None
        self.__controller_handle = lambda t, q, v, sensors_data, u: u.fill(0.0)
        super().__init__(
            self.__compute_command, self.internal_dynamics)

    def __compute_command(self,
                          t: float,
                          q: np.ndarray,
                          v: np.ndarray,
                          sensors_data: jiminy.sensorsData,
                          u: np.ndarray) -> None:
        """
        @brief Internal controller callback, should not be called directly.
        """
        if self.__pbar is not None:
            self.__pbar.update(t - self.__pbar.n)
        self.__controller_handle(t, q, v, sensors_data, u)

    def initialize(self, robot: BaseJiminyRobot) -> None:
        """
        @brief    TODO
        """
        if self.is_initialized:
            raise RuntimeError("Controller already initialized.")
        self.__robot = robot
        return_code = super().initialize(self.__robot)
        if return_code == jiminy.hresult_t.SUCCESS:
            raise ValueError("Impossible to instantiate the controller. "
                "There is something wrong with the robot.")

    def set_controller_handle(self, controller_handle: Callable) -> None:
        """
        @brief Set the controller callback function to use.

        @param compute_command  Controller entry-point, implemented as a
                                callback function. It must have the following
                                signature:
                                  controller_handle(
                                      t: float,
                                      q: np.ndarray,
                                      v: np.ndarray,
                                      sensors_data: jiminy_py.core.sensorsData,
                                      u_command: np.ndarray) -> None
        """

        try:
            t = 0.0
            y, dy = np.zeros(self.__robot.nq), np.zeros(self.__robot.nv)
            sensors_data = self.__robot.sensors_data
            u_command = np.zeros(self.__robot.nmotors)
            controller_handle(t, y, dy, sensors_data, u_command)
            self.__controller_handle = controller_handle
        except Exception as e:
            raise RuntimeError("The controller handle has a wrong signature. "
                "It is expected:\ncontroller_handle(t, y, dy, sensorsData, "
                "u_command) -> None") from e

    def reset(self) -> None:
        """
        @brief Reset the controller. Not intended to be called manually.
        """
        super().reset()
        self.close_progress_bar()

    def set_progress_bar(self, tf: float) -> None:
        """
        @brief Reset the progress bar. It must be called manually after calling
               `reset` method to enable automatic progress bar update.
        """
        self.__pbar = tqdm(total=tf, bar_format=("{percentage:3.0f}%|{bar}| "
            "{n:.2f}/{total_fmt} [{elapsed}<{remaining}]"))

    def close_progress_bar(self) -> None:
        if self.__pbar is not None:
            self.__pbar.update(self.__pbar.total - self.__pbar.n)
            self.__pbar.close()
            self.__pbar = None

    def internal_dynamics(self,
                          t: float,
                          q: np.ndarray,
                          v: np.ndarray,
                          sensors_data: jiminy.sensorsData,
                          u_command: np.ndarray) -> None:
        """
        @brief Internal dynamics of the robot.

        @details Overload this method to implement a custom internal dynamics
                 for the robot. Note that is results in an overhead of about
                 100% of the simulation in most cases, which is not often not
                 acceptable in production, but still useful for prototyping.
                 One way to solve this issue would be to compile it using
                 CPython.

        @remark  This method is time-continuous as it is designed to implement
                 physical laws.
        """
        pass
