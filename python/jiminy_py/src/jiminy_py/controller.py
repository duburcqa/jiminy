## @file src/jiminy_py/controller.py
import numpy as np
from typing import Callable

from . import core as jiminy
from .robot import BaseJiminyRobot


class BaseJiminyController(jiminy.ControllerFunctor):
    """
    @brief Base class to instantiate a Jiminy controller based on a
           callable function.

    @details This class is primarily helpful for those who want to
             implement a custom internal dynamics with hysteresis for
             prototyping.
    """
    def __init__(self, compute_command: Callable):
        """
        @brief    TODO
        """
        self.__robot = None
        super().__init__(compute_command, self.internal_dynamics)

    def initialize(self, robot: BaseJiminyRobot):
        """
        @brief    TODO
        """
        self.__robot = robot
        return_code = super().initialize(self.__robot)

        if return_code == jiminy.hresult_t.SUCCESS:
            raise ValueError("Impossible to instantiate the controller. "
                "There is something wrong with the robot.")

    def internal_dynamics(self,
                          t: float,
                          q: np.ndarray,
                          v: np.ndarray,
                          sensors_data: jiminy.sensorsData,
                          u_command: np.ndarray):
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
