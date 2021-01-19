import numpy as np
from typing import Callable

from . import core as jiminy
from .robot import BaseJiminyRobot


ObserverHandleType = Callable[[
    float, np.ndarray, np.ndarray, jiminy.sensorsData], None]
ControllerHandleType = Callable[[
    float, np.ndarray, np.ndarray, jiminy.sensorsData, np.ndarray], None]


class BaseJiminyObserverController(jiminy.BaseController):
    """Base class to instantiate a Jiminy observer and/or controller based on
    callables that can be changed on-the-fly.

    .. note::
        Native Python implementation of internal dynamics would be extremely
        slow, so it is recommended to compile this class using a JIT compiler
        such a Numba. Doing it is left to the user.
    """
    def __init__(self) -> None:
        # Initialize base controller
        super().__init__()

        # Define some internal buffers
        self.__must_refresh_observer = True
        self.__controller_handle = \
            lambda t, q, v, sensors_data, u: u.fill(0.0)
        self.__observer_handle = lambda t, q, v, sensors_data: None

    def initialize(self, robot: BaseJiminyRobot) -> None:
        """Initialize the controller.

        :param robot: Jiminy robot to control.
        """
        if self.is_initialized:
            raise RuntimeError("Controller already initialized.")
        return_code = super().initialize(robot)
        if return_code != jiminy.hresult_t.SUCCESS:
            raise ValueError(
                "Impossible to instantiate the controller. There is "
                "something wrong with the robot.")

    def set_observer_handle(self,
                            observer_handle: ObserverHandleType,
                            unsafe: bool = False) -> None:
        r"""Set the observer callback function.

        By default, the observer update period is the same of the observer. One
        is responsible to implement custom breakpoint point management if it
        must be different.

        :param compute_command:
            .. raw:: html

                Observer entry-point, implemented as a callback function. It
                must have the following signature:

            | observer_handle\(**t**: float,
            |                  **q**: np.ndarray,
            |                  **v**: np.ndarray,
            |                  **sensors_data**: jiminy_py.core.sensorsData
            |                  \) -> None
        :param unsafe: Whether or not to check if the handle is valid.
        """
        try:
            if not unsafe:
                t = 0.0
                y, dy = np.zeros(self.robot.nq), np.zeros(self.robot.nv)
                sensors_data = self.robot.sensors_data
                observer_handle(t, y, dy, sensors_data)
        except Exception as e:
            raise RuntimeError(
                "The observer handle has wrong signature. It is expected:"
                "\ncontroller_handle(t, y, dy, sensorsData) -> None"
                ) from e
        self.__observer_handle = observer_handle

    def set_controller_handle(self,
                              controller_handle: ControllerHandleType,
                              unsafe: bool = False) -> None:
        r"""Set the controller callback function.

        :param compute_command:
            .. raw:: html

                Controller entry-point, implemented as a callback function. It
                must have the following signature:

            | controller_handle\(**t**: float,
            |                    **q**: np.ndarray,
            |                    **v**: np.ndarray,
            |                    **sensors_data**: jiminy_py.core.sensorsData,
            |                    **u_command**: np.ndarray
            |                    \) -> None
        :param unsafe: Whether or not to check if the handle is valid.
        """
        try:
            if not unsafe:
                t = 0.0
                y, dy = np.zeros(self.robot.nq), np.zeros(self.robot.nv)
                sensors_data = self.robot.sensors_data
                u_command = np.zeros(self.robot.nmotors)
                controller_handle(t, y, dy, sensors_data, u_command)
        except Exception as e:
            raise RuntimeError(
                "The controller handle has wrong signature. It is expected:"
                "\ncontroller_handle(t, y, dy, sensorsData, u_command) -> None"
                ) from e
        self.__controller_handle = controller_handle

    def compute_command(self,
                        t: float,
                        q: np.ndarray,
                        v: np.ndarray,
                        u: np.ndarray) -> None:
        """Internal controller callback, should not be called directly.
        """
        if self.__must_refresh_observer:
            self.__observer_handle(t, q, v, self.sensors_data)
        self.__controller_handle(t, q, v, self.sensors_data, u)
        self.__must_refresh_observer = True

    def refresh_observation(self,
                            t: float,
                            q: np.ndarray,
                            v: np.ndarray,
                            sensors_data: jiminy.sensorsData) -> None:
        """Refresh observer.
        """
        if self.__must_refresh_observer:
            self.__observer_handle(t, q, v, sensors_data)
        self.__must_refresh_observer = False


BaseJiminyObserverController.internal_dynamics.__doc__ = \
    """Internal dynamics of the robot.

    Overload this method to implement a custom internal dynamics for the robot.

    .. warning:::
        It is likely to result in an overhead of about 100% of the simulation,
        which is not often not acceptable in production, but still useful for
        prototyping. One way to solve this issue would be to compile it using
        CPython.

    .. note:::
        This method is time-continuous as it is designed to implement physical
        laws.
    """
