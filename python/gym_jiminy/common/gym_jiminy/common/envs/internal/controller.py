""" TODO: Write documentation.
"""
from typing import Optional, Dict

import numpy as np

import jiminy_py.core as jiminy

from ...bases import ObserverHandleType, ControllerHandleType


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
        self.__sensors_data: Dict[str, np.ndarray] = dict(self.sensors_data)
        self.__observer_handle: Optional[ObserverHandleType] = None
        self.__controller_handle: Optional[ControllerHandleType] = None

    @property
    def has_observer(self) -> bool:
        """Check whether a valid observer handle has been set.
        """
        return self.__observer_handle is not None

    @property
    def has_controller(self) -> bool:
        """Check whether a valid controller handle has been set.
        """
        return self.__controller_handle is not None

    def initialize(self,  # type: ignore[override]
                   robot: jiminy.Robot) -> None:
        """Initialize the controller.

        :param robot: Jiminy robot to control.
        """
        return_code = super().initialize(robot)
        if return_code != jiminy.hresult_t.SUCCESS:
            raise ValueError(
                "Impossible to instantiate the controller. There is "
                "something wrong with the robot.")

    def reset(self, reset_dynamic_telemetry: bool = False) -> jiminy.hresult_t:
        """ TODO: Write documentation.
        """
        return_code = super().reset(reset_dynamic_telemetry)
        if return_code == jiminy.hresult_t.SUCCESS:
            self.__sensors_data = dict(self.sensors_data)
        return return_code

    def set_observer_handle(self,
                            observer_handle: Optional[ObserverHandleType],
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
        :param unsafe: Whether to check if the handle is valid.
        """
        try:
            if not unsafe and observer_handle is not None:
                t, q, v = 0.0, np.zeros(self.robot.nq), np.zeros(self.robot.nv)
                sensors_data: Dict[str, np.ndarray] = dict(
                    self.robot.sensors_data)
                observer_handle(t, q, v, sensors_data)
            self.__observer_handle = observer_handle
        except Exception as e:
            raise RuntimeError(
                "The observer handle has wrong signature. It is expected:\n"
                "controller_handle(t, q, v, sensors_data) -> None"
                ) from e

    def set_controller_handle(self,
                              controller_handle: Optional[
                                  ControllerHandleType],
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
            |                    **command**: np.ndarray
            |                    \) -> None
        :param unsafe: Whether to check if the handle is valid.
        """
        try:
            if not unsafe and controller_handle is not None:
                t, q, v = 0.0, np.zeros(self.robot.nq), np.zeros(self.robot.nv)
                sensors_data: Dict[str, np.ndarray] = dict(
                    self.robot.sensors_data)
                command = np.zeros(self.robot.nmotors)
                controller_handle(t, q, v, sensors_data, command)
            self.__controller_handle = controller_handle
        except Exception as e:
            raise RuntimeError(
                "The controller handle has wrong signature. It is expected:\n"
                "controller_handle(t, q, v, sensors_data, command) -> None"
                ) from e

    def compute_command(self,  # type: ignore[override]
                        t: float,
                        q: np.ndarray,
                        v: np.ndarray,
                        command: np.ndarray) -> None:
        """Internal controller callback, should not be called directly.
        """
        if self.__must_refresh_observer and self.has_observer:
            self.__observer_handle(  # type: ignore[misc]
                t, q, v, self.__sensors_data)
        if self.has_controller:
            self.__controller_handle(  # type: ignore[misc]
                t, q, v, self.__sensors_data, command)
        self.__must_refresh_observer = True

    def refresh_observation(self,
                            t: float,
                            q: np.ndarray,
                            v: np.ndarray,
                            sensors_data: Dict[str, np.ndarray]) -> None:
        """Refresh observer.
        """
        if self.__must_refresh_observer and self.has_observer:
            self.__observer_handle(  # type: ignore[misc]
                t, q, v, sensors_data)
        self.__must_refresh_observer = False


BaseJiminyObserverController.internal_dynamics.__doc__ = \
    """Internal dynamics of the robot.

    Overload this method to implement a custom internal dynamics for the robot.

    .. warning::
        It is likely to result in an overhead of about 100% of the simulation,
        which is not often not acceptable in production, but still useful for
        prototyping. One way to solve this issue would be to compile it using
        CPython.

    .. note::
        This method is time-continuous as it is designed to implement physical
        laws.
    """
