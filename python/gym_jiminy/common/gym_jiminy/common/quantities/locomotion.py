"""Quantities mainly relevant for locomotion tasks on floating-base robots.
"""
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from ..bases import InterfaceJiminyEnv, AbstractQuantity
from ..utils import fill


@dataclass(unsafe_hash=True)
class CenterOfMass(AbstractQuantity[np.ndarray]):
    """Position, Velocity or Acceleration of the center of mass of the robot as
    a whole.
    """

    kinematic_level: pin.KinematicLevel
    """Kinematic level to compute, ie position, velocity or acceleration.
    """

    def __init__(
            self,
            env: InterfaceJiminyEnv,
            parent: Optional[AbstractQuantity],
            kinematic_level: pin.KinematicLevel = pin.POSITION
            ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :para kinematic_level: Desired kinematic level, ie position, velocity
                               or acceleration.
        """
        # Backup some user argument(s)
        self.kinematic_level = kinematic_level

        # Call base implementation
        super().__init__(env, parent, requirements={})

        # Pre-allocate memory for the CoM quantity
        self._com_data: np.ndarray = np.array([])

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Refresh CoM quantity proxy based on kinematic level
        if self.kinematic_level == pin.POSITION:
            self._com_data = self.pinocchio_data.com[0]
        elif self.kinematic_level == pin.VELOCITY:
            self._com_data = self.pinocchio_data.vcom[0]
        else:
            self._com_data = self.pinocchio_data.acom[0]

    def refresh(self) -> np.ndarray:
        # Jiminy does not compute the CoM acceleration automatically
        if self.kinematic_level == pin.ACCELERATION:
            pin.centerOfMass(self.pinocchio_model,
                             self.pinocchio_data,
                             self.kinematic_level)

        # Return proxy directly without copy
        return self._com_data


@dataclass(unsafe_hash=True)
class ZeroMomentPoint(AbstractQuantity[np.ndarray]):
    """Zero Moment Point (ZMP), also called Divergent Component of Motion
    (DCM).

    This quantity only makes sense for legged robots whose the inverted
    pendulum is a relevant approximate dynamic model. It is involved in various
    dynamic stability metrics (usually only on flat ground), such as N-steps
    capturability. More precisely, the robot will keep balance if the ZMP is
    maintained inside the support polygon.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[AbstractQuantity]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        """
        # Call base implementation
        super().__init__(env, parent, requirements={"com": (CenterOfMass, {})})

        # Weight of the robot
        self._robot_weight: float = -1

        # Proxy for the derivative of the spatial centroidal momentum
        self.dhg: Tuple[np.ndarray, np.ndarray] = (np.ndarray([]),) * 2

        # Pre-allocate memory for the ZMP
        self._zmp = np.zeros(2)

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Compute the weight of the robot
        gravity = abs(self.pinocchio_model.gravity.linear[2])
        robot_mass = self.pinocchio_data.mass[0]
        self._robot_weight = robot_mass * gravity

        # Refresh derivative of the spatial centroidal momentum
        self.dhg = ((dhg := self.pinocchio_data.dhg).linear, dhg.angular)

        # Re-initialized pre-allocated memory buffer
        fill(self._zmp, 0)

    def refresh(self) -> np.ndarray:
        # Extract intermediary quantities for convenience
        (dhg_linear, dhg_angular), com = self.dhg, self.com

        # Compute the vertical force applied by the robot
        f_z = dhg_linear[2] + self._robot_weight

        # Compute the center of pressure
        self._zmp[:] = com[:2] * (self._robot_weight / f_z)
        if abs(f_z) > np.finfo(np.float32).eps:
            self._zmp[0] -= (dhg_angular[1] + dhg_linear[0] * com[2]) / f_z
            self._zmp[1] += (dhg_angular[0] - dhg_linear[1] * com[2]) / f_z

        return self._zmp
