import numpy as np
from typing import Optional
from dataclasses import dataclass

from .generic import CenterOfMass
from ..bases import InterfaceJiminyEnv, AbstractQuantity
from ..utils import fill


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

        # Proxy for the derivative of the spatial centroidal momentum
        self.dhg = np.ndarray([])

        # Pre-allocate memory for the ZMP
        self._zmp = np.zeros(2)

    def initialize(self) -> None:
        super().initialize()
        self._gravity = abs(self.pinocchio_model.gravity.linear[2])
        self._robot_mass = self.pinocchio_data.mass[0]
        self._robot_weight = self._robot_mass * self._gravity
        self.dhg = self.pinocchio_data.dhg
        fill(self._zmp, 0)

    def refresh(self) -> np.ndarray:
        dhg, com = self.dhg, self.com
        f_z = dhg.linear[2] + self._robot_weight
        self._zmp[:] = com[:2] * (self._robot_weight / f_z)
        if abs(f_z) > np.finfo(np.float32).eps:
            self._zmp[0] -= (dhg.angular[1] + dhg.linear[0] * com[2]) / f_z
            self._zmp[1] += (dhg.angular[0] - dhg.linear[1] * com[2]) / f_z
        return self._zmp
