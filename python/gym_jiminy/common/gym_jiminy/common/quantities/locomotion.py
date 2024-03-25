import numpy as np
from dataclasses import dataclass

from jiminy_py.simulator import Simulator

from .generic import CenterOfMass
from ..bases import AbstractQuantity
from ..utils import fill


@dataclass(unsafe_hash=True)
class ZeroMomentPoint(AbstractQuantity[np.ndarray]):
    """ TODO: Write documentation.
    """
    def __init__(self, simulator: Simulator) -> None:
        """ TODO: Write documentation.
        """
        super().__init__(simulator, requirements={"com": (CenterOfMass, {})})
        self.dhg = np.ndarray([])
        self._zmp = np.zeros(2)

    def initialize(self) -> None:
        """ TODO: Write documentation.
        """
        super().initialize()
        self._gravity = abs(self.pinocchio_model.gravity.linear[2])
        self._robot_mass = self.pinocchio_data.mass[0]
        self._robot_weight = self._robot_mass * self._gravity
        self.dhg = self.pinocchio_data.dhg
        fill(self._zmp, 0)

    def refresh(self) -> np.ndarray:
        """ TODO: Write documentation.
        """
        dhg, com = self.dhg, self.com
        f_z = dhg.linear[2] + self._robot_weight
        self._zmp[:] = com[:2] * (self._robot_weight / f_z)
        if abs(f_z) > np.finfo(np.float32).eps:
            self._zmp[0] -= (dhg.angular[1] + dhg.linear[0] * com[2]) / f_z
            self._zmp[1] += (dhg.angular[0] - dhg.linear[1] * com[2]) / f_z
        return self._zmp
