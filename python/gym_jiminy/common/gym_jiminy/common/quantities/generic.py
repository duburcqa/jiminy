import numpy as np
from attrs import define

from jiminy_py.simulator import Simulator
import pinocchio as pin

from ..bases import AbstractQuantity


@define(unsafe_hash=True)
class CenterOfMass(AbstractQuantity[np.ndarray]):
    """ TODO: Write documentation.
    """

    kinematic_level: pin.KinematicLevel
    """ TODO: Write documentation.
    """

    def __init__(
            self,
            simulator: Simulator,
            kinematic_level: pin.KinematicLevel = pin.KinematicLevel.POSITION
            ) -> None:
        """ TODO: Write documentation.
        """
        super().__init__(simulator, requirements={})
        self.kinematic_level = kinematic_level
        self._value: np.ndarray = np.array([])

    def initialize(self) -> None:
        """ TODO: Write documentation.
        """
        super().initialize()
        if self.kinematic_level == pin.KinematicLevel.POSITION:
            self.value = self.pinocchio_data.com[0]
        elif self.kinematic_level == pin.KinematicLevel.VELOCITY:
            self.value = self.pinocchio_data.vcom[0]
        else:
            self.value = self.pinocchio_data.acom[0]

    def refresh(self) -> np.ndarray:
        """ TODO: Write documentation.
        """
        # Jiminy does not compute the CoM acceleration automatically
        if self.kinematic_level == pin.KinematicLevel.ACCELERATION:
            pin.centerOfMass(self.pinocchio_model,
                             self.pinocchio_data,
                             self.kinematic_level)
        return self.value
