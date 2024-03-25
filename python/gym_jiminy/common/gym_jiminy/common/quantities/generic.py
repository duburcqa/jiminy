from dataclasses import dataclass

import numpy as np

from jiminy_py.core import array_copyto
from jiminy_py.simulator import Simulator
import pinocchio as pin

from ..bases import AbstractQuantity
from ..utils import fill, transforms_to_vector


@dataclass(unsafe_hash=True)
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


@dataclass(unsafe_hash=True)
class AverageFrameVelocity(AbstractQuantity[np.ndarray]):
    """ TODO: Write documentation.
    """

    frame_name: str
    """ TODO: Write documentation.
    """

    reference_frame: pin.ReferenceFrame
    """ TODO: Write documentation.
    """

    def __init__(self,
                 simulator: Simulator,
                 frame_name: str,
                 reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL
                 ) -> None:
        """ TODO: Write documentation.
        """
        # Backup user arguments
        self.frame_name = frame_name
        self.reference_frame = reference_frame

        # Call base implementation
        super().__init__(simulator, requirements={})

        # Define specialize difference operator on SE3 Lie group
        self._se3_diff = partial(pin.LieGroup.difference, pin.liegroups.SE3())

        # Pre-allocate memory to store current and previous frame pose
        self._xyzquat_prev, self._xyzquat = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for _ in range(2))
        self._pose = (np.zeros(3), np.eye(3))

        # Pre-allocate memory for return value
        self._value: np.ndarray = np.zeros(6)

    def initialize(self) -> None:
        """ TODO: Write documentation.
        """
        # Call base implementation
        super().initialize()

        # Extract proxy to current frame pose for efficiency
        frame_index = self.pinocchio_model.getFrameIdx(self.frame_name)
        oMf = self.pinocchio_data.oMf[frame_index]
        self._pose = (oMf.translation, oMf.rotation)

        # Re-initialize pre-allocated buffers
        transforms_to_vector((self._pose,), self._xyzquat)
        array_copyto(self._xyzquat_prev, self._xyzquat)
        fill(self._value, 0)

    def refresh(self) -> np.ndarray:
        """ TODO: Write documentation.
        """
        # Convert current transform to (XYZ, Quat) convention
        transforms_to_vector((self._pose,), self._xyzquat)

        # Compute average frame velocity in local frame since previous step
        self._value[:] = self._se3_diff(self._xyzquat_prev, self._xyzquat)
        self._value /= self.step_dt  # FIXME: `step_dt` not available

        # FIXME: `reference_frame` is ignored for now

        # Backup current frame pose
        array_copyto(self._xyzquat_prev, self._xyzquat)

        return self._value
