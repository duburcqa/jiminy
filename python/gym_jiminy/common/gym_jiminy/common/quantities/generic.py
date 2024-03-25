from functools import partial
from dataclasses import dataclass

import numpy as np

from jiminy_py.core import array_copyto
import pinocchio as pin

from ..bases import InterfaceJiminyEnv, AbstractQuantity
from ..utils import fill, transforms_to_vector, quat_multiply


@dataclass(unsafe_hash=True)
class CenterOfMass(AbstractQuantity[np.ndarray]):
    """ TODO: Write documentation.
    """

    kinematic_level: pin.KinematicLevel
    """ TODO: Write documentation.
    """

    def __init__(
            self,
            env: InterfaceJiminyEnv,
            kinematic_level: pin.KinematicLevel = pin.KinematicLevel.POSITION
            ) -> None:
        """ TODO: Write documentation.
        """
        super().__init__(env, requirements={})
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
class AverageVelocityFrame(AbstractQuantity[np.ndarray]):
    """ TODO: Write documentation.
    """

    frame_name: str
    """ TODO: Write documentation.
    """

    reference_frame: pin.ReferenceFrame
    """ TODO: Write documentation.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 frame_name: str,
                 reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL
                 ) -> None:
        """ TODO: Write documentation.
        """
        # Backup user arguments
        self.frame_name = frame_name
        self.reference_frame = reference_frame

        # Make sure at requested reference frame is supported
        if reference_frame not in (pin.ReferenceFrame.LOCAL,
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
            raise ValueError(
                "Reference frame must be 'LOCAL' or 'LOCAL_WORLD_ALIGNED'.")

        # Call base implementation
        super().__init__(env, requirements={})

        # Define specialize difference operator on SE3 Lie group
        self._se3_diff = partial(pin.LieGroup.difference, pin.liegroups.SE3())

        # Inverse step size
        self._inv_step_dt = 0.0

        # Pre-allocate memory to store current and previous frame pose
        self._xyzquat_prev, self._xyzquat = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for _ in range(2))
        self._pose = (np.zeros(3), np.eye(3))

        # Pre-allocate memory for return value
        self._value: np.ndarray = np.zeros(6)

        # Reshape linear plus angular velocity vector to vectorize rotation
        self._v_lin_ang = np.reshape(self._value, (2, 3)).T

    def initialize(self) -> None:
        """ TODO: Write documentation.
        """
        # Call base implementation
        super().initialize()

        # Compute inverse step size
        self._inv_step_dt = 1.0 / self.env.step_dt

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
        self._value *= self._inv_step_dt

        # Translate local velocity to world frame
        if pin.ReferenceFrame.LOCAL_WORLD_ALIGNED:
            # TODO: x2 speedup can be expected using `np.dot` with  `nb.jit`
            _, rot_mat = self._pose
            self._v_lin_ang[:] = rot_mat @ self._v_lin_ang

        # Backup current frame pose
        array_copyto(self._xyzquat_prev, self._xyzquat)

        return self._value
