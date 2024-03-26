from functools import partial
from dataclasses import dataclass

import numpy as np

from jiminy_py.core import array_copyto
import pinocchio as pin

from ..bases import InterfaceJiminyEnv, AbstractQuantity
from ..utils import fill, transforms_to_vector


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
            kinematic_level: pin.KinematicLevel = pin.POSITION
            ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :para kinematic_level: Desired kinematic level, ie position, velocity
                               or acceleration.
        """
        # Backup some user argument(s)
        self.kinematic_level = kinematic_level

        # Call base implementation
        super().__init__(env, requirements={})

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
class AverageSpatialVelocityFrame(AbstractQuantity[np.ndarray]):
    """Average spatial velocity of a given frame at the end of an agent step.

    The average spatial velocity is obtained by finite difference. More
    precisely, it is defined here as the ratio of the geodesic distance in SE3
    Lie  group between the pose of the frame at the end of previous and current
    step over the time difference between them. Notably, under this definition,
    the linear average velocity jointly depends on rate of change of the
    translation and rotation of the frame, which may be undesirable in some
    cases. Alternatively, the double geodesic distance could be used instead to
    completely decouple the translation from the rotation.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    reference_frame: pin.ReferenceFrame
    """Whether the spatial velocity must be computed in local reference frame
    or re-aligned with world axes.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 frame_name: str,
                 reference_frame: pin.ReferenceFrame = pin.LOCAL
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param frame_name: Name of the frame on which to operate.
        :param reference_frame:
            Whether the spatial velocity must be computed in local reference
            frame (aka 'pin.LOCAL') or re-aligned with world axes (aka
            'pin.LOCAL_WORLD_ALIGNED').
        """
        # Make sure at requested reference frame is supported
        if reference_frame not in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED):
            raise ValueError("Reference frame must be either 'pin.LOCAL' or "
                             "'pin.LOCAL_WORLD_ALIGNED'.")

        # Backup some user argument(s)
        self.frame_name = frame_name
        self.reference_frame = reference_frame

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

        # Pre-allocate memory for the spatial velocity
        self._v_spatial: np.ndarray = np.zeros(6)

        # Reshape linear plus angular velocity vector to vectorize rotation
        self._v_lin_ang = np.reshape(self._v_spatial, (2, 3)).T

    def initialize(self) -> None:
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
        fill(self._v_spatial, 0)

    def refresh(self) -> np.ndarray:
        # Convert current transform to (XYZ, Quat) convention
        transforms_to_vector((self._pose,), self._xyzquat)

        # Compute average frame velocity in local frame since previous step
        self._v_spatial[:] = self._se3_diff(self._xyzquat_prev, self._xyzquat)
        self._v_spatial *= self._inv_step_dt

        # Translate local velocity to world frame
        if self.reference_frame == pin.LOCAL_WORLD_ALIGNED:
            # TODO: x2 speedup can be expected using `np.dot` with  `nb.jit`
            _, rot_mat = self._pose
            self._v_lin_ang[:] = rot_mat @ self._v_lin_ang

        # Backup current frame pose
        array_copyto(self._xyzquat_prev, self._xyzquat)

        return self._v_spatial
