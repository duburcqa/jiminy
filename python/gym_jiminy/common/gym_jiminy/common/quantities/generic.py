"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
from functools import partial
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

import numpy as np

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)
import pinocchio as pin

from ..bases import InterfaceJiminyEnv, AbstractQuantity
from ..utils import fill, transforms_to_vector, matrix_to_rpy


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
                 parent: Optional[AbstractQuantity],
                 frame_name: str,
                 reference_frame: pin.ReferenceFrame = pin.LOCAL
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
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
        super().__init__(env, parent, requirements={})

        # Define specialize difference operator on SE3 Lie group
        self._se3_diff = partial(
            pin.LieGroup.difference,
            pin.liegroups.SE3())  # pylint: disable=no-member

        # Inverse step size
        self._inv_step_dt = 0.0

        # Pre-allocate memory to store current and previous frame pose vector
        self._xyzquat_prev, self._xyzquat = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for _ in range(2))

        # Define proxy to the current frame pose (translation, rotation matrix)
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

        # Refresh proxy to current frame pose
        frame_index = self.pinocchio_model.getFrameId(self.frame_name)
        transform = self.pinocchio_data.oMf[frame_index]
        self._pose = (transform.translation, transform.rotation)

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


@dataclass(unsafe_hash=True)
class _BatchEulerAnglesFrame(AbstractQuantity[Dict[str, np.ndarray]]):
    """Euler angles (Roll-Pitch-Yaw) representation of the orientation of all
    frames involved in quantities relying on it and are active since last reset
    of computation tracking if shared cache is available, its parent otherwise.

    This quantity only provides a performance benefit when managed by some
    `QuantityManager`. It is not supposed to be instantiated manually but use
    internally by `EulerAnglesFrame` as a requirement for vectorization of
    computations for all frames at once.

    The orientation of all frames is exposed to the user as a dictionary whose
    keys are the individual frame names. Internally, data are stored in batched
    2D contiguous array for efficiency. The first dimension are the 3 Euler
    angles (roll, pitch, yaw) components, while the second one are individual
    frames with the same ordering as 'self.frame_names'.

    The expected maximum speedup wrt computing Euler angles individually is
    about x15, which is achieved asymptotically for more than 100 frames. It is
    already x5 faster for 5 frames, x7 for 10 frames, and x9 for 20 frames.

    .. note::
        This quantity does not allow for specifying frames directly. There is
        no way to get the orientation of multiple frames at once for now.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[AbstractQuantity]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        """
        # Call base implementation
        super().__init__(env, parent, requirements={})

        # Initialize the ordered list of frame names
        self.frame_names: Set[str] = set()

        # Store all rotation matrices at once
        self._rot_mat_batch: np.ndarray = np.array([])

        # Define proxy for slices of the batch storing all rotation matrices
        self._rot_mat_views: List[np.ndarray] = []

        # Define proxy for the rotation matrices of all frames
        self._rot_mat_list: List[np.ndarray] = []

        # Store Roll-Pitch-Yaw of all frames at once
        self._rpy_batch: np.ndarray = np.array([])

        # Mapping from frame name to individual Roll-Pitch-Yaw slices
        self._rpy_map: Dict[str, np.ndarray] = {}

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Update the list of frame names based on its cache owner list.
        # Note that only active quantities are shared for efficiency. The list
        # of active quantity may change dynamically. Re-initializing the class
        # to take into account changes of the active set must be decided by
        # `EulerAnglesFrame`.
        assert isinstance(self.parent, EulerAnglesFrame)
        self.frame_names = {self.parent.frame_name}
        if self.cache:
            for owner in self.cache.owners:
                # We only consider active instances of `_BatchEulerAnglesFrame`
                # instead of their corresponding parent `EulerAnglesFrame`.
                # This is necessary because a derived quantity may feature
                # `_BatchEulerAnglesFrame` as a requirement without actually
                # relying on it depending on whether it is part of the optimal
                # computation path at the time being or not.
                if owner.is_active(any_cache_owner=False):
                    assert isinstance(owner.parent, EulerAnglesFrame)
                    self.frame_names.add(owner.parent.frame_name)

        # Re-allocate memory as the number of frames is not known in advance.
        # Note that Fortran memory layout (column-major) is used for speed up
        # because it preserves contiguity when copying frame data.
        nframes = len(self.frame_names)
        self._rot_mat_batch = np.zeros((3, 3, nframes), order='F')
        self._rpy_batch = np.zeros((3, nframes))

        # Refresh proxies
        self._rot_mat_views.clear()
        self._rot_mat_list.clear()
        for i, frame_name in enumerate(self.frame_names):
            frame_index = self.pinocchio_model.getFrameId(frame_name)
            rot_matrix = self.pinocchio_data.oMf[frame_index].rotation
            self._rot_mat_views.append(self._rot_mat_batch[..., i])
            self._rot_mat_list.append(rot_matrix)

        # Re-assign mapping from frame name to their corresponding data
        self._rpy_map = dict(zip(self.frame_names, self._rpy_batch.T))

    def refresh(self) -> Dict[str, np.ndarray]:
        # Copy all rotation matrices in contiguous buffer
        multi_array_copyto(self._rot_mat_views, self._rot_mat_list)

        # Convert all rotation matrices at once to Roll-Pitch-Yaw
        matrix_to_rpy(self._rot_mat_batch, self._rpy_batch)

        # Return proxy directly without copy
        return self._rpy_map


@dataclass(unsafe_hash=True)
class EulerAnglesFrame(AbstractQuantity[np.ndarray]):
    """Euler angles (Roll-Pitch-Yaw) representation of the orientation of a
    given frame in world reference frame at the end of an agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[AbstractQuantity],
                 frame_name: str,
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name

        # Call base implementation
        super().__init__(
            env, parent, requirements={"data": (_BatchEulerAnglesFrame, {})})

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation.
        # The quantity is now considered active at this point.
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            # Must reset the tracking for shared computation systematically,
            # just in case the optimal computation path has changed.
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        # Return a slice of batched data.
        # Note that mapping from frame name to frame index in batched data
        # cannot be pre-computed as it may changed dynamically.
        return self.data[self.frame_name]
