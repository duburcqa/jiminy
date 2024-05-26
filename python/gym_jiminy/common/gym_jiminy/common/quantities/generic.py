"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
import warnings
from functools import partial
from dataclasses import dataclass
from typing import (
    List, Dict, Set, Optional, Protocol, Sequence, runtime_checkable)

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    multi_array_copyto)
import pinocchio as pin

from ..bases import (
    InterfaceJiminyEnv, InterfaceQuantity, AbstractQuantity, QuantityEvalMode)
from ..utils import (
    fill, matrix_to_rpy, matrix_to_quat, quat_to_matrix,
    quat_interpolate_middle)

from .transform import StackedQuantity


@runtime_checkable
class FrameQuantity(Protocol):
    """Protocol that must be satisfied by all quantities associated with one
    particular frame.

    This protocol is used when aggregating individual frame-level quantities
    in a larger batch for computation vectorization on all frames at once.
    Intermediate quantities managing these batches will make sure that all
    their parents derive from one of the supported protocols, which includes
    this one.
    """
    frame_name: str


@runtime_checkable
class MultiFrameQuantity(Protocol):
    """Protocol that must be satisfied by all quantities associated with
    a particular set of frames for which the same batched intermediary
    quantities must be computed.

    This protocol is involved in automatic computation vectorization. See
    `FrameQuantity` documentation for details.
    """
    frame_names: Sequence[str]


@dataclass(unsafe_hash=True)
class _MultiFramesRotationMatrix(AbstractQuantity[np.ndarray]):
    """3D rotation matrix of the orientation of all frames involved in
    quantities relying on it and are active since last reset of computation
    tracking if shared cache is available, its parent otherwise.

    This quantity only provides a performance benefit when managed by some
    `QuantityManager`. It is not supposed to be instantiated manually but use
    as requirement of some other quantity for computation vectorization on all
    frames at once.

    The data associated with each frame is exposed to the user as a batched 3D
    contiguous array. The two first dimensions are rotation matrix elements,
    while the last one are individual frames with the same ordering as
    'self.frame_names'.

    .. note::
        This quantity does not allow for specifying frames directly. There is
        no way to get the orientation of multiple frames at once for now.
    """

    identifier: int
    """Uniquely identify its parent type.

    This implies that quantities specifying `_MultiFramesRotationMatrix` as a
    requirement will shared a unique batch with all the other ones having the
    same type but not the others. This is essential to provide data access as a
    batched ND contiguous array.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: InterfaceQuantity,
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a parent has been specified
        assert isinstance(parent, (FrameQuantity, MultiFrameQuantity))

        # Set unique identifier based on parent type
        self.identifier = hash(type(parent))

        # Call base implementation
        super().__init__(
            env, parent, requirements={}, mode=mode, auto_refresh=False)

        # Initialize the ordered list of frame names
        self.frame_names: Set[str] = set()

        # Store all rotation matrices at once
        self._rot_mat_batch: np.ndarray = np.array([])

        # Define proxy for slices of the batch storing all rotation matrices
        self._rot_mat_slices: List[np.ndarray] = []

        # Define proxy for the rotation matrices of all frames
        self._rot_mat_list: List[np.ndarray] = []

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Update the frame names based on the cache owners of this quantity.
        # Note that only active quantities are considered for efficiency, which
        # may change dynamically. Delegating this responsibility to cache
        # owners may be possible but difficult to implement because
        # `frame_names` must be cleared first before re-registering themselves,
        # just in case of optimal computation graph has changed, not only once
        # to avoid getting rid of quantities that just registered themselves.
        # Nevertheless, whenever re-initializing this quantity to take into
        # account changes of the active set must be decided by cache owners.
        assert isinstance(self.parent, (FrameQuantity, MultiFrameQuantity))
        if isinstance(self.parent, FrameQuantity):
            self.frame_names = {self.parent.frame_name}
        else:
            self.frame_names = set(self.parent.frame_names)
        if self.has_cache:
            for owner in self.cache.owners:
                # We only consider active `_MultiFramesEulerAngles` instances
                # instead of their parents. This is necessary because a derived
                # quantity may feature `_MultiFramesEulerAngles` as requirement
                # without actually relying on it depending on whether it is
                # part of the optimal computation path at that time.
                if owner.is_active(any_cache_owner=False):
                    assert isinstance(
                        owner.parent, (FrameQuantity, MultiFrameQuantity))
                    if isinstance(owner.parent, FrameQuantity):
                        self.frame_names.add(owner.parent.frame_name)
                    else:
                        self.frame_names.union(owner.parent.frame_names)

        # Re-allocate memory as the number of frames is not known in advance.
        # Note that Fortran memory layout (column-major) is used for speed up
        # because it preserves contiguity when copying frame data.
        nframes = len(self.frame_names)
        self._rot_mat_batch = np.zeros((3, 3, nframes), order='F')

        # Refresh proxies
        self._rot_mat_slices.clear()
        self._rot_mat_list.clear()
        for i, frame_name in enumerate(self.frame_names):
            frame_index = self.pinocchio_model.getFrameId(frame_name)
            rot_matrix = self.pinocchio_data.oMf[frame_index].rotation
            self._rot_mat_slices.append(self._rot_mat_batch[..., i])
            self._rot_mat_list.append(rot_matrix)

    def refresh(self) -> np.ndarray:
        # Copy all rotation matrices in contiguous buffer
        multi_array_copyto(self._rot_mat_slices, self._rot_mat_list)

        # Return proxy directly without copy
        return self._rot_mat_batch


@dataclass(unsafe_hash=True)
class _MultiFramesEulerAngles(InterfaceQuantity[Dict[str, np.ndarray]]):
    """Euler angles (Roll-Pitch-Yaw) of the orientation of all frames involved
    in quantities relying on it and are active since last reset of computation
    tracking if shared cache is available, its parent otherwise.

    It is not supposed to be instantiated manually but use internally by
    `FrameEulerAngles`. See `_MultiFramesRotationMatrix` documentation.

    The orientation of all frames is exposed to the user as a dictionary whose
    keys are the individual frame names. Internally, data are stored in batched
    2D contiguous array for efficiency. The first dimension gathers the 3 Euler
    angles (roll, pitch, yaw), while the second one are individual frames with
    the same ordering as 'self.frame_names'.

    The expected maximum speedup wrt computing Euler angles individually is
    about x15, which is achieved asymptotically for more than 100 frames. It is
    already x5 faster for 5 frames, x7 for 10 frames, and x9 for 20 frames.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: "FrameEulerAngles",
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: `FrameEulerAngles` instance from which this quantity is
                       a requirement.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a suitable parent has been provided
        assert isinstance(parent, FrameEulerAngles)

        # Backup some user argument(s)
        self.mode = mode

        # Initialize the ordered list of frame names.
        # Note that this must be done BEFORE calling base `__init__`, otherwise
        # `isinstance(..., (FrameQuantity, MultiFrameQuantity))` will fail.
        self.frame_names: Set[str] = set()

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                rot_mat_batch=(_MultiFramesRotationMatrix, dict(
                    mode=mode))),
            auto_refresh=False)

        # Store Roll-Pitch-Yaw of all frames at once
        self._rpy_batch: np.ndarray = np.array([])

        # Mapping from frame name to individual Roll-Pitch-Yaw slices
        self._rpy_map: Dict[str, np.ndarray] = {}

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Update the frame names based on the cache owners of this quantity
        assert isinstance(self.parent, FrameEulerAngles)
        self.frame_names = {self.parent.frame_name}
        if self.has_cache:
            for owner in self.cache.owners:
                if owner.is_active(any_cache_owner=False):
                    assert isinstance(owner.parent, FrameEulerAngles)
                    self.frame_names.add(owner.parent.frame_name)

        # Re-allocate memory as the number of frames is not known in advance
        nframes = len(self.frame_names)
        self._rpy_batch = np.zeros((3, nframes), order='F')

        # Re-assign mapping from frame name to their corresponding data
        self._rpy_map = dict(zip(self.frame_names, self._rpy_batch.T))

    def refresh(self) -> Dict[str, np.ndarray]:
        # Convert all rotation matrices at once to Roll-Pitch-Yaw
        matrix_to_rpy(self.rot_mat_batch, self._rpy_batch)

        # Return proxy directly without copy
        return self._rpy_map


@dataclass(unsafe_hash=True)
class FrameEulerAngles(InterfaceQuantity[np.ndarray]):
    """Euler angles (Roll-Pitch-Yaw) of the orientation of a given frame in
    world reference frame at the end of the agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_name: str,
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(data=(_MultiFramesEulerAngles, dict(mode=mode))),
            auto_refresh=False)

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation.
        # The quantity is now considered active at this point.
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            # Must reset the tracking for shared computation systematically,
            # just in case the optimal computation path has changed to the
            # point that relying on batched quantity is no longer relevant.
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        # Return a slice of batched data.
        # Note that mapping from frame name to frame index in batched data
        # cannot be pre-computed as it may changed dynamically.
        return self.data[self.frame_name]


@dataclass(unsafe_hash=True)
class _MultiFramesXYZQuat(AbstractQuantity[Dict[str, np.ndarray]]):
    """Vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of the
    transform of all frames involved in quantities relying on it and are active
    since last reset of computation tracking if shared cache is available, its
    parent otherwise.

    It is not supposed to be instantiated manually but use internally by
    `FrameXYZQuat`. See `_MultiFramesRotationMatrix` documentation.

    The transform of all frames is exposed to the user as a dictionary whose
    keys are the individual frame names. Internally, data are stored in batched
    2D contiguous array for efficiency. The first dimension gathers the 6
    components (X, Y, Z, QuatX, QuatY, QuatZ, QuatW), while the second one are
    individual frames with the same ordering as 'self.frame_names'.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: "FrameXYZQuat",
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: `FrameXYZQuat` instance from which this quantity
                       is a requirement.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a suitable parent has been provided
        assert isinstance(parent, FrameXYZQuat)

        # Initialize the ordered list of frame names
        self.frame_names: Set[str] = set()

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                rot_mat_batch=(_MultiFramesRotationMatrix, dict(
                    mode=mode))),
            mode=mode,
            auto_refresh=False)

        # Define proxy for slices of the batch storing all translation vectors
        self._translation_slices: List[np.ndarray] = []

        # Define proxy for the translation vectors of all frames
        self._translation_list: List[np.ndarray] = []

        # Store XYZQuat of all frames at once
        self._xyzquat_batch: np.ndarray = np.array([])

        # Mapping from frame name to individual XYZQuat slices
        self._xyzquat_map: Dict[str, np.ndarray] = {}

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Update the frame names based on the cache owners of this quantity
        assert isinstance(self.parent, FrameXYZQuat)
        self.frame_names = {self.parent.frame_name}
        if self.has_cache:
            for owner in self.cache.owners:
                if owner.is_active(any_cache_owner=False):
                    assert isinstance(owner.parent, FrameXYZQuat)
                    self.frame_names.add(owner.parent.frame_name)

        # Re-allocate memory as the number of frames is not known in advance
        nframes = len(self.frame_names)
        self._xyzquat_batch = np.zeros((7, nframes), order='F')

        # Refresh proxies
        self._translation_slices.clear()
        self._translation_list.clear()
        for i, frame_name in enumerate(self.frame_names):
            frame_index = self.pinocchio_model.getFrameId(frame_name)
            translation = self.pinocchio_data.oMf[frame_index].translation
            self._translation_slices.append(self._xyzquat_batch[:3, i])
            self._translation_list.append(translation)

        # Re-assign mapping from frame name to their corresponding data
        self._xyzquat_map = dict(zip(self.frame_names, self._xyzquat_batch.T))

    def refresh(self) -> Dict[str, np.ndarray]:
        # Copy all translations in contiguous buffer
        multi_array_copyto(self._translation_slices, self._translation_list)

        # Convert all rotation matrices at once to XYZQuat representation
        matrix_to_quat(self.rot_mat_batch, self._xyzquat_batch[-4:])

        # Return proxy directly without copy
        return self._xyzquat_map


@dataclass(unsafe_hash=True)
class FrameXYZQuat(InterfaceQuantity[np.ndarray]):
    """Vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of the
    transform of a given frame in world reference frame at the end of the
    agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_name: str,
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(data=(_MultiFramesXYZQuat, dict(mode=mode))),
            auto_refresh=False)

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        # Return a slice of batched data
        return self.data[self.frame_name]


@dataclass(unsafe_hash=True)
class AverageFrameSpatialVelocity(InterfaceQuantity[np.ndarray]):
    """Average spatial velocity of a given frame at the end of the agent step.

    The average spatial velocity is obtained by finite difference. More
    precisely, it is defined here as the ratio of the geodesic distance in SE3
    Lie Group between the pose of the frame at the end of previous and current
    step over the time difference between them. Notably, under this definition,
    the linear average velocity jointly depends on rate of change of the
    translation and rotation of the frame, which may be undesirable in some
    cases. Alternatively, the double geodesic distance could be used instead to
    completely decouple the translation from the rotation.

    .. note::
        The local frame for which the velocity is expressed is defined as the
        midpoint interpolation between the previous and current frame pose.
        This definition is arbitrary, in a sense that any other point for an
        interpolation ratio going from 0.0 (previous pose) to 1.0 (current
        pose) would be equally valid.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    reference_frame: pin.ReferenceFrame
    """Whether the spatial velocity must be computed in local reference frame
    or re-aligned with world axes.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_name: str,
                 *,
                 reference_frame: pin.ReferenceFrame = pin.LOCAL,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        :param reference_frame:
            Whether the spatial velocity must be computed in local reference
            frame (aka 'pin.LOCAL') or re-aligned with world axes (aka
            'pin.LOCAL_WORLD_ALIGNED').
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure at requested reference frame is supported
        if reference_frame not in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED):
            raise ValueError("Reference frame must be either 'pin.LOCAL' or "
                             "'pin.LOCAL_WORLD_ALIGNED'.")

        # Backup some user argument(s)
        self.frame_name = frame_name
        self.reference_frame = reference_frame
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(xyzquat_stack=(StackedQuantity, dict(
                quantity=(FrameXYZQuat, dict(
                    frame_name=frame_name, mode=mode)),
                num_stack=2))),
            auto_refresh=False)

        # Define specialize difference operator on SE3 Lie group
        self._se3_diff = partial(
            pin.LieGroup.difference,
            pin.liegroups.SE3())  # pylint: disable=no-member

        # Inverse step size
        self._inv_step_dt = 0.0

        # Allocate memory for the average quaternion and rotation matrix
        self._quat_mean = np.zeros(4)
        self._rot_mat_mean = np.eye(3)

        # Pre-allocate memory for the spatial velocity
        self._v_spatial: np.ndarray = np.zeros(6)

        # Reshape linear plus angular velocity vector to vectorize rotation
        self._v_lin_ang = np.reshape(self._v_spatial, (2, 3)).T

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Compute inverse step size
        self._inv_step_dt = 1.0 / self.env.step_dt

        # Re-initialize pre-allocated buffers
        fill(self._v_spatial, 0)

    def refresh(self) -> np.ndarray:
        # Fetch previous and current XYZQuat representation of frame transform.
        # It will raise an exception if not enough data is available at this
        # point. This should never occur in practice as it will be fine at
        # the end of the first step already, before the reward and termination
        # conditions are evaluated.
        xyzquat_prev, xyzquat = self.xyzquat_stack

        # Compute average frame velocity in local frame since previous step
        self._v_spatial[:] = self._se3_diff(xyzquat_prev, xyzquat)
        self._v_spatial *= self._inv_step_dt

        # Translate local velocity to world frame
        if self.reference_frame == pin.LOCAL_WORLD_ALIGNED:
            # Define world frame as the "middle" between prev and next pose.
            # The orientation difference has an effect on the translation
            # difference, but not the other way around. Here, we only care
            # about the middle rotation, so we can consider SO3 Lie Group
            # algebra instead of SE3.
            quat_interpolate_middle(
                xyzquat_prev[-4:], xyzquat[-4:], self._quat_mean)
            quat_to_matrix(self._quat_mean, self._rot_mat_mean)

            # TODO: x2 speedup can be expected using `np.dot` with `nb.jit`
            self._v_lin_ang[:] = self._rot_mat_mean @ self._v_lin_ang

        return self._v_spatial


@dataclass(unsafe_hash=True)
class ActuatedJointPositions(AbstractQuantity[np.ndarray]):
    """Concatenation of the current position of all the actuated joints
    of the robot.

    In practice, all actuated joints must be 1DoF for now. The principal angle
    is used in case of revolute unbounded revolute joints.

    .. warning::
        Revolute unbounded joints are not supported for now.

    .. warning::
        Data is extracted from the true configuration vector instead of using
        sensor data. As a result, this quantity is appropriate for computing
        reward components and termination conditions but must be avoided in
        observers and controllers.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements={},
            mode=mode,
            auto_refresh=False)

        # Mechanical joint position indices.
        # Note that it will only be used in last resort if it can be written as
        # a slice. Indeed, "fancy" indexing returns a copy of the original data
        # instead of a view, which requires fetching data at every refresh.
        self.position_indices: List[int] = []

        # Buffer storing mechanical joint positions
        self.data = np.array([])

        # Whether mechanical joint positions must be updated at every refresh
        self._must_refresh = False

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Refresh mechanical joint position indices
        self.position_indices.clear()
        for motor in self.env.robot.motors:
            joint_index = self.pinocchio_model.getJointId(motor.joint_name)
            joint = self.pinocchio_model.joints[joint_index]
            joint_type = jiminy.get_joint_type(joint)
            if joint_type == jiminy.JointModelType.ROTARY_UNBOUNDED:
                raise ValueError(
                    "Revolute unbounded joints are not supported for now.")
            self.position_indices += range(joint.idx_q, joint.idx_q + joint.nq)

        # Determine whether data can be extracted from state by reference
        position_first = min(self.position_indices)
        position_last = max(self.position_indices)
        self._must_refresh = True
        if self.mode == QuantityEvalMode.TRUE:
            try:
                if (np.array(self.position_indices) == np.arange(
                        position_first, position_last + 1)).all():
                    self._must_refresh = False
                else:
                    warnings.warn(
                        "Consider using the same ordering for motors and "
                        "joints for optimal performance.")
            except ValueError:
                pass

        # Try extracting mechanical joint positions by reference if possible
        if self._must_refresh:
            self.data = np.full((len(self.position_indices),), float("nan"))
        else:
            self.data = self.state.q[slice(position_first, position_last + 1)]

    def refresh(self) -> np.ndarray:
        # Update mechanical joint positions only if necessary
        if self._must_refresh:
            self.state.q.take(self.position_indices, None, self.data, "clip")

        return self.data
