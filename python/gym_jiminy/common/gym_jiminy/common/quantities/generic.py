"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
from copy import deepcopy
from collections import deque
from functools import partial
from dataclasses import dataclass
from typing import (
    List, Dict, Set, Any, Optional, Protocol, Sequence, Tuple, TypeVar, Union,
    Generic, Callable, runtime_checkable)
from typing_extensions import TypeAlias

import numpy as np

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    multi_array_copyto)
import pinocchio as pin

from ..bases import (
    InterfaceJiminyEnv, InterfaceQuantity, AbstractQuantity, QuantityCreator,
    QuantityEvalMode)
from ..utils import (
    fill, matrix_to_rpy, matrix_to_quat, quat_to_matrix,
    quat_interpolate_middle)


EllipsisType: TypeAlias = Any  # TODO: `EllipsisType` introduced in Python 3.10

ValueT = TypeVar('ValueT')
OtherValueT = TypeVar('OtherValueT')
YetAnotherValueT = TypeVar('YetAnotherValueT')


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
class _MultiFramesEulerAngles(AbstractQuantity[Dict[str, np.ndarray]]):
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
            mode=mode,
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
class StackedQuantity(InterfaceQuantity[Tuple[ValueT, ...]]):
    """Keep track of a given quantity over time by automatically stacking its
    value once per environment step since last reset.

    .. note::
        A new entry is added to the stack right before evaluating the reward
        and termination conditions. Internal simulation steps, observer and
        controller updates are ignored.
    """

    quantity: InterfaceQuantity
    """Base quantity whose value must be stacked over time since last reset.
    """

    num_stack: Optional[int]
    """Maximum number of values that keep in memory before starting to discard
    the oldest one (FIFO). None if unlimited.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[ValueT],
                 *,
                 num_stack: Optional[int] = None) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose values
                         must be stacked, plus all its constructor keyword-
                         arguments except environment 'env' and parent 'parent.
        :param num_stack: Maximum number of values that keep in memory before
                          starting to discard the oldest one (FIFO). None if
                          unlimited.
        """
        # Backup user arguments
        self.num_stack = num_stack

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(data=quantity),
                         auto_refresh=True)

        # Keep track of the quantity that must be stacked once instantiated
        self.quantity = self.requirements["data"]

        # Allocate deque buffer
        self._deque: deque = deque(maxlen=self.num_stack)

        # Keep track of the last time the quantity has been stacked
        self._num_steps_prev = -1

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Clear buffer
        self._deque.clear()

        # Reset step counter
        self._num_steps_prev = -1

    def refresh(self) -> Tuple[ValueT, ...]:
        # Append value to the queue only once per step and only if a simulation
        # is running. Note that data must be deep-copied to make sure it does
        # not get altered afterward.
        if self.env.is_simulation_running:
            num_steps = self.env.num_steps
            if num_steps != self._num_steps_prev:
                assert num_steps == self._num_steps_prev + 1
                self._deque.append(deepcopy(self.data))
                self._num_steps_prev += 1

        # Return the whole stack as a tuple to preserve the integrity of the
        # underlying container and make the API robust to internal changes.
        return tuple(self._deque)


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
class MaskedQuantity(InterfaceQuantity[np.ndarray]):
    """Extract elements from a given quantity whose value is a N-dimensional
    array along an axis.

    Elements will be extract by copy unless the indices of the elements to
    extract to be written equivalently by a slice (ie they are evenly spaced),
    and the array can be flattened while preserving memory contiguity if 'axis'
    is `None`.
    """

    quantity: InterfaceQuantity
    """Base quantity whose elements must be extracted.
    """

    indices: Tuple[int, ...]
    """Indices of the elements to extract.
    """

    axis: Optional[int]
    """Axis over which to extract elements. `None` to consider flattened array.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[np.ndarray],
                 key: Union[Sequence[int], Sequence[bool]],
                 *,
                 axis: Optional[int] = None) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose values
                         must be extracted, plus all its constructor keyword-
                         arguments except environment 'env' and parent 'parent.
        :param key: Sequence of indices or boolean mask that will be used to
                    extract elements from the quantity along one axis.
        :param axis: Axis over which to extract elements. `None` to consider
                     flattened array.
                     Optional: `None` by default.
        """
        # Check if indices or boolean mask has been provided
        if all(isinstance(e, bool) for e in key):
            key = tuple(np.flatnonzero(key))
        elif not all(isinstance(e, int) for e in key):
            raise ValueError(
                "Argument 'key' invalid. It must either be a boolean mask, or "
                "a sequence of indices.")

        # Backup user arguments
        self.indices = tuple(key)
        self.axis = axis

        # Make sure that at least one index must be extracted
        if not self.indices:
            raise ValueError(
                "No indices to extract from quantity. Data would be empty.")

        # Check if the indices are evenly spaced
        self._slices: Tuple[Union[slice, EllipsisType], ...] = ()
        stride: Optional[int] = None
        if len(self.indices) == 1:
            stride = 1
        if len(self.indices) > 1 and all(e >= 0 for e in self.indices):
            spacing = np.unique(np.diff(self.indices))
            if spacing.size == 1:
                stride = spacing[0]
        if stride is not None:
            slice_ = slice(self.indices[0], self.indices[-1] + 1, stride)
            if axis is None:
                self._slices = (slice_,)
            elif axis > 0:
                self._slices = (*((slice(None),) * axis), slice_)
            else:
                self._slices = (
                    Ellipsis, slice_, *((slice(None),) * (- axis - 1)))

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(data=quantity),
                         auto_refresh=False)

        # Keep track of the quantity from which data must be extracted
        self.quantity = self.requirements["data"]

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

    def refresh(self) -> np.ndarray:
        # Extract elements from quantity
        if not self._slices:
            # Note that `take` is faster than classical advanced indexing via
            # `operator[]` (`__getitem__`) because the latter is more generic.
            # Notably, `operator[]` supports boolean mask but `take` does not.
            return self.data.take(self.indices, axis=self.axis)
        if self.axis is None:
            # `reshape` must be used instead of `flat` to get a view that can
            # be sliced without copy.
            return self.data.reshape((-1,))[self._slices]
        return self.data[self._slices]


@dataclass(unsafe_hash=True)
class BinaryOpQuantity(InterfaceQuantity[ValueT],
                       Generic[ValueT, OtherValueT, YetAnotherValueT]):
    """Apply a given binary operator between two quantities.

    This quantity is mainly useful for computing the error between the value of
    a given quantity evaluated at the current simulation state and the state of
    at the current simulation time for the reference trajectory being selected.
    """

    quantity_left: InterfaceQuantity[OtherValueT]
    """Left-hand side quantity that will be forwarded to the binary operator.
    """

    quantity_right: InterfaceQuantity[YetAnotherValueT]
    """Right-hand side quantity that will be forwarded to the binary operator.
    """

    op: Callable[[OtherValueT, YetAnotherValueT], ValueT]
    """Callable taking right- and left-hand side quantities as input argument.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity_left: QuantityCreator[OtherValueT],
                 quantity_right: QuantityCreator[YetAnotherValueT],
                 op: Callable[[OtherValueT, YetAnotherValueT], ValueT]
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity_left: Tuple gathering the class of the quantity that
                              must be passed to left-hand side of the binary
                              operator, plus all its constructor keyword-
                              arguments except environment 'env' and parent
                              'parent.
        :param quantity_right: Quantity that must be passed to right-hand side
                               of the binary operator as a tuple
                               (class, keyword-arguments). See `quantity_left`
                               argument for details.
        :param op: Any callable taking the right- and left-hand side quantities
                   as input argument. For example `operator.sub` to compute the
                   difference.
        """
        # Backup some user argument(s)
        self.op = op

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                value_left=quantity_left, value_right=quantity_right),
            auto_refresh=False)

        # Keep track of the left- and right-hand side quantities for hashing
        self.quantity_left = self.requirements["value_left"]
        self.quantity_right = self.requirements["value_right"]

    def refresh(self) -> ValueT:
        return self.op(self.value_left, self.value_right)
