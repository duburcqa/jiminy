# pylint: disable=redefined-builtin
"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
import warnings
from enum import IntEnum
from functools import partial
from dataclasses import dataclass
from typing import (
    List, Dict, Optional, Protocol, Sequence, Tuple, Union, Callable,
    runtime_checkable)

import numpy as np
import numba as nb

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    multi_array_copyto)
import pinocchio as pin
import hppfcl as fcl

from ..bases import (
    InterfaceJiminyEnv, InterfaceQuantity, AbstractQuantity, StateQuantity,
    QuantityEvalMode)
from ..utils import (
    matrix_to_rpy, matrix_to_quat, quat_apply, remove_yaw_from_quat,
    quat_interpolate_middle)

from .transform import (
    StackedQuantity, MaskedQuantity, UnaryOpQuantity, BinaryOpQuantity)


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
    frame_names: Tuple[str, ...]


def aggregate_frame_names(quantity: InterfaceQuantity) -> Tuple[
        Tuple[str, ...],
        Dict[Union[str, Tuple[str, ...]], Union[int, Tuple[()], slice]]]:
    """Generate a sequence of frame names that contains all the sub-sequences
    specified by the parents of all the cache owners of a given quantity.

    Ideally, the generated sequence should be the shortest possible. Since
    finding the optimal sequence is a complex problem, a heuristic is used
    instead. It consists in aggregating first all multi-frame quantities
    sequentially after ordering them by decreasing length, followed by all
    single-frame quantities.

    .. note::
        Only active quantities are considered for best performance, which may
        change dynamically. Delegating this responsibility to cache owners may
        be possible but difficult to implement because `frame_names` must be
        cleared first before re-registering themselves, just in case of optimal
        computation graph has changed, not only once to avoid getting rid of
        quantities that just registered themselves. Nevertheless, whenever
        re-initializing this quantity to take into account changes of the
        active set must be decided by cache owners.

    :param quantity: Quantity whose parent implements either `FrameQuantity` or
                     `MultiFrameQuantity` protocol. All the parents of all its
                     cache owners must also implement one of these protocol.
    """
    # Make sure that parent quantity implement multi- or single-frame protocol
    assert isinstance(quantity.parent, (FrameQuantity, MultiFrameQuantity))
    quantities = (quantity.cache.owners if quantity.has_cache else (quantity,))

    # First, order all multi-frame quantities by decreasing length
    frame_names_chunks: List[Tuple[str, ...]] = []
    for owner in quantities:
        parent = owner.parent
        assert parent is not None
        if parent.is_active(any_cache_owner=False):
            if isinstance(parent, MultiFrameQuantity):
                frame_names_chunks.append(parent.frame_names)

    # Next, process ordered multi-frame quantities sequentially.
    # For each of them, we first check if its set of frames is completely
    # included in the current full set. If so, then there is nothing do not and
    # we can move to the next quantity. If not, then we check if a part of its
    # tail or head is contained at the beginning or end of the full set
    # respectively. If so, only the missing part is prepended or appended
    # respectively. If not, then the while set of frames is appended to the
    # current full set before moving to the next quantity.
    frame_names: List[str] = []
    frame_names_chunks = sorted(frame_names_chunks, key=len)[::-1]
    for frame_names_ in map(list, frame_names_chunks):
        nframes, nframes_ = map(len, (frame_names, frame_names_))
        for i in range(nframes - nframes_ + 1):
            # Check if the sub-chain is completely included in the
            # current full set.
            if frame_names_ == frame_names[i:(i + nframes_)]:
                break
        else:
            for i in range(1, nframes_ + 1):
                # Check if part of the frame names matches with the
                # tail of the current full set. If so, append the
                # disjoint head only.
                if (frame_names[(nframes - nframes_ + i):] ==
                        frame_names_[:(nframes_ - i)]):
                    frame_names += frame_names_[(nframes_ - i):]
                    break
                # Check if part of the frame names matches with the
                # head of the current full set. If so, prepend the
                # disjoint tail only.
                if frame_names[:(nframes_ - i)] == frame_names_[i:]:
                    frame_names = frame_names_[:i] + frame_names
                    break

    # Finally, loop over all single-frame quantities.
    # If a frame name is missing, then it is appended to the current full set.
    # Otherwise, we just move to the next quantity.
    frame_name_chunks: List[str] = []
    for owner in quantities:
        parent = owner.parent
        assert parent is not None
        if parent.is_active(any_cache_owner=False):
            if isinstance(parent, FrameQuantity):
                frame_name_chunks.append(parent.frame_name)
                frame_name = frame_name_chunks[-1]
                if frame_name not in frame_names:
                    frame_names.append(frame_name)
    frame_names = tuple(frame_names)

    # Compute mapping from frame names to their corresponding indices in the
    # generated sequence of frame names.
    # The indices are stored as a slice for non-empty multi-frame quantities,
    # as an empty tuple for empty multi-frame quantities, or as an integer for
    # single-frame quantities.
    frame_slices: Dict[
        Union[str, Tuple[str, ...]], Union[int, Tuple[()], slice]] = {}
    nframes = len(frame_names)
    for frame_names_ in frame_names_chunks:
        if frame_names_ in frame_slices:
            continue
        if not frame_names_:
            frame_slices[frame_names_] = ()
            continue
        nframes_ = len(frame_names_)
        for i in range(nframes - nframes_ + 1):
            if frame_names_ == frame_names[i:(i + nframes_)]:
                break
        frame_slices[frame_names_] = slice(i, i + nframes_)
    for frame_name in frame_name_chunks:
        if frame_name in frame_slices:
            continue
        frame_slices[frame_name] = frame_names.index(frame_name)

    return frame_names, frame_slices


@dataclass(unsafe_hash=True)
class _BatchedFramesRotationMatrix(
        AbstractQuantity[Dict[Union[str, Tuple[str, ...]], np.ndarray]]):
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

        # Call base implementation
        super().__init__(
            env, parent, requirements={}, mode=mode, auto_refresh=False)

        # Initialize the ordered list of frame names
        self.frame_names: Tuple[str, ...] = ()

        # Store all rotation matrices at once
        self._rot_mat_batch: np.ndarray = np.array([])

        # Define proxy for views of the batch storing all rotation matrices
        self._rot_mat_views: List[np.ndarray] = []

        # Define proxy for the rotation matrices of all frames
        self._rot_mat_list: List[np.ndarray] = []

        # Mapping from frame names to views of batched rotation matrices
        self._rot_mat_map: Dict[Union[str, Tuple[str, ...]], np.ndarray] = {}

    def initialize(self) -> None:
        # Deactivate all cache owners first since only one is tracking frames
        for quantity in (self.cache.owners if self.has_cache else (self,)):
            quantity._is_active = False

        # Call base implementation
        super().initialize()

        # Update the frame names based on the cache owners of this quantity
        self.frame_names, frame_slices = aggregate_frame_names(self)

        # Re-allocate memory as the number of frames is not known in advance.
        # Note that Fortran memory layout (column-major) is used for speed up
        # because it preserves contiguity when copying frame data.
        # Anyway, C memory layout (row-major) does not make sense in this case
        # since chunks of columns are systematically extracted, which means
        # that the returned array would NEVER be contiguous.
        nframes = len(self.frame_names)
        self._rot_mat_batch = np.zeros((3, 3, nframes), order='F')

        # Refresh proxies
        self._rot_mat_views.clear()
        self._rot_mat_list.clear()
        for i, frame_name in enumerate(self.frame_names):
            frame_index = self.pinocchio_model.getFrameId(frame_name)
            rot_matrix = self.pinocchio_data.oMf[frame_index].rotation
            self._rot_mat_views.append(self._rot_mat_batch[..., i])
            self._rot_mat_list.append(rot_matrix)

        # Re-assign mapping from frame names to their corresponding data
        self._rot_mat_map = {
            key: self._rot_mat_batch[:, :, frame_slice]
            for key, frame_slice in frame_slices.items()}

    def refresh(self) -> Dict[Union[str, Tuple[str, ...]], np.ndarray]:
        # Copy all rotation matrices in contiguous buffer
        multi_array_copyto(self._rot_mat_views, self._rot_mat_list)

        # Return proxy directly without copy
        return self._rot_mat_map


class OrientationType(IntEnum):
    """Specify the desired vector representation of the frame orientations.
    """

    MATRIX = 0
    """3D rotation matrix.
    """

    EULER = 1
    """Euler angles (Roll, Pitch, Yaw).
    """

    QUATERNION = 2
    """Quaternion coordinates (QuatX, QuatY, QuatZ, QuatW).
    """

    ANGLE_AXIS = 3
    """Angle-Axis representation (theta * AxisX, theta * AxisY, theta * AxisZ).
    """


# Define proxies for fast lookup
_MATRIX, _EULER, _QUATERNION, _ANGLE_AXIS = (  # pylint: disable=invalid-name
    OrientationType)


@dataclass(unsafe_hash=True)
class _BatchedFramesOrientation(
        InterfaceQuantity[Dict[Union[str, Tuple[str, ...]], np.ndarray]]):
    """Vector representation of the orientation in world reference frame of all
    frames involved in quantities relying on it and are active since last reset
    of computation tracking if shared cache is available, its parent otherwise.

    The vector representation of the orientation of all the frames are stacked
    in a single contiguous N-dimensional array whose last dimension corresponds
    to the individual frames.

    The orientation of all frames is exposed to the user as a dictionary whose
    keys are the individual frame names. Internally, data are stored in batched
    2D contiguous array for efficiency. The first dimension gathers the 3 Euler
    angles (roll, pitch, yaw), while the second one are individual frames with
    the same ordering as 'self.frame_names'.

    This quantity is used internally by `FrameOrientation`. It is not supposed
    to be instantiated manually. See `_BatchedFramesRotationMatrix`
    documentation for details.

    In the particular case of Euler angle representation, the expected maximum
    speedup wrt computing Euler angles individually is about x15, which is
    achieved asymptotically for more than 100 frames. Still, it is already x5
    faster for 5 frames, x7 for 10 frames, and x9 for 20 frames.
    """

    type: OrientationType
    """Selected vector representation of the orientation for all frames.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Union["FrameOrientation", "MultiFrameOrientation"],
                 type: OrientationType,
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: `FrameOrientation` or `MultiFrameOrientation` instance
                       from which this quantity is a requirement.
        :param type: Desired vector representation of the orientation for all
                     frames. Note that `OrientationType.ANGLE_AXIS` is not
                     supported for now.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a suitable parent has been provided
        assert isinstance(parent, (FrameOrientation, MultiFrameOrientation))

        # Make sure that the specified orientation representation is supported
        if type not in (OrientationType.MATRIX,
                        OrientationType.EULER,
                        OrientationType.QUATERNION):
            raise ValueError(
                "This quantity only supports orientation representations "
                "'MATRIX', 'EULER', and 'QUATERNION'.")

        # Backup some user argument(s)
        self.type = type
        self.mode = mode

        # Initialize the ordered list of frame names.
        # Note that this must be done BEFORE calling base `__init__`, otherwise
        # `isinstance(..., (FrameQuantity, MultiFrameQuantity))` will fail.
        self.frame_names: Tuple[str, ...] = ()

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                rot_mat_map=(_BatchedFramesRotationMatrix, dict(mode=mode))),
            auto_refresh=False)

        # Mapping from frame names managed by this specific instance to their
        # corresponding indices in the generated sequence of frame names.
        self._frame_slices: Tuple[Tuple[
            Union[str, Tuple[str, ...]], Union[int, Tuple[()], slice]], ...
            ] = ()

        # Store the representation of the orientation of all frames at once
        self._data_batch: np.ndarray = np.array([])

        # Mapping from chunks of frame names to vector representation views
        self._data_map: Dict[Union[str, Tuple[str, ...]], np.ndarray] = {}

    def initialize(self) -> None:
        # Deactivate all cache owners first since only one is tracking frames
        for quantity in (self.cache.owners if self.has_cache else (self,)):
            quantity._is_active = False

        # Call base implementation
        super().initialize()

        # Update the frame names based on the cache owners of this quantity
        self.frame_names, frame_slices_map = aggregate_frame_names(self)

        # Re-assign mapping of chunk of frame names being managed
        self._frame_slices = tuple(frame_slices_map.items())

        # Re-allocate memory as the number of frames is not known in advance
        nframes = len(self.frame_names)
        if self.type in (OrientationType.EULER, OrientationType.ANGLE_AXIS):
            self._data_batch = np.zeros((3, nframes), order='F')
        elif self.type == OrientationType.QUATERNION:
            self._data_batch = np.zeros((4, nframes), order='F')

        # Re-assign mapping from chunks of frame names to corresponding data
        if self.type is not OrientationType.MATRIX:
            self._data_map = {
                key: self._data_batch[..., frame_slice]
                for key, frame_slice in frame_slices_map.items()}

    def refresh(self) -> Dict[Union[str, Tuple[str, ...]], np.ndarray]:
        # Get the complete batch of rotation matrices managed by this instance
        value = self.rot_mat_map.get()
        rot_mat_batch = value[self.frame_names]

        # Convert all rotation matrices at once to the desired representation
        if self.type == _EULER:
            matrix_to_rpy(rot_mat_batch, self._data_batch)
        elif self.type == _QUATERNION:
            matrix_to_quat(rot_mat_batch, self._data_batch)
        else:
            # Slice data.
            # Note that it cannot be pre-computed once and for all because
            # the batched data reference may changed dynamically.
            self._data_map = {
                key: rot_mat_batch[..., frame_slice]
                for key, frame_slice in self._frame_slices}

        # Return proxy directly without copy
        return self._data_map


@dataclass(unsafe_hash=True)
class FrameOrientation(InterfaceQuantity[np.ndarray]):
    """Vector representation of the orientation of a given frame in world
    reference frame at the end of the agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    type: OrientationType
    """Desired vector representation of the orientation.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
                 type: OrientationType = OrientationType.MATRIX,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        :param type: Desired vector representation of the orientation.
                     Optional: 'OrientationType.MATRIX' by default.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.type = type
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(_BatchedFramesOrientation, dict(
                    type=type,
                    mode=mode))),
            auto_refresh=False)

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            # Must reset the tracking for shared computation systematically,
            # just in case the optimal computation path has changed to the
            # point that relying on batched quantity is no longer relevant.
            self.data.reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        value = self.data.get()
        return value[self.frame_name]


@dataclass(unsafe_hash=True)
class MultiFrameOrientation(InterfaceQuantity[np.ndarray]):
    """Vector representation of the orientation of a given set of frames in
    world reference frame at the end of the agent step.

    The vector representation of the orientation of all the frames are stacked
    in a single contiguous N-dimensional array whose last dimension corresponds
    to the individual frames. See `_BatchedFramesOrientation` documentation for
    details.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames on which to operate.
    """

    type: OrientationType
    """Selected vector representation of the orientation for all frames.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Sequence[str],
                 *,
                 type: OrientationType = OrientationType.MATRIX,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_names: Name of the frames on which to operate.
        :param type: Desired vector representation of the orientation for all
                     frames.
                     Optional: 'OrientationType.MATRIX' by default.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure that the user did not pass a single frame name
        assert not isinstance(frame_names, str)

        # Backup some user argument(s)
        self.frame_names = tuple(frame_names)
        self.type = type
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(_BatchedFramesOrientation, dict(
                    type=type,
                    mode=mode))),
            auto_refresh=False)

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation.
        # The quantity is now considered active at this point.
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            self.data.reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        # Return a slice of batched data.
        # Note that mapping from frame names to frame index in batched data
        # cannot be pre-computed as it may changed dynamically.
        value = self.data.get()
        return value[self.frame_names]


@dataclass(unsafe_hash=True)
class _BatchedFramesPosition(
        AbstractQuantity[Dict[Union[str, Tuple[str, ...]], np.ndarray]]):
    """Position vector (X, Y, Z) of all frames involved in quantities relying
    on it and are active since last reset of computation tracking if shared
    cache is available, its parent otherwise.

    It is not supposed to be instantiated manually but use internally by
    `FramePosition`. See `_BatchedFramesRotationMatrix` documentation.

    The positions of all frames are exposed to the user as a dictionary whose
    keys are the individual frame names and/or set of frame names as a tuple.
    Internally, data are stored in batched 2D contiguous array for efficiency.
    The first dimension gathers the 3 components (X, Y, Z), while the second
    one are individual frames with the same ordering as 'self.frame_names'.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Union["FramePosition", "MultiFramePosition"],
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: `FramePosition` or `MultiFramePosition` instance from
                       which this quantity is a requirement.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a suitable parent has been provided
        assert isinstance(parent, (FramePosition, MultiFramePosition))

        # Initialize the ordered list of frame names
        self.frame_names: Tuple[str, ...] = ()

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements={},
            mode=mode,
            auto_refresh=False)

        # Define proxy for the position vectors of all frames
        self._pos_refs: List[np.ndarray] = []

        # Store the position of all frames at once
        self._pos_batch: np.ndarray = np.array([])

        # Define proxy for views of the batch storing all translation vectors
        self._pos_views: List[np.ndarray] = []

        # Mapping from chunks of frame names to individual position views
        self._pos_map: Dict[Union[str, Tuple[str, ...]], np.ndarray] = {}

    def initialize(self) -> None:
        # Deactivate all cache owners first since only one is tracking frames
        for quantity in (self.cache.owners if self.has_cache else (self,)):
            quantity._is_active = False

        # Call base implementation
        super().initialize()

        # Update the frame names based on the cache owners of this quantity
        self.frame_names, frame_slices = aggregate_frame_names(self)

        # Re-allocate memory as the number of frames is not known in advance
        nframes = len(self.frame_names)
        self._pos_batch = np.zeros((3, nframes), order='F')

        # Refresh proxies
        self._pos_views.clear()
        self._pos_refs.clear()
        for i, frame_name in enumerate(self.frame_names):
            frame_index = self.pinocchio_model.getFrameId(frame_name)
            translation = self.pinocchio_data.oMf[frame_index].translation
            self._pos_views.append(self._pos_batch[:, i])
            self._pos_refs.append(translation)

        # Re-assign mapping from frame names to their corresponding data
        self._pos_map = {
            key: self._pos_batch[:, frame_slice]
            for key, frame_slice in frame_slices.items()}

    def refresh(self) -> Dict[Union[str, Tuple[str, ...]], np.ndarray]:
        # Copy all translations in contiguous buffer
        multi_array_copyto(self._pos_views, self._pos_refs)

        # Return proxy directly without copy
        return self._pos_map


@dataclass(unsafe_hash=True)
class FramePosition(InterfaceQuantity[np.ndarray]):
    """Position vector (X, Y, Z) of a given frame in world reference frame at
    the end of the agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(_BatchedFramesPosition, dict(mode=mode))),
            auto_refresh=False)

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            self.data.reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        value = self.data.get()
        return value[self.frame_name]


@dataclass(unsafe_hash=True)
class MultiFramePosition(InterfaceQuantity[np.ndarray]):
    """Position vector (X, Y, Z) of a given set of frames in world reference
    frame at the end of the agent step.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Sequence[str],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frames on which to operate.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure that the user did not pass a single frame name
        assert not isinstance(frame_names, str)

        # Backup some user argument(s)
        self.frame_names = tuple(frame_names)
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(_BatchedFramesPosition, dict(mode=mode))),
            auto_refresh=False)

    def initialize(self) -> None:
        # Check if the quantity is already active
        was_active = self._is_active

        # Call base implementation
        super().initialize()

        # Force re-initializing shared data if the active set has changed
        if not was_active:
            self.data.reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        value = self.data.get()
        return value[self.frame_names]


@dataclass(unsafe_hash=True)
class FrameXYZQuat(InterfaceQuantity[np.ndarray]):
    """Spatial vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of
    the transform of a given frame in world reference frame at the end of the
    agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                position=(FramePosition, dict(
                    frame_name=frame_name,
                    mode=mode)),
                quat=(FrameOrientation, dict(
                    frame_name=frame_name,
                    type=OrientationType.QUATERNION,
                    mode=mode))),
            auto_refresh=False)

        # Pre-allocate memory for storing the pose XYZQuat of all frames
        self._xyzquat = np.zeros((7,))

        # Define position and orientation memory views for fast assignment
        self._xyzquat_views = (self._xyzquat[:3], self._xyzquat[-4:])

    def refresh(self) -> np.ndarray:
        # Compute the position and orientation of all frames at once
        xyz_quat = (self.position.get(), self.quat.get())

        # Copy data in contiguous buffer
        multi_array_copyto(self._xyzquat_views, xyz_quat)

        return self._xyzquat


@dataclass(unsafe_hash=True)
class MultiFrameXYZQuat(InterfaceQuantity[np.ndarray]):
    """Spatial vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of
    the transform of a given set of frames in world reference frame at the end
    of the agent step.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Sequence[str],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frames on which to operate.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure that the user did not pass a single frame name
        assert not isinstance(frame_names, str)

        # Backup some user argument(s)
        self.frame_names = tuple(frame_names)
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                positions=(MultiFramePosition, dict(
                    frame_names=frame_names,
                    mode=mode)),
                quats=(MultiFrameOrientation, dict(
                    frame_names=frame_names,
                    type=OrientationType.QUATERNION,
                    mode=mode))),
            auto_refresh=False)

        # Pre-allocate memory for storing the pose XYZQuat of all frames
        self._xyzquats = np.zeros((7, len(frame_names)), order='C')

        # Define position and orientation memory views for fast assignment
        self._xyzquats_views = (self._xyzquats[:3], self._xyzquats[-4:])

    def refresh(self) -> np.ndarray:
        # Compute the position and orientation of all frames at once
        xyz_quat_batch = (self.positions.get(), self.quats.get())

        # Copy data in contiguous buffer
        multi_array_copyto(self._xyzquats_views, xyz_quat_batch)

        return self._xyzquats


@dataclass(unsafe_hash=True)
class MultiFrameMeanXYZQuat(InterfaceQuantity[np.ndarray]):
    """Spatial vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of
    the average transform of a given set of frames in world reference frame at
    the end of the agent step.

    Broadly speaking, the average is defined as the value minimizing the mean
    error wrt every individual elements, considering some distance metric. In
    this case, the average position (X, Y, Z) and orientation as a quaternion
    vector (QuatX, QuatY, QuatZ, QuatW) are computed separately (double
    geodesic). It has the advantage to be much easier to compute, and to
    decouple the translation from the rotation, which is desirable when
    defining reward components weighting differently position or orientation
    errors. See `quaternion_average` for details about the distance metric
    being used to compute the average orientation.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Sequence[str],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frames on which to operate.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure that the user did not pass a single frame name
        assert not isinstance(frame_names, str)

        # Backup some user argument(s)
        self.frame_names = tuple(frame_names)
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                positions=(MultiFramePosition, dict(
                    frame_names=frame_names,
                    mode=mode)),
                quats=(MultiFrameOrientation, dict(
                    frame_names=frame_names,
                    type=OrientationType.QUATERNION,
                    mode=mode))),
            auto_refresh=False)

        # Jit-able method specialization of `np.mean` for `axis=-1`
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def position_average(value: np.ndarray, out: np.ndarray) -> None:
            """Compute the mean of an array over its last axis only.

            :param value: N-dimensional array from which the last axis will be
                          reduced.
            :param out: Pre-allocated array in which to store the result.
            """
            out[:] = np.sum(value, -1) / value.shape[-1]

        self._position_average = position_average

        # Jit-able specialization of `quat_average` for 2D matrices
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def quat_average_2d(quat: np.ndarray, out: np.ndarray) -> None:
            """Compute the average of a batch of quaternions [qx, qy, qz, qw].

            .. note::
                Jit-able specialization of `quat_average` for 2D matrices, with
                further optimization for the special case where there is only 2
                quaternions.

            :param quat: N-dimensional (N >= 2) array whose first dimension
                         gathers the 4 quaternion coordinates [qx, qy, qz, qw].
            :param out: Pre-allocated array in which to store the result.
            """
            num_quats = quat.shape[1]
            if num_quats == 1:
                out[:] = quat
            elif num_quats == 2:
                quat_interpolate_middle(quat[:, 0], quat[:, 1], out)
            else:
                _, eigvec = np.linalg.eigh(quat @ quat.T)
                out[:] = eigvec[..., -1]

        self._quat_average = quat_average_2d

        # Pre-allocate memory for the mean for mean pose vector XYZQuat
        self._xyzquat_mean = np.zeros((7,))

        # Define position and orientation proxies for fast access
        self._position_mean_view = self._xyzquat_mean[:3]
        self._quat_mean_view = self._xyzquat_mean[3:]

    def refresh(self) -> np.ndarray:
        # Compute the mean translation
        self._position_average(self.positions.get(), self._position_mean_view)

        # Compute the mean quaternion
        self._quat_average(self.quats.get(), self._quat_mean_view)

        return self._xyzquat_mean


@dataclass(unsafe_hash=True)
class MultiFrameCollisionDetection(InterfaceQuantity[bool]):
    """Check if some geometry objects are colliding with each other.

    It takes into account some safety margins by which their volume will be
    inflated / deflated.

    .. note::
        Jiminy enforces all collision geometries to be either primitive shapes
        or convex polyhedra for efficiency. In practice, tf meshes where
        specified in the original URDF file, then they will be converted into
        their respective convex hull.
    """

    frame_names: Tuple[str, ...]
    """Name of the bodies of the robot to consider for collision detection.

    All the geometry objects sharing with them the same parent joint will be
    taking into account.
    """

    security_margin: float
    """Signed distance below which a pair of geometry objects is stated in
    collision.

    This can be interpreted as inflating or deflating the geometry objects by
    the safety margin depending on whether it is positive or negative
    respectively. Therefore, the actual geometry objects do no have to be in
    contact to be stated in collision if the satefy margin is positive. On the
    contrary, the penetration depth must be large enough if the security margin
    is positive.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Sequence[str],
                 *,
                 security_margin: float = 0.0) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_names: Name of the bodies of the robot to consider for
                            collision detection. All the geometry objects
                            sharing with them the same parent joint will be
                            taking into account.
        :param security_margin: Signed distance below which a pair of geometry
                                objects is stated in collision.
                                Optional: 0.0 by default.
        """
        # Backup some user-arguments
        self.frame_names = tuple(frame_names)
        self.security_margin = security_margin

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements={},
            auto_refresh=False)

        # Initialize a broadphase manager for each collision group
        self._collision_groups = [
            fcl.DynamicAABBTreeCollisionManager() for _ in frame_names]

        # Initialize pair-wise collision requests between groups of bodies
        self._requests: List[Tuple[
            fcl.BroadPhaseCollisionManager,
            fcl.BroadPhaseCollisionManager,
            fcl.CollisionCallBackBase]] = []
        for i in range(len(frame_names)):
            for j in range(i + 1, len(frame_names)):
                manager_1 = self._collision_groups[i]
                manager_2 = self._collision_groups[j]
                callback = fcl.CollisionCallBackDefault()
                request: fcl.CollisionRequest = (
                    callback.data.request)  # pylint: disable=no-member
                request.gjk_initial_guess = jiminy.GJKInitialGuess.CachedGuess
                # request.gjk_variant = fcl.GJKVariant.NesterovAcceleration
                # request.break_distance = 0.1
                request.gjk_tolerance = 1e-6
                request.distance_upper_bound = 1e-6
                request.num_max_contacts = 1
                request.security_margin = security_margin
                self._requests.append((manager_1, manager_2, callback))

        # Store callable responsible to updating transform of colision objects
        self._transform_updates: List[Callable[[], None]] = []

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Define robot proxy for convenience
        robot = self.env.robot

        # Clear all collision managers
        for manager in self._collision_groups:
            manager.clear()

        # Get the list of parent joint indices mapping
        frame_indices_map: Dict[int, int] = {}
        for i, frame_name in enumerate(self.frame_names):
            frame_index = robot.pinocchio_model.getFrameId(frame_name)
            frame = robot.pinocchio_model.frames[frame_index]
            frame_indices_map[frame.parent] = i

        # Add collision objects to their corresponding manager
        self._transform_updates.clear()
        for i, geom in enumerate(robot.collision_model.geometryObjects):
            j = frame_indices_map.get(geom.parentJoint)
            if j is not None:
                obj = fcl.CollisionObject(geom.geometry)
                self._collision_groups[j].registerObject(obj)
                pose = robot.collision_data.oMg[i]
                translation, rotation = pose.translation, pose.rotation
                self._transform_updates += (
                    partial(obj.setTranslation, translation),
                    partial(obj.setRotation, rotation))

        # Initialize collision detection facilities
        for manager in self._collision_groups:
            manager.setup()

    def refresh(self) -> bool:
        # Update collision object placement
        for transform_update in self._transform_updates:
            transform_update()

        # Update all collision managers
        # for manager in self._collision_groups:
        #     manager.update()

        # Check collision for all candidate pairs
        for manager_1, manager_2, callback in self._requests:
            manager_1.collide(manager_2, callback)
            if callback.data.result.isCollision():
                return True
        return False


@dataclass(unsafe_hash=True)
class _DifferenceFrameXYZQuat(InterfaceQuantity[np.ndarray]):
    """Motion vector representation (VX, VY, VZ, WX, WY, WZ) of the finite
    difference between the pose of a given frame at the end of previous and
    current agent steps.

    The finite difference is defined here as the geodesic distance in SE3 Lie
    Group. Under this definition, the rate of change of the translation depends
    on rate of change of the orientation of the frame, which may be undesirable
    in some cases. Alternatively, the double geodesic distance could be used
    instead to completely decouple the position from the orientation.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
        :param reference_frame:
            Whether the spatial velocity must be computed in local reference
            frame (aka 'pin.LOCAL') or re-aligned with world axes (aka
            'pin.LOCAL_WORLD_ALIGNED').
            Optional: 'pinocchio.ReferenceFrame.LOCAL' by default.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                xyzquat_stack=(StackedQuantity, dict(
                    quantity=(FrameXYZQuat, dict(
                        frame_name=frame_name,
                        mode=mode)),
                    max_stack=2))),
            auto_refresh=False)

        # Define specialize difference operator on SE3 Lie group
        self._difference = (
            pin.liegroups.SE3().difference)  # pylint: disable=no-member

        # Pre-allocate memory to store the pose difference
        self._data: np.ndarray = np.zeros(6)

    def refresh(self) -> np.ndarray:
        # Fetch previous and current XYZQuat representation of frame transform.
        # It will raise an exception if not enough data is available at this
        # point. This should never occur in practice as it will be fine at
        # the end of the first step already, before the reward and termination
        # conditions are evaluated.
        xyzquat_prev, xyzquat = self.xyzquat_stack.get()

        # Compute average frame velocity in local frame since previous step
        self._data[:] = self._difference(xyzquat_prev, xyzquat)

        return self._data


@dataclass(unsafe_hash=True)
class AverageFrameXYZQuat(InterfaceQuantity[np.ndarray]):
    """Spatial vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of
    the average pose of a given frame over the whole agent step.

    The average frame pose is obtained by integration of the average velocity
    over the whole agent step, backward in time from the state at the end of
    the step to the midpoint. See `_DifferenceFrameXYZQuat` documentation for
    details.

    .. note::
        There is a coupling between the rate of change of the orientation over
        the agent step and the position of the midpoint. Depending on the
        application, it may be desirable to decouple the translation from the
        rotation completely by performing computation on the Cartesian Product
        of the 3D Euclidean space R3 times the Special Orthogonal Group SO3.
        The resulting distance metric is referred to as double geodesic and
        does not correspond to the actual shortest path anymore.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                xyzquat_next=(FrameXYZQuat, dict(
                    frame_name=frame_name,
                    mode=mode)),
                xyzquat_diff=(_DifferenceFrameXYZQuat, dict(
                    frame_name=frame_name,
                    mode=mode))),
            auto_refresh=False)

        # Define specialize integrate operator on SE3 Lie group
        self._integrate = (
            pin.liegroups.SE3().integrate)  # pylint: disable=no-member

    def refresh(self) -> np.ndarray:
        # Interpolate the average spatial velocity at midpoint
        return self._integrate(
            self.xyzquat_next.get(), - 0.5 * self.xyzquat_diff.get())


@dataclass(unsafe_hash=True)
class AverageFrameRollPitch(InterfaceQuantity[np.ndarray]):
    """Quaternion representation of the average Yaw-free orientation from the
    Roll-Pitch-Yaw decomposition of a given frame over the whole agent step.

    .. seealso::
        See `remove_yaw_from_quat` and `AverageFrameXYZQuat` for details about
        the Roll-Pitch-Yaw decomposition and how the average frame pose is
        defined respectively.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_name = frame_name
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                quat_mean=(MaskedQuantity, dict(
                    quantity=(AverageFrameXYZQuat, dict(
                        frame_name=frame_name,
                        mode=mode)),
                    axis=0,
                    keys=(3, 4, 5, 6)))),
            auto_refresh=False)

        # Twist-free average orientation of the base as a quaternion
        self._quat_no_yaw_mean = np.zeros((4,))

    def refresh(self) -> np.ndarray:
        # Compute Yaw-free average orientation
        remove_yaw_from_quat(self.quat_mean.get(), self._quat_no_yaw_mean)

        return self._quat_no_yaw_mean


@dataclass(unsafe_hash=True)
class FrameSpatialAverageVelocity(InterfaceQuantity[np.ndarray]):
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
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
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
            Optional: 'pinocchio.ReferenceFrame.LOCAL' by default.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
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
            requirements=dict(
                xyzquat_diff=(_DifferenceFrameXYZQuat, dict(
                    frame_name=frame_name,
                    mode=mode)),
                quat_mean=(MaskedQuantity, dict(
                    quantity=(AverageFrameXYZQuat, dict(
                        frame_name=frame_name,
                        mode=mode)),
                    axis=0,
                    keys=(3, 4, 5, 6)))),
            auto_refresh=False)

        # Inverse time difference from previous to next state
        self._inv_step_dt = 1.0 / self.env.step_dt

        # Pre-allocate memory for the spatial velocity
        self._v_spatial: np.ndarray = np.zeros(6)

        # Reshape linear plus angular velocity vector to vectorize rotation
        self._v_lin_ang = self._v_spatial.reshape((2, 3)).T

    def refresh(self) -> np.ndarray:
        # Compute average frame velocity in local frame since previous step
        np.multiply(
            self.xyzquat_diff.get(), self._inv_step_dt, self._v_spatial)

        # Translate local velocity to world frame
        if self.reference_frame == pin.LOCAL_WORLD_ALIGNED:
            # Define world frame as the "middle" between prev and next pose.
            # Here, we only care about the middle rotation, so we can consider
            # SO3 Lie Group algebra instead of SE3.
            quat_apply(self.quat_mean.get(), self._v_lin_ang, self._v_lin_ang)

        return self._v_spatial


@dataclass(unsafe_hash=True)
class MultiActuatedJointKinematic(AbstractQuantity[np.ndarray]):
    """Current position, velocity or acceleration of all the actuated joints
    of the robot in motor order, before or after the mechanical transmissions.

    In practice, all actuated joints must be 1DoF for now. In the case of
    revolute unbounded revolute joints, the principal angle 'theta' is used to
    encode the position, not the polar coordinates `(cos(theta), sin(theta))`.

    .. warning::
        Data is extracted from the true configuration vector instead of using
        sensor data. As a result, this quantity is appropriate for computing
        reward components and termination conditions but must be avoided in
        observers and controllers, unless `mode=QuantityEvalMode.REFERENCE`.

    .. warning::
        Revolute unbounded joints are not supported for now.
    """

    kinematic_level: pin.KinematicLevel
    """Kinematic level to consider, ie position, velocity or acceleration.
    """

    is_motor_side: bool
    """Whether the compute kinematic data on motor- or joint-side, ie before or
    after their respective mechanical transmision.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 kinematic_level: pin.KinematicLevel = pin.POSITION,
                 is_motor_side: bool = False,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param kinematic_level: Desired kinematic level, ie position, velocity
                                or acceleration.
        :param is_motor_side: Whether the compute kinematic data on motor- or
                              joint-side, ie before or after the mechanical
                              transmisions.
                              Optional: False by default.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Backup some of the user-arguments
        self.kinematic_level = kinematic_level
        self.is_motor_side = is_motor_side

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                state=(StateQuantity, dict(
                    update_kinematics=False))),
            mode=mode,
            auto_refresh=False)

        # Mechanical joint position indices.
        # Note that it will only be used in last resort if it can be written as
        # a slice. Indeed, "fancy" indexing returns a copy of the original data
        # instead of a view, which requires fetching data at every refresh.
        self.kinematic_indices: List[int] = []

        # Keep track of the mechanical reduction ratio for all the motors
        self._joint_to_motor_ratios = np.array([])

        # Buffer storing mechanical joint positions
        self._data = np.array([])

        # Whether mechanical joint positions must be updated at every refresh
        self._must_refresh = False

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Make sure that the state data meet requirements
        state = self.state.get()
        if ((self.kinematic_level == pin.ACCELERATION and state.a is None) or
                (self.kinematic_level >= pin.VELOCITY and state.v is None)):
            raise RuntimeError(
                "Available state data do not meet requirements for kinematic "
                f"level '{self.kinematic_level}'.")

        # Refresh mechanical joint position indices and reduction ratio
        joint_to_motor_ratios = []
        self.kinematic_indices.clear()
        for motor in self.robot.motors:
            joint = self.pinocchio_model.joints[motor.joint_index]
            joint_type = jiminy.get_joint_type(joint)
            if joint_type == jiminy.JointModelType.ROTARY_UNBOUNDED:
                raise ValueError(
                    "Revolute unbounded joints are not supported for now.")
            if self.kinematic_level == pin.KinematicLevel.POSITION:
                kin_first, kin_last = joint.idx_q, joint.idx_q + joint.nq
            else:
                kin_first, kin_last = joint.idx_v, joint.idx_v + joint.nv
            motor_options = motor.get_options()
            mechanical_reduction = motor_options["mechanicalReduction"]
            joint_to_motor_ratios.append(mechanical_reduction)
            self.kinematic_indices += range(kin_first, kin_last)
        self._joint_to_motor_ratios = np.array(joint_to_motor_ratios)

        # Determine whether data can be extracted from state by reference
        kin_first = min(self.kinematic_indices)
        kin_last = max(self.kinematic_indices)
        self._must_refresh = True
        if self.mode == QuantityEvalMode.TRUE:
            try:
                if np.all(np.array(self.kinematic_indices) == np.arange(
                        kin_first, kin_last + 1)):
                    self._must_refresh = False
                elif sorted(self.kinematic_indices) != self.kinematic_indices:
                    warnings.warn(
                        "Consider using the same ordering for motors and "
                        "joints for optimal performance.")
            except ValueError:
                pass

        # Try extracting mechanical joint positions by reference if possible
        if self._must_refresh:
            self._data = np.full((len(self.kinematic_indices),), float("nan"))
        else:
            state = self.state.get()
            if self.kinematic_level == pin.KinematicLevel.POSITION:
                self._data = state.q[slice(kin_first, kin_last + 1)]
            elif self.kinematic_level == pin.KinematicLevel.VELOCITY:
                self._data = state.v[slice(kin_first, kin_last + 1)]
            else:
                self._data = state.a[slice(kin_first, kin_last + 1)]

    def refresh(self) -> np.ndarray:
        # Update mechanical joint positions only if necessary
        state = self.state.get()
        if self._must_refresh:
            if self.kinematic_level == pin.KinematicLevel.POSITION:
                data = state.q
            elif self.kinematic_level == pin.KinematicLevel.VELOCITY:
                data = state.v
            else:
                data = state.a
            data.take(self.kinematic_indices, None, self._data, "clip")

        # Translate encoder data at joint level
        if self.is_motor_side:
            self._data *= self._joint_to_motor_ratios

        return self._data


class EnergyGenerationMode(IntEnum):
    """Specify what happens to the energy generated by motors when breaking.
    """

    CHARGE = 0
    """The energy flows back to the battery to charge them without any kind of
    losses in the process if negative overall.
    """

    LOST_EACH = 1
    """The generated energy by each motor individually is lost by thermal
    dissipation, without flowing back to the battery nor powering other motors
    consuming energy if any.
    """

    LOST_GLOBAL = 2
    """The energy is lost by thermal dissipation without flowing back to the
    battery if negative overall.
    """

    PENALIZE = 3
    """The generated energy by each motor individually is treated as consumed.
    """


# Define proxies for fast lookup
_CHARGE, _LOST_EACH, _LOST_GLOBAL, _PENALIZE = map(int, EnergyGenerationMode)


@dataclass(unsafe_hash=True)
class AverageMechanicalPowerConsumption(InterfaceQuantity[float]):
    """Average mechanical power consumption by all the motors over a sliding
    time window.
    """

    max_stack: int
    """Time horizon over which values of the instantaneous power consumption
    will be stacked for computing the average.
    """

    generator_mode: EnergyGenerationMode
    """Specify what happens to the energy generated by motors when breaking.
    See `EnergyGenerationMode` documentation for details.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(
            self,
            env: InterfaceJiminyEnv,
            parent: Optional[InterfaceQuantity],
            *,
            horizon: float,
            generator_mode: EnergyGenerationMode = EnergyGenerationMode.CHARGE,
            mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the average.
        :param generator_mode: Specify what happens to the energy generated by
                               motors when breaking.
                               Optional: `EnergyGenerationMode.CHARGE` by
                               default.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Convert horizon in stack length, assuming constant env timestep
        max_stack = max(int(np.ceil(horizon / env.step_dt)), 1)

        # Backup some of the user-arguments
        self.max_stack = max_stack
        self.generator_mode = generator_mode
        self.mode = mode

        # Jit-able method computing the total instantaneous power consumption
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def _compute_power(generator_mode: int,  # EnergyGenerationMode
                           motor_velocities: np.ndarray,
                           motor_efforts: np.ndarray) -> float:
            """Compute the total instantaneous mechanical power consumption of
            all motors.

            :param generator_mode: Specify what happens to the energy generated
                                   by motors when breaking.
            :param motor_velocities: Velocity of all the motors before
                                     transmission as a 1D array. The order must
                                     be consistent with the motor indices.
            :param motor_efforts: Effort of all the motors before transmission
                                  as a 1D array. The order must be consistent
                                  with the motor indices.
            """
            if generator_mode in (_CHARGE, _LOST_GLOBAL):
                total_power = np.dot(motor_velocities, motor_efforts)
                if generator_mode == _CHARGE:
                    return total_power
                return max(total_power, 0.0)
            motor_powers = motor_velocities * motor_efforts
            if generator_mode == _LOST_EACH:
                return np.sum(np.maximum(motor_powers, 0.0))
            return np.sum(np.abs(motor_powers))

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                total_power_stack=(StackedQuantity, dict(
                    quantity=(BinaryOpQuantity, dict(
                        quantity_left=(UnaryOpQuantity, dict(
                            quantity=(StateQuantity, dict(
                                update_kinematics=False,
                                mode=self.mode)),
                            op=lambda state: state.command)),
                        quantity_right=(MultiActuatedJointKinematic, dict(
                            kinematic_level=pin.KinematicLevel.VELOCITY,
                            is_motor_side=True,
                            mode=self.mode)),
                        op=partial(_compute_power, int(self.generator_mode)))),
                    max_stack=self.max_stack,
                    as_array=True,
                    mode='slice'))),
            auto_refresh=False)

    def refresh(self) -> float:
        return np.mean(self.total_power_stack.get())
