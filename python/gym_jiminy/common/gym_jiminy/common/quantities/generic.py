# pylint: disable=redefined-builtin
"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
import warnings
from enum import Enum
from dataclasses import dataclass
from typing import (
    List, Dict, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable)

import numpy as np
import numba as nb

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)
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
class MultiFramesQuantity(Protocol):
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
                     `MultiFramesQuantity` protocol. All the parents of all its
                     cache owners must also implement one of these protocol.
    """
    # Make sure that parent quantity implement multi- or single-frame protocol
    assert isinstance(quantity.parent, (FrameQuantity, MultiFramesQuantity))
    quantities = (quantity.cache.owners if quantity.has_cache else (quantity,))

    # First, order all multi-frame quantities by decreasing length
    frame_names_chunks: List[Tuple[str, ...]] = []
    for owner in quantities:
        if owner.parent.is_active(any_cache_owner=False):
            if isinstance(owner.parent, MultiFramesQuantity):
                frame_names_chunks.append(owner.parent.frame_names)

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
        if owner.parent.is_active(any_cache_owner=False):
            if isinstance(owner.parent, FrameQuantity):
                frame_name_chunks.append(owner.parent.frame_name)
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
        assert isinstance(parent, (FrameQuantity, MultiFramesQuantity))

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


class Orientation(Enum):
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

    type: Orientation
    """Selected vector representation of the orientation for all frames.
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
                 parent: Union["FrameOrientation", "MultiFramesOrientation"],
                 type: Orientation,
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: `FrameOrientation` or `MultiFramesOrientation` instance
                       from which this quantity is a requirement.
        :param type: Desired vector representation of the orientation for all
                     frames.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a suitable parent has been provided
        assert isinstance(parent, (FrameOrientation, MultiFramesOrientation))

        # Backup some user argument(s)
        self.type = type
        self.mode = mode

        # Initialize the ordered list of frame names.
        # Note that this must be done BEFORE calling base `__init__`, otherwise
        # `isinstance(..., (FrameQuantity, MultiFramesQuantity))` will fail.
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
        if self.type == Orientation.EULER:
            self._data_batch = np.zeros((3, nframes), order='C')
        elif self.type == Orientation.QUATERNION:
            self._data_batch = np.zeros((4, nframes), order='C')

        # Re-assign mapping from chunks of frame names to corresponding data
        if self.type in (Orientation.EULER, Orientation.QUATERNION):
            self._data_map = {
                key: self._data_batch[..., frame_slice]
                for key, frame_slice in frame_slices_map.items()}

    def refresh(self) -> Dict[Union[str, Tuple[str, ...]], np.ndarray]:
        # Get the complete batch of rotation matrices managed by this instance
        rot_mat_batch = self.rot_mat_map[self.frame_names]

        # Convert all rotation matrices at once to the desired representation
        if self.type == Orientation.EULER:
            matrix_to_rpy(rot_mat_batch, self._data_batch)
        elif self.type == Orientation.QUATERNION:
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
    """Euler angles (Roll-Pitch-Yaw) of the orientation of a given frame in
    world reference frame at the end of the agent step.
    """

    frame_name: str
    """Name of the frame on which to operate.
    """

    type: Orientation
    """Desired vector representation of the orientation.
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
                 type: Orientation = Orientation.MATRIX,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_name: Name of the frame on which to operate.
        :param type: Desired vector representation of the orientation.
                     Optional: 'Orientation.MATRIX' by default.
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
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        return self.data[self.frame_name]


@dataclass(unsafe_hash=True)
class MultiFramesOrientation(InterfaceQuantity[np.ndarray]):
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

    type: Orientation
    """Selected vector representation of the orientation for all frames.
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
                 frame_names: Sequence[str],
                 *,
                 type: Orientation = Orientation.MATRIX,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_names: Name of the frames on which to operate.
        :param type: Desired vector representation of the orientation for all
                     frames.
                     Optional: 'Orientation.MATRIX' by default.
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
            # Must reset the tracking for shared computation systematically,
            # just in case the optimal computation path has changed to the
            # point that relying on batched quantity is no longer relevant.
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        # Return a slice of batched data.
        # Note that mapping from frame names to frame index in batched data
        # cannot be pre-computed as it may changed dynamically.
        return self.data[self.frame_names]


@dataclass(unsafe_hash=True)
class _BatchedFramesPosition(
        AbstractQuantity[Dict[Union[str, Tuple[str, ...]], np.ndarray]]):
    """Position (X, Y, Z) of all frames involved in quantities relying on it
    and are active since last reset of computation tracking if shared cache is
    available, its parent otherwise.

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
                 parent: Union["FramePosition", "MultiFramesPosition"],
                 mode: QuantityEvalMode) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: `FramePosition` or `MultiFramesPosition` instance from
                       which this quantity is a requirement.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Make sure that a suitable parent has been provided
        assert isinstance(parent, (FramePosition, MultiFramesPosition))

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
        self._pos_batch = np.zeros((3, nframes), order='C')

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
    """Position (X, Y, Z) of a given frame in world reference frame at the end
    of the agent step.
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
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        return self.data[self.frame_name]


@dataclass(unsafe_hash=True)
class MultiFramesPosition(InterfaceQuantity[np.ndarray]):
    """Position (X, Y, Z) of a given set of frames in world reference frame at
    the end of the agent step.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames on which to operate.
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
            self.requirements["data"].reset(reset_tracking=True)

    def refresh(self) -> np.ndarray:
        return self.data[self.frame_names]


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
                    type=Orientation.QUATERNION,
                    mode=mode))),
            auto_refresh=False)

        # Pre-allocate memory for storing the pose XYZQuat of all frames
        self._xyzquat = np.zeros((7,))

    def refresh(self) -> np.ndarray:
        # Copy the position of all frames at once in contiguous buffer
        array_copyto(self._xyzquat[:3], self.position)

        # Copy the quaternion of all frames at once in contiguous buffer
        array_copyto(self._xyzquat[-4:], self.quat)

        return self._xyzquat


@dataclass(unsafe_hash=True)
class MultiFramesXYZQuat(InterfaceQuantity[np.ndarray]):
    """Vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of the
    transform of a given set of frames in world reference frame at the end of
    the agent step.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames on which to operate.
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
                positions=(MultiFramesPosition, dict(
                    frame_names=frame_names,
                    mode=mode)),
                quats=(MultiFramesOrientation, dict(
                    frame_names=frame_names,
                    type=Orientation.QUATERNION,
                    mode=mode))),
            auto_refresh=False)

        # Pre-allocate memory for storing the pose XYZQuat of all frames
        self._xyzquats = np.zeros((7, len(frame_names)), order='C')

    def refresh(self) -> np.ndarray:
        # Copy the position of all frames at once in contiguous buffer
        array_copyto(self._xyzquats[:3], self.positions)

        # Copy the quaternion of all frames at once in contiguous buffer
        array_copyto(self._xyzquats[-4:], self.quats)

        return self._xyzquats


@dataclass(unsafe_hash=True)
class MultiFramesMeanXYZQuat(InterfaceQuantity[np.ndarray]):
    """Vector representation (X, Y, Z, QuatX, QuatY, QuatZ, QuatW) of the
    average transform of a given set of frames in world reference frame at the
    end of the agent step.

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
    """Specify on which state to evaluate this quantity. See `Mode`
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
                positions=(MultiFramesPosition, dict(
                    frame_names=frame_names,
                    mode=mode)),
                quats=(MultiFramesOrientation, dict(
                    frame_names=frame_names,
                    type=Orientation.QUATERNION,
                    mode=mode))),
            auto_refresh=False)

        # Define jit-able specialization of `np.mean` for `axis=-1`
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def position_average(value: np.ndarray, out: np.ndarray) -> None:
            """Compute the mean of an array over its last axis only.

            :param value: N-dimensional array from which the last axis will be
                          reduced.
            :param out: A pre-allocated array into which the result is stored.
            """
            out[:] = np.sum(value, -1) / value.shape[-1]

        self._position_average_fun = position_average

        # Define jit-able specialization of `quat_average` for 2D matrices
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def quat_average_2d(quat: np.ndarray, out: np.ndarray) -> None:
            """Compute the average of a batch of quaternions [qx, qy, qz, qw].

            .. note::
                Jit-able specialization of `quat_average` for 2D matrices, with
                further optimization for the special case where there is only 2
                quaternions.

            :param quat: N-dimensional (N >= 2) array whose first dimension
                         gathers the 4 quaternion coordinates [qx, qy, qz, qw].
            :param out: Pre-allocated array into which the result is stored.
            """
            num_quats = quat.shape[1]
            if num_quats == 1:
                out[:] = quat
            elif num_quats == 2:
                quat_interpolate_middle(quat[:, 0], quat[:, 1], out)
            else:
                _, eigvec = np.linalg.eigh(quat @ quat.T)
                out[:] = eigvec[..., -1]

        self._quat_average_fun = quat_average_2d

        # Pre-allocate memory for the mean for mean pose vector XYZQuat
        self._xyzquat_mean = np.zeros((7,))

        # Define position and orientation proxies for fast access
        self._position_mean_view = self._xyzquat_mean[:3]
        self._quat_mean_view = self._xyzquat_mean[3:]

    def refresh(self) -> np.ndarray:
        # Compute the mean translation
        self._position_average_fun(self.positions, self._position_mean_view)

        # Compute the mean quaternion
        self._quat_average_fun(self.quats, self._quat_mean_view)

        return self._xyzquat_mean


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
            requirements=dict(xyzquat_stack=(StackedQuantity, dict(
                quantity=(FrameXYZQuat, dict(
                    frame_name=frame_name, mode=mode)),
                num_stack=2))),
            auto_refresh=False)

        # Define specialize difference operator on SE3 Lie group
        self._se3_diff = (
            pin.liegroups.SE3().difference)  # pylint: disable=no-member

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
class ActuatedJointsPosition(AbstractQuantity[np.ndarray]):
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
