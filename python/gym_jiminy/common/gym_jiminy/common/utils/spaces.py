""" TODO: Write documentation.
"""
from itertools import zip_longest
from collections import OrderedDict
from collections.abc import Iterable
from typing import (
    Any, Optional, Union, Sequence, TypeVar, Mapping as MappingT,
    Iterable as IterableT, no_type_check, cast)

import numba as nb
import numpy as np
from numpy import typing as npt
from numpy.core.umath import (  # type: ignore[attr-defined]
    copyto as _array_copyto)

import tree
import gymnasium as gym


ValueT = TypeVar('ValueT')
StructNested = Union[MappingT[str, 'StructNested[ValueT]'],
                     IterableT['StructNested[ValueT]'],
                     ValueT]
FieldNested = StructNested[str]
DataNested = StructNested[np.ndarray]

DataNestedT = TypeVar('DataNestedT', bound=DataNested)


global_rng = np.random.default_rng()


@nb.jit(nopython=True, nogil=True, inline='always')
def _array_clip(value: np.ndarray,
                low: np.ndarray,
                high: np.ndarray) -> np.ndarray:
    return value.clip(low, high)


def _unflatten_as(structure: StructNested[Any],
                  flat_sequence: Sequence[DataNested]) -> DataNested:
    """Unflatten a sequence into a given structure.

    .. seealso::
        This method is the same as 'tree.unflatten_as' without runtime checks.

    :param structure: Arbitrarily nested structure.
    :param flat_sequence: Sequence to unflatten.

    :returns: 'flat_sequence' unflattened into 'structure'.
    """
    if not tree.is_nested(structure):
        return flat_sequence[0]
    _, packed = tree._packed_nest_with_indices(structure, flat_sequence, 0)
    return tree._sequence_like(structure, packed)


def _clip_or_copy(value: np.ndarray, space: gym.Space) -> np.ndarray:
    """Clip value if associated to 'gym.spaces.Box', otherwise return a copy.

    :param value: Value to clip.
    :param space: `gym.Space` associated with 'value'.
    """
    if isinstance(space, gym.spaces.Box):
        return _array_clip(value, space.low, space.high)
    return value.copy()


def sample(low: Union[float, np.ndarray] = -1.0,
           high: Union[float, np.ndarray] = 1.0,
           dist: str = 'uniform',
           scale: Union[float, np.ndarray] = 1.0,
           enable_log_scale: bool = False,
           shape: Optional[Sequence[int]] = None,
           rg: Optional[np.random.Generator] = None
           ) -> np.ndarray:
    """Randomly sample values from a given distribution.

    .. note:
        If 'low', 'high', and 'scale' are floats, then the output is float if
        'shape' is None, otherwise it has type `np.ndarray` and shape 'shape'.
        Similarly, if any of 'low', 'high', and 'scale' are `np.ndarray`, then
        its shape follows the broadcasting rules between these variables.

    :param low: Lower value for bounded distribution, negative-side standard
                deviation otherwise.
                Optional: -1.0 by default.
    :param high: Upper value for bounded distribution, positive-side standard
                 deviation otherwise.
                 Optional: 1.0 by default.
    :param dist: Name of the statistical distribution from which to draw
                 samples. It must be a member function for `np.random`.
                 Optional: 'uniform' by default.
    :param scale: Shrink the standard deviation of the distribution around the
                  mean by this factor.
                  Optional: No scaling by default?
    :param enable_log_scale: The sampled values are power of 10.
    :param shape: Enforce of the sampling shape. Only available if 'low',
                  'high' and 'scale' are floats. `None` to disable.
                  Optional: Disabled by default.
    :param rg: Custom random number generator from which to draw samples.
               Optional: Default to `np.random`.
    """
    # Make sure the distribution is supported
    if dist not in ('uniform', 'normal'):
        raise NotImplementedError(
            f"'{dist}' distribution type is not supported for now.")

    # Extract mean and deviation from min/max
    mean = 0.5 * (low + high)
    dev = 0.5 * scale * (high - low)

    # Get sample shape.
    # Better use dev than mean since it works even if only scale is array.
    if isinstance(dev, np.ndarray):
        if shape is None:
            shape = dev.shape
        else:
            try:
                shape = list(shape)
                np.broadcast(np.empty(shape, dtype=[]), dev)
            except ValueError as e:
                raise ValueError(
                    f"'shape' {shape} must be broadcastable with 'low', "
                    f"'high' and 'scale' {dev.shape} if specified.") from e

    # Sample from normalized distribution.
    # Note that some distributions are not normalized by default.
    distrib_fn = getattr(rg or global_rng, dist)
    if dist == 'uniform':
        value = distrib_fn(low=-1.0, high=1.0, size=shape)
    else:
        value = distrib_fn(size=shape)

    # Set mean and deviation
    value = mean + dev * value

    # Revert log scale if appropriate
    if enable_log_scale:
        value = 10 ** value

    return np.asarray(value)


def is_bounded(space_nested: gym.Space) -> bool:
    """Check wether a `gym.Space` has finite bounds.

    :param space: `gym.Space` on which to operate.
    """
    for space in tree.flatten(space_nested):
        is_bounded_fn = getattr(space, "is_bounded", None)
        if is_bounded_fn is not None and not is_bounded_fn():
            return False
    return True


@no_type_check
def zeros(space: gym.Space[DataNestedT],
          dtype: npt.DTypeLike = None) -> DataNestedT:
    """Allocate data structure from `gym.Space` and initialize it to zero.

    :param space: `gym.Space` on which to operate.
    :param dtype: Can be specified to overwrite original space dtype.
                  Optional: None by default
    """
    # Note that it is not possible to take advantage of dm-tree because the
    # output type for collections (OrderedDict or Tuple) is not the same as the
    # input one (gym.Space). This feature request would be too specific.
    if isinstance(space, gym.spaces.Dict):
        value = OrderedDict()
        for field, subspace in dict.items(space.spaces):
            value[field] = zeros(subspace, dtype=dtype)
        return value
    if isinstance(space, gym.spaces.Tuple):
        return tuple(zeros(subspace, dtype=dtype) for subspace in space.spaces)
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=dtype or space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        # Note that np.array of 0 dim is returned in order to be mutable
        return np.array(0, dtype=dtype or np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros_like(space.nvec, dtype=dtype or np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.zeros(space.n, dtype=dtype or np.int8)
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


def fill(data: DataNested, fill_value: Union[float, int, np.number]) -> None:
    """Set every element of 'data' from `gym.Space` to scalar 'fill_value'.

    :param data: Data structure to update.
    :param fill_value: Value used to fill any scalar from the leaves.
    """
    for value in tree.flatten(data):
        try:
            value.fill(fill_value)
        except AttributeError as e:
            raise ValueError(
                "Leaves of 'data' structure must have type `np.ndarray`."
                ) from e


def set_value(data: DataNested, value: DataNested) -> None:
    """Partially set 'data' from `gym.Space` to 'value'.

    It avoids memory allocation, so that memory pointers of 'data' remains
    unchanged. As direct consequences, it is necessary to preallocate memory
    beforehand, and to work with fixed shape buffers.

    .. note::
        If 'data' is a dictionary, 'value' must be a subtree of 'data', whose
        leaves must be broadcast-able with the ones of 'data'.

    :param data: Data structure to partially update.
    :param value: Subtree of data only containing fields to update.
    """
    if isinstance(data, np.ndarray):
        try:
            data.flat[:] = value
        except TypeError as e:
            raise TypeError(f"Cannot broadcast '{value}' to '{data}'.") from e
    elif isinstance(data, dict):
        assert isinstance(value, dict)
        for field, subval in value.items():
            set_value(data[field], subval)
    elif isinstance(data, Iterable):
        assert isinstance(value, Iterable)
        for subdata, subval in zip_longest(data, value):
            set_value(subdata, subval)
    else:
        raise ValueError(
            "Leaves of 'data' structure must have type `np.ndarray`."
            )


def copyto(src: DataNestedT, dest: DataNestedT) -> None:
    """Copy arbitrarily nested data structure of 'np.ndarray' to a given
    pre-allocated destination.

    It avoids memory allocation completely, so that memory pointers of 'data'
    remains unchanged. As direct consequences, it is necessary to preallocate
    memory beforehand, and it only supports arrays of fixed shape.

    :param data: Data structure to update.
    :param value: Data to copy.
    """
    if isinstance(src, np.ndarray):
        _array_copyto(src, dest)
    else:
        for data, value in zip(tree.flatten(src), tree.flatten(dest)):
            _array_copyto(data, value)


def copy(data: DataNestedT) -> DataNestedT:
    """Shallow copy recursively 'data' from `gym.Space`, so that only leaves
    are still references.

    :param data: Hierarchical data structure to copy without allocation.
    """
    return cast(DataNestedT, _unflatten_as(data, tree.flatten(data)))


def clip(data: DataNestedT,
         space_nested: gym.Space[DataNestedT],
         check: bool = True) -> DataNestedT:
    """Clamp value from `gym.Space` to make sure it is within bounds.

    .. note:
        None of the leaves of the returned data structured is sharing memory
        with the original one, even if clipping had no effect or was not
        applicable. This alleviate the need of calling 'deepcopy' afterward.

    :param space: `gym.Space` on which to operate.
    :param data: Data to clip.
    """
    if check:
        return tree.map_structure(_clip_or_copy, data, space_nested)
    return cast(DataNestedT, _unflatten_as(data, [
        _clip_or_copy(value, space) for value, space in zip(
            tree.flatten(data), tree.flatten(space_nested))]))
