""" TODO: Write documentation.
"""
from functools import partial
from itertools import zip_longest
from collections import OrderedDict
from collections.abc import Iterable
from typing import (
    Any, Optional, Union, Sequence, TypeVar, Dict, Mapping as MappingT,
    Iterable as IterableT, Tuple, SupportsFloat, Callable, no_type_check, cast)

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


@no_type_check
@nb.jit(nopython=True, nogil=True, inline='always')
def _array_clip(value: np.ndarray,
                low: Union[np.ndarray, SupportsFloat],
                high: Union[np.ndarray, SupportsFloat]) -> np.ndarray:
    """Element-wise out-of-place clipping of array elements.

    :param value: Array holding values to clip.
    :param low: lower bound.
    :param high: upper bound.
    """
    if value.ndim:
        return np.minimum(np.maximum(value, low), high)
    # Surprisingly, calling '.item()' on python scalars is supported by numba
    return np.array(min(max(value.item(), low.item()), high.item()))


@no_type_check
@nb.jit(nopython=True, nogil=True, inline='always')
def _array_contains(value: np.ndarray,
                    low: Union[np.ndarray, SupportsFloat],
                    high: Union[np.ndarray, SupportsFloat]) -> np.ndarray:
    """Check that all array elements are withing bounds.

    :param value: Array holding values to check.
    :param low: lower bound.
    :param high: upper bound.
    """
    if value.ndim:
        return np.logical_and(low <= value, value <= high).all()
    return low.item() <= value.item() <= high.item()


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


def get_bounds(space: gym.Space) -> Tuple[
        Union[np.ndarray, SupportsFloat], Union[np.ndarray, SupportsFloat]]:
    """Get the lower and upper bounds of a given 'gym.Space' if applicable,
    raises any exception otherwise.

    :param space: `gym.Space` on which to operate.

    :returns: Lower and upper bounds as a tuple.
    """
    if isinstance(space, gym.spaces.Box):
        return (space.low, space.high)
    if isinstance(space, gym.spaces.Discrete):
        return (space.start, space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return (0, space.nvec)
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


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
          dtype: npt.DTypeLike = None,
          enforce_bounds: bool = True) -> DataNestedT:
    """Allocate data structure from `gym.Space` and initialize it to zero.

    :param space: `gym.Space` on which to operate.
    :param dtype: Can be specified to overwrite original space dtype.
                  Optional: None by default
    """
    # Note that it is not possible to take advantage of dm-tree because the
    # output type for collections (OrderedDict or Tuple) is not the same as the
    # input one (gym.Space). This feature request would be too specific.
    value = None
    if isinstance(space, gym.spaces.Dict):
        value = OrderedDict()
        for field, subspace in dict.items(space.spaces):
            value[field] = zeros(subspace, dtype=dtype)
        return value
    if isinstance(space, gym.spaces.Tuple):
        value = tuple(zeros(subspace, dtype=dtype)
                      for subspace in space.spaces.values())
    elif isinstance(space, gym.spaces.Box):
        value = np.zeros(space.shape, dtype=dtype or space.dtype)
    elif isinstance(space, gym.spaces.Discrete):
        # Note that np.array of 0 dim is returned in order to be mutable
        value = np.array(0, dtype=dtype or np.int64)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        value = np.zeros_like(space.nvec, dtype=dtype or np.int64)
    elif isinstance(space, gym.spaces.MultiBinary):
        value = np.zeros(space.n, dtype=dtype or np.int8)
    if value is not None:
        if enforce_bounds:
            value = clip(value, space)
        return value
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


def build_copyto(dst: DataNested) -> Callable[[DataNested], None]:
    """Specialize 'copyto' for a given pre-allocated destination.

    :param dst: Hierarchical data structure to update.
    """
    if isinstance(dst, np.ndarray):
        try:
            return partial(_array_copyto, dst)
        except Exception as e:
            raise ValueError("All leaves must have tpe 'np.ndarray'.") from e
    assert isinstance(dst, dict)

    def _seq_calls(funcs: Sequence[Callable[[DataNested], None]],
                   src_nested: Dict[str, DataNested]) -> None:
        """Copy arbitrarily nested data structure of 'np.ndarray' specialized
        for some pre-allocated destination.

        :param src_nested: Data with the same hierarchy than the destination.
        """
        src: DataNested
        for func, src in zip(funcs, dict.values(src_nested)):
            func(src)

    return partial(_seq_calls, [build_copyto(value) for value in dst.values()])


def copyto(dst: DataNested, src: DataNested) -> None:
    """Copy arbitrarily nested data structure of 'np.ndarray' to a given
    pre-allocated destination.

    It avoids memory allocation completely, so that memory pointers of 'data'
    remains unchanged. As direct consequences, it is necessary to preallocate
    memory beforehand, and it only supports arrays of fixed shape.

    .. note::
        Unlike the function returned by 'build_copyto', only the flattened data
        structure needs to match, not the original one. This means that the
        source and/or destination can be flattened already when provided.

    :param dst: Hierarchical data structure to update, possibly flattened.
    :param value: Hierarchical data to copy, possibly flattened.
    """
    for data, value in zip(tree.flatten(dst), tree.flatten(src)):
        _array_copyto(data, value)


def copy(data: DataNestedT) -> DataNestedT:
    """Shallow copy recursively 'data' from `gym.Space`, so that only leaves
    are still references.

    :param data: Hierarchical data structure to copy without allocation.
    """
    return cast(DataNestedT, _unflatten_as(data, tree.flatten(data)))


def build_clip(data: DataNested,
               space: gym.Space[DataNested]) -> Callable[[], DataNested]:
    """Specialize 'clip' for some pre-allocated data.

    .. warning::
        This method is much faster than 'clip' but it requires updating
        pre-allocated memory instead of allocated new one as it is usually
        the case without careful memory management.

    :param data: Data to clip.
    :param space: `gym.Space` on which to operate.
    """
    if not isinstance(space, gym.spaces.Dict):
        try:
            return partial(_array_clip, data, *get_bounds(space))
        except NotImplementedError:
            assert isinstance(data, np.ndarray)
            return data.copy
    assert isinstance(data, dict)

    def _setitem(field: str,
                 func: Callable[[], DataNested],
                 out: Dict[str, DataNested]) -> None:
        """Set a given field of a nested data structure to the value return by
        a function with no input argument.

        :param field: Field to set.
        :param func: Function to call.
        :param out: Nested data structure.
        """
        out[field] = func()

    def _seq_calls(func1: Callable[[DataNested], None],
                   func2: Callable[[DataNested], None],
                   out: DataNested) -> None:
        """Call two functions sequentially in order while passing the same
        input argument to both of them.

        :param func1: First function.
        :param func2: Second function.
        :param out: Input argument to forward.
        """
        func1(out)
        func2(out)

    func = None
    for field, subspace in dict.items(space.spaces):
        op = partial(_setitem, field, build_clip(data[field], subspace))
        func = op if func is None else partial(_seq_calls, func, op)
    if func is None:
        return lambda: OrderedDict()

    # Define the chain of functions operating on a given out
    def _clip_impl(func: Callable[[DataNested], None]) -> DataNested:
        """Clip arbitrarily nested data structure of 'np.ndarray' specialized
        for some pre-allocated data.
        """
        out: DataNested = OrderedDict()
        func(out)
        return out

    return partial(_clip_impl, func)


def clip(data: DataNested,
         space: gym.Space[DataNested]) -> DataNested:
    """Clip data from `gym.Space` to make sure it is within bounds.

    .. note:
        None of the leaves of the returned data structured is sharing memory
        with the original one, even if clipping had no effect. This alleviate
        the need of calling 'deepcopy' afterward.

    :param data: Data to clip.
    :param space: `gym.Space` on which to operate.
    """
    if not isinstance(space, gym.spaces.Dict):
        try:
            return _array_clip(data, *get_bounds(space))
        except NotImplementedError:
            assert isinstance(data, np.ndarray)
            return data.copy()
    assert isinstance(data, dict)

    out: Dict[str, DataNested] = OrderedDict()
    for field, subspace in dict.items(space.spaces):
        out[field] = clip(data[field], subspace)
    return out


def build_contains(data: DataNested,
                   space: gym.Space[DataNested]) -> Callable[[], bool]:
    """Specialize 'contains' for a given pre-allocated data structure.

    :param data: Pre-allocated data structure to check.
    :param space: `gym.Space` on which to operate.
    """
    if not isinstance(space, gym.spaces.Dict):
        return partial(_array_contains, data, *get_bounds(space))
    assert isinstance(data, dict)

    def _all(func1: Callable[[], bool],
             func2: Callable[[], bool]) -> bool:
        """Check if two functions with no input argument are returning True.

        :param func1: First function.
        :param func2: Second function.
        """
        return func1() and func2()

    func = None
    for field, subspace in dict.items(space.spaces):
        try:
            op = build_contains(data[field], subspace)
            func = op if func is None else partial(_all, func, op)
        except NotImplementedError:
            pass
    return func or (lambda: True)


def contains(data: DataNested, space: gym.Space[DataNested]) -> bool:
    """Check if all leaves of a nested data structure are within bounds of
    their respective `gym.Space`.

    By design, it is always `True` for all spaces but `gym.spaces.Box`,
    `gym.spaces.Discrete` and `gym.spaces.MultiDiscrete`.

    :param data: Data structure to check.
    :param space: `gym.Space` on which to operate.
    """
    if not isinstance(space, gym.spaces.Dict):
        try:
            return _array_contains(data, *get_bounds(space))
        except NotImplementedError:
            return True
    assert isinstance(data, dict)

    return all(contains(data[field], subspace)
               for field, subspace in dict.items(space.spaces))
