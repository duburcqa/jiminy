""" TODO: Write documentation.
"""
from collections import OrderedDict
from typing import Optional, Union, Dict, Sequence, TypeVar

import gym
import tree
import numpy as np
from numpy.random.mtrand import _rand as global_randstate


ValueT = TypeVar('ValueT')
StructNested = Union[Dict[str, 'StructNested'],  # type: ignore[misc]
                     Sequence['StructNested'],  # type: ignore[misc]
                     ValueT]
FieldNested = StructNested[str]  # type: ignore[misc]
DataNested = StructNested[np.ndarray]  # type: ignore[misc]


if tuple(map(int, (gym.__version__.split(".", 4)[:3]))) < (0, 23, 0):
    def _space_nested_raw(space_nested: gym.Space) -> gym.Space:
        """Replace any `gym.spaces.Dict|Tuple` by a native collection type for
        inter-operability with gym<0.23.0.

        .. note::
            It is necessary because non primitive objects must inherit from
            `collection.abc.Mapping|Sequence` for `dm-tree` to operate on them.
        """
        return tree.traverse(
            lambda space: _space_nested_raw(space.spaces) if isinstance(
                space, (gym.spaces.Dict, gym.spaces.Tuple)) else None,
            space_nested)
else:
    def _space_nested_raw(space_nested: gym.Space) -> gym.Space:
        return space_nested


def sample(low: Union[float, np.ndarray] = -1.0,
           high: Union[float, np.ndarray] = 1.0,
           dist: str = 'uniform',
           scale: Union[float, np.ndarray] = 1.0,
           enable_log_scale: bool = False,
           shape: Optional[Sequence[int]] = None,
           rg: Optional[Union[
               np.random.Generator, np.random.RandomState]] = None
           ) -> Union[float, np.ndarray]:
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
    # Note that some distributions are not normalized by default
    if rg is None:
        rg = global_randstate
    distrib_fn = getattr(rg, dist)
    if dist == 'uniform':
        value = distrib_fn(low=-1.0, high=1.0, size=shape)
    else:
        value = distrib_fn(size=shape)

    # Set mean and deviation
    value = mean + dev * value

    # Revert log scale if appropriate
    if enable_log_scale:
        value = 10 ** value

    return value


def is_bounded(space_nested: gym.Space) -> bool:
    """Check wether a `gym.Space` has finite bounds.

    :param space: `gym.Space` on which to operate.
    """
    for space in tree.flatten(_space_nested_raw(space_nested)):
        is_bounded_fn = getattr(space, "is_bounded", None)
        if is_bounded_fn is not None and not is_bounded_fn():
            return False
    return True


def zeros(space: gym.Space, dtype: Optional[type] = None) -> DataNested:
    """Allocate data structure from `gym.Space` and initialize it to zero.

    :param space: `gym.Space` on which to operate.
    :param dtype: Must be specified to overwrite original space dtype.
    """
    # Note that it is not possible to take advantage of dm-tree because the
    # output type for collections (OrderedDict or Tuple) is not the same as
    # the input one (gym.Space). This feature request would be too specific.
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=dtype or space.dtype)
    if isinstance(space, gym.spaces.Dict):
        value = OrderedDict()
        for field, subspace in dict.items(space.spaces):
            value[field] = zeros(subspace, dtype=dtype)
        return value
    if isinstance(space, gym.spaces.Tuple):
        return tuple(zeros(subspace, dtype=dtype) for subspace in space.spaces)
    if isinstance(space, gym.spaces.Discrete):
        # Note that np.array of 0 dim is returned in order to be mutable
        return np.array(0, dtype=dtype or np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros_like(space.nvec, dtype=dtype or np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.zeros(space.n, dtype=dtype or np.int8)
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


def fill(data: DataNested, fill_value: float) -> None:
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
    unchanged. A direct consequences, it is necessary to preallocate memory
    beforehand, and to work with fixed shape buffers.

    .. note::
        If 'data' is a dictionary, 'value' must be a subtree of 'data',
        whose leaves must be broadcastable with the ones of 'data'.

    :param data: Data structure to partially update.
    :param value: Unset of data only containing fields to update.
    """
    for data_i, value_i in zip(tree.flatten(data), tree.flatten(value)):
        try:
            data_i.flat[:] = value_i
        except AttributeError as e:
            raise ValueError(
                "Leaves of 'data' structure must have type `np.ndarray`."
                ) from e


def copy(data: DataNested) -> DataNested:
    """Shallow copy recursively 'data' from `gym.Space`, so that only leaves
    are still references.

    :param data: Hierarchical data structure to copy without allocation.
    """
    return tree.unflatten_as(data, tree.flatten(data))


def clip(space_nested: gym.Space,
         data: DataNested) -> DataNested:
    """Clamp value from `gym.Space` to make sure it is within bounds.

    :param space: `gym.Space` on which to operate.
    :param data: Data to clamp.
    """
    return tree.map_structure(
        lambda value, space:
            np.minimum(np.maximum(value, space.low), space.high)
        if isinstance(space, gym.spaces.Box) else value,
        data, _space_nested_raw(space_nested))
