""" TODO: Write documentation.
"""
from collections import OrderedDict
from typing import Optional, Union, Dict, Sequence, List

import numpy as np
import numba as nb
import gym
from gym import spaces

import jiminy_py.core as jiminy


SpaceDictNested = Union[  # type: ignore
    Dict[str, 'SpaceDictNested'], np.ndarray]  # type: ignore
ListStrRecursive = List[Union[str, 'ListStrRecursive']]  # type: ignore
FieldDictNested = Union[  # type: ignore
    Dict[str, 'FieldDictNested'], ListStrRecursive]  # type: ignore


def sample(low: Union[float, np.ndarray] = -1.0,
           high: Union[float, np.ndarray] = 1.0,
           dist: str = 'uniform',
           scale: Union[float, np.ndarray] = 1.0,
           enable_log_scale: bool = False,
           shape: Optional[Sequence[int]] = None,
           rg: Optional[np.random.RandomState] = None
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
                  Optional: Disable by default.
    :param rg: Custom random number generator from which to draw samples.
               Optional: Default to `np.random`.
    """
    # Make sure the distribution is supported
    if dist not in ('uniform', 'normal'):
        raise NotImplementedError(
            f"'{dist}' distribution type is not supported for now.")

    # Extract mean and deviation from min/max
    mean = (low + high) / 2
    dev = scale * (high - low) / 2

    # Get sample shape.
    # Better use dev than mean since it works even if only scale is array.
    if isinstance(dev, np.ndarray):
        if shape is None:
            shape = dev.shape
        else:
            try:
                shape = list(shape)
                np.broadcast(
                    np.empty(shape, dtype=[]), np.empty(dev.shape, dtype=[]))
            except ValueError as e:
                raise ValueError(
                    f"'shape' {shape} must be broadcastable with 'low' and "
                    f"'high' {dev.shape} if specified.") from e

    # Sample from normalized distribution.
    # Note that some distributions are not normalized by default
    if rg is None:
        rg = np.random
    distrib_fn = getattr(rg, dist)
    val = distrib_fn(size=shape)
    if dist == 'uniform':
        val = 2.0 * (val - 0.5)

    # Set mean and deviation
    val = mean + dev * val

    # Revert log scale if appropriate
    if enable_log_scale:
        val = 10 ** val

    return val


def is_bounded(space: gym.Space) -> bool:
    """Check wether `gym.spaces.Space` has finite bounds.

    :param space: Gym.Space on which to operate.
    """
    if isinstance(space, spaces.Box):
        return space.is_bounded()
    if isinstance(space, spaces.Dict):
        return any(not is_bounded(subspace) for subspace in space.values())
    if isinstance(space, spaces.Tuple):
        return any(not is_bounded(subspace) for subspace in space)
    if isinstance(space, (
            spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


def zeros(space: gym.Space,
          dtype: Optional[type] = None) -> Union[SpaceDictNested, int]:
    """Allocate data structure from `gym.space.Space` and initialize it to zero.

    :param space: Gym.Space on which to operate.
    :param dtype: Must be specified to overwrite original space dtype.
    """
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=dtype or space.dtype)
    if isinstance(space, spaces.Dict):
        value = OrderedDict()
        for field, subspace in dict.items(space.spaces):
            value[field] = zeros(subspace, dtype=dtype)
        return value
    if isinstance(space, spaces.Tuple):
        return tuple(zeros(subspace, dtype=dtype) for subspace in space.spaces)
    if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete)):
        return np.array(0)  # Using np.array of 0 dim to be mutable
    if isinstance(space, spaces.MultiBinary):
        return np.zeros(space.n, dtype=np.int8)
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


def fill(data: SpaceDictNested,
         fill_value: float) -> None:
    """Set every element of 'data' from `Gym.Space` to scalar 'fill_value'.

    :param data: Data structure to update.
    :param fill_value: Value used to fill any scalar from the leaves.
    """
    if isinstance(data, np.ndarray):
        data.fill(fill_value)
    elif isinstance(data, dict):
        for subdata in dict.values(data):
            fill(subdata, fill_value)
    elif isinstance(data, (tuple, list)):
        for subdata in data:
            fill(subdata, fill_value)
    else:
        if hasattr(data, '__dict__') or hasattr(data, '__slots__'):
            raise NotImplementedError(
                f"Data of type {type(data)} is not supported.")
        raise ValueError("Data of immutable type is not supported.")


def set_value(data: SpaceDictNested,
              value: SpaceDictNested) -> None:
    """Partially set 'data' from `Gym.Space` to 'value'.

    It avoids memory allocation, so that memory pointers of 'data' remains
    unchanged. A direct consequences, it is necessary to preallocate memory
    beforehand, and to work with fixed shape buffers.

    .. note::
        If 'data' is a dictionary, 'value' must be a subtree of 'data',
        whose leaf values must be broadcastable with the ones of 'data'.

    :param data: Data structure to partially update.
    :param value: Unset of data only containing fields to update.
    """
    if isinstance(data, np.ndarray):
        try:
            data.flat[:] = value
        except TypeError as e:
            raise TypeError(f"Cannot cast '{data}' to '{value}'.") from e
    elif isinstance(data, dict):
        for field, subval in dict.items(value):
            set_value(data[field], subval)
    elif isinstance(data, (tuple, list)):
        for subdata, subval in zip(data, value):
            fill(subdata, subval)
    else:
        raise NotImplementedError(
            f"Data of type {type(data)} is not supported.")


def copy(data: SpaceDictNested) -> SpaceDictNested:
    """Shadow copy recursively 'data' from `Gym.Space`, so that only leaves
    are still references.

    :param data: Hierarchical data structure to copy without allocation.
    """
    if isinstance(data, dict):
        value = OrderedDict()
        for field, subdata in dict.items(data):
            value[field] = copy(subdata)
        return value
    if isinstance(data, (tuple, list)):
        return data.__class__(copy(subdata) for subdata in data)
    return data


def clip(space: gym.Space, value: SpaceDictNested) -> SpaceDictNested:
    """Clamp value from Gym.Space to make sure it is within bounds.

    :param space: Gym.Space on which to operate.
    :param value: Value to clamp.
    """
    if isinstance(space, spaces.Box):
        return np.core.multiarray.clip(value, space.low, space.high)
    if isinstance(space, spaces.Dict):
        out = OrderedDict()
        for field, subspace in dict.items(space.spaces):
            out[field] = clip(subspace, value[field])
        return out
    if isinstance(space, spaces.Tuple):
        return (clip(subspace, subvalue)
                for subspace, subvalue in zip(space, value))
    if isinstance(space, (
            spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return value  # No need to clip those spaces.
    raise NotImplementedError(
        f"Gym.Space of type {type(space)} is not supported.")


@nb.jit(nopython=True, nogil=True)
def is_breakpoint(t: float, dt: float, eps: float) -> bool:
    """Check if 't' is multiple of 'dt' at a given precision 'eps'.

    :param t: Current time.
    :param dt: Timestep.
    :param eps: Precision.

    :meta private:
    """
    if dt < eps:
        return True
    dt_next = dt - t % dt
    if (dt_next <= eps / 2) or ((dt - dt_next) < eps / 2):
        return True
    return False


def get_fieldnames(space: spaces.Space,
                   namespace: str = "") -> FieldDictNested:
    """ Get generic fieldnames from `gym.spaces.Space`, so that it can be used
    in conjunction with `register_variables`, to register any value from gym
    Space to the telemetry conveniently.

    :param space: Gym.Space on which to operate.
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disable by default.
    """
    # Fancy action space: Trying to be clever and infer meaningful
    # telemetry names based on action space
    if isinstance(space, (spaces.Box, spaces.Tuple, spaces.MultiBinary)):
        # No information: fallback to basic numbering
        return [".".join(filter(None, (namespace, str(i))))
                for i in range(len(space))]
    if isinstance(space, (
            spaces.Discrete, spaces.MultiDiscrete)):
        # The space is already scalar. Namespace alone as fieldname is enough.
        return [namespace]
    if isinstance(space, spaces.Dict):
        assert space.spaces, "Dict space cannot be empty."
        out: List[Union[Dict[str, FieldDictNested], str]] = []
        for field, subspace in dict.items(space.spaces):
            if isinstance(subspace, spaces.Dict):
                out.append({field: get_fieldnames(subspace, namespace)})
            else:
                out.append(field)
        return out
    raise NotImplementedError(
        f"Gym.Space of type {type(space)} is not supported.")


def register_variables(controller: jiminy.AbstractController,
                       fields: FieldDictNested,
                       data: SpaceDictNested,
                       namespace: str = "") -> bool:
    """Register data from `Gym.Space` to the telemetry of a controller.

    .. warning::
        Variables are registered by reference. Consequently, the user is
        responsible to manage the lifetime of the data to avoid it being
        garbage collected, and to make sure the variables are updated
        when necessary, and only when it is.

    :param controller: Robot's controller of the simulator used to register
                       variables to the telemetry.
    :param field: Nested variable names, as returned by `get_fieldnames`
                  method. It can be a nested list or/and dict. The leaf are
                  str corresponding to the name of each scalar data.
    :param data: Data from `Gym.spaces.Space` to register. Note that the
                 telemetry stores pointers to the underlying memory, so it
                 only supports np.float64, and make sure to reassign data
                 using `np.core.multiarray.copyto` for `[:]` operator.
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disable by default.

    :returns: Whether or not the registration has been successful.
    """
    # Make sure data is at least 1d if numpy array, to avoid undefined `len`
    if isinstance(data, np.ndarray):
        data = np.atleast_1d(data)  # By reference

    # Make sure fields and data are consistent
    assert fields and len(fields) == len(data), (
        "fields and data are inconsistent.")

    # Default case: data is already a numpy array. Can be registered directly.
    if isinstance(data, np.ndarray):
        if np.issubsctype(data, np.float64):
            for i, field in enumerate(fields):
                assert not isinstance(fields, dict)
                if isinstance(fields[i], list):
                    fields[i] = [".".join(filter(None, (namespace, subfield)))
                                 for subfield in field]
                else:
                    fields[i] = ".".join(filter(None, (namespace, field)))
            hresult = controller.register_variables(fields, data)
            return hresult == jiminy.hresult_t.SUCCESS
        return False

    # Fallback to looping over fields and data iterators
    is_success = True
    if isinstance(fields, dict):
        fields = list(fields.values())
    if isinstance(data, dict):
        data = list(data.values())
    for subfields, value in zip(fields, data):
        assert isinstance(subfields, (dict, list))
        is_success = register_variables(
            controller, subfields, value, namespace)
        if not is_success:
            break
    return is_success
