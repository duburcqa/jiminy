""" TODO: Write documentation.
"""
from typing import Optional, Union, Dict, Sequence

import numpy as np
import numba as nb
import gym

import jiminy_py.core as jiminy


SpaceDictNested = Union[  # type: ignore
    Dict[str, 'SpaceDictNested'], np.ndarray]  # type: ignore
ListStrRecursive = Sequence[Union[str, 'ListStrRecursive']]  # type: ignore
FieldDictNested = Union[  # type: ignore
    Dict[str, 'FieldDictNested'], ListStrRecursive]  # type: ignore


def sample(low: Union[float, np.ndarray] = -1.0,
           high: Union[float, np.ndarray] = 1.0,
           dist: str = 'uniform',
           scale: Union[float, np.ndarray] = 1.0,
           enable_log_scale: bool = False,
           size: Optional[Sequence[int]] = None,
           rg: Optional[np.random.RandomState] = None):
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
        if size is None:
            size = dev.shape
        else:
            raise ValueError(
                "One cannot specify 'size' if 'low' and 'high' are vectors.")

    # Sample from normalized distribution.
    # Note that some distributions are not normalized by default
    if rg is None:
        rg = np.random
    distrib_fn = getattr(rg, dist)
    val = distrib_fn(size=size)
    if dist == 'uniform':
        val = 2.0 * (val - 0.5)

    # Set mean and deviation
    val = mean + dev * val

    # Revert log scale if appropriate
    if enable_log_scale:
        val = 10 ** val

    return val


def zeros(space: gym.Space) -> Union[SpaceDictNested, int]:
    """Set to zero data from `Gym.Space`.
    """
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Dict):
        value = {}
        for field, subspace in space.spaces.items():
            value[field] = zeros(subspace)
        return value
    if isinstance(space, gym.spaces.Discrete):
        return np.array(0)  # Using np.array of 0 dim to be mutable
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


def fill(data: SpaceDictNested,
         fill_value: float) -> None:
    """Set every element of 'data' from `Gym.Space` to scalar 'fill_value'.
    """
    if isinstance(data, np.ndarray):
        data.fill(fill_value)
    elif isinstance(data, dict):
        for sub_data in data.values():
            fill(sub_data, fill_value)
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
    beforehand, and to work with fixed size buffers.

    .. note::
        If 'data' is a dictionary, 'value' must be a subtree of 'data',
        whose leaf values must be broadcastable with the ones of 'data'.
    """
    if isinstance(data, np.ndarray):
        try:
            np.core.umath.copyto(data, value)
        except TypeError as e:
            raise TypeError(f"Cannot cast '{data}' to '{value}'.") from e
    elif isinstance(data, dict):
        for field, sub_val in value.items():
            set_value(data[field], sub_val)
    else:
        raise NotImplementedError(
            f"Data of type {type(data)} is not supported.")


def copy(data: SpaceDictNested) -> SpaceDictNested:
    """Shadow copy recursively 'data' from `Gym.Space`, so that only leaves
    are still references.
    """
    if isinstance(data, dict):
        value = {}
        for field, sub_data in data.items():
            value[field] = copy(sub_data)
        return value
    return data


def clip(space: gym.Space, value: SpaceDictNested) -> SpaceDictNested:
    """Clamp value from Gym.Space to make sure it is within bounds.
    """
    if isinstance(space, gym.spaces.Box):
        return np.core.umath.clip(value, space.low, space.high)
    if isinstance(space, gym.spaces.Dict):
        return dict(
            (k, clip(subspace, value[k]))
            for k, subspace in space.spaces.items())
    if isinstance(space, gym.spaces.Discrete):
        return value  # No need to clip Discrete space.
    raise NotImplementedError(
        f"Gym.Space of type {type(space)} is not supported by this "
        "method.")


@nb.jit(nopython=True, nogil=True)
def _is_breakpoint(t: float, dt: float, eps: float) -> bool:
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


def register_variables(controller: jiminy.AbstractController,
                       field: FieldDictNested,
                       data: SpaceDictNested,
                       namespace: Optional[str] = None) -> bool:
    """Register data from `Gym.Space` to the telemetry of a controller.

    .. warning::
        Variables are registered by reference. Consequently, the user is
        responsible to manage the lifetime of the data to avoid it being
        garbage collected, and to make sure the variables are updated
        when necessary, and only when it is.

    :returns: Whether or not the registration has been successful.
    """
    assert field and len(field) == len(data), (
        "field and data are inconsistent.")
    if isinstance(field, dict):
        is_success = True
        for subfield, value in zip(field.values(), data.values()):
            hresult = register_variables(
                controller, subfield, value, namespace)
            is_success = is_success and (hresult == jiminy.hresult_t.SUCCESS)
    elif isinstance(field[0], str):
        if namespace is not None:
            field = [".".join((namespace, name)) for name in field]
        hresult = controller.register_variables(field, data)
        is_success = (hresult == jiminy.hresult_t.SUCCESS)
    elif isinstance(field[0], (list, tuple)):
        is_success = True
        for subfield, value in zip(field, data):
            hresult = register_variables(
                controller, subfield, value, namespace)
            is_success = is_success and (hresult == jiminy.hresult_t.SUCCESS)
    else:
        raise ValueError(f"Unsupported field type '{type(field)}'.")
    return is_success
