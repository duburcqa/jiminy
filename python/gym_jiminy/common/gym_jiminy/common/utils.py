""" TODO: Write documentation.
"""
from collections import OrderedDict
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


def zeros(space: gym.Space) -> Union[SpaceDictNested, int]:
    """Set to zero data from `Gym.Space`.
    """
    if isinstance(space, gym.spaces.Dict):
        value = OrderedDict()
        for field, subspace in space.spaces.items():
            value[field] = zeros(subspace)
        return value
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return 0
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported by this method.")


def fill(data: SpaceDictNested,
         fill_value: float) -> None:
    """Set every element of 'data' from `Gym.Space` to scalar 'fill_value'.
    """
    if isinstance(data, dict):
        for sub_data in data.values():
            fill(sub_data, fill_value)
    elif isinstance(data, np.ndarray):
        data.fill(fill_value)
    else:
        raise NotImplementedError(
            f"Data of type {type(data)} is not supported by this method.")


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
    if isinstance(data, dict):
        for field, sub_val in value.items():
            set_value(data[field], sub_val)
    elif isinstance(data, np.ndarray):
        np.copyto(data, value)
    else:
        raise NotImplementedError(
            f"Data of type {type(data)} is not supported by this method.")


def copy(data: SpaceDictNested) -> SpaceDictNested:
    """Shadow copy recursively 'data' from `Gym.Space`, so that only leaves
    are still references.
    """
    if isinstance(data, dict):
        value = data.__class__()
        for field, sub_data in data.items():
            value[field] = copy(sub_data)
        return value
    return data


def _clamp(space: gym.Space, value: SpaceDictNested) -> SpaceDictNested:
    """Clamp an element from Gym.Space to make sure it is within bounds.

    :meta private:
    """
    if isinstance(space, gym.spaces.Dict):
        return OrderedDict(
            (k, _clamp(subspace, value[k]))
            for k, subspace in space.spaces.items())
    if isinstance(space, gym.spaces.Box):
        return np.clip(value, space.low, space.high)
    if isinstance(space, gym.spaces.Discrete):
        return np.clip(value, 0, space.n)
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
    assert data is not None and len(field) == len(data), (
        "field and data are inconsistent.")
    if isinstance(field, dict):
        is_success = True
        for subfield, value in zip(field.values(), data.values()):
            hresult = register_variables(
                controller, subfield, value, namespace)
            is_success = is_success and (hresult == jiminy.hresult_t.SUCCESS)
    elif isinstance(field, list) and isinstance(field[0], list):
        is_success = True
        for subfield, value in zip(field, data):
            hresult = register_variables(
                controller, subfield, value, namespace)
            is_success = is_success and (hresult == jiminy.hresult_t.SUCCESS)
    else:
        if namespace is not None:
            field = [".".join((namespace, name)) for name in field]
        hresult = controller.register_variables(field, data)
        is_success = (hresult == jiminy.hresult_t.SUCCESS)
    return is_success
