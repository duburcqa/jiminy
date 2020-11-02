""" TODO: Write documentation.
"""
from collections import OrderedDict
from typing import Optional, Union, Dict, List

import numpy as np
import numba as nb
import gym

import jiminy_py.core as jiminy


SpaceDictRecursive = Union[  # type: ignore
    Dict[str, 'SpaceDictRecursive'], np.ndarray]  # type: ignore
ListStrRecursive = List[Union[str, 'ListStrRecursive']]  # type: ignore
FieldDictRecursive = Union[  # type: ignore
    Dict[str, 'FieldDictRecursive'], ListStrRecursive]  # type: ignore


def _clamp(space: gym.Space, x: SpaceDictRecursive) -> SpaceDictRecursive:
    """Clamp an element from Gym.Space to make sure it is within bounds.

    :meta private:
    """
    if isinstance(space, gym.spaces.Dict):
        return OrderedDict(
            (k, _clamp(subspace, x[k]))
            for k, subspace in space.spaces.items())
    if isinstance(space, gym.spaces.Box):
        return np.clip(x, space.low, space.high)
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


def set_value(data: SpaceDictRecursive,
              fill_value: SpaceDictRecursive) -> None:
    """Set to zero data from `Gym.Space`.
    """
    if isinstance(data, dict):
        for sub_data, sub_val in zip(data.values(), fill_value.values()):
            set_value(sub_data, sub_val)
    elif isinstance(data, np.ndarray):
        data[:] = fill_value
    else:
        raise NotImplementedError(
            f"Data of type {type(data)} is not supported by this method.")


def set_zeros(data: SpaceDictRecursive) -> None:
    """Set to zero data from `Gym.Space`.
    """
    if isinstance(data, dict):
        for value in data.values():
            set_zeros(value)
    elif isinstance(data, np.ndarray):
        data.fill(0.0)
    else:
        raise NotImplementedError(
            f"Data of type {type(data)} is not supported by this method.")


def register_variables(controller: jiminy.AbstractController,
                       field: FieldDictRecursive,
                       data: SpaceDictRecursive,
                       namespace: Optional[str] = None) -> None:
    """Register data from `Gym.Space` to the telemetry of a controller.

    .. warning::
        Variables are registered by reference. Consequently, the user is
        responsible to manage the lifetime of the data to avoid it being
        garbage collected, and to make sure the variables are updated
        when necessary, and only when it is.
    """
    assert data is not None and len(field) == len(data), (
        "field and data are inconsistent.")
    if isinstance(field, dict):
        for subfield, value in zip(field.values(), data.values()):
            register_variables(controller, subfield, value, namespace)
    elif isinstance(field, list) and isinstance(field[0], list):
        for subfield, value in zip(field, data):
            register_variables(controller, subfield, value, namespace)
    else:
        if namespace is not None:
            field = [".".join((namespace, name)) for name in field]
        controller.register_variables(field, data)
