""" TODO: Write documentation.
"""
from typing import Union, Dict, List

import numpy as np
import numba as nb
from gym import spaces

import jiminy_py.core as jiminy

from .spaces import FieldDictNested, SpaceDictNested


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
    """Get generic fieldnames from `gym.spaces.Space`, so that it can be used
    in conjunction with `register_variables`, to register any value from gym
    Space to the telemetry conveniently.

    :param space: Gym.Space on which to operate.
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disabled by default.
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
                 using `np.copyto` or `[:]` operator (faster).
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disabled by default.

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
