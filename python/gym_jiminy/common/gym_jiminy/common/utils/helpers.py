""" TODO: Write documentation.
"""
import logging
from typing import Union, Dict, List, ValuesView, Tuple, Iterable

import numpy as np
import numba as nb

import jiminy_py.core as jiminy

from .spaces import FieldDictNested, SpaceDictNested


logger = logging.getLogger(__name__)


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


def get_fieldnames(variable: SpaceDictNested,
                   namespace: str = "") -> FieldDictNested:
    """Get generic fieldnames from `gym.spaces.Space`, so that it can be used
    in conjunction with `register_variables`, to register any value from gym
    Space to the telemetry conveniently.

    :param space: Gym.Space on which to operate.
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disabled by default.
    """
    if isinstance(variable, np.ndarray):
        # Empty: return empty list
        if variable.size < 1:
            return []
        # Scalar: namespace alone as fieldname is enough
        if variable.size == 1:
            return [namespace]
        # Tensor: basic numbering
        return np.array([
            ".".join(filter(None, (namespace, str(i))))
            for i in range(variable.size)]).reshape(variable.shape).tolist()

    # Fancy action spaces: trying to be clever and infer meaningful telemetry
    # names based on action space.
    if isinstance(variable, (dict, tuple)):
        if not variable:
            return []
        if isinstance(variable, tuple):
            value_items: Iterable[Tuple[str, SpaceDictNested]] = (
                (str(i), value) for i, value in enumerate(variable))
        else:
            value_items = dict.items(variable)
        out: List[Union[Dict[str, FieldDictNested], str]] = []
        for field, value in value_items:
            if isinstance(value, (dict, tuple)):
                out.append({field: get_fieldnames(value, namespace)})
            else:
                out.append(field)
        return out

    raise NotImplementedError(
        f"'variable' type `{type(variable)}` is not supported.")


def register_variables(controller: jiminy.AbstractController,
                       fields: Union[
                           ValuesView[FieldDictNested], FieldDictNested],
                       data: SpaceDictNested) -> bool:
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
    :param data: Data from `gym.spaces.Space` to register. Note that the
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
    assert len(fields) == len(data), (
        f"fields ({fields}) and data ({data}) are inconsistent.")

    # Default case: data is already a numpy array. Can be registered directly.
    if isinstance(data, np.ndarray):
        if np.issubsctype(data, np.float64):
            assert isinstance(fields, list)
            hresult = controller.register_variables(fields, data)
            return hresult == jiminy.hresult_t.SUCCESS
        else:
            logger.warning(
                f"Variable of dtype '{data.dtype}' cannot be registered to "
                "the telemetry and must have dtype 'np.float64' instead.")
        return False

    # Fallback to looping over fields and data iterators
    is_success = True
    if isinstance(fields, dict):
        fields = fields.values()
    for subfields, value in zip(fields, data.values()):
        assert isinstance(subfields, (dict, list))
        is_success = register_variables(controller, subfields, value)
        if not is_success:
            break
    return is_success
