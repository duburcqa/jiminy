""" TODO: Write documentation.
"""
import logging
from typing import Union, ValuesView

import gym
import tree
import numpy as np
import numba as nb

import jiminy_py.core as jiminy

from .spaces import FieldNested, DataNested, zeros


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


def get_fieldnames(structure: Union[FieldNested, DataNested],
                   namespace: str = "") -> FieldNested:
    """Generate generic fieldnames from `gym..Space`, so that it can be used
    in conjunction with `register_variables`, to register any value from gym
    Space to the telemetry conveniently.

    :param space: Gym.Space on which to operate.
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disabled by default.
    """
    # Create dummy data structure if gym.Space is provided
    if isinstance(structure, gym.Space):
        structure = zeros(structure)

    fieldnames = []
    for fieldname_path, data in tree.flatten_with_path(structure):
        assert isinstance(data, np.ndarray), (
            "'structure' ({structure}) must have leaves of type `np.ndarray`.")
        if data.size < 1:
            # Empty: return empty list
            fieldname = []
        elif data.size == 1:
            # Scalar: fieldname path is enough
            fieldname = [".".join(filter(None, (namespace, *fieldname_path)))]
        else:
            # Tensor: basic numbering
            fieldname = np.array([
                ".".join(filter(None, (namespace, *fieldname_path, str(i))))
                for i in range(data.size)]).reshape(data.shape).tolist()
        fieldnames.append(fieldname)

    return tree.unflatten_as(structure, fieldnames)


def register_variables(controller: jiminy.AbstractController,
                       fieldnames: Union[
                           ValuesView[FieldNested], FieldNested],
                       data: DataNested) -> bool:
    """Register data from `Gym.Space` to the telemetry of a controller.

    .. warning::
        Variables are registered by reference. Consequently, the user is
        responsible to manage the lifetime of the data to avoid it being
        garbage collected, and to make sure the variables are updated
        when necessary, and only when it is.

    :param controller: Robot's controller of the simulator used to register
                       variables to the telemetry.
    :param fieldnames: Nested variable names, as returned by `get_fieldnames`
                       method. It can be a nested list or/and dict. The leaves
                       are str corresponding to the name of each scalar data.
    :param data: Data from `gym.spaces.Space` to register. Note that the
                 telemetry stores pointers to the underlying memory, so it
                 only supports np.float64, and make sure to reassign data
                 using `np.copyto` or `[:]` operator (faster).

    :returns: Whether or not the registration has been successful.
    """
    # pylint: disable=cell-var-from-loop
    for fieldname, value in zip(
            tree.flatten_up_to(data, fieldnames),
            tree.flatten(data)):
        if np.issubsctype(value, np.float64):
            assert isinstance(fieldname, list), (
                "'fieldname' ({fieldname}) should be a list of strings.")
            hresult = controller.register_variables(fieldname, value)
            if hresult != jiminy.hresult_t.SUCCESS:
                return False
        else:
            logger.warning(
                f"Variable of dtype '{value.dtype}' cannot be registered to "
                "the telemetry and must have dtype 'np.float64' instead.")
            return False
    return True
