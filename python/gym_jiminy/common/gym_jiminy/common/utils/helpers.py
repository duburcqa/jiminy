""" TODO: Write documentation.
"""
import math
import logging
from typing import Sequence, Union, ValuesView

import gymnasium as gym
import tree
import numpy as np
import numba as nb

import jiminy_py.core as jiminy

from .spaces import FieldNested, DataNested, zeros


logger = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True, inline='always')
def is_breakpoint(t: float, dt: float, eps: float) -> bool:
    """Check if 't' is multiple of 'dt' at a given precision 'eps'.

    :param t: Current time.
    :param dt: Timestep.
    :param eps: Precision.

    :meta private:
    """
    if dt < eps:
        return True
    dt_prev = t % dt
    return (dt_prev < eps / 2) or (dt - dt_prev <= eps / 2)


@nb.jit(nopython=True, cache=True, inline='always')
def is_nan(value: np.ndarray) -> bool:
    """Check if any value of a numpy array is nan.

    .. warning::
        This method does not implement any short-circuit mechanism as it is
        optimized for arrays that are unlikely to contain nan values.

    :param value: N-dimensional array.
    """
    if value.ndim:
        return np.isnan(value).any()
    return math.isnan(value.item())


def get_fieldnames(structure: Union[gym.Space[DataNested], DataNested],
                   namespace: str = "") -> FieldNested:
    """Generate generic fieldnames for a given nested data structure, so that
    it can be used in conjunction with `register_variables`, to register any
    value from gym space to the telemetry conveniently.

    :param structure: Nested data structure on which to operate.
    :param namespace: Namespace used to prepend fields, using '.' delimiter.
                      Empty string to disable.
                      Optional: Disabled by default.
    """
    # Create dummy data structure if gym.Space is provided
    if isinstance(structure, gym.Space):
        structure = zeros(structure)

    fieldnames = []
    fieldname_path: Sequence[Union[str, int]]
    for fieldname_path, data in tree.flatten_with_path(structure):
        fieldname_path = (namespace, *fieldname_path)
        assert isinstance(data, np.ndarray), (
            "'structure' ({structure}) must have leaves of type `np.ndarray`.")
        if data.size < 1:
            # Empty: return empty list
            fieldname = []
        elif data.size == 1:
            # Scalar: fieldname path is enough
            fieldname = [".".join(map(str, filter(None, fieldname_path)))]
        else:
            # Tensor: basic numbering
            fieldname = np.array([
                ".".join(map(str, filter(None, (*fieldname_path, i))))
                for i in range(data.size)]).reshape(data.shape).tolist()
        fieldnames.append(fieldname)

    return tree.unflatten_as(structure, fieldnames)


def register_variables(controller: jiminy.AbstractController,
                       fieldnames: Union[
                           ValuesView[FieldNested], FieldNested],
                       data: DataNested) -> bool:
    """Register data from `Gym.Space` to the telemetry of a controller.

    .. warning::
        Variables are registered by reference. This is necessary because, under
        the hood, Jiminy telemetry stores pointers to the underlying memory for
        efficiency. Consequently, the user is responsible to manage the
        lifetime of the data to avoid it being garbage collected, and to make
        sure the variables are updated by reassigning its value instead of
        re-allocating memory, using either `np.copyto`, `[:]` operator, or
        `jiminy.array_copyto` (from  slowest to fastest).

    .. warning::
        The telemetry only supports `np.float64` or `np.int64` dtypes.

    :param controller: Robot's controller of the simulator used to register
                       variables to the telemetry.
    :param fieldnames: Nested variable names, as returned by `get_fieldnames`
                       method. It can be a nested list or/and dict. The leaves
                       are str corresponding to the name of each scalar data.
    :param data: Data from `gym.spaces.Space` to register.

    :returns: Whether the registration has been successful.
    """
    for fieldname, value in zip(
            tree.flatten_up_to(data, fieldnames),
            tree.flatten(data)):
        if any(np.issubsctype(value, type) for type in (np.float64, np.int64)):
            assert isinstance(fieldname, list), (
                f"'fieldname' ({fieldname}) should be a list of strings.")
            hresult = controller.register_variables(fieldname, value)
            if hresult != jiminy.hresult_t.SUCCESS:
                return False
        else:
            logger.warning(
                "Variables of dtype '%s' cannot be registered to the "
                "telemetry. It must have dtype 'np.float64' or 'np.int64'.",
                value.dtype)
            return False
    return True
