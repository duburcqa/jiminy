"""Miscellaneous utilities that have no place anywhere else but are very useful
nonetheless.
"""
import math
import logging
from functools import partial
from typing import Any, List, Sequence, ValuesView, Optional, Union, Protocol

import gymnasium as gym
import numpy as np
import numba as nb

import jiminy_py.core as jiminy
from jiminy_py import tree

from .spaces import FieldNested, DataNested, ArrayOrScalar, zeros


LOGGER = logging.getLogger(__name__)

GLOBAL_RNG = np.random.default_rng()


FieldNestedSequence = Sequence[Union['FieldNestedSequence', str]]
FieldNestedList = List[Union[FieldNestedSequence, str]]


class RandomDistribution(Protocol):
    """Protocol that must be satisfied for passing a generic callable as
    custom statistical distribution to `sample` method.
    """
    def __call__(self, rg: np.random.Generator, *args: Any, **kwargs: Any
                 ) -> ArrayOrScalar:
        ...


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
                   namespace: str = "") -> FieldNestedList:
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

    fieldnames: FieldNestedList = []
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
                ".".join(map(str, (*filter(None, fieldname_path), i)))
                for i in range(data.size)]).reshape(data.shape).tolist()
        fieldnames.append(fieldname)

    return tree.unflatten_as(structure, fieldnames)


def register_variables(controller: jiminy.AbstractController,
                       fieldnames: Union[
                           ValuesView[FieldNested], FieldNested],
                       data: DataNested) -> None:
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
    :param data: Data from `gym.Space` to register.
    """
    for fieldname, value in zip(
            tree.flatten_up_to(data, fieldnames), tree.flatten(data)):
        assert isinstance(fieldname, list), (
            f"'fieldname' ({fieldname}) should be a list of strings.")
        controller.register_variables(fieldname, value)


def sample(low: Union[float, np.ndarray] = -1.0,
           high: Union[float, np.ndarray] = 1.0,
           dist: Union[str, RandomDistribution] = 'uniform',
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
    :param dist: The statistical from which to draw samples, either provided as
                 a pre-defined string or a callable. For strings, then it must
                 be a member function of `np.random.Generator` (only 'uniform'
                 and 'normal' are supported for now). For callables, it must
                 corresponds to a standardized distribution and satisfying
                 `gym_jiminy.common.utils.RandomDistribution` protocol. This is
                 especially useful for specifying custom parameters of complex
                 distributions such as Beta. Using `functools.partial` is
                 recommended, eg `partial(np.random.Generator.Beta, a=1, b=8)`.
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
    # Compute mean and deviation from low and high arguments
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
                    f"'shape' {shape} must be broadcast-able with 'low', "
                    f"'high' and 'scale' {dev.shape} if specified.") from e

    # Define "standardized" distribution callable if only its name was provided
    if isinstance(dist, str):
        if dist not in ('uniform', 'normal'):
            raise NotImplementedError(
                f"'{dist}' distribution type is not supported for now.")
        dist_fn = getattr(np.random.Generator, dist)
        if dist == 'uniform':
            # The uniform distribution is NOT standardized by default
            dist_fn = partial(dist_fn, low=-1.0, high=1.0)
    else:
        dist_fn = dist

    # Generate samples from distribution.
    # Make sure that the result is always returned as np.ndarray.
    value = np.asarray(dist_fn(rg or GLOBAL_RNG, size=shape))

    # Apply mean and standard deviation transformation
    value = mean + dev * value

    # Revert log scale if requested
    if enable_log_scale:
        value = 10 ** value

    return value
