"""Utilities operating over complex `Gym.Space`s associated with arbitrarily
nested data structure of `np.ndarray` and heavily optimized for speed.

They combine static control flow pre-computed for given space and eventually
some pre-allocated values with Just-In-Time (JIT) compiling via Numba when
possible for optimal performance.
"""
import math
from functools import partial
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence, MutableSequence
from typing import (
    Any, List, Dict, Optional, Union, Sequence as SequenceT, Tuple, Literal,
    Mapping as MappingT, SupportsFloat, TypeVar, Type, Callable, no_type_check,
    overload)

import numba as nb
import numpy as np
from numpy import typing as npt

import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    EncoderSensor, EffortSensor, array_copyto, multi_array_copyto)
from jiminy_py import tree


ValueT = TypeVar('ValueT')
ValueInT = TypeVar('ValueInT')
ValueOutT = TypeVar('ValueOutT')


StructNested = Union[MappingT[str, 'StructNested[ValueT]'],
                     SequenceT['StructNested[ValueT]'],
                     ValueT]
FieldNested = StructNested[str]
DataNested = StructNested[np.ndarray]
DataNestedT = TypeVar('DataNestedT', bound=DataNested)

ArrayOrScalar = Union[np.ndarray, SupportsFloat]


@no_type_check
@nb.jit(nopython=True, cache=True, fastmath=True)
def _array_clip(value: np.ndarray,
                low: Optional[ArrayOrScalar],
                high: Optional[ArrayOrScalar]) -> np.ndarray:
    """Element-wise out-of-place clipping of array elements.

    :param value: Array holding values to clip.
    :param low: Optional lower bound.
    :param high: Optional upper bound.
    """
    # Note that in-place clipping is actually slower than out-of-place in
    # Numba when 'fastmath' compilation flag is set.

    # Short circuit if there is neither low or high bounds
    if low is None and high is None:
        return value.copy()

    # Generic case.
    # Note that chaining `np.minimum` with `np.maximum` yields to better
    # performance than `np.clip` when 'fastmath' compilation flag is set.
    if value.ndim:
        if low is not None and high is not None:
            return np.minimum(np.maximum(value, low), high)
        if low is not None:
            return np.maximum(value, low)
        return np.minimum(value, high)

    # Scalar case.
    # Strangely, calling '.item()' on Python scalars is supported by Numba.
    out = value.item()
    if low is not None:
        out = max(out, low.item())
    if high is not None:
        out = min(out, high.item())
    return np.array(out)


@no_type_check
@nb.jit(nopython=True, cache=True, fastmath=True)
def _array_contains(value: np.ndarray,
                    low: Optional[ArrayOrScalar],
                    high: Optional[ArrayOrScalar]) -> bool:
    """Check that all array elements are withing bounds, up to some tolerance
    threshold.

    :param value: Array holding values to check.
    :param low: Optional lower bound.
    :param high: Optional upper bound.
    """
    value_ = np.asarray(value)
    if value_.ndim:
        value_1d = np.atleast_1d(value_)
        # Reversed bound check because 'all' is always true for empty arrays
        if low is not None and not (low <= value_1d).all():
            return False
        if high is not None and not (value_1d <= high).all():
            return False
        return True
    if low is not None and (low.item() > value_.item()):
        return False
    if high is not None and (value_.item() > high.item()):
        return False
    return True


def get_robot_state_space(robot: jiminy.Robot,
                          use_theoretical_model: bool = False,
                          ignore_velocity_limit: bool = True
                          ) -> gym.spaces.Dict:
    """Get the state space associated with a given robot.

    .. warning:
        This method is not meant to be overloaded in general since the
        definition of the state space is mostly consensual. One must rather
        overload `_initialize_observation_space` to customize the observation
        space as a whole.

    :param robot: Jiminy robot to consider.
    :param use_theoretical_model: Whether to compute the state space associated
                                  with the theoretical model instead of the
                                  extended simulation model.
    :param ignore_velocity_limit: Whether to ignore the velocity bounds
                                  specified in model.
    """
    # Define some proxies for convenience
    pinocchio_model = robot.pinocchio_model
    position_limit_lower = pinocchio_model.lowerPositionLimit
    position_limit_upper = pinocchio_model.upperPositionLimit
    velocity_limit = pinocchio_model.velocityLimit

    # Deduce bounds associated the theoretical model from the extended one
    if use_theoretical_model:
        position_limit_lower, position_limit_upper = map(
            robot.get_theoretical_position_from_extended,
            (position_limit_lower, position_limit_upper))
        velocity_limit = (
            robot.get_theoretical_velocity_from_extended(velocity_limit))

    # Ignore velocity bounds in requested
    if ignore_velocity_limit:
        velocity_limit = np.full_like(velocity_limit, float("inf"))

    # Aggregate position and velocity bounds to define state space
    return gym.spaces.Dict(OrderedDict(
        q=gym.spaces.Box(low=position_limit_lower,
                         high=position_limit_upper,
                         dtype=np.float64),
        v=gym.spaces.Box(low=float("-inf"),
                         high=float("inf"),
                         shape=(robot.pinocchio_model.nv,),
                         dtype=np.float64)))


def get_robot_measurements_space(robot: jiminy.Robot) -> gym.spaces.Dict:
    """Get the sensor space associated with a given robot.

    It gathers the sensors data in a dictionary. It maps each available type of
    sensor to the associated data matrix. Rows correspond to the sensor type's
    fields, and columns correspond to each individual sensor.

    .. note:
        The mapping between row `i` of data matrix and associated sensor type's
        field is given by:

        .. code-block:: python

            field = getattr(jiminy_py.core, key).fieldnames[i]

        The mapping between column `j` of data matrix and associated sensor
        name and object are given by:

        .. code-block:: python

            sensor = env.robot.sensors[key][j]

    :param robot: Jiminy robot to consider.
    """
    # Make sure that the robot is initialized
    assert robot.is_initialized

    # Define some proxies for convenience
    position_limit_lower = robot.pinocchio_model.lowerPositionLimit
    position_limit_upper = robot.pinocchio_model.upperPositionLimit

    # Initialize the bounds of the sensor space
    sensor_measurements = robot.sensor_measurements
    sensor_space_lower = OrderedDict(
        (key, np.full(value.shape, -np.inf))
        for key, value in sensor_measurements.items())
    sensor_space_upper = OrderedDict(
        (key, np.full(value.shape, np.inf))
        for key, value in sensor_measurements.items())

    # Replace inf bounds of the encoder sensor space
    for sensor in robot.sensors.get(EncoderSensor.type, ()):
        # Get the position bounds of the sensor.
        # Note that for rotary unbounded encoders, the sensor bounds cannot be
        # extracted from the motor because only the principal value of the
        # angle is observed by the sensor.
        assert isinstance(sensor, EncoderSensor)
        joint = robot.pinocchio_model.joints[sensor.joint_index]
        joint_type = jiminy.get_joint_type(joint)
        if joint_type == jiminy.JointModelType.ROTARY_UNBOUNDED:
            sensor_position_lower = - np.pi
            sensor_position_upper = + np.pi
        else:
            try:
                motor = robot.motors[sensor.motor_index]
                sensor_position_lower = motor.position_limit_lower
                sensor_position_upper = motor.position_limit_upper
            except IndexError:
                sensor_position_lower = position_limit_lower[joint.idx_q]
                sensor_position_upper = position_limit_upper[joint.idx_q]

        # Update the bounds accordingly
        sensor_space_lower[EncoderSensor.type][0, sensor.index] = (
            sensor_position_lower)
        sensor_space_upper[EncoderSensor.type][0, sensor.index] = (
            sensor_position_upper)

    # Replace inf bounds of the effort sensor space
    for sensor in robot.sensors.get(EffortSensor.type, ()):
        assert isinstance(sensor, EffortSensor)
        motor = robot.motors[sensor.motor_index]
        sensor_space_lower[EffortSensor.type][0, sensor.index] = (
            - motor.effort_limit)
        sensor_space_upper[EffortSensor.type][0, sensor.index] = (
            motor.effort_limit)

    return gym.spaces.Dict(OrderedDict(
        (key, gym.spaces.Box(low=min_val, high=max_val, dtype=np.float64))
        for (key, min_val), max_val in zip(
            sensor_space_lower.items(), sensor_space_upper.values())))


def get_bounds(space: gym.Space,
               tol_abs: float = 0.0,
               tol_rel: float = 0.0,
               ) -> Tuple[Optional[ArrayOrScalar], Optional[ArrayOrScalar]]:
    """Get the lower and upper bounds of a given 'gym.Space' if any.

    :param space: `gym.Space` on which to operate.
    :param tol_abs: Absolute tolerance.
                    Optional: 0.0 by default
    :param tol_rel: Relative tolerance. It will be ignored if either the lower
                    or upper is not specified.
                    Optional: 0.0 by default.

    :returns: Lower and upper bounds as a tuple.
    """
    # Extract lower and upper bounds depending on the gym space
    dtype: npt.DTypeLike
    low: Optional[ArrayOrScalar] = None
    high: Optional[ArrayOrScalar] = None
    if isinstance(space, gym.spaces.Box):
        low, high = space.low, space.high
        dtype = low.dtype
    if isinstance(space, gym.spaces.Discrete):
        low, high = space.start, space.n
        dtype = np.dtype(int)
    if isinstance(space, gym.spaces.MultiDiscrete):
        low, high = 0, space.nvec
        dtype = np.dtype(int)

    # Take into account the absolute and relative tolerances
    # assert tol_abs >= 0.0 and tol_rel >= 0.0
    if tol_abs or tol_rel and (low is not None or high is not None):
        tol_nd = np.full_like(low, tol_abs)
        if tol_rel and low is not None and high is not None:
            tol_nd = np.maximum(
                (high - low) * tol_rel, tol_nd)  # type: ignore[operator]
        if low is not None:
            low = (low - tol_nd).astype(dtype)
        if high is not None:
            high = (high + tol_nd).astype(dtype)

    return low, high


@no_type_check
def zeros(space: gym.Space[DataNestedT],
          dtype: npt.DTypeLike = None,
          enforce_bounds: bool = True) -> DataNestedT:
    """Allocate data structure from `gym.Space` and initialize it to zero.

    :param space: `gym.Space` on which to operate.
    :param dtype: Can be specified to overwrite original space dtype.
                  Optional: None by default
    """
    # Note that it is not possible to take advantage of `jiminy_py.tree`
    # because the output type for collections (OrderedDict or Tuple) is not the
    # same as the input one (gym.Space).
    value = None
    if isinstance(space, gym.spaces.Dict):
        value = OrderedDict()
        for field, subspace in space.items():
            value[field] = zeros(subspace, dtype=dtype)
        return value
    if isinstance(space, gym.spaces.Tuple):
        value = tuple(zeros(subspace, dtype=dtype) for subspace in space)
    elif isinstance(space, gym.spaces.Box):
        value = np.zeros(space.shape, dtype=dtype or space.dtype)
    elif isinstance(space, gym.spaces.Discrete):
        # Note that np.array of 0 dim is returned in order to be mutable
        value = np.array(0, dtype=dtype or np.int64)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        value = np.zeros_like(space.nvec, dtype=dtype or np.int64)
    elif isinstance(space, gym.spaces.MultiBinary):
        value = np.zeros(space.n, dtype=dtype or np.int8)
    if value is not None:
        if enforce_bounds:
            value = clip(value, space)
        return value
    if not isinstance(space, gym.Space):
        raise ValueError(
            "All spaces must derived from `gym.Space`, including tuple and "
            "dict containers.")
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


def fill(data: DataNested, fill_value: Union[float, int, np.number]) -> None:
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


def copyto(dst: DataNested, src: DataNested) -> None:
    """Copy arbitrarily nested data structure of 'np.ndarray' to a given
    pre-allocated destination.

    It avoids memory allocation completely, so that memory pointers of 'data'
    remains unchanged. As direct consequences, it is necessary to preallocate
    memory beforehand, and it only supports arrays of fixed shape.

    .. note::
        Unlike the function returned by 'build_copyto', only the flattened data
        structure needs to match, not the original one. This means that the
        source and/or destination can be flattened already when provided.
        Beware values must be sorted by keys in case of nested dict.

    :param dst: Hierarchical data structure to update, possibly flattened.
    :param value: Hierarchical data to copy, possibly flattened.
    """
    multi_array_copyto(tree.flatten(dst), tree.flatten(src))


@overload
def copy(data: DataNestedT) -> DataNestedT:
    ...


@overload
def copy(data: gym.Space[DataNestedT]) -> gym.Space[DataNestedT]:
    ...


def copy(data: Union[DataNestedT, gym.Space[DataNestedT]]
         ) -> Union[DataNestedT, gym.Space[DataNestedT]]:
    """Shallow copy recursively 'data' from `gym.Space`, so that only leaves
    are still references.

    :param data: Hierarchical data structure to copy without allocation.
    """
    return tree.unflatten_as(data, tree.flatten(data))


@no_type_check
def clip(data: DataNested, space: gym.Space[DataNested]) -> DataNested:
    """Clip data from `gym.Space` to make sure it is within bounds.

    .. note:
        None of the leaves of the returned data structured is sharing memory
        with the original one, even if clipping had no effect. This alleviate
        the need of calling 'deepcopy' afterward.

    :param data: Data to clip.
    :param space: `gym.Space` on which to operate.
    """
    data_type = type(data)
    if tree.issubclass_mapping(data_type):
        return data_type({
            field: clip(data[field], subspace)
            for field, subspace in space.items()})
    if tree.issubclass_sequence(data_type):
        return data_type([
            clip(data[i], subspace)
            for i, subspace in enumerate(space)])
    return _array_clip(data, *get_bounds(space))


@no_type_check
def contains(data: DataNested,
             space: gym.Space[DataNested],
             tol_abs: float = 0.0,
             tol_rel: float = 0.0) -> bool:
    """Check if all leaves of a nested data structure are within bounds of
    their respective `gym.Space`, up to some tolerance threshold. If both
    absolute and relative tolerances are provided, then satisfying only one of
    the two criteria is considered sufficient.

    By design, it is always `True` for all spaces but `gym.spaces.Box`,
    `gym.spaces.Discrete` and `gym.spaces.MultiDiscrete`.

    :param data: Data structure to check.
    :param space: `gym.Space` on which to operate.
    :param tol_abs: Absolute tolerance.
    :param tol_rel: Relative tolerance.
    """
    data_type = type(data)
    if tree.issubclass_mapping(data_type):
        return all(contains(data[field], subspace, tol_abs, tol_rel)
                   for field, subspace in space.items())
    if tree.issubclass_sequence(data_type):
        return all(contains(data[i], subspace, tol_abs, tol_rel)
                   for i, subspace in enumerate(space))
    return _array_contains(data, *get_bounds(space, tol_abs, tol_rel))


@no_type_check
def build_reduce(fn: Callable[..., ValueInT],
                 op: Optional[Callable[[ValueOutT, ValueInT], ValueOutT]],
                 dataset: SequenceT[DataNested],
                 space: Optional[gym.Space[DataNested]],
                 arity: Optional[Literal[0, 1]],
                 *args: Any,
                 initializer: Optional[Callable[[], ValueOutT]] = None,
                 forward_bounds: bool = True,
                 tol_abs: float = 0.0,
                 tol_rel: float = 0.0) -> Callable[..., ValueOutT]:
    """Generate specialized callable applying transform and reduction on all
    leaves of given nested space.

    .. note::
        Original ordering of the leaves is preserved. More precisely, both
        transform and reduction will be applied recursively in keys order.

    .. warning::
        It is assumed without checking that all nested data structures are
        consistent together and with the space if provided. It holds true both
        data known at generation-time or runtime. It is only required for data
        that may be provided at runtime to include the original data structure,
        so it may contain additional branches which will be ignored.

    .. warning::
        Providing additional data at runtime is supported but impede
        performance. Arity larger than 1 is not supported because the code path
        could not be fully specialized, causing dramatic slowdown.

    .. warning::
        There is no built-in 'short-circuit' mechanism, which means that it
        will go through all leaves systematically unless the reduction operator
        itself raises an exception.

    :param fn: Transform applied to every leaves of the nested data structures
               before performing the actual reduction. This function can
               perform in-place or out-of-place operations without restriction.
               `None` is not supported because it would be irrelevant. Note
               that if tracking the hierarchy during reduction is not
               necessary, then it would be way more efficient to first flatten
               the pre-allocated nested data structure once for all, and then
               perform reduction on this flattened view using the standard
               'functools.reduce' method. Still, flattening at runtime using
               'flatten' would still much slower than a specialized nested
               reduction.
    :param op: Optional reduction operator applied cumulatively on all leaves
               after transform. See 'functools.reduce' documentation for
               details. `None` to only apply transform on all leaves without
               reduction. This is useful when apply in-place transform.
    :param dataset: Pre-allocated nested data structure. Optional if the space
                    is provided.
    :param space: Container space on which to operate (eg `gym.spaces.Dict` or
                  `gym.spaces.Tuple`). Optional iif the nested data structure
                  is provided.
    :param arity: Arity of the generated callable. `None` to indicate that it
                  must be determined at runtime, which is slower.
    :param args: Extra arguments to systematically forward as transform input
                 for all leaves. Note that, as for Python built-ins methods,
                 keywords are not supported for the sake of efficiency.
    :param initializer: Function used to compute the initial value before
                        starting reduction. Optional if the reduction operator
                        has same input and output types. If `None`, then the
                        value corresponding to the first leaf after transform
                        will be used instead.
    :param forward_bounds: Whether to forward the lower and upper bounds of the
                           `gym.Space` associated with each leaf as transform
                           input. In this case, they will be added after the
                           data structure provided at runtime but before other
                           extra arguments if any. It is up to the user to make
                           sure all leaves have bounds, otherwise it will raise
                           an exception at generation-time. This argument is
                           ignored if not space is specified.
    :param tol_abs: Absolute tolerance added to the lower and upper bounds of
                    the `gym.Space` associated with each leaf.
                    Optional: 0.0 by default.
    :param tol_rel: Relative tolerance added to the lower and upper bounds of
                    the `gym.Space` associated with each leaf.
                    Optional: 0.0 by default.

    :returns: Fully-specialized reduction callable.
    """
    # pylint: disable=unused-argument

    def _build_reduce(
            arity: Literal[0, 1],
            is_initialized: bool,
            fn_1: Union[Callable[..., ValueInT], Callable[..., ValueOutT]],
            field_1: Union[str, int],
            fn_2: Union[Callable[..., ValueInT], Callable[..., ValueOutT]],
            field_2: Union[str, int],
            ) -> Callable[..., ValueOutT]:
        """Internal method generating a specialized callable performing a
        single reduction operation on either leaf transform and/or already
        branch reduction.

        :param arity: Arity of the generated callable.
        :param is_initialized: Whether the output has already been initialized
                               at this point. The first reduction is the only
                               one to initialize the output, either by calling
                               the initializer if provided or passing directly
                               the output of first transform call otherwise.
        :param fn_1: Leaf transform or branch reduction to call last.
        :param field_1: Pass the value corresponding this key as input argument
                        for nested data structure provided at runtime if any
                        iif callable 'fn_1' is a leaf transform.
        :param is_out_1: Whether callable 'fn_1' is already a branch reduction.
        :param fn_2: Leaf transform or branch reduction to call first.
        :param field_2: Same as 'field_1' for callable 'fn_2'.
        :param is_out_2: Same as 'is_out_1' for callable 'fn_2'.

        :returns: Specialized branch reduction callable requiring passing the
                  current reduction output as input if some reduction operator
                  has been specified.
        """
        # Extract extra arguments from functor if necessary to preserve order
        has_args = False
        is_out_1, is_out_2 = fn_1.func is not fn, fn_2.func is not fn
        if not is_out_1:
            fn_1, dataset, args_1 = fn_1.func, fn_1.args[:-1], fn_1.args[-1]
            has_args |= bool(args_1)
            if arity == 0:
                fn_1 = partial(fn_1, *dataset, *args_1)
            elif dataset:
                fn_1 = partial(fn_1, *dataset)
        if not is_out_2:
            fn_2, dataset, args_2 = fn_2.func, fn_2.args[:-1], fn_2.args[-1]
            has_args |= bool(args_2)
            if arity == 0:
                fn_2 = partial(fn_2, *dataset, *args_2)
            elif dataset:
                fn_2 = partial(fn_2, *dataset)

        # Specialization if no op is specified
        if op is None:
            if arity == 0:
                def _reduce(fn_1, fn_2):
                    fn_2()
                    fn_1()
                return partial(_reduce, fn_1, fn_2)
            if is_out_1 and is_out_2:
                def _reduce(fn_1, fn_2, delayed):
                    fn_2(delayed)
                    fn_1(delayed)
                return partial(_reduce, fn_1, fn_2)
            if is_out_1 and not is_out_2:
                if has_args:
                    def _reduce(fn_1, fn_2, field_2, args_2, delayed):
                        fn_2(delayed[field_2], *args_2)
                        fn_1(delayed)
                    return partial(_reduce, fn_1, fn_2, field_2, args_2)

                def _reduce(fn_1, fn_2, field_2, delayed):
                    fn_2(delayed[field_2])
                    fn_1(delayed)
                return partial(_reduce, fn_1, fn_2, field_2)
            if not is_out_1 and is_out_2:
                if has_args:
                    def _reduce(fn_1, field_1, args_1, fn_2, delayed):
                        fn_2(delayed)
                        fn_1(delayed[field_1], *args_1)
                    return partial(_reduce, fn_1, field_1, args_1, fn_2)

                def _reduce(fn_1, field_1, fn_2, delayed):
                    fn_2(delayed)
                    fn_1(delayed[field_1])
                return partial(_reduce, fn_1, field_1, fn_2)
            if has_args:
                def _reduce(
                        fn_1, field_1, args_1, fn_2, field_2, args_2, delayed):
                    fn_2(delayed[field_2], *args_2)
                    fn_1(delayed[field_1], *args_1)
                return partial(
                    _reduce, fn_1, field_1, args_1, fn_2, field_2, args_2)

            def _reduce(fn_1, field_1, fn_2, field_2, delayed):
                fn_2(delayed[field_2])
                fn_1(delayed[field_1])
            return partial(_reduce, fn_1, field_1, fn_2, field_2)

        # Specialization if op is specified
        if arity == 0:
            if is_initialized:
                if is_out_1 and is_out_2:
                    def _reduce(fn_1, fn_2, out):
                        return fn_1(fn_2(out))
                    return partial(_reduce, fn_1, fn_2)
                if is_out_1 and not is_out_2:
                    def _reduce(op, fn_1, fn_2, out):
                        return fn_1(op(out, fn_2()))
                elif not is_out_1 and is_out_2:
                    def _reduce(op, fn_1, fn_2, out):
                        return op(fn_2(out), fn_1())
                else:
                    def _reduce(op, fn_1, fn_2, out):
                        return op(op(out, fn_2()), fn_1())
                return partial(_reduce, op, fn_1, fn_2)
            if is_out_1 and not is_out_2:
                def _reduce(fn_1, fn_2, out):
                    return fn_1(fn_2())
                return partial(_reduce, fn_1, fn_2)
            if not is_out_1 and not is_out_2:
                def _reduce(op, fn_1, fn_2, out):
                    return op(fn_2(), fn_1())
            return partial(_reduce, op, fn_1, fn_2)
        if is_initialized:
            if is_out_1 and is_out_2:
                def _reduce(fn_1, fn_2, out, delayed):
                    return fn_1(fn_2(out, delayed), delayed)
                return partial(_reduce, fn_1, fn_2)
            if is_out_1 and not is_out_2:
                def _reduce(op, fn_1, fn_2, field_2, args_2, out, delayed):
                    return fn_1(
                        op(out, fn_2(delayed[field_2], *args_2)), delayed)
                return partial(_reduce, op, fn_1, fn_2, field_2, args_2)
            if not is_out_1 and is_out_2:
                def _reduce(op, fn_1, field_1, args_1, fn_2, out, delayed):
                    return op(
                        fn_2(out, delayed), fn_1(delayed[field_1], *args_1))
                return partial(_reduce, op, fn_1, field_1, args_1, fn_2)

            def _reduce(
                    op, fn_1, field_1, args_1, fn_2, field_2, args_2, out,
                    delayed):
                return op(op(out, fn_2(delayed[field_2], *args_2)),
                          fn_1(delayed[field_1], *args_1))
            return partial(
                _reduce, op, fn_1, field_1, args_1, fn_2, field_2, args_2)
        if is_out_1 and not is_out_2:
            def _reduce(fn_1, fn_2, field_2, args_2, out, delayed):
                return fn_1(fn_2(delayed[field_2], *args_2), delayed)
            return partial(_reduce, fn_1, fn_2, field_2, args_2)

        def _reduce(  # type: ignore[no-redef]
                op, fn_1, field_1, args_1, fn_2, field_2, args_2, out,
                delayed):
            return op(fn_2(delayed[field_2], *args_2),
                      fn_1(delayed[field_1], *args_1))
        return partial(
            _reduce, op, fn_1, field_1, args_1, fn_2, field_2, args_2)

    def _build_forward(
            arity: Literal[0, 1],
            parent: Optional[Union[str, int]],
            is_initialized: bool,
            post_fn: Union[Callable[..., ValueInT], Callable[..., ValueOutT]],
            field: Optional[Union[str, int]],
            ) -> Union[Callable[..., ValueInT], Callable[..., ValueOutT]]:
        """Internal method generating a specialized callable forwarding the
        value associated with a given key for nested data structure provided at
        runtime if any as input argument of some leaf transform or branch
        reduction callable.

        The callable is not a reduction at this point, so doing it here since
        it is the very last moment before main entry-point returns.

        :param arity: Arity of the generated callable.
        :param is_initialized: Whether the output has already been initialized.
        :param parent: Parent key to forward.
        :param post_fn: Leaf transform or branch reduction.

        :returns: Specialized key-forwarding callable.
        """
        is_out = post_fn.func is not fn
        if not is_out:
            # Extract extra arguments from functor to preserve arguments order
            dataset, args = post_fn.args[:-1], post_fn.args[-1]
            post_fn = post_fn.func
            has_args = bool(args)
            if arity == 0:
                post_fn = partial(post_fn, *dataset, *args)
            elif dataset:
                post_fn = partial(post_fn, *dataset)

            # Specialization if no op is specified
            if op is None:
                if arity == 0:
                    def _forward(post_fn):
                        post_fn()
                    return partial(_forward, post_fn)
                if has_args:
                    if parent is None and field is None:
                        def _forward(post_fn, args, delayed):
                            post_fn(delayed, *args)
                        return partial(_forward, post_fn, args)
                    if (parent is None) ^ (field is None):
                        def _forward(post_fn, field, args, delayed):
                            post_fn(delayed[field], *args)
                        return partial(
                            _forward, post_fn, parent or field, args)

                    def _forward(post_fn, parent, field, args, delayed):
                        post_fn(delayed[parent][field], *args)
                    return partial(_forward, post_fn, parent, field, args)
                if parent is None and field is None:
                    def _forward(post_fn, delayed):
                        post_fn(delayed)
                    return partial(_forward, post_fn)
                if (parent is None) ^ (field is None):
                    def _forward(post_fn, field, delayed):
                        post_fn(delayed[field])
                    return partial(_forward, post_fn, parent or field)

                def _forward(post_fn, parent, field, delayed):
                    post_fn(delayed[parent][field])
                return partial(_forward, post_fn, parent, field)

            # Specialization if op is specified
            if arity == 0:
                if is_initialized:
                    def _forward(op, post_fn, out):
                        return op(out, post_fn())
                    return partial(_forward, op, post_fn)

                def _forward(post_fn, out):
                    return post_fn()
                return partial(_forward, post_fn)
            if is_initialized:
                if parent is None and field is None:
                    def _forward(op, post_fn, args, out, delayed):
                        return op(out, post_fn(delayed, *args))
                    return partial(_forward, op, post_fn, args)
                if (parent is None) ^ (field is None):
                    def _forward(op, post_fn, field, args, out, delayed):
                        return op(out, post_fn(delayed[field], *args))
                    return partial(
                        _forward, op, post_fn, parent or field, args)

                def _forward(op, post_fn, parent, field, args, out, delayed):
                    return op(out, post_fn(delayed[parent][field], *args))
                return partial(_forward, op, post_fn, parent, field, args)
            if parent is None and field is None:
                def _forward(post_fn, args, out, delayed):
                    return post_fn(delayed, *args)
                return partial(_forward, post_fn, args)
            if (parent is None) ^ (field is None):
                def _forward(post_fn, field, args, out, delayed):
                    return post_fn(delayed[field], *args)
                return partial(_forward, post_fn, parent or field, args)

            def _forward(post_fn, parent, field, args, out, delayed):
                return post_fn(delayed[parent][field], *args)
            return partial(_forward, post_fn, parent, field, args)

        # No key to forward for main entry-point of zero arity
        if parent is None or arity == 0:
            return post_fn

        # Forward key in all other cases
        if op is None:
            def _forward(post_fn, field, delayed):
                return post_fn(delayed[field])
        else:
            def _forward(post_fn, field, out, delayed):
                return post_fn(out, delayed[field])
        return partial(_forward, post_fn, parent)

    def _build_transform_and_reduce(
            arity: Literal[0, 1],
            parent: Optional[Union[str, int]],
            is_initialized: bool,
            dataset: SequenceT[DataNested],
            space: Optional[gym.Space[DataNested]]) -> Optional[
                Union[Callable[..., ValueInT], Callable[..., ValueOutT]]]:
        """Internal method for generating specialized callable applying
        transform and reduction on all leaves of a nested space recursively.

        :param arity: Arity of the generated callable.
        :param parent: Key of parent space mapping to space if any, `None`
                       otherwise.
        :param is_initialized: Whether the output has already been initialized
                               at this point. See `_build_reduce` for details.
        :param data: Possibly nested pre-allocated data.
        :param space: Possibly nested space on which to operate.

        :returns: Specialized transform if the space is a actually a leaf,
                  otherwise a specialized transform and reduction callable
                  still requiring passing the current reduction output as input
                  if some reduction operator has been specified. `None` if
                  nested data structure if empty.
        """
        # Determine top-level keys if nested data structure
        keys: Optional[Union[SequenceT[int], SequenceT[str]]] = None
        space_or_data = space
        if space_or_data is None and dataset:
            space_or_data = dataset[0]
        if isinstance(space_or_data, Mapping):
            keys = space_or_data.keys()
        elif isinstance(space_or_data, Sequence):
            keys = range(len(space_or_data))
        else:
            assert isinstance(space_or_data, (gym.Space, np.ndarray))

        # Return specialized transform if leaf
        if keys is None:
            post_fn = fn if not dataset else partial(fn, *dataset)
            post_args = args
            if forward_bounds and space is not None:
                post_args = (
                    *get_bounds(space, tol_abs, tol_rel), *post_args)
            post_fn = partial(post_fn, post_args)
            if parent is None:
                post_fn = _build_forward(
                    arity, parent, is_initialized, post_fn, None)
            return post_fn
        if not keys:
            return None

        # Generate transform and reduce method if branch
        field_prev, field, out_fn = None, None, None
        for field in keys:
            values = [data[field] for data in dataset]
            subspace = None if space is None else space[field]
            must_initialize = not is_initialized and len(keys) == 1
            post_fn = _build_transform_and_reduce(
                arity, field, not must_initialize, values, subspace)
            if post_fn is None:
                continue
            if out_fn is None:
                out_fn = post_fn
            else:
                out_fn = _build_reduce(
                    arity, is_initialized, post_fn, field, out_fn, field_prev)
                is_initialized = True
            field_prev = field
        if out_fn is None:
            return None
        return _build_forward(arity, parent, is_initialized, out_fn, field)

    def _dispatch(
            post_fn_0: Callable[[], ValueOutT],
            post_fn_1: Callable[[DataNested], ValueOutT],
            *delayed: Tuple[DataNested]) -> ValueOutT:
        """Internal method for handling unknown arity at generation-time.

        :param post_fn_0: Nullary specialized transform and reduce callable.
        :param post_fn_1: Unary specialized transform and reduce callable.
        :param delayed: Optional nested data structure any provided at runtime.

        :returns: Specialized transform and reduce callable of dynamic arity.
        """
        if not delayed:
            return post_fn_0()
        return post_fn_1(delayed[0])

    def _build_init(
            arity: Literal[0, 1],
            post_fn: Callable[..., ValueOutT]) -> Callable[..., ValueOutT]:
        """Internal method generating a specialized callable initializing the
        output if a reduction operator and a dedicated initializer has been
        specified.

        :param post_fn: Specialized transform and reduce callable.

        :returns: Specialized transform and reduce callable only taking nested
                  data structures as input.
        """
        if post_fn is None:
            if initializer is None:
                return lambda *args, **kwargs: None
            return initializer
        if op is None:
            return post_fn
        if initializer is None:
            return partial(post_fn, None)
        if arity == 0:
            def _initialize(post_fn, initializer):
                return post_fn(initializer())
        else:
            def _initialize(post_fn, initializer, delayed):
                return post_fn(initializer(), delayed)
        return partial(_initialize, post_fn, initializer)

    # Check that the combination of input arguments are valid
    if space is None and not dataset:
        raise TypeError("At least one dataset or the space must be specified.")
    if arity not in (0, 1, None):
        raise TypeError("Arity must be either 0, 1 or `None`.")
    if isinstance(fn, partial):
        raise TypeError("Transform function cannot be 'partial' instance.")

    # Generate transform and reduce callable of various arity if necessary
    all_fn = [None, None]
    for i in (0, 1):
        if arity is not None and i != arity:
            continue
        is_initialized = op is not None and initializer is not None
        all_fn[i] = _build_init(i, _build_transform_and_reduce(
            i, None, is_initialized, dataset, space))

    # Return callable of requested arity if specified, dynamic dispatch if not
    if arity is None:
        return partial(_dispatch, *all_fn)
    return all_fn[arity]


@no_type_check
def build_map(fn: Callable[..., ValueT],
              data: Optional[DataNested],
              space: Optional[gym.Space[DataNested]],
              arity: Optional[Literal[0, 1]],
              *args: Any,
              forward_bounds: bool = True,
              tol_abs: float = 0.0,
              tol_rel: float = 0.0) -> Callable[[], StructNested[ValueT]]:
    """Generate specialized callable returning applying out-of-place transform
    to all leaves of given nested space.

    .. warning::
        This method systematically allocates memory to store the resulting
        nested data structure, which is costly. If pre-allocation is possible,
        it would more efficient to use `build_reduce` without operator instead.

    .. warning::
        Providing additional data at runtime is supported but impede
        performance. Arity larger than 1 is not supported because the code path
        could not be fully specialized, causing dramatic slowdown.

    .. warning::
        It is assumed without check that all nested data structures are
        consistent together and with the space if provided. It holds true both
        data known at generation-time or runtime. Yet, it is only required for
        data provided at runtime if any to include the original data structure,
        so it may contain additional branches which will be ignored.

    :param fn: Transform applied to every leaves of the nested data structures.
               This function is supposed to allocate its own memory while
               performing some out-of-place operations then return the outcome.
    :param data: Pre-allocated nested data structure. Optional iif the space is
                 provided. This enables generating specialized random sampling
                 methods for instance.
    :param space: `gym.spaces.Dict` on which to operate. Optional iif the
                  nested data structure is provided.
    :param arity: Arity of the generated callable. `None` to indicate that it
                  must be determined at runtime, which is slower.
    :param args: Extra arguments to systematically forward as transform input
                 for all leaves. Note that, as for Python built-ins methods,
                 keywords are not supported for the sake of efficiency.
    :param forward_bounds: Whether to forward the lower and upper bounds of the
                           `gym.Space` associated with each leaf as transform
                           input. In this case, they will be added after the
                           data structure provided at runtime but before other
                           extra arguments if any. It is up to the user to make
                           sure all leaves have bounds, otherwise it will raise
                           an exception at generation-time. This argument is
                           ignored if not space is specified.
                           Optional: `True` by default.
    :param tol_abs: Absolute tolerance added to the lower and upper bounds of
                    the `gym.Space` associated with each leaf.
                    Optional: 0.0 by default.
    :param tol_rel: Relative tolerance added to the lower and upper bounds of
                    the `gym.Space` associated with each leaf.
                    Optional: 0.0 by default.

    :returns: Fully-specialized mapping callable.
    """
    def _build_setitem(
            arity: Literal[0, 1],
            self_fn: Optional[Callable[..., Dict[str, StructNested[ValueT]]]],
            value_fn: Callable[..., StructNested[ValueT]],
            key: Optional[Union[str, int]]
            ) -> Callable[..., Dict[str, StructNested[ValueT]]]:
        """Internal method generating a specialized item assignment callable
        responsible for populating a parent transformed nested data structure
        with either some child branch already transformed or some leaf to be
        transformed.

        This method aims to be composed with itself for recursively creating
        the whole transformed nested data structure.

        :param arity: Arity of the generated callable.
        :param self_fn: Parent branch transform.
        :param value_fn: Child leaf or branch transform.
        :param key: Field of the parent transformed nested data structure that
                    must be populated with the output of the child transform.

        :returns: Specialized item assignment callable.
        """
        # Extract extra arguments from functor if necessary to preserve order
        is_out, has_args = False, False
        if isinstance(value_fn, partial):
            is_out = value_fn.func is not fn
            if not is_out:
                dataset, args = value_fn.args[:-1], value_fn.args[-1]
                value_fn = value_fn.func
                has_args = bool(args)
                if arity == 0:
                    value_fn = partial(value_fn, *dataset, *args)
                elif dataset:
                    value_fn = partial(value_fn, *dataset)

        is_mapping = isinstance(key, str)
        if arity == 0:
            if key is None:
                return value_fn
            if is_mapping:
                def _setitem(self_fn, value_fn, key):
                    self = self_fn()
                    self[key] = value_fn()
                    return self
                return partial(_setitem, self_fn, value_fn, key)

            def _setitem(self_fn, value_fn):
                self = self_fn()
                self.append(value_fn())
                return self
            return partial(_setitem, self_fn, value_fn)
        if has_args:
            if key is None:
                def _setitem(value_fn, args, delayed):
                    return value_fn(delayed, *args)
                return partial(_setitem, value_fn, args)
            if is_mapping:
                def _setitem(self_fn, value_fn, key, args, delayed):
                    self = self_fn(delayed)
                    self[key] = value_fn(delayed[key], *args)
                    return self
                return partial(_setitem, self_fn, value_fn, key, args)

            def _setitem(self_fn, value_fn, key, args, delayed):
                self = self_fn(delayed)
                self.append(value_fn(delayed[key], *args))
                return self
            return partial(_setitem, self_fn, value_fn, key, args)
        if key is None:
            return value_fn
        if is_mapping:
            def _setitem(self_fn, value_fn, key, delayed):
                self = self_fn(delayed)
                self[key] = value_fn(delayed[key])
                return self
            return partial(_setitem, self_fn, value_fn, key)

        def _setitem(  # type: ignore[no-redef]
                self_fn, value_fn, key, delayed):
            self = self_fn(delayed)
            self.append(value_fn(delayed[key]))
            return self
        return partial(_setitem, self_fn, value_fn, key)

    def _build_map(
            arity: Literal[0, 1],
            parent: Optional[str],
            data: Optional[DataNested],
            space: Optional[gym.Space[DataNested]]
            ) -> Callable[..., Dict[str, StructNested[ValueT]]]:
        """Internal method for generating specialized callable applying
        out-of-place transform to all leaves of given nested space.

        :param arity: Arity of the generated callable.
        :param parent: Key of parent space mapping to space if any, `None`
                       otherwise.
        :param data: Possibly nested pre-allocated data.
        :param space: Possibly nested space on which to operate.

        :returns: Specialized leaf or branch transform.
        """
        # Determine top-level keys if nested data structure
        keys: Optional[Union[SequenceT[int], SequenceT[str]]] = None
        space_or_data = data if data is not None else space
        if isinstance(space_or_data, Mapping):
            keys = space_or_data.keys()
            if isinstance(space_or_data, gym.spaces.Dict):
                if data is None:
                    container_cls = OrderedDict
                else:
                    container_cls = type(space_or_data)
            elif isinstance(space_or_data, MutableMapping):
                container_cls = type(space_or_data)
            else:
                container_cls = dict
        elif isinstance(space_or_data, Sequence):
            keys = range(len(space_or_data))
            if isinstance(space_or_data, gym.spaces.Tuple):
                if data is None:
                    container_cls = list
                else:
                    container_cls = type(space_or_data)
            elif isinstance(space_or_data, MutableSequence):
                container_cls = type(space_or_data)
            else:
                container_cls = list
        else:
            assert isinstance(space_or_data, (gym.Space, np.ndarray))

        # Return specialized transform if leaf
        if keys is None:
            post_fn = fn if data is None else partial(fn, data)
            post_args = args
            if forward_bounds and space is not None:
                post_args = (
                    *get_bounds(space, tol_abs, tol_rel), *post_args)
            post_fn = partial(post_fn, post_args)
            if parent is None:
                post_fn = _build_setitem(arity, None, post_fn, None)
            return post_fn

        # Create new empty container to all transformed values.
        # FIXME: Immutable containers should be instantiated at the end.
        def _create(cls: Type[ValueT], *args: Any) -> ValueT:
            # pylint: disable=unused-argument
            return cls()
        out_fn = partial(_create, container_cls)

        # Apply map recursively while preserving order using monadic operations
        for field in keys:
            value = None if data is None else data[field]
            subspace = None if space is None else space[field]
            post_fn = _build_map(arity, field, value, subspace)
            out_fn = _build_setitem(arity, out_fn, post_fn, field)

        return out_fn

    def _dispatch(
            post_fn_0: Callable[[], Dict[str, StructNested[ValueT]]],
            post_fn_1: Callable[
                [Dict[str, DataNested]], Dict[str, StructNested[ValueT]]],
            *delayed: Tuple[Dict[str, DataNested]]
            ) -> Dict[str, StructNested[ValueT]]:
        """Internal method for handling unknown arity at generation-time.

        :param post_fn_0: Nullary specialized map callable.
        :param post_fn_1: Unary specialized map callable.
        :param delayed: Optional nested data structure any provided at runtime.

        :returns: Specialized map callable of dynamic arity.
        """
        if not delayed:
            return post_fn_0()
        return post_fn_1(delayed[0])

    # Check that the combination of input arguments are valid
    if space is None and data is None:
        raise TypeError("At least data or space must be specified.")
    if arity not in (0, 1, None):
        raise TypeError("Arity must be either 0, 1 or `None`.")
    if isinstance(fn, partial):
        raise TypeError("Transform function cannot be 'partial' instance.")

    # Generate transform and reduce callable of various arity if necessary
    all_fn = [None, None]
    for i in (0, 1):
        if arity is not None and i != arity:
            continue
        all_fn[i] = _build_map(i, None, data, space)

    # Return callable of requested arity if specified, dynamic dispatch if not
    if arity is None:
        return partial(_dispatch, *all_fn)
    return all_fn[arity]


def build_copyto(dst: DataNested) -> Callable[[DataNested], None]:
    """Generate specialized `copyto` method for a given pre-allocated
    destination.

    Note that the key ordering of source and destination data structures do NOT
    have to match, only the hierarchy has to be the same. Beside, additional
    subtrees of the output data structure will be ignored if any.

    :param dst: Nested data structure to be updated.
    """
    # Special case if parent is not a container
    dst_type = type(dst)
    if not (tree.issubclass_mapping(dst_type) or
            tree.issubclass_sequence(dst_type)):
        return partial(array_copyto, dst)

    # Build specialized flattening method, appending all leaves in a buffer
    src_flat: List[np.ndarray] = []
    flatten = build_reduce(
        src_flat.append, None, (), dst, 1, forward_bounds=False)

    # Flatten the nested data structure to update on-the-spot for efficiency
    dst_flat = tree.flatten(dst)

    # Define helper that gathers all operations
    def _flatten_and_copyto(src_flat: List[np.ndarray],
                            flatten: Callable[[DataNested], None],
                            dst_flat: Sequence[np.ndarray],
                            src: DataNested) -> None:
        """Internal method that flattens the input data structure before
        copying data from source to destination all at once.

        :param src_flat: Buffer storing the result of the specialized
                         flattening operator.
        :param flatten: Flattening operator specialized for a given output data
                        structure.
        :param src_flat: Pre-computed flattened output data structure.
        :param src: Nested input data structure whose output data structure is
                    a subtree.
        """
        # Populate buffer with flattened input data structure
        src_flat.clear()
        flatten(src)

        # Copy from source to destination all arrays at once
        multi_array_copyto(dst_flat, src_flat)

    return partial(_flatten_and_copyto, src_flat, flatten, dst_flat)


def build_clip(data: DataNested,
               space: gym.Space[DataNested]) -> Callable[[], DataNested]:
    """Generate specialized `clip` method for some pre-allocated nested data
    structure and corresponding space.

    :param data: Nested data structure whose leaves must be clipped.
    :param space: `gym.Space` on which to operate.
    """
    return build_map(_array_clip, data, space, 0)


def build_contains(data: DataNested,
                   space: gym.Space[DataNested],
                   tol_abs: float = 0.0,
                   tol_rel: float = 0.0) -> Callable[[], bool]:
    """Generate specialized `contains` method for some pre-allocated nested
    data structure and corresponding space.

    :param data: Pre-allocated nested data structure whose leaves must be
                 within bounds if defined and ignored otherwise.
    :param space: `gym.Space` on which to operate.
    :param tol_abs: Absolute tolerance for floating point equality check.
                    Optional: `0.0` by default.
    :param tol_rel: Relative tolerance for floating point aprox equality check.
                    Optional: `0.0` by default.
    """
    # Define a special exception involved in short-circuit mechanism
    class ShortCircuitContains(Exception):
        """Internal exception involved in short-circuit mechanism.
        """

    @nb.jit(nopython=True, cache=True)
    def _contains_or_raises(value: np.ndarray,
                            low: Optional[ArrayOrScalar],
                            high: Optional[ArrayOrScalar]) -> bool:
        """Thin wrapper around original `_array_contains` method to raise
        an exception if the test fails. It enables short-circuit mechanism
        to abort checking remaining leaves if any.

        Short-circuit mechanism not only speeds-up scenarios where at least one
        leaf does not met requirements and also the other scenarios where all
        tests passes since it is no longer necessary to specify the reduction
        operator 'operator.and' to keep track of the result.

        :param value: Array holding values to check.
        :param low: Lower bound.
        :param high: Upper bound.
        """
        if not _array_contains(value, low, high):
            raise ShortCircuitContains("Short-circuit exception.")
        return True

    def _exception_handling(out_fn: Callable[[], bool]) -> bool:
        """Internal method for short-circuit exception handling.

        :param out_fn: specialized contain callable raising short-circuit
                        exception as soon as one leaf fails the test.

        :returns: `True` if all leaves are within bounds of their
                    respective space, `False` otherwise.
        """
        try:
            out_fn()
        except ShortCircuitContains:
            return False
        return True

    return partial(_exception_handling, build_reduce(
        _contains_or_raises,
        None,
        (data,),
        space,
        arity=0,
        forward_bounds=True,
        tol_abs=tol_abs,
        tol_rel=tol_rel))


def build_normalize(space: gym.Space[DataNested],
                    dst: DataNested,
                    src: Optional[DataNested] = None,
                    *,
                    ignore_unbounded: bool = False,
                    is_reversed: bool = False) -> Callable[..., None]:
    """Generate a normalization or de-normalization method specialized for a
    given pre-allocated destination.

    .. note::
        The generated method applies element-wise de-normalization to all
        elements of the leaf spaces having finite bounds. For those that does
        not, it simply copies the value from 'src' to 'dst'.

    .. warning::
        This method requires all leaf spaces to have type `gym.spaces.Box`
        with dtype 'np.floating'.

    :param dst: Nested data structure to updated.
    :param space: Original (de-normalized) `gym.Space` on which to operate.
    :param src: Normalized nested data if 'is_reversed' is True, original data
                (de-normalized) otherwise. `None` to pass it at runtime.
                Optional: `None` by default.
    :param is_reversed: True to de-normalize, False to normalize.
    """
    @nb.jit(nopython=True, cache=True)
    def _array_normalize(dst: np.ndarray,
                         src: np.ndarray,
                         low: np.ndarray,
                         high: np.ndarray,
                         is_reversed: bool) -> None:
        """Element-wise normalization or de-normalization of array.

        :param dst: Pre-allocated array into which the result must be stored.
        :param src: Input array.
        :param low: Lower bound.
        :param high: Upper bound.
        :param is_reversed: True to de-normalize, False to normalize.
        """
        for i, (lo, hi, val) in enumerate(zip(low.flat, high.flat, src.flat)):
            if not np.isfinite(lo) or not np.isfinite(hi):
                dst.flat[i] = val
            elif is_reversed:
                dst.flat[i] = (lo + hi - val * (lo - hi)) / 2
            else:
                dst.flat[i] = (lo + hi - 2 * val) / (lo - hi)

    # Make sure that all leaves are `gym.space.Box` with `floating` dtype
    for subspace in tree.flatten(space):
        assert isinstance(subspace, gym.spaces.Box)
        assert np.issubdtype(subspace.dtype, np.floating)
        if not ignore_unbounded and not subspace.is_bounded():
            raise RuntimeError(
                "All leaf spaces must be bounded if `ignore_unbounded=False`.")

    dataset = [dst,]
    if src is not None:
        dataset.append(src)
    return build_reduce(
        _array_normalize, None, dataset, space, 2 - len(dataset), is_reversed)


def build_flatten(data_nested: DataNested,
                  data_flat: Optional[DataNested] = None,
                  *, is_reversed: Optional[bool] = None
                  ) -> Callable[..., None]:
    """Generate a flattening or un-flattening method specialized for some
    pre-allocated nested data.

    .. note::
        Multi-dimensional leaf spaces are supported. Values will be flattened
        in 1D vectors using 'C' order (row-major). It ignores the actual memory
        layout the leaves of 'data_nested' and they are not required to have
        the same dtype as 'data_flat'.

    :param data_nested: Nested data structure.
    :param data_flat: Flat array consistent with the nested data structure.
                      Optional iif `is_reversed` is `True`.
                      Optional: `None` by default.
    :param is_reversed: True to update 'data_flat' (flattening), 'data_nested'
                        otherwise (un-flattening).
                        Optional: True if 'data_flat' is specified, False
                        otherwise.
    """
    # Make sure that the input arguments are valid
    if is_reversed is None:
        is_reversed = data_flat is None
    assert is_reversed or data_flat is not None

    # Flatten nested data while preserving leaves ordering
    data_leaves = tree.flatten(data_nested)

    # Compute slices to split destination in accordance with nested data.
    # It will be passed to `build_reduce` as an input dataset. It is kind of
    # hacky since only passing `DataNested` instances is officially supported,
    # but it is currently the easiest way to keep track of some internal state
    # and specify leaf-specific constants.
    start_indices, stop_indices = [], []
    idx_start = 0
    for data in data_leaves:
        idx_end = idx_start + max(math.prod(data.shape), 1)
        start_indices.append(idx_start)
        stop_indices.append(idx_end)
        idx_start = idx_end

    @nb.jit(nopython=True, cache=True)
    def _flatten(data: np.ndarray,
                 idx_start: int,
                 idx_end: int,
                 data_flat: np.ndarray,
                 is_reversed: bool) -> None:
        """Synchronize the flatten and un-flatten representation of the data
        associated with the same leaf space.

        In practice, it assigns the value of a 1D array slice to some multi-
        dimensional array, or the other way around.

        :param data: Multi-dimensional array that will be either updated or
                     copied as a whole depending on 'is_reversed'.
        :param idx_start: First index of the slice of 'data_flat' to
                          synchronized with 'data'.
        :param idx_end: One-after-last index of the slice of 'data_flat' to
                        synchronized with 'data'.
        :param data_flat: 1D array from which to extract that will be either
                          updated or copied depending on 'is_reversed'.
        :param is_reversed: True to update the multi-dimensional array 'data'
                            by copying the value from slice 'flat_slice' of
                            vector 'data_flat', False for doing the contrary.
        """
        # Note that passing slices as input argument is very slow in numba
        if is_reversed:
            data.ravel()[:] = data_flat[idx_start:idx_end]
        else:
            data_flat[idx_start:idx_end] = data.ravel()

    args = (is_reversed,)
    if data_flat is not None:
        args = (data_flat, *args)  # type: ignore[assignment]
    arity = 2 - len(args)
    out_fn = build_reduce(_flatten, None, (
        data_leaves, start_indices, stop_indices), None, arity, *args)
    if data_flat is None:
        def _repeat(out_fn: Callable[[DataNested], None],
                    n_leaves: int,
                    delayed: DataNested) -> None:
            """Dispatch flattened data provided at runtime to each transform
            '_flatten' specialized for all leaves of the original nested space.

            In practice, it simply repeats the flattened data as many times as
            the number of leaves of the original nested space before passing
            them altogether in a tuple as input argument of a function.

            :param out_fn: Flattening or un-flattening method already
                           specialized for a given pre-allocated nested data.
            :param n_leaves: Total number of leaves in original nested space.
            :param delayed: Flattened data provided at runtime.
            """
            out_fn((delayed,) * n_leaves)

        out_fn = partial(_repeat, out_fn, len(data_leaves))
    return out_fn
