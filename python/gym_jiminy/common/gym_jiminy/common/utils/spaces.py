""" TODO: Write documentation.
"""
import math
from functools import partial
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence, MutableSequence
from typing import (
    Any, Dict, Optional, Union, Sequence as SequenceT, Mapping as MappingT,
    Iterable as IterableT, Tuple, Literal, SupportsFloat, TypeVar, Type,
    Callable, Protocol, no_type_check, cast)

import numba as nb
import numpy as np
from numpy import typing as npt

import tree
import gymnasium as gym

from jiminy_py.core import array_copyto  # pylint: disable=no-name-in-module


GLOBAL_RNG = np.random.default_rng()


ValueT = TypeVar('ValueT')
ValueInT = TypeVar('ValueInT')
ValueOutT = TypeVar('ValueOutT')

StructNested = Union[MappingT[str, 'StructNested[ValueT]'],
                     IterableT['StructNested[ValueT]'],
                     ValueT]
FieldNested = StructNested[str]
DataNested = StructNested[np.ndarray]
DataNestedT = TypeVar('DataNestedT', bound=DataNested)

ArrayOrScalar = Union[np.ndarray, SupportsFloat]
ArrayOrScalarT = TypeVar('ArrayOrScalarT', bound=ArrayOrScalar)


class RandomDistribution(Protocol):
    """Protocol to be satisfied when passing generic statistical distribution
    callable to `sample` method.
    """
    def __call__(self, rg: np.random.Generator, *args: Any, **kwargs: Any
                 ) -> ArrayOrScalar:
        ...


@no_type_check
@nb.jit(nopython=True, cache=True)
def _array_clip(value: np.ndarray,
                low: Optional[ArrayOrScalar],
                high: Optional[ArrayOrScalar]) -> np.ndarray:
    """Element-wise out-of-place clipping of array elements.

    :param value: Array holding values to clip.
    :param low: Optional lower bound.
    :param high: Optional upper bound.
    """
    if low is None and high is None:
        return value.copy()
    if value.ndim:
        out_nd = value
        if low is not None:
            out_nd = np.maximum(out_nd, low)
        if high is not None:
            out_nd = np.minimum(out_nd, high)
        return out_nd
    # Surprisingly, calling '.item()' on python scalars is supported by numba
    out_0d = value.item()
    if low is not None:
        out_0d = max(out_0d, low.item())
    if high is not None:
        out_0d = min(out_0d, high.item())
    return np.array(out_0d)


@no_type_check
@nb.jit(nopython=True, cache=True)
def _array_contains(value: np.ndarray,
                    low: Optional[ArrayOrScalar],
                    high: Optional[ArrayOrScalar],
                    tol_abs: float,
                    tol_rel: float) -> bool:
    """Check that all array elements are withing bounds, up to some tolerance
    threshold. If both absolute and relative tolerances are provided, then
    satisfying only one of the two criteria is considered sufficient.

    :param value: Array holding values to check.
    :param low: Optional lower bound.
    :param high: Optional upper bound.
    :param tol_abs: Absolute tolerance.
    :param tol_rel: Relative tolerance. It will be ignored if either the lower
                    or upper is not specified.
    """
    if value.ndim:
        tol_nd = np.full_like(value, tol_abs)
        if low is not None and high is not None and tol_rel > 0.0:
            tol_nd = np.maximum((high - low) * tol_rel, tol_nd)
        # Reversed bound check because 'all' is always true for empty arrays
        if low is not None and not (low - tol_nd <= value).all():
            return False
        if high is not None and not (value <= high + tol_nd).all():
            return False
        return True
    tol_0d = tol_abs
    if low is not None and high is not None and tol_rel > 0.0:
        tol_0d = max((high.item() - low.item()) * tol_rel, tol_0d)
    if low is not None and (low.item() - tol_0d > value.item()):
        return False
    if high is not None and (value.item() > high.item() + tol_0d):
        return False
    return True


def _unflatten_as(structure: StructNested[Any],
                  flat_sequence: SequenceT[DataNested]) -> DataNested:
    """Unflatten a sequence into a given structure.

    .. seealso::
        This method is the same as 'tree.unflatten_as' without runtime checks.

    :param structure: Arbitrarily nested structure.
    :param flat_sequence: Sequence to unflatten.

    :returns: 'flat_sequence' unflattened into 'structure'.
    """
    if not tree.is_nested(structure):
        return flat_sequence[0]
    _, packed = tree._packed_nest_with_indices(structure, flat_sequence, 0)
    return tree._sequence_like(structure, packed)


def get_bounds(space: gym.Space
               ) -> Tuple[Optional[ArrayOrScalar], Optional[ArrayOrScalar]]:
    """Get the lower and upper bounds of a given 'gym.Space' if any.

    :param space: `gym.Space` on which to operate.

    :returns: Lower and upper bounds as a tuple.
    """
    if isinstance(space, gym.spaces.Box):
        return space.low, space.high
    if isinstance(space, gym.spaces.Discrete):
        return space.start, space.n
    if isinstance(space, gym.spaces.MultiDiscrete):
        return 0, space.nvec
    return None, None


def sample(low: Union[float, np.ndarray] = -1.0,
           high: Union[float, np.ndarray] = 1.0,
           dist: Union[str, RandomDistribution] = 'uniform',
           scale: Union[float, np.ndarray] = 1.0,
           enable_log_scale: bool = False,
           shape: Optional[SequenceT[int]] = None,
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


def is_bounded(space_nested: gym.Space) -> bool:
    """Check wether a `gym.Space` has finite bounds.

    :param space: `gym.Space` on which to operate.
    """
    for space in tree.flatten(space_nested):
        is_bounded_fn = getattr(space, "is_bounded", None)
        if is_bounded_fn is not None and not is_bounded_fn():
            return False
    return True


@no_type_check
def zeros(space: gym.Space[DataNestedT],
          dtype: npt.DTypeLike = None,
          enforce_bounds: bool = True) -> DataNestedT:
    """Allocate data structure from `gym.Space` and initialize it to zero.

    :param space: `gym.Space` on which to operate.
    :param dtype: Can be specified to overwrite original space dtype.
                  Optional: None by default
    """
    # Note that it is not possible to take advantage of dm-tree because the
    # output type for collections (OrderedDict or Tuple) is not the same as the
    # input one (gym.Space). This feature request would be too specific.
    value = None
    if isinstance(space, gym.spaces.Dict):
        value = OrderedDict()
        for field, subspace in space.spaces.items():
            value[field] = zeros(subspace, dtype=dtype)
        return value
    if isinstance(space, gym.spaces.Tuple):
        value = tuple(zeros(subspace, dtype=dtype)
                      for subspace in space.spaces.values())
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
    for data, value in zip(*map(tree.flatten, (dst, src))):
        array_copyto(data, value)


def copy(data: DataNestedT) -> DataNestedT:
    """Shallow copy recursively 'data' from `gym.Space`, so that only leaves
    are still references.

    :param data: Hierarchical data structure to copy without allocation.
    """
    return cast(DataNestedT, _unflatten_as(data, tree.flatten(data)))


def clip(data: DataNested, space: gym.Space[DataNested]) -> DataNested:
    """Clip data from `gym.Space` to make sure it is within bounds.

    .. note:
        None of the leaves of the returned data structured is sharing memory
        with the original one, even if clipping had no effect. This alleviate
        the need of calling 'deepcopy' afterward.

    :param data: Data to clip.
    :param space: `gym.Space` on which to operate.
    """
    # FIXME: Add support of `gym.spaces.Tuple`
    if not isinstance(space, gym.spaces.Dict):
        return _array_clip(data, *get_bounds(space))
    assert isinstance(data, dict)

    out: Dict[str, DataNested] = OrderedDict()
    for field, subspace in space.spaces.items():
        out[field] = clip(data[field], subspace)
    return out


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
    if not isinstance(space, gym.spaces.Dict):
        return _array_contains(data, *get_bounds(space), tol_abs, tol_rel)
    assert isinstance(data, dict)

    return all(contains(data[field], subspace, tol_abs, tol_rel)
               for field, subspace in space.spaces.items())


@no_type_check
def build_reduce(fn: Callable[..., ValueInT],
                 op: Optional[Callable[[ValueOutT, ValueInT], ValueOutT]],
                 dataset: SequenceT[DataNested],
                 space: Optional[gym.spaces.Dict],
                 arity: Optional[Literal[0, 1]],
                 *args: Any,
                 initializer: Optional[Callable[[], ValueOutT]] = None,
                 forward_bounds: bool = True) -> Callable[..., ValueOutT]:
    """Generate specialized callable applying transform and reduction on all
    leaves of given nested space.

    .. note::
        Original ordering of the leaves is preserved. More precisely, both
        transform and reduction will be applied recursively in keys order.

    .. warning::
        It is assumed without check that all nested data structures are
        consistent together and with the space if provided. It holds true both
        data known at generation-time or runtime. Yet, it is only required for
        data provided at runtime if any to include the original data structure,
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
               before performing the actual reduction. This function can do
               some in-place or out-of-place operations without restriction.
               `None` is not supported because it would be irrelevant. It is
               way more efficient to flatten the pre-allocated nested data
               structure once for all and perform reduction on this flattened
               view using 'functools.reduce' method instead. Note that flatten
               at runtime using 'tree.flatten' would still much faster than
               a specialized nested reduction doing list concatenation.
    :param op: Optional reduction operator applied cumulatively on all leaves
               after transform. See 'functools.reduce' documentation for
               details. `None` to only apply transform on all leaves without
               reduction. This is useful when apply in-place transform.
    :param data: Pre-allocated nested data structure. Optional if the space is
                 provided but hardly relevant.
    :param space: `gym.spaces.Dict` on which to operate. Optional iif the
                  nested data structure is provided.
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
        is_out_1, is_out_2 = fn_1.func is not fn, fn_2.func is not fn
        if not is_out_1:
            fn_1, dataset, args_1 = fn_1.func, fn_1.args[:-1], fn_1.args[-1]
            has_args = bool(args_1)
            if arity == 0:
                fn_1 = partial(fn_1, *dataset, *args_1)
            elif dataset:
                fn_1 = partial(fn_1, *dataset)
        if not is_out_2:
            fn_2, dataset, args_2 = fn_2.func, fn_2.args[:-1], fn_2.args[-1]
            has_args = bool(args_2)
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
        if parent is None and not is_out:
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
                    if field is None:
                        def _forward(post_fn, args, delayed):
                            post_fn(delayed, *args)
                        return partial(_forward, post_fn, args)

                    def _forward(post_fn, field, args, delayed):
                        post_fn(delayed[field], *args)
                    return partial(_forward, post_fn, field, args)
                if field is None:
                    def _forward(post_fn, delayed):
                        post_fn(delayed)
                    return partial(_forward, post_fn)

                def _forward(post_fn, field, delayed):
                    post_fn(delayed[field])
                return partial(_forward, post_fn, field)

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
                if field is None:
                    def _forward(op, post_fn, args, out, delayed):
                        return op(out, post_fn(delayed, *args))
                    return partial(_forward, op, post_fn, args)

                def _forward(op, post_fn, field, args, out, delayed):
                    return op(out, post_fn(delayed[field], *args))
                return partial(_forward, op, post_fn, field, args)
            if field is None:
                def _forward(post_fn, args, out, delayed):
                    return post_fn(delayed, *args)
                return partial(_forward, post_fn, args)

            def _forward(post_fn, field, args, out, delayed):
                return post_fn(delayed[field], *args)
            return partial(_forward, post_fn, field, args)

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
                post_args = (*get_bounds(space), *post_args)
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
              forward_bounds: bool = True
              ) -> Callable[[], StructNested[ValueT]]:
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
                post_args = (*get_bounds(space), *post_args)
            post_fn = partial(post_fn, post_args)
            if parent is None:
                post_fn = _build_setitem(arity, None, post_fn, None)
            return post_fn

        # Create new empty container to all transformed values.
        # TODO: The actual container should be created at the end if unmutable.
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

    :param dst: Nested data structure to be updated.
    """
    return build_reduce(array_copyto, None, (dst,), None, 1)


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
    """
    # Define a special exception involved in short-circuit mechanism
    class ShortCircuitContains(Exception):
        """Internal exception involved in short-circuit mechanism.
        """

    @nb.jit(nopython=True, cache=True)
    def _contains_or_raises(value: np.ndarray,
                            low: Optional[ArrayOrScalar],
                            high: Optional[ArrayOrScalar],
                            tol_abs: float,
                            tol_rel: float) -> bool:
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
        :param tol_abs: Absolute tolerance.
        :param tol_rel: Relative tolerance.
        """
        if not _array_contains(value, low, high, tol_abs, tol_rel):
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
        _contains_or_raises, None, (data,), space, 0, tol_abs, tol_rel))


def build_normalize(space: gym.Space[DataNested],
                    dst: DataNested,
                    src: Optional[DataNested] = None,
                    *, is_reversed: bool = False) -> Callable[..., None]:
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

    dataset = [dst,]
    if src is not None:
        dataset.append(src)
    return build_reduce(
        _array_normalize, None, dataset, space, 2 - len(dataset), is_reversed)


def build_flatten(data_nested: DataNested,
                  data_flat: Optional[DataNested] = None,
                  *, is_reversed: bool = False) -> Callable[..., None]:
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
    :param is_reversed: True to update 'data_flat', 'data_nested' otherwise.
    """
    # Make sure that the input arguments are valid
    assert is_reversed or data_flat is not None

    # Flatten nested data while preserving leaves ordering
    data_leaves = build_reduce(
        lambda x: x, lambda x, y: x.append(y) or x, (data_nested,), None, 0,
        initializer=list)()

    # Compute slices to split destination in accordance with nested data.
    # It will be passed to `build_reduce` as an input dataset. It is kind of
    # hacky since only passing `DataNested` instances is officially supported,
    # but it is currently the easiest way to keep track of some internal state
    # and specify leaf-specific constants.
    flat_slices = []
    idx_start = 0
    for data in data_leaves:
        idx_end = idx_start + max(math.prod(data.shape), 1)
        flat_slices.append((idx_start, idx_end))
        idx_start = idx_end

    @nb.jit(nopython=True, cache=True)
    def _flatten(data: np.ndarray,
                 flat_slice: Tuple[int, int],
                 data_flat: np.ndarray,
                 is_reversed: bool) -> None:
        """Synchronize the flatten and un-flatten representation of the data
        associated with the same leaf space.

        In practice, it assigns the value of a 1D array slice to some multi-
        dimensional array, or the other way around.

        :param data: Multi-dimensional array that will be either updated or
                     copied as a whole depending on 'is_reversed'.
        :param flat_slice: Start and stop indices of the slice of 'data_flat'
                           to synchronized with 'data'.
        :param data_flat: 1D array from which to extract that will be either
                          updated or copied depending on 'is_reversed'.
        :param is_reversed: True to update the multi-dimensional array 'data'
                            by copying the value from slice 'flat_slice' of
                            vector 'data_flat', False for doing the contrary.
        """
        # For some reason, passing a slice as input argument is much slower
        # in numba than creating it inside the method.
        if is_reversed:
            data.ravel()[:] = data_flat[slice(*flat_slice)]
        else:
            data_flat[slice(*flat_slice)] = data.ravel()

    args = (is_reversed,)
    if data_flat is not None:
        args = (data_flat, *args)  # type: ignore[assignment]
    out_fn = build_reduce(
        _flatten, None, (data_leaves, flat_slices), None, 2 - len(args), *args)
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
