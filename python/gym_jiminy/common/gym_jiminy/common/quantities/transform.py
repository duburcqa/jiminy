"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
import sys
import warnings
from dataclasses import dataclass
from types import EllipsisType
from typing import (
    Any, Optional, Sequence, Tuple, TypeVar, Union, Generic, ClassVar,
    Callable, Literal, List, overload, cast)

import numpy as np

from jiminy_py import tree
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)

from ..bases import InterfaceJiminyEnv, InterfaceQuantity, QuantityCreator
from ..utils import DataNested, build_reduce


ValueT = TypeVar('ValueT')
OtherValueT = TypeVar('OtherValueT')
YetAnotherValueT = TypeVar('YetAnotherValueT')


@dataclass(unsafe_hash=True)
class StackedQuantity(
        InterfaceQuantity[OtherValueT], Generic[ValueT, OtherValueT]):
    """Keep track of a given quantity over time by automatically stacking its
    value once per environment step since last reset.

    .. note::
        A new entry is added to the stack right before evaluating the reward
        and termination conditions. Internal simulation steps, observer and
        controller updates are ignored.
    """

    quantity: InterfaceQuantity[ValueT]
    """Base quantity whose value must be stacked over time since last reset.
    """

    max_stack: int
    """Maximum number of values that keep in memory before starting to discard
    the oldest one (FIFO). `sys.maxsize` if unlimited.
    """

    as_array: bool
    """Whether to return data as a tuple or a contiguous N-dimensional array
    whose last dimension gathers the value of individual timesteps.
    """

    is_wrapping: bool
    """Whether to wrap the stack around (i.e. starting filling data back from
    the start when full) when full instead of shifting data to the left.
    """

    allow_update_graph: ClassVar[bool] = False
    """Disable dynamic computation graph update.
    """

    @overload
    def __init__(self: "StackedQuantity[ValueT, List[ValueT]]",
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[ValueT],
                 *,
                 max_stack: int,
                 is_wrapping: bool,
                 as_array: Literal[False]) -> None:
        ...

    @overload
    def __init__(self: "StackedQuantity[Union[np.ndarray, float], np.ndarray]",
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[Union[np.ndarray, float]],
                 *,
                 max_stack: int,
                 is_wrapping: bool,
                 as_array: Literal[True]) -> None:
        ...

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[Any],
                 *,
                 max_stack: int = sys.maxsize,
                 is_wrapping: bool = False,
                 as_array: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose values
                         must be stacked, plus all its constructor keyword-
                         arguments except environment 'env' and 'parent'.
        :param max_stack: Maximum number of values that keep in memory before
                          starting to discard the oldest one (FIFO).
                          Optional: The maxium sequence length by default, ie
                          `sys.maxsize` (2^63 - 1).
        :param is_wrapping: Whether to wrap the stack around (i.e. starting
                            filling data back from the start when full) when
                            full instead of shifting data to the left. Note
                            that wrapping around is much faster for large stack
                            but does not preserve temporal ordering.
                            Optional: False by default.
        :param as_array: Whether to return data as a list or a contiguous
                         N-dimensional array whose last dimension gathers the
                         value of individual timesteps.
        """
        # Make sure that the input arguments are valid
        if max_stack > 10000 and (as_array and not is_wrapping):
            warnings.warn(
                "Very large stack length is strongly discourages for "
                "`as_array=True` and `is_wrapping=False`.")

        # Backup user arguments
        self.max_stack = max_stack
        self.is_wrapping = is_wrapping
        self.as_array = as_array

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(quantity=quantity),
                         auto_refresh=True)

        # Define specialized flattening operators for efficiency
        self._use_deepcopy = False
        self._dst_flat: List[np.ndarray] = []
        self._src_flat: List[np.ndarray] = []
        self._flatten_dst: Callable[[DataNested], None] = lambda data: None
        self._flatten_src: Callable[[DataNested], None] = lambda data: None

        # Allocate stack buffer.
        # Note that using a plain old list is more efficient than dequeue in
        # practice. Although front deletion is very fast compared to list,
        # casting deque to tuple or list is very slow, which ultimately
        # prevail. The matter gets worst as the maximum length gets longer.
        self._value_list: List[ValueT] = []

        # Continuous memory to store the whole stack if requested.
        # Note that it will be allocated lazily since the dimension of the
        # quantity is not known in advance.
        self._data = np.array([])

        # Define proxy to number of steps of current episode for fast access
        self._num_steps = np.array(-1)

        # Keep track of the last time the quantity has been stacked
        self._num_steps_prev = -1

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Refresh proxy
        self._num_steps = self.env.num_steps

        # Clear stack buffer
        self._value_list.clear()

        # Get current value of base quantity
        value = self.quantity.get()

        # Try to define specialized operators based on value.
        # This would succeeded if and only if all leaves are `np.ndarray`.
        # Do not try again if deepcopy mode was previously enabled.
        if not self.as_array and not self._use_deepcopy:
            try:
                self._flatten_dst = build_reduce(fn=self._dst_flat.append,
                                                 op=None,
                                                 dataset=(),
                                                 space=value,
                                                 arity=1,
                                                 forward_bounds=False)
                self._flatten_src = build_reduce(fn=self._src_flat.append,
                                                 op=None,
                                                 dataset=(),
                                                 space=value,
                                                 arity=1,
                                                 forward_bounds=False)
            except AssertionError:
                # Falling back to generic deepcopy
                self._use_deepcopy = True

        # Initialize buffers if necessary
        if self.as_array:
            # Make sure that the value of the quantity is supported
            if not isinstance(value, (int, float, np.ndarray, np.number)):
                raise ValueError(
                    "'as_array=True' is only supported by quantities "
                    "returning N-dimensional arrays as value.")
            _value = np.asarray(value)

            # Allocate contiguous memory if necessary
            self._data = np.zeros(
                (*_value.shape, self.max_stack), order='F', dtype=_value.dtype)

        # Reset step counter
        self._num_steps_prev = -1

    def refresh(self) -> OtherValueT:
        # Check if there is anything to do
        must_refresh = True
        num_steps = self._num_steps.item()
        if self.env.is_simulation_running:
            # Early return if the stack if already up to date
            if num_steps == self._num_steps_prev:
                must_refresh = False

            # Make sure that no steps are missing in the stack
            elif num_steps != self._num_steps_prev + 1:
                raise RuntimeError(
                    "Previous step missing in the stack. Please reset the "
                    "environment after adding this quantity.")

        # Extract contiguous slice of (future) available data if necessary
        if self.as_array:
            data = self._data
            num_stack = num_steps + 1
            if num_stack < self.max_stack:
                data = self._data[..., :num_stack]

        # Get current index if wrapping around
        if self.is_wrapping:
            index = num_steps % self.max_stack

        # Append current value of the quantity to the history buffer or update
        # aggregated continuous array directly if necessary.
        is_stack_full = num_steps >= self.max_stack
        if must_refresh:
            # Get the current value of the quantity
            value = self.quantity.get()

            # Append value to the history or aggregate data directly
            if self.as_array:
                if self.is_wrapping:
                    array_copyto(data[..., index], value)
                else:
                    # Shift all available data one timestep to the left.
                    # Operate on (future) available data only for efficiency.
                    if is_stack_full:
                        array_copyto(data[..., :-1], data[..., 1:])

                    # Update most recent value in stack with the current one
                    array_copyto(data[..., -1], value)
            else:
                # Remove oldest value in the stack if full
                update_buffer = is_stack_full and not self._use_deepcopy
                update_in_place = update_buffer and self.is_wrapping
                if update_in_place:
                    buffer = self._value_list[index]
                elif update_buffer:
                    buffer = self._value_list.pop(0)

                # Copy of the current value, while avoiding memory allocation
                # if possible for efficiency. Note that data must be
                # "deep-copied" to make sure it does not get altered afterward.
                if update_buffer:
                    # pylint: disable=used-before-assignment
                    try:
                        self._dst_flat.clear()
                        self._flatten_dst(buffer)
                        self._src_flat.clear()
                        self._flatten_src(value)  # type: ignore[arg-type]
                        multi_array_copyto(self._dst_flat, self._src_flat)
                    except AssertionError:
                        # The value of the quantity has changed its memory
                        # layout wrt initialization. Enabling generic deepcopy
                        # fallback from now on.
                        buffer = tree.deepcopy(value)
                        self._use_deepcopy = True
                else:
                    buffer = tree.deepcopy(value)

                # Add copied value to the stack if necessary
                if not update_in_place:
                    if is_stack_full and self.is_wrapping:
                        self._value_list.insert(0, buffer)
                    else:
                        self._value_list.append(buffer)

            # Increment step counter
            self._num_steps_prev += 1

        # Return aggregate data if requested
        if self.as_array:
            return cast(OtherValueT, data)

        # Return the whole stack as a list to preserve the integrity of the
        # underlying container and make the API robust to internal changes.
        return cast(OtherValueT, tuple(self._value_list))


@dataclass(unsafe_hash=True)
class MaskedQuantity(InterfaceQuantity[np.ndarray]):
    """Extract a pre-defined set of elements from a given quantity whose value
    is a N-dimensional array along an axis.

    Elements will be extract by copy unless the indices of the elements to
    extract to be written equivalently by a slice (ie they are evenly spaced),
    and the array can be flattened while preserving memory contiguity if 'axis'
    is `None`, which means that the result will be different between C- and F-
    contiguous arrays.
    """

    quantity: InterfaceQuantity[np.ndarray]
    """Base quantity whose elements must be extracted.
    """

    indices: Tuple[Union[int, EllipsisType], ...]
    """Indices of the elements to extract.
    """

    axis: Optional[int]
    """Axis over which to extract elements. `None` to consider flattened array.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[np.ndarray],
                 keys: Union[Sequence[Union[int, EllipsisType]],
                             Sequence[bool]],
                 *,
                 axis: Optional[int] = 0) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose values
                         must be extracted, plus any keyword-arguments of its
                         constructor except 'env' and 'parent'.
        :param keys: Sequence of indices or boolean mask that will be used to
                     extract elements from the quantity along one axis.
                     Ellipsis can be specified to automatically extract any
                     indices in between surrounding indices or at both ends.
                     Ellipsis on the right end is only supported for indices
                     with constant stride.
        :param axis: Axis over which to extract elements. `None` to consider
                     flattened array.
                     Optional: First axis by default.
        """
        # Convert boolean mask to indices
        if any(isinstance(e, (bool, np.bool_)) for e in keys):
            if not all(isinstance(e, (bool, np.bool_)) for e in keys):
                raise ValueError(
                    "Interleave boolean mask with ellipsis is not supported.")
            keys = tuple(np.flatnonzero(keys))  # type: ignore[arg-type]

        # Convert keys to tuple while removing consecutive ellipsis if any
        keys = tuple(
            e if e is Ellipsis else int(e)
            for e, _next in zip(keys, (*keys[1:], object()))
            if e is not Ellipsis or _next != e)

        # Replace intermediary ellipsis by indices if possible.
        # Note that it is important to do this substitution BEFORE storing
        # indices as attribute, otherwise masked quantities whose keys are
        # different be actually corresponds to identicial indices would be
        # identified as different as recomputed, e.g. (1, 2, 3) vs (..., 3).
        if any(e is Ellipsis for e in keys):
            for i in range(len(keys))[1:-1][::-1]:
                if keys[i] is Ellipsis:
                    indices = range(
                        keys[i - 1], keys[i + 1])  # type: ignore[arg-type]
                    keys = (*keys[:(i - 1)], *indices, *keys[(i + 1):])
            if len(keys) > 1 and keys[0] is Ellipsis:
                assert isinstance(keys[1], int)
                keys = (*range(0, keys[1]), *keys[1:])

        # Make sure that at least one index must be extracted
        if not keys:
            raise ValueError(
                "No indices to extract from quantity. Data would be empty.")

        # Make sure that at least one index must be extracted
        if keys == (Ellipsis,):
            raise ValueError(
                "Specifying `keys=(...,)` is not allowed as it has no effect.")

        # Check if indices or ellipsis has been provided
        if not all((e is Ellipsis) or isinstance(e, int) for e in keys):
            raise ValueError(
                "Argument 'keys' invalid. It must either be a boolean mask, "
                "or a sequence of indices and ellipsis.")

        # Backup user arguments
        self.indices = keys
        self.axis = axis

        # Check if the indices are evenly spaced
        stride: Optional[int] = None
        keys_heads, key_tail = cast(Tuple[int, ...], keys[:-1]), keys[-1]
        if len(keys) == 1:
            stride = 1
        elif all(e >= 0 for e in keys if e is not Ellipsis):
            if key_tail is Ellipsis:
                spaces = np.array((*np.diff(keys_heads), 1))
            else:
                spaces = np.diff((*keys_heads, key_tail))
            try:
                (stride,) = np.unique(spaces)
            except ValueError as e:
                if key_tail is Ellipsis:
                    raise ValueError(
                        "Ellipsis on the right end is only supported for "
                        "sequence of indices with constant stride.") from e

        # Convert indices to slices if possible
        self._slices: Tuple[Union[slice, EllipsisType], ...] = ()
        if stride is not None:
            slice_ = slice(keys[0],
                           None if key_tail is Ellipsis else key_tail + 1,
                           stride)
            if axis is None:
                self._slices = (slice_,)
            elif axis >= 0:
                self._slices = (*((slice(None),) * axis), slice_)
            else:
                self._slices = (
                    Ellipsis, slice_, *((slice(None),) * (- axis - 1)))

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(quantity=quantity),
                         auto_refresh=False)

    def refresh(self) -> np.ndarray:
        # Get current value of base quantity
        value = self.quantity.get()

        # Extract elements from quantity
        if not self._slices:
            # Note that `take` is faster than classical advanced indexing via
            # `operator[]` (`__getitem__`) because the latter is more generic.
            # Notably, `operator[]` supports boolean mask but `take` does not.
            return value.take(
                self.indices, self.axis)  # type: ignore[arg-type]
        if self.axis is None:
            # `ravel` must be used instead of `flat` to get a view that can
            # be sliced without copy.
            return value.ravel(order="K")[self._slices]
        return value[self._slices]


@dataclass(unsafe_hash=True)
class ConcatenatedQuantity(InterfaceQuantity[np.ndarray]):
    """Concatenate a set of quantities whose value are N-dimensional arrays
    along a given axis.

    All the quantities must have the same shape, except for the dimension
    corresponding to concatenation axis.
    """

    quantities: Tuple[InterfaceQuantity[np.ndarray], ...]
    """Base quantities whose values must be concatenated.
    """

    axis: int
    """Axis over which to concatenate values.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantities: Sequence[QuantityCreator[np.ndarray]],
                 *,
                 axis: int = 0) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantities: Sequence of tuples, each of which gathering the
                           class of the quantity whose values must be
                           extracted, plus any keyword-arguments of its
                           constructor except 'env' and 'parent'.
        :param axis: Axis over which to concatenate values.
                     Optional: First axis by default.
        """
        # Backup user arguments
        self.axis = axis

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements={
                            str(i): quantity
                            for i, quantity in enumerate(quantities)
                         },
                         auto_refresh=False)

        # Define proxies for fast access
        if len(quantities) < 2:
            raise ValueError(
                "Specifying less than 2 quantities is not allowed.")
        self.quantities = tuple(self.requirements.values())

        # Continuous memory to store the result
        # Note that it will be allocated lazily since the dimension of the
        # quantity is not known in advance.
        self._data = np.array([])

        # Store slices of data associated with each individual quantity
        self._data_slices: List[np.ndarray] = []

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Get current value of all the quantities
        values = [quantity.get() for quantity in self.quantities]

        # Allocate contiguous memory
        self._data = np.concatenate(values, axis=self.axis)

        # Compute slices of data
        self._data_slices.clear()
        idx_start = 0
        for data in values:
            idx_end = idx_start + data.shape[self.axis]
            self._data_slices.append(self._data[
                (*((slice(None),) * self.axis), slice(idx_start, idx_end))])
            idx_start = idx_end

    def refresh(self) -> np.ndarray:
        # Refresh the contiguous buffer
        multi_array_copyto(self._data_slices,
                           [quantity.get() for quantity in self.quantities])

        return self._data


@dataclass(unsafe_hash=True)
class UnaryOpQuantity(InterfaceQuantity[ValueT],
                      Generic[ValueT, OtherValueT]):
    """Apply a given unary operator to a quantity.

    This quantity is useful to translate quantities from world frame to local
    odometry frame. It may also be used to convert multi-variate quantities as
    scalar, typically by computing the L^p-norm.
    """

    quantity: InterfaceQuantity[OtherValueT]
    """Quantity that will be forwarded to the unary operator.
    """

    op: Callable[[OtherValueT], ValueT]
    """Callable taking any value of the quantity as input argument.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[OtherValueT],
                 op: Callable[[OtherValueT], ValueT]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose value
                         must be passed as argument of the unary operator, plus
                         any keyword-arguments of its constructor except 'env'
                         and 'parent'.
        :param op: Any callable taking any value of the quantity as input
                   argument. For example `partial(np.linalg.norm, ord=2)` to
                   compute the difference.
        """
        # Backup some user argument(s)
        self.op = op

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(quantity=quantity),
            auto_refresh=False)

    def refresh(self) -> ValueT:
        return self.op(self.quantity.get())


@dataclass(unsafe_hash=True)
class BinaryOpQuantity(InterfaceQuantity[ValueT],
                       Generic[ValueT, OtherValueT, YetAnotherValueT]):
    """Apply a given binary operator between two quantities.

    This quantity is mainly useful for computing the error between the value of
    a given quantity evaluated at the current simulation state and the state of
    at the current simulation time for the reference trajectory being selected.
    """

    quantity_left: InterfaceQuantity[OtherValueT]
    """Left-hand side quantity that will be forwarded to the binary operator.
    """

    quantity_right: InterfaceQuantity[YetAnotherValueT]
    """Right-hand side quantity that will be forwarded to the binary operator.
    """

    op: Callable[[OtherValueT, YetAnotherValueT], ValueT]
    """Callable taking left- and right-hand side quantities as input argument.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity_left: QuantityCreator[OtherValueT],
                 quantity_right: QuantityCreator[YetAnotherValueT],
                 op: Callable[[OtherValueT, YetAnotherValueT], ValueT]
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity_left: Tuple gathering the class of the quantity that
                              must be passed to left-hand side of the binary
                              operator, plus all its constructor keyword-
                              arguments except environment 'env' and parent
                              'parent.
        :param quantity_right: Quantity that must be passed to right-hand side
                               of the binary operator as a tuple
                               (class, keyword-arguments). See `quantity_left`
                               argument for details.
        :param op: Any callable taking the right- and left-hand side quantities
                   as input argument. For example `operator.sub` to compute the
                   difference.
        """
        # Backup some user argument(s)
        self.op = op

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                quantity_left=quantity_left, quantity_right=quantity_right),
            auto_refresh=False)

    def refresh(self) -> ValueT:
        return self.op(self.quantity_left.get(), self.quantity_right.get())


@dataclass(unsafe_hash=True)
class MultiAryOpQuantity(InterfaceQuantity[ValueT]):
    """Apply a given n-ary operator to the values of a given set of quantities.
    """

    quantities: Tuple[InterfaceQuantity[Any], ...]
    """Sequence of quantities that will be forwarded to the n-ary operator in
    this exact order.
    """

    op: Callable[[Sequence[Any]], ValueT]
    """Callable taking the packed sequence of values for all the specified
    quantities as input argument.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantities: Sequence[QuantityCreator[Any]],
                 op: Callable[[Sequence[Any]], ValueT]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantities: Ordered sequence of n pairs, each gathering the
                           class of a quantity whose value must be passed as
                           argument of the n-ary operator, plus any
                           keyword-arguments of its constructor except 'env'
                           and 'parent'.
        :param op: Any callable taking the packed sequence of values for all
                   the quantities as input argument, in the exact order they
                   were originally specified.
        """
        # Backup some user argument(s)
        self.op = op

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements={
                f"quantity_{i}": quantity
                for i, quantity in enumerate(quantities)},
            auto_refresh=False)

        # Keep track of the instantiated quantities for identity check
        self.quantities = tuple(self.requirements.values())

    def refresh(self) -> ValueT:
        return self.op([quantity.get() for quantity in self.quantities])
