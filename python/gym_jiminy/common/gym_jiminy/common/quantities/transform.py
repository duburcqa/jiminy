"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from types import EllipsisType
from typing import (
    Any, Optional, Sequence, Tuple, TypeVar, Union, Generic, ClassVar,
    Callable, Literal, List, overload, cast)

import numpy as np

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)

from ..bases import InterfaceJiminyEnv, InterfaceQuantity, QuantityCreator


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

    mode: Literal['slice', 'zeros']
    """Fallback strategy in case of incomplete stack. "slice" returns only
    available data, "zeros" returns a zero-padded fixed-length stack.
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
                 as_array: Literal[False],
                 mode: Literal['slice', 'zeros']) -> None:
        ...

    @overload
    def __init__(self: "StackedQuantity[Union[np.ndarray, float], np.ndarray]",
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[Union[np.ndarray, float]],
                 *,
                 max_stack: int,
                 as_array: Literal[True],
                 mode: Literal['slice', 'zeros']) -> None:
        ...

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[Any],
                 *,
                 max_stack: int = sys.maxsize,
                 as_array: bool = False,
                 mode: Literal['slice', 'zeros'] = 'slice') -> None:
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
        :param as_array: Whether to return data as a list or a contiguous
                         N-dimensional array whose last dimension gathers the
                         value of individual timesteps.
        :param mode: Fallback strategy in case of incomplete stack.
                     'zeros' is only supported by quantities returning
                     fixed-size N-D array.
                     Optional: 'slice' by default.
        """
        # Make sure that the input arguments are valid
        if max_stack > 10000 and (mode != 'slice' or as_array):
            warnings.warn(
                "Very large stack length is strongly discourages for "
                "`mode != 'slice'` or `as_array=True`.")

        # Backup user arguments
        self.max_stack = max_stack
        self.as_array = as_array
        self.mode = mode

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(quantity=quantity),
                         auto_refresh=True)

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
        self._data_views: Tuple[np.ndarray, ...] = ()

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

        # Initialize buffers if necessary
        if self.as_array or self.mode == 'zeros':
            # Get current value of base quantity
            value = self.quantity.get()

            # Make sure that the quantity is an array or a scalar
            if not isinstance(value, (int, float, np.ndarray)):
                raise ValueError(
                    "'as_array=True' is only supported by quantities "
                    "returning N-dimensional arrays as value.")
            value = np.asarray(value)

            # Full the queue with zero if necessary
            if self.mode == 'zeros':
                for _ in range(self.max_stack):
                    self._value_list.append(
                        np.zeros_like(value))  # type: ignore[arg-type]

            # Allocate stack memory if necessary
            if self.as_array:
                self._data = np.zeros((*value.shape, self.max_stack),
                                      order='F',
                                      dtype=value.dtype)
                self._data_views = tuple(
                    self._data[..., i] for i in range(self.max_stack))

        # Reset step counter
        self._num_steps_prev = -1

    def refresh(self) -> OtherValueT:
        # Append value to the queue only once per step and only if a simulation
        # is running. Note that data must be deep-copied to make sure it does
        # not get altered afterward.
        value_list = self._value_list
        if self.env.is_simulation_running:
            num_steps = self._num_steps.item()
            if num_steps != self._num_steps_prev:
                if num_steps != self._num_steps_prev + 1:
                    raise RuntimeError(
                        "Previous step missing in the stack. Please reset the "
                        "environment after adding this quantity.")
                value = self.quantity.get()
                if isinstance(value, np.ndarray):
                    # Avoid memory allocation if possible, which is much faster
                    if len(value_list) == self.max_stack:
                        buffer = value_list.pop(0)
                        array_copyto(buffer, value)
                        value_list.append(buffer)
                    else:
                        value_list.append(
                            value.copy())  # type: ignore[arg-type]
                else:
                    if len(value_list) == self.max_stack:
                        del value_list[0]
                    value_list.append(deepcopy(value))
                self._num_steps_prev += 1

        # Aggregate data in contiguous array only if requested
        if self.as_array:
            is_padded = self.mode == 'zeros'
            offset = - self._num_steps_prev - 1
            data, data_views = self._data, self._data_views
            if offset > - self.max_stack:
                if is_padded:
                    value_list = value_list[offset:]
                else:
                    data = data[..., offset:]
                data_views = self._data_views[offset:]
            multi_array_copyto(data_views, value_list)
            return cast(OtherValueT, data)

        # Return the whole stack as a list to preserve the integrity of the
        # underlying container and make the API robust to internal changes.
        return cast(OtherValueT, value_list)


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

    indices: Tuple[int, ...]
    """Indices of the elements to extract.
    """

    axis: Optional[int]
    """Axis over which to extract elements. `None` to consider flattened array.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[np.ndarray],
                 keys: Union[Sequence[int], Sequence[bool]],
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
        :param axis: Axis over which to extract elements. `None` to consider
                     flattened array.
                     Optional: First axis by default.
        """
        # Check if indices or boolean mask has been provided
        if all(isinstance(e, bool) for e in keys):
            keys = tuple(np.flatnonzero(keys))
        elif not all(isinstance(e, int) for e in keys):
            raise ValueError(
                "Argument 'keys' invalid. It must either be a boolean mask, "
                "or a sequence of indices.")

        # Backup user arguments
        self.indices = tuple(keys)
        self.axis = axis

        # Make sure that at least one index must be extracted
        if not self.indices:
            raise ValueError(
                "No indices to extract from quantity. Data would be empty.")

        # Check if the indices are evenly spaced
        self._slices: Tuple[Union[slice, EllipsisType], ...] = ()
        stride: Optional[int] = None
        if len(self.indices) == 1:
            stride = 1
        if len(self.indices) > 1 and all(e >= 0 for e in self.indices):
            spacing = np.unique(np.diff(self.indices))
            if spacing.size == 1:
                stride = spacing[0]
        if stride is not None:
            slice_ = slice(self.indices[0], self.indices[-1] + 1, stride)
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
            return value.take(self.indices, self.axis)
        if self.axis is None:
            # `ravel` must be used instead of `flat` to get a view that can
            # be sliced without copy.
            return value.ravel(order="K")[self._slices]
        return value[self._slices]


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
