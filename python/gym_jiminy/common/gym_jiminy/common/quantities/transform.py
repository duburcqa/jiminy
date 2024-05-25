"""Generic quantities that may be relevant for any kind of robot, regardless
its topology (multiple or single branch, fixed or floating base...) and the
application (locomotion, grasping...).
"""
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import (
    Any, Optional, Sequence, Tuple, TypeVar, Union, Generic, Callable)
from typing_extensions import TypeAlias

import numpy as np


from ..bases import InterfaceJiminyEnv, InterfaceQuantity, QuantityCreator


EllipsisType: TypeAlias = Any  # TODO: `EllipsisType` introduced in Python 3.10

ValueT = TypeVar('ValueT')
OtherValueT = TypeVar('OtherValueT')
YetAnotherValueT = TypeVar('YetAnotherValueT')


@dataclass(unsafe_hash=True)
class StackedQuantity(InterfaceQuantity[Tuple[ValueT, ...]]):
    """Keep track of a given quantity over time by automatically stacking its
    value once per environment step since last reset.

    .. note::
        A new entry is added to the stack right before evaluating the reward
        and termination conditions. Internal simulation steps, observer and
        controller updates are ignored.
    """

    quantity: InterfaceQuantity
    """Base quantity whose value must be stacked over time since last reset.
    """

    num_stack: Optional[int]
    """Maximum number of values that keep in memory before starting to discard
    the oldest one (FIFO). None if unlimited.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 quantity: QuantityCreator[ValueT],
                 *,
                 num_stack: Optional[int] = None) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose values
                         must be stacked, plus all its constructor keyword-
                         arguments except environment 'env' and parent 'parent.
        :param num_stack: Maximum number of values that keep in memory before
                          starting to discard the oldest one (FIFO). None if
                          unlimited.
        """
        # Backup user arguments
        self.num_stack = num_stack

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(data=quantity),
                         auto_refresh=True)

        # Keep track of the quantity that must be stacked once instantiated
        self.quantity = self.requirements["data"]

        # Allocate deque buffer
        self._deque: deque = deque(maxlen=self.num_stack)

        # Keep track of the last time the quantity has been stacked
        self._num_steps_prev = -1

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Clear buffer
        self._deque.clear()

        # Reset step counter
        self._num_steps_prev = -1

    def refresh(self) -> Tuple[ValueT, ...]:
        # Append value to the queue only once per step and only if a simulation
        # is running. Note that data must be deep-copied to make sure it does
        # not get altered afterward.
        if self.env.is_simulation_running:
            num_steps = self.env.num_steps
            if num_steps != self._num_steps_prev:
                assert num_steps == self._num_steps_prev + 1
                self._deque.append(deepcopy(self.data))
                self._num_steps_prev += 1

        # Return the whole stack as a tuple to preserve the integrity of the
        # underlying container and make the API robust to internal changes.
        return tuple(self._deque)


@dataclass(unsafe_hash=True)
class MaskedQuantity(InterfaceQuantity[np.ndarray]):
    """Extract a pre-defined set of elements from a given quantity whose value
    is a N-dimensional array along an axis.

    Elements will be extract by copy unless the indices of the elements to
    extract to be written equivalently by a slice (ie they are evenly spaced),
    and the array can be flattened while preserving memory contiguity if 'axis'
    is `None`.
    """

    quantity: InterfaceQuantity
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
                 key: Union[Sequence[int], Sequence[bool]],
                 *,
                 axis: Optional[int] = None) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity: Tuple gathering the class of the quantity whose values
                         must be extracted, plus all its constructor keyword-
                         arguments except environment 'env' and parent 'parent.
        :param key: Sequence of indices or boolean mask that will be used to
                    extract elements from the quantity along one axis.
        :param axis: Axis over which to extract elements. `None` to consider
                     flattened array.
                     Optional: `None` by default.
        """
        # Check if indices or boolean mask has been provided
        if all(isinstance(e, bool) for e in key):
            key = tuple(np.flatnonzero(key))
        elif not all(isinstance(e, int) for e in key):
            raise ValueError(
                "Argument 'key' invalid. It must either be a boolean mask, or "
                "a sequence of indices.")

        # Backup user arguments
        self.indices = tuple(key)
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
            elif axis > 0:
                self._slices = (*((slice(None),) * axis), slice_)
            else:
                self._slices = (
                    Ellipsis, slice_, *((slice(None),) * (- axis - 1)))

        # Call base implementation
        super().__init__(env,
                         parent,
                         requirements=dict(data=quantity),
                         auto_refresh=False)

        # Keep track of the quantity from which data must be extracted
        self.quantity = self.requirements["data"]

    def refresh(self) -> np.ndarray:
        # Extract elements from quantity
        if not self._slices:
            # Note that `take` is faster than classical advanced indexing via
            # `operator[]` (`__getitem__`) because the latter is more generic.
            # Notably, `operator[]` supports boolean mask but `take` does not.
            return self.data.take(self.indices, self.axis)
        if self.axis is None:
            # `reshape` must be used instead of `flat` to get a view that can
            # be sliced without copy.
            return self.data.reshape((-1,))[self._slices]
        return self.data[self._slices]


@dataclass(unsafe_hash=True)
class UnaryOpQuantity(InterfaceQuantity[ValueT],
                      Generic[ValueT, OtherValueT]):
    """Apply a given unary operator to a quantity.

    This quantity is useful to translate quantities from world frame to local
    odometry frame. It may also be used to convert multi-variate quantities as
    scalar, typically by computing the Lp-norm.
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
                         all its constructor keyword-arguments except
                         environment 'env' and parent 'parent.
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
            requirements=dict(data=quantity),
            auto_refresh=False)

        # Keep track of the left- and right-hand side quantities for hashing
        self.quantity = self.requirements["data"]

    def refresh(self) -> ValueT:
        return self.op(self.data)


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
    """Callable taking right- and left-hand side quantities as input argument.
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
                value_left=quantity_left, value_right=quantity_right),
            auto_refresh=False)

        # Keep track of the left- and right-hand side quantities for hashing
        self.quantity_left = self.requirements["value_left"]
        self.quantity_right = self.requirements["value_right"]

    def refresh(self) -> ValueT:
        return self.op(self.value_left, self.value_right)
