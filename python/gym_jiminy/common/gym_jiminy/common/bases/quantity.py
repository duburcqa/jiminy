"""This module promotes quantities as first-class objects.

Defining quantities this way allows for standardization of common intermediary
metrics for computation of reward components and and termination conditions, eg
the center of pressure or the average spatial velocity of a frame. Overall, it
greatly reduces code duplication and bugs.

Apart from that, it offers a dedicated quantity manager that is responsible for
optimizing the computation path, which is expected to significantly increase
the step collection throughput. This speedup is achieved by caching already
computed that did not changed since then, computing redundant intermediary
quantities only once per step, and gathering similar quantities in a large
batch to leverage vectorization of math instructions.
"""
import weakref
from weakref import ReferenceType
from abc import ABC, abstractmethod
from collections.abc import MutableSet
from functools import partial
from typing import (
    Any, Dict, List, Optional, Tuple, Generic, TypeVar, Type, Iterator,
    Callable, cast)

from .interfaces import InterfaceJiminyEnv


ValueT = TypeVar('ValueT')

QuantityCreator = Tuple[Type["AbstractQuantity"], Dict[str, Any]]


class WeakMutableCollection(MutableSet, Generic[ValueT]):
    """Mutable unordered list container storing weak reference to objects.
    Elements will be discarded when no strong reference to the value exists
    anymore, and a user-specified callback will be triggered if any.

    Internally, it is implemented as a set for which uniqueness is
    characterized by identity instead of equality operator.
    """
    def __init__(self, callback: Optional[Callable[[
            "WeakMutableCollection[ValueT]", ReferenceType
            ], None]] = None) -> None:
        """
        :param callback: Callback that will be triggered every time an element
                         is discarded from the container.
                         Optional: None by default.
        """
        self._callback = callback
        self._ref_list: List[ReferenceType] = []

    def __callback__(self, ref: ReferenceType) -> None:
        """Internal method that will be called every time an element must be
        discarded from the containers, either because it was requested by the
        user or because no strong reference to the value exists anymore.

        If a callback has been specified by the user, it will be triggered
        after removing the weak reference from the container.
        """
        self._ref_list.remove(ref)
        if self._callback is not None:
            self._callback(self, ref)

    def __contains__(self, obj: Any) -> bool:
        """Dunder method to check if a weak reference to a given object is
        already stored in the container, which is characterized by identity
        instead of equality operator.

        :param obj: Object to look for in the container.
        """
        return any(ref() is obj for ref in self._ref_list)

    def __iter__(self) -> Iterator[ValueT]:
        """Dunder method that returns an iterator over the objects of the
        container for which a string reference still exist.
        """
        for ref in self._ref_list:
            obj = ref()
            if obj is not None:
                yield obj

    def __len__(self) -> int:
        """Dunder method that returns the length of the container.
        """
        return len(self._ref_list)

    def add(self, value: ValueT) -> None:
        """Add a new element to the container if not already contained.

        This has no effect if the element is already present.

        :param obj: Object to add to the container.
        """
        if value not in self:
            self._ref_list.append(weakref.ref(value, self.__callback__))

    def discard(self, value: ValueT) -> None:
        """Remove an element from the container if stored in it.

        This method does not raise an exception when the element is missing.

        :param obj: Object to remove from the container.
        """
        if value not in self:
            self.__callback__(weakref.ref(value))


class SharedCache(Generic[ValueT]):
    """Basic thread local shared cache.

    Its API mimics `std::optional` from the Standard C++ library. All it does
    is encapsulating any Python object as a mutable variable, plus exposing a
    simple mechanism for keeping track of all "owners" of the cache.

    .. warning::
        This implementation is not thread safe.
    """

    owners: WeakMutableCollection["AbstractQuantity"]
    """Owners of the shared buffer, ie quantities relying on it to store the
    result of their evaluation. This information may be useful for determining
    the most efficient computation path overall.

    .. note::
        Quantities must add themselves to this list when passing them a shared
        cache instance.

    .. note::
        Internally, it stores weak references to avoid holding alive quantities
        that could be garbage collected otherwise. `WeakSet` cannot be used
        because owners are objects all having the same hash, eg "identical"
        quantities.
    """

    def __init__(self) -> None:
        """
        """
        # Cached value if any
        self._value: Optional[ValueT] = None

        # Whether a value is stored in cached
        self._has_value: bool = False

        # Initialize "owners" of the shared buffer.
        # Define callback to reset part of the computation graph whenever a
        # quantity owning the cache gets garbage collected, namely all
        # quantities that may assume at some point the existence of this
        # deleted owner to find the adjust their computation path.
        def _callback(self: WeakMutableCollection["AbstractQuantity"],
                      ref: ReferenceType   # pylint: disable=unused-argument
                      ) -> None:
            for owner in self:
                while owner.parent is not None:
                    owner = owner.parent
                owner.reset(reset_tracking=True)

        self.owners = WeakMutableCollection(_callback)

    @property
    def has_value(self) -> bool:
        """Whether a value is stored in cache.
        """
        return self._has_value

    def reset(self) -> None:
        """Clear value stored in cache if any.
        """
        self._value = None
        self._has_value = False

    def set(self, value: ValueT) -> None:
        """Set value in cache, silently overriding the existing value if any.

        .. warning:
            Beware the value is stored by reference for efficiency. It is up to
            the user to copy it if necessary.

        :param value: Value to store in cache.
        """
        self._value = value
        self._has_value = True

    def get(self) -> ValueT:
        """Return cached value if any, otherwise raises an exception.
        """
        if self._has_value:
            return cast(ValueT, self._value)
        raise ValueError(
            "No value has been stored. Please call 'set' before 'get'.")


class AbstractQuantity(ABC, Generic[ValueT]):
    """Interface for quantities that involved in reward or termination
    conditions evaluation.

    .. note::
        Quantities are meant to be managed automatically via `QuantityManager`.
        Dealing with quantities manually is error-prone, and as such, is
        strongly discourage. Nonetheless, power-user that understand the risks
        are allowed to do it.

    .. warning::
        Mutual dependency between quantities is disallowed.

    .. warning::
        The user is responsible for implementing the dunder methods `__eq__`
        and `__hash__` that characterize identical quantities. This property is
        used internally by `QuantityManager` to synchronize cache  between
        them. It is advised to use decorator `@dataclass(unsafe_hash=True)` for
        convenience, but it can also be done manually.
    """

    requirements: Dict[str, "AbstractQuantity"]
    """Intermediary quantities on which this quantity may rely on for its
    evaluation at some point, depending on the optimal computation path at
    runtime. There values will be exposed to the user as usual properties.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional["AbstractQuantity"],
                 requirements: Dict[str, QuantityCreator]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param requirements: Intermediary quantities on which this quantity
                             depends for its evaluation, as a dictionary
                             whose keys are tuple gathering their respective
                             class and all their constructor keyword-arguments
                             except the environment 'env'.
        """
        # Backup some of user argument(s)
        self.env = env
        self.parent = parent

        # Instantiate intermediary quantities if any
        self.requirements: Dict[str, AbstractQuantity] = {
            name: cls(env, self, **kwargs)
            for name, (cls, kwargs) in requirements.items()}

        # Define some proxies for fast access
        self.pinocchio_model = env.robot.pinocchio_model
        self.pinocchio_data = env.robot.pinocchio_data

        # Shared cache handling
        self._cache: Optional[SharedCache[ValueT]] = None
        self._has_cache = False

        # Track whether the quantity has been called since previous reset
        self._is_active = False

        # Whether the quantity must be re-initialized
        self._is_initialized: bool = False

        # Add getter of all intermediary quantities dynamically.
        # This approach is hacky but much faster than any of other official
        # approach, ie implementing custom a `__getattribute__` method or even
        # worst a custom `__getattr__` method.
        def get_value(name: str, quantity: AbstractQuantity) -> Any:
            return quantity.requirements[name].get()

        for name in self.requirements.keys():
            setattr(type(self), name, property(partial(get_value, name)))

    def __getattr__(self, name: str) -> Any:
        """Get access to intermediary quantities as first-class properties,
        without having to do it through `requirements`.

        .. warning::
            Accessing quantities this way is convenient, but unfortunately
            much slower than do it through `requirements` manually. As a
            result, this approach is mainly intended for ease of use while
            prototyping.

        :param name: Name of the requested quantity.
        """
        return self.__getattribute__('requirements')[name].get()

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return [*super().__dir__(), *self.requirements.keys()]

    @property
    def cache(self) -> SharedCache[ValueT]:
        """Get shared cache if available, otherwise raises an exception.

        .. warning::
            This method is not meant to be overloaded.
        """
        if not self._has_cache:
            raise RuntimeError(
                "No shared cache has been set for this quantity. Make sure it "
                "is managed by some `QuantityManager` instance.")
        return cast(SharedCache[ValueT], self._cache)

    @cache.setter
    def cache(self, cache: Optional[SharedCache[ValueT]]) -> None:
        """Set optional cache variable. When specified, it is used to store
        evaluated quantity and retrieve its value later one.

        .. warning::
            Value is stored by reference for efficiency. It is up to the user
            to the copy to retain its current value for later use if necessary.

        .. note::
            One may overload this method to encapsulate the cache in a custom
            wrapper with specialized 'get' and 'set' methods before passing it
            to the base implementation. For instance, to enforce copy of the
            cached value before returning it.
        """
        # Withdraw this quantity from the owners of its current cache if any
        if self._cache is not None:
            self._cache.owners.discard(self)

        # Declare this quantity as owner of the cache if specified
        if cache is not None:
            cache.owners.add(self)

        # Update internal cache attribute
        self._cache = cache
        self._has_cache = cache is not None

    def is_active(self, any_cache_owner: bool = False) -> bool:
        """Whether this quantity is considered active, namely `initialize` has
        been called at least once since previous tracking reset.

        :param any_owner: False to check only if this exact instance is active,
                          True if any of the identical quantities (sharing the
                          same cache) is considered sufficient.
                          Optional: False by default.
        """
        if not any_cache_owner or self._cache is None:
            return self._is_active
        return any(owner._is_active for owner in self._cache.owners)

    def get(self) -> ValueT:
        """Get cached value of requested quantity if available, otherwise
        evaluate it and store it in cache.

        This quantity is considered active as soon as this method has been
        called at least once since previous tracking reset. The method
        `is_active` will be return true even before calling `initialize`.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Get value in cache if available.
        # Note that direct access to internal `_value` attribute is preferred
        # over the public API `get` for speedup. The same cannot be done for
        # `has_value` as it would prevent mocking it during running unit tests
        # or benchmarks.
        if (self._has_cache and
                self._cache.has_value):  # type: ignore[union-attr]
            self._is_active = True
            return self._cache._value  # type: ignore[union-attr,return-value]

        # Evaluate quantity
        try:
            if not self._is_initialized:
                self.initialize()
                assert (self._is_initialized and
                        self._is_active)  # type: ignore[unreachable]
            value = self.refresh()
        except RecursionError as e:
            raise LookupError(
                "Mutual dependency between quantities is disallowed.") from e

        # Return value after storing it in shared cache if available
        if self._has_cache:
            self._cache.set(value)  # type: ignore[union-attr]
        return value

    def reset(self, reset_tracking: bool = False) -> None:
        """Consider that the quantity must be re-initialized before being
        evaluated once again.

        If shared cache is available, then it will be cleared and all identity
        quantities will jointly be reset.

        .. note::
            This method must be called right before performing agent steps,
            otherwise this quantity will not be refreshed if it was evaluated
            previously.

        .. warning::
            This method is not meant to be overloaded.

        :param reset_tracking: Do not consider this quantity as active anymore
                               until the `get` method gets called once again.
                               Optional: False by default.
        """
        # No longer consider this exact instance as initialized
        self._is_initialized = False

        # No longer consider this exact instance as active if requested
        if reset_tracking:
            self._is_active = False

        # Reset all requirements first
        for quantity in self.requirements.values():
            quantity.reset(reset_tracking)

        # More work has to be done if shared cache is available and has value
        if self._has_cache:
            # Early return if shared cache has no value
            if not self.cache.has_value:
                return

            # Invalidate cache before looping over all identical properties
            self.cache.reset()

            # Reset all identical quantities
            for owner in self.cache.owners:
                owner.reset()

    def initialize(self) -> None:
        """Initialize internal buffers.

        This is typically useful to refresh shared memory proxies or to
        re-initialize pre-allocated buffers.

        .. warning::
            Intermediary quantities 'requirements' are NOT initialized
            automatically because they can be initialized lazily in most cases,
            or are optional depending on the most efficient computation path at
            run-time. It is up to the developer implementing quantities to take
            care of it.

        .. note::
            This method must be called before starting a new episode.

        .. note::
            Lazy-initialization is used for efficiency, ie `initialize` will be
            called before the first time `refresh` has to be called, which may
            never be the case if cache is shared between multiple identical
            instances of the same quantity.
        """
        # Refresh some proxies
        self.pinocchio_model = self.env.robot.pinocchio_model
        self.pinocchio_data = self.env.robot.pinocchio_data

        # The quantity is now considered initialized and active unconditionally
        self._is_initialized = True
        self._is_active = True

    @abstractmethod
    def refresh(self) -> ValueT:
        """Evaluate this quantity based on the agent state at the end of the
        current agent step.
        """
