from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    Any, Dict, List, Optional, Tuple, Iterator, Generic, TypeVar, Type, cast)

from .interfaces import InterfaceJiminyEnv


ValueT = TypeVar('ValueT')
OtherT = TypeVar('OtherT')

QuantityCreator = Tuple[Type["AbstractQuantity"], Dict[str, Any]]


# Forces `SharedCache.has_value` to always return `False`.
# Overriding this value is mainly useful for profiling.
DISABLE_CACHING = False


class SharedCache(Generic[ValueT]):
    """Basic thread local shared cache.

    Its API mimics `std::optional` from the Standard C++ library. All it does
    is encapsulating any Python object as a mutable variable, plus exposing a
    simple mechanism for keeping track of all "owners" of the cache.

    .. warning::
        This implementation is not thread safe.
    """
    __slots__ = ("_value", "_has_value", "owners")

    owners: List["AbstractQuantity"]
    """Owners of the shared buffer, ie quantities relying on it to store the
    result of their evaluation. This information may be useful for determining
    the most efficient computation path overall.

    .. note::
        Quantities must add themselves to this list when passing them a shared
        cache instance.

    .. note::
        A set cannot be used because owners are all objects having the same
        hash, eg "identical" quantities.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        """
        # Cached value if any
        self._value: Optional[ValueT] = None

        # Whether a value is stored in cached
        self._has_value: bool = False

        # Initialize "owners" of the shared buffer
        self.owners: List["AbstractQuantity"] = []

    @property
    def has_value(self) -> bool:
        """Whether a value is stored in cache.
        """
        return not DISABLE_CACHING and self._has_value

    def reset(self) -> None:
        """Clear value stored in cache if any.
        """
        self._value = None
        self._has_value = False

    def set(self, value: ValueT) -> None:
        """Set value in cache if none, otherwise raises an exception.

        .. warning:
            Beware the value is stored by reference for efficiency. It is up to
            the user to copy it if necessary.

        :param value: Value to store in cache.
        """
        if self._has_value:
            raise ValueError(
                "A value is already stored. Please call 'reset' before 'set'.")
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

        # Shared cache handling
        self._cache: Optional[SharedCache[ValueT]] = None
        self._has_cache = False

        # Track whether the quantity has been called since previous reset
        self._is_active = False

        # Whether the quantity must be re-initialized
        self._is_initialized: bool = False

    def __getattr__(self, name: str) -> Any:
        """Get access to intermediary quantities as first-class properties,
        without having to do it through `requirements`.

        .. warning::
            Getting quantities this way is convenient but unfortunately much
            slower than do it through `requirements` manually. It takes 40ns on
            Python 3.12 and a whooping 180ns on Python 3.11. As a result, this
            approach is mainly intended for ease of use while prototyping.

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
            raise AttributeError(
                "No shared cache has been set for this quantity. Make sure it "
                "is managed by some `QuantityManager` instance.")
        return cast(SharedCache[ValueT], self._cache)

    @cache.setter
    def cache(self, cache: SharedCache[ValueT]) -> None:
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
            self._cache.owners.remove(self)

        # Declare this quantity as owner of the cache
        cache.owners.append(self)

        # Update internal cache attribute
        self._cache = cache
        self._has_cache = True

    @property
    def is_active(self) -> bool:
        """Whether this quantity is considered active, namely `initialize` has
        been called at least once since previous tracking reset, either by this
        exact instance or any identical quantity if shared cache is available.
        """
        if self._cache is None:
            return self._is_active
        return any(owner._is_active for owner in self._cache.owners)

    def get(self) -> ValueT:
        """Get cached value of requested quantity if available, otherwise
        evaluate it and store it in cache.

        This quantity is considered active as soon as this method has been
        called at least once since previous tracking reset. The corresponding
        property `is_active` will be true even before calling `initialize`.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Get value in cache if available
        if self._has_cache and self.cache.has_value:
            self._is_active = True
            return self.cache.get()

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
            self.cache.set(value)
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
            quantity.reset()

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


class QuantityManager(Mapping):
    """This class centralizes the evaluation of all quantities involved in
    reward or termination conditions evaluation to redundant and unnecessary
    computations.

    It is responsible for making sure all quantities are evaluated on the same
    environment, and internal buffers are re-initialized whenever necessary.

    .. note::
        Individual quantities can be accessed either as instance properties or
        items of a dictionary. Choosing one or the other is only a matter of
        taste since both options have been heavily optimized to  minimize
        overhead and should be equally efficient.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 quantity_creators: Dict[str, QuantityCreator]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param quantity_creators:
            All quantities to jointly manage, as a dictionary whose keys are
            tuple gathering their respective class and all their constructor
            keyword-arguments except the environment 'env'.
        """
        # Instantiate and store all top-level quantities to manage
        self.quantities: Dict[str, AbstractQuantity] = {
            name: cls(env, None, **kwargs)
            for name, (cls, kwargs) in quantity_creators.items()}

        # Get the complete list of all quantities involved in computations
        i = 0
        self._quantities_all = list(self.quantities.values())
        while i < len(self._quantities_all):
            quantity = self._quantities_all[i]
            self._quantities_all += quantity.requirements.values()
            i += 1

        # Set a shared cache entry for all quantities
        self._caches: Dict[AbstractQuantity, SharedCache] = {}
        for quantity in self._quantities_all:
            cache = self._caches.setdefault(quantity, SharedCache())
            quantity.cache = cache

    def reset(self, reset_tracking: bool = False) -> None:
        """Consider that all managed quantity must be re-initialized before
        being able to evaluate them once again.

        .. note::
            The cache is cleared automatically by the quantities themselves.

        :param reset_tracking: Do not consider any quantity as active anymore.
                               Optional: False by default.
        """
        for quantity in self.quantities.values():
            quantity.reset(reset_tracking)

    def clear(self) -> None:
        """Clear internal cache storing already evaluated quantities.

        .. note::
            This method is supposed to be called right calling `step` of
            `BaseJiminyEnv`.
        """
        for cache in self._caches.values():
            cache.reset()

    def __getattr__(self, name: str) -> Any:
        """Get access managed quantities as first-class properties.

        .. warning::
            Getting quantities this way is convenient but unfortunately much
            slower than do it through `__getitem__`. Using this method is not
            recommend in production, especially on Python<3.12.

        :param name: Name of the requested quantity.
        """
        return self[name]

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return [*super().__dir__(), *self.quantities.keys()]

    def __getitem__(self, name: str) -> Any:
        """Get cached value of requested quantity if available, otherwise
        evaluate it and store it in cache.
        """
        return self.quantities[name].get()

    def __iter__(self) -> Iterator[str]:
        """Iterate over names of managed quantities.
        """
        return iter(self.quantities)

    def __len__(self) -> int:
        """Number of quantities being managed.
        """
        return len(self.quantities)
