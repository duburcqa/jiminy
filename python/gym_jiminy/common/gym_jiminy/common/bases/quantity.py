from abc import ABC, abstractmethod
from functools import partial
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Iterator, Generic, TypeVar

from .interfaces import InterfaceJiminyEnv


ValueT = TypeVar('ValueT')

QuantityCreator = Tuple["AbstractQuantity", Dict[str, Any]]


class ThreadLocalSharedCache(Generic[ValueT]):
    """Basic thread local shared cache.

    Its API mimics `std::optional` from the Standard C++ library. All it does
    is encapsulating any Python object as a mutable variable, plus exposing a
    simple mechanism for keeping track of all "owners" of the cache.

    .. warning::
        This implementation is not thread safe.
    """
    __slots__ = ("_value", "_has_value", "owners")

    def __init__(self) -> None:
        """
        """
        # Cached value if any
        self._value: Optional[ValueT] = None

        # Whether a value is stored in cached
        self._has_value: bool = False

        # Keep track of all "owners" of the shared buffer. This information may
        # be useful for determining the most efficient computation path. It is
        # up to user to update this list accordingly.
        # Note that 'set' cannot be used because owners are all objects having
        # the same hash, eg "identical" quantities.
        self.owners: List["AbstractQuantity"] = []

    def reset(self):
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
            # Assert(s) for type checker
            # assert self._value is not None
            return self._value
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
        The user is responsible for implementing the dunder method `__hash__`
        that uniquely define identical quantities as it is used internally by
        `QuantityManager` to synchronize cache between them. It is advised to
        use decorator `@dataclass(unsafe_hash=True)` for convenience, but it
        can also be done manually.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 requirements: Dict[str, QuantityCreator]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param requirements: Intermediary quantities on which this quantity
                             depends for its evaluation, as a dictionary
                             whose keys are tuple gathering their respective
                             class and all their constructor keyword-arguments
                             except the environment 'env'.
        """
        # Backup some of user argument(s)
        self.env = env

        # Instantiate intermediary quantities if any
        self.requirements: Dict[str, AbstractQuantity] = {
            name: cls(env, **kwargs)
            for name, (cls, kwargs) in requirements.items()}

        # Add getter of all intermediary quantities dynamically.
        # This approach is kind of hacky but much faster than any of other
        # official approach, ie implementing custom `__getattribute__` or
        # even slower custom `__getattr__`.
        def get_value(name: str, manager: AbstractQuantity) -> Any:
            return self.requirements[name].get()

        for name in self.requirements.keys():
            setattr(type(self), name, property(partial(get_value, name)))

        # Shared cache handling
        self._cache: Optional[ThreadLocalSharedCache[ValueT]] = None
        self._has_cache = False

        # Whether the quantity must be re-initialized
        self._is_initialized: bool = False

    @property
    def cache(self) -> ThreadLocalSharedCache[ValueT]:
        """Get shared cache if available, otherwise raises an exception.

        .. warning::
            This method is not meant to be overloaded.
        """
        if not self._has_cache:
            raise AttributeError(
                "No shared cache has been set for this quantity. Make sure it "
                "is managed by some `QuantityManager` instance.")
        # Assert(s) for type checker
        # assert self._cache is not None
        return self._cache

    @cache.setter
    def cache(self, cache: ThreadLocalSharedCache[ValueT]) -> None:
        """Set optional cache variable. When specified, it is used to store
        evaluated quantity and retrieve its value later one.

        .. warning::
            Value is stored by reference for efficiency. It is up to the user
            to the copy to retain its current value for later use if necessary.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Withdraw this quantity from the owners of its current cache if any
        if self._cache is not None:
            self._cache.owners.remove(self)

        # Declare this quantity as owner of the cache
        cache.owners.append(self)

        # Update internal cache attribute
        self._cache = cache
        self._has_cache = True

    def get(self) -> ValueT:
        """Get cached value of requested quantity if available, otherwise
        evaluate it and store it in cache.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Get value in cache if available.
        # Note that asking for forgiven rather than permission should be faster
        # if the user hits the cache at least 2 or 3 times.
        if self._has_cache:
            try:
                return self.cache.get()
            except ValueError:
                pass

        # Evaluate quantity
        try:
            if not self._is_initialized:
                self.initialize()
                assert self._is_initialized
            value = self.refresh()
        except RecursionError as e:
            raise LookupError(
                "Mutual dependency between quantities is disallowed.") from e

        # Return value after storing in cache if enabled
        if self._has_cache:
            self.cache.set(value)
        return value

    def reset(self) -> None:
        """Consider that the quantity must be re-initialized before being
        evaluated once again.

        .. note::
            This method must be called right before performing agent steps,
            otherwise this quantity will not be refreshed if it was evaluated
            previously.

        .. warning::
            This method is not meant to be overloaded.
        """
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize internal buffers.

        This is typically useful to refresh shared memory proxies or to
        re-initialize pre-allocated buffers.

        .. note::
            This method must be called before starting a new episode.

        .. note::
            Lazy-initialization is used for efficiency, ie `initialize` will be
            called before the first time `refresh` has to be called, which may
            never be the case if cache is shared between multiple identical
            instances of the same quantity.
        """
        self.pinocchio_model = self.env.pinocchio_model
        self.pinocchio_data = self.env.pinocchio_data
        self._is_initialized = True

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
        :param quantity_creators:
            All quantities to jointly manage, as a dictionary whose keys are
            tuple gathering their respective class and all their constructor
            keyword-arguments except the environment 'env'.
        """
        # Instantiate and store all top-level quantities to manage
        self.quantities: Dict[str, AbstractQuantity] = {
            name: cls(env, **kwargs)
            for name, (cls, kwargs) in quantity_creators.items()}

        # Add getter of all top-level quantities dynamically
        def get_quantity(name: str, manager: QuantityManager) -> Any:
            return self.quantities[name].get()

        for name in self.quantities.keys():
            setattr(type(self), name, property(partial(get_quantity, name)))

        # Get the complete list of all quantities involved in computations
        i = 0
        self._quantities_all = list(self.quantities.values())
        while i < len(self._quantities_all):
            quantity = self._quantities_all[i]
            self._quantities_all += quantity.requirements.values()
            i += 1

        # Set a shared cache entry for all quantities
        self._caches: Dict[AbstractQuantity, ThreadLocalSharedCache] = {}
        for quantity in self._quantities_all:
            cache = self._caches.setdefault(quantity, ThreadLocalSharedCache())
            quantity.cache = cache

    def reset(self) -> None:
        """Consider that all managed quantity must be re-initialized before
        being able to evaluate them once again.
        """
        for quantity in self._quantities_all[::-1]:
            quantity.reset()
        self.clear()

    def clear(self) -> None:
        """Clear internal cache storing already evaluated quantities.

        .. note::
            This method is supposed to be called right calling `step` of
            `BaseJiminyEnv`.
        """
        for cache in self._caches.values():
            cache.reset()

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
