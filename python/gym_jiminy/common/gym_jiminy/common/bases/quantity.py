from abc import ABC, abstractmethod
from collections.abc import Mapping
from itertools import chain
from typing import (
    Any, Dict, Tuple, Iterator, Iterable, Optional, Generic, TypeVar)

from jiminy_py.simulator import Simulator


ValueT = TypeVar('ValueT')

QuantityCreator = Tuple["AbstractQuantity", Dict[str, Any]]


class OptionalValue(Generic[ValueT]):
    """ TODO: Write documentation.
    """
    __slots__ = ("_value",)

    def __init__(self) -> None:
        """ TODO: Write documentation.
        """
        self._value: Optional[ValueT] = None

    def reset(self):
        """ TODO: Write documentation.
        """
        self._value = None

    def has_value(self) -> bool:
        """ TODO: Write documentation.
        """
        return self._value is not None

    def set(self, value: ValueT) -> None:
        """ TODO: Write documentation.
        """
        self._value = value

    def get(self) -> ValueT:
        """ TODO: Write documentation.
        """
        if self.has_value():
            return self._value
        raise ValueError("No value.")


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
        use decorator `attrs.define(frozen=True)` for convenience, but it can
        also be done manually.
    """
    def __init__(self,
                 simulator: Simulator,
                 requirements: Dict[str, QuantityCreator]) -> None:
        """
        :param simulator: Simulator already fully initialized with up-to-date
                          robot (model, controller, and hardware) and options.
        :param requirements: Intermediary quantities on which the current
                             quantity depends for its evaluation.
        """
        self.simulator = simulator
        self.requirements: Dict[str, AbstractQuantity] = {
            name: cls(simulator, **kwargs)
            for name, (cls, kwargs) in requirements.items()}
        self._cache: Optional[OptionalValue] = None
        self._is_initialized: bool = False

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to intermediary quantities as first-class
        properties, without having to do it through `requirements`.

        :param name: Name of the requested quantity.
        """
        return self.__getattribute__('requirements')[name].get()

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), self.requirements.keys())

    def set_cache(self, cache: OptionalValue) -> None:
        """Set optional cache variable. When specified, it is used to store
        evaluated quantity and retrieve its value later one.

        .. warning::
            Value is stored by reference for efficiency. It is up to the user
            to the copy to retain its current value for later use if necessary.

        .. warning::
            This method is not meant to be overloaded.
        """
        self._cache = cache

    def get(self) -> ValueT:
        """Get cached value of requested quantity if available, otherwise
        evaluate it and store it in cache.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Get value in cache if available
        is_cache_enabled = self._cache is not None
        if is_cache_enabled and self._cache.has_value():
            return self._cache.get()

        # Evaluate quantity
        try:
            if not self._is_initialized:
                self.initialize()
                assert self._is_initialized
            value = self.refresh()
        except RecursionError as e:
            raise LookupError(
                "Mutual dependency between quantities is disallowed.") from e
        if value is None:
            raise ValueError("Evaluated quantity must not be none.")

        # Return value after storing in cache if enabled
        if is_cache_enabled:
            self._cache.set(value)
        return value

    def reset(self) -> None:
        """Consider that the quantity must be re-initialized before being able
        to evaluate them once again

        .. warning::
            This method is not meant to be overloaded.
        """
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize internal buffers for fast access to shared memory or to
        avoid redundant computations.

        .. note::
            Lazy-initialization is used for efficiency, ie `initialize` will be
            called before the first time `refresh` has to be called, which may
            never be the case if cache is shared between multiple identical
            instances of the same quantity.
        """
        self.pinocchio_model = self.simulator.pinocchio_model
        self.pinocchio_data = self.simulator.pinocchio_data
        self._is_initialized = True

    @abstractmethod
    def refresh(self) -> ValueT:
        """Evaluate quantity at the current simulation state.
        """


class QuantityManager(Mapping):
    """This class centralizes the evaluation of all quantities involved in
    reward or termination conditions evaluation to redundant and unnecessary
    computations.

    It is responsible for making sure all quantities are evaluated on the same
    simulator, and internal buffers are re-initialized whenever necessary.
    """
    def __init__(self,
                 simulator: Simulator,
                 quantity_creators: Dict[str, QuantityCreator]) -> None:
        """ TODO: Write documentation.

        :param simulator: Simulator already fully initialized with up-to-date
                          robot (model, controller, and hardware) and options.
        """
        # Instantiate and store all top-level quantities to manage
        self.quantities: Dict[str, AbstractQuantity] = {
            name: cls(simulator, **kwargs)
            for name, (cls, kwargs) in quantity_creators.items()}

        # Get the complete list of all quantities involved in computations
        i = 0
        self._quantities_all = list(self.quantities.values())
        while i < len(self._quantities_all):
            quantity = self._quantities_all[i]
            self._quantities_all += quantity.requirements.values()
            i += 1

        # Set a shared cache entry for all quantities
        self._caches: Dict[AbstractQuantity, OptionalValue] = {}
        for quantity in self._quantities_all:
            cache = self._caches.setdefault(quantity, OptionalValue())
            quantity.set_cache(cache)

    def reset(self) -> None:
        """Consider that all managed quantity must be re-initialized before
        being able to evaluate them once again
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

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access managed quantities as first-class properties.

        :param name: Name of the requested quantity.
        """
        return self.__getattribute__('quantities')[name].get()

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), self.quantities.keys())

    def __getitem__(self, name: str) -> Any:
        """Get cached value of requested quantity if available, otherwise
        evaluate it and store it in cache.
        """
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over names of managed quantities.
        """
        return iter(self.quantities)

    def __len__(self) -> int:
        """Number of quantities being managed.
        """
        return len(self.quantities)
