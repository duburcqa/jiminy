from abc import ABC, abstractmethod
from functools import partial
from collections.abc import Mapping
from typing import Any, Dict, Tuple, Iterator, Optional, Generic, TypeVar

from .interfaces import InterfaceJiminyEnv


ValueT = TypeVar('ValueT')

QuantityCreator = Tuple["AbstractQuantity", Dict[str, Any]]


class OptionalValue(Generic[ValueT]):
    """ TODO: Write documentation.
    """
    __slots__ = ("_value", "_has_value")

    def __init__(self) -> None:
        """ TODO: Write documentation.
        """
        self._value: Optional[ValueT] = None
        self._has_value: bool = False

    def reset(self):
        """ TODO: Write documentation.
        """
        self._value = None
        self._has_value = False

    def has_value(self) -> bool:
        """ TODO: Write documentation.
        """
        return self._has_value

    def set(self, value: ValueT) -> None:
        """ TODO: Write documentation.
        """
        self._value = value
        self._has_value = True

    def get(self) -> ValueT:
        """ TODO: Write documentation.
        """
        if self._has_value:
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
        use decorator `@dataclass(unsafe_hash=True)` for convenience, but it
        can also be done manually.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 requirements: Dict[str, QuantityCreator]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param requirements: Intermediary quantities on which the current
                             quantity depends for its evaluation.
        """
        self.env = env
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

        self._cache: Optional[OptionalValue] = None
        self._is_initialized: bool = False

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
        try:
            return self._cache.get()
        except (AttributeError, ValueError):
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
        if self._cache is not None:
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
        self.pinocchio_model = self.env.pinocchio_model
        self.pinocchio_data = self.env.pinocchio_data
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
    environment, and internal buffers are re-initialized whenever necessary.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 quantity_creators: Dict[str, QuantityCreator]) -> None:
        """ TODO: Write documentation.

        :param env: Base or wrapped jiminy environment.
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
