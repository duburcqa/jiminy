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
import re
import weakref
from enum import IntEnum
from weakref import ReferenceType
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from collections.abc import MutableSet
from dataclasses import dataclass, replace
from functools import wraps
from typing import (
    Any, Dict, List, Optional, Tuple, Generic, TypeVar, Type, Iterator,
    Collection, Callable, Literal, ClassVar, TYPE_CHECKING)

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)
from jiminy_py.dynamics import State, Trajectory, update_quantities
import pinocchio as pin

from .interfaces import InterfaceJiminyEnv


ValueT = TypeVar('ValueT')


class WeakMutableCollection(MutableSet, Generic[ValueT]):
    """Mutable unordered list container storing weak reference to objects.
    Elements will be discarded when no strong reference to the value exists
    anymore, and a user-specified callback will be triggered if any.

    Internally, it is implemented as a set for which uniqueness is
    characterized by identity instead of equality operator.
    """

    __slots__ = ("_callback", "_weakrefs")

    def __init__(self, callback: Optional[Callable[[
            "WeakMutableCollection[ValueT]", ReferenceType
            ], None]] = None) -> None:
        """
        :param callback: Callback that will be triggered every time an element
                         is discarded from the container.
                         Optional: None by default.
        """
        self._callback = callback
        self._weakrefs: List[ReferenceType] = []

    def __callback__(self, ref: ReferenceType) -> None:
        """Internal method that will be called every time an element must be
        discarded from the containers, either because it was requested by the
        user or because no strong reference to the value exists anymore.

        If a callback has been specified by the user, it will be triggered
        after removing the weak reference from the container.
        """
        # Even though a temporary weak reference is provided for removal, the
        # identity check is performed on the object being stored. If the latter
        # has already been deleted, then one of the object in the list that
        # has been deleted while be removed. It is not a big deal if it was
        # actually the right weak reference since all of them will be removed
        # in the end, so it is not a big deal.
        value = ref()
        for i, ref_i in enumerate(self._weakrefs):
            if value is ref_i():
                del self._weakrefs[i]
                break
        if self._callback is not None:
            self._callback(self, ref)

    def __contains__(self, obj: Any) -> bool:
        """Dunder method to check if a weak reference to a given object is
        already stored in the container, which is characterized by identity
        instead of equality operator.

        :param obj: Object to look for in the container.
        """
        return any(ref() is obj for ref in self._weakrefs)

    def __iter__(self) -> Iterator[ValueT]:
        """Dunder method that returns an iterator over the objects of the
        container for which a reference still exist.
        """
        for ref in self._weakrefs:
            obj = ref()
            if obj is not None:
                yield obj

    def __len__(self) -> int:
        """Dunder method that returns the length of the container.
        """
        return len(self._weakrefs)

    def add(self, value: ValueT) -> None:
        """Add a new element to the container if not already contained.

        This has no effect if the element is already present.

        :param obj: Object to add to the container.
        """
        if value not in self:
            self._weakrefs.append(weakref.ref(value, self.__callback__))

    def discard(self, value: ValueT) -> None:
        """Remove an element from the container if stored in it.

        This method does not raise an exception when the element is missing.

        :param obj: Object to remove from the container.
        """
        if value in self:
            self.__callback__(weakref.ref(value))


class QuantityStateMachine(IntEnum):
    """Specify the current state of a given (unique) quantity, which determines
    the steps to perform for retrieving its current value.
    """

    IS_RESET = 0
    """The quantity at hand has just been reset. The quantity must first be
    initialized, then refreshed and finally stored in cached before to retrieve
    its value.
    """

    IS_INITIALIZED = 1
    """The quantity at hand has been initialized but never evaluated for the
    current robot state. Its value must still be refreshed and stored in cache
    before to retrieve it.
    """

    IS_CACHED = 2
    """The quantity at hand has been evaluated and its value stored in cache.
    As such, its value can be retrieve from cache directly.
    """


# Define proxies for fast lookup
_IS_RESET, _IS_INITIALIZED, _IS_CACHED = (  # pylint: disable=invalid-name
    QuantityStateMachine)


class SharedCache(Generic[ValueT]):
    """Basic thread local shared cache.

    Its API mimics `std::optional` from the Standard C++ library. All it does
    is encapsulating any Python object as a mutable variable, plus exposing a
    simple mechanism for keeping track of all "owners" of the cache.

    .. warning::
        This implementation is not thread safe.
    """

    __slots__ = (
        "_value", "_weakrefs", "_owner", "_auto_refresh", "sm_state", "owners")

    owners: Collection["InterfaceQuantity[ValueT]"]
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

        # Whether auto-refresh is requested
        self._auto_refresh = True

        # Basic state machine management
        self.sm_state: QuantityStateMachine = QuantityStateMachine.IS_RESET

        # Initialize "owners" of the shared buffer.
        # Define callback to reset part of the computation graph whenever a
        # quantity owning the cache gets garbage collected, namely all
        # quantities that may assume at some point the existence of this
        # deleted owner to adjust their computation path.
        def _callback(
                self: WeakMutableCollection["InterfaceQuantity[ValueT]"],
                ref: ReferenceType) -> None:  # pylint: disable=unused-argument
            owner: Optional["InterfaceQuantity[ValueT]"]
            for owner in self:
                # Stop going up in parent chain if dynamic computation graph
                # update is disable for efficiency.
                while (owner.allow_update_graph and
                        owner.parent is not None and owner.parent.has_cache):
                    owner = owner.parent
                owner.reset(reset_tracking=True)

        # Initialize weak reference to owning quantities
        self._weakrefs = WeakMutableCollection(_callback)

        # Maintain alive owning quantities upon reset
        self.owners = self._weakrefs
        self._owner: Optional["InterfaceQuantity[ValueT]"] = None

    def add(self, owner: "InterfaceQuantity[ValueT]") -> None:
        """Add a given quantity instance to the set of co-owners associated
        with the shared cache at hand.

        .. warning::
            All shared cache co-owners must be instances of the same unique
            quantity. An exception will be thrown if an attempt is made to add
            a quantity instance that does not satisfy this condition.

        :param owner: Quantity instance to add to the set of co-owners.
        """
        # Make sure that the quantity is not already part of the co-owners
        if id(owner) in map(id, self.owners):
            raise ValueError(
                "The specified quantity instance is already an owner of this "
                "shared cache.")

        # Make sure that the new owner is consistent with the others if any
        if any(owner != _owner for _owner in self._weakrefs):
            raise ValueError(
                "Quantity instance inconsistent with already existing shared "
                "cache owners.")

        # Add quantity instance to shared cache owners
        self._weakrefs.add(owner)

        # Refresh owners
        if self.sm_state is not QuantityStateMachine.IS_RESET:
            self.owners = tuple(self._weakrefs)

    def discard(self, owner: "InterfaceQuantity[ValueT]") -> None:
        """Remove a given quantity instance from the set of co-owners
        associated with the shared cache at hand.

        :param owner: Quantity instance to remove from the set of co-owners.
        """
        # Make sure that the quantity is part of the co-owners
        if id(owner) not in map(id, self.owners):
            raise ValueError(
                "The specified quantity instance is not an owner of this "
                "shared cache.")

        # Restore "dynamic" owner list as it may be involved in quantity reset
        self.owners = self._weakrefs

        # Remove quantity instance from shared cache owners
        self._weakrefs.discard(owner)

        # Refresh owners.
        # Note that one must keep tracking the quantity instance being used in
        # computations, aka 'self._owner', even if it is no longer an actual
        # shared cache owner. This is necessary because updating it would
        # require resetting the state machine, which is not an option as it
        # would mess up with quantities storing history since initialization.
        if self.sm_state is not QuantityStateMachine.IS_RESET:
            self.owners = tuple(self._weakrefs)

    def reset(self,
              ignore_auto_refresh: bool = False,
              reset_state_machine: bool = False) -> None:
        """Clear value stored in cache if any.

        :param ignore_auto_refresh: Whether to skip automatic refresh of all
                                    co-owner quantities of this shared cache.
                                    Optional: False by default.
        :param reset_state_machine: Whether to reset completely the state
                                    machine of the underlying quantity, ie not
                                    considering it initialized anymore.
                                    Optional: False by default.
        """
        # Clear cache
        if self.sm_state is _IS_CACHED:
            self.sm_state = _IS_INITIALIZED

        # Special branch if case quantities must be reset on the way
        if reset_state_machine:
            # Reset the state machine completely
            self.sm_state = _IS_RESET

            # Update list of owning quantities
            self.owners = self._weakrefs
            self._owner = None

            # Reset auto-refresh buffer
            self._auto_refresh = True

        # Refresh automatically if not already proven useless and not ignored
        if not ignore_auto_refresh and self._auto_refresh:
            for owner in self.owners:
                if owner.auto_refresh:
                    owner.get()
                    break
            else:
                self._auto_refresh = False

    def get(self) -> ValueT:
        """Return cached value if any, otherwise evaluate it and store it.
        """
        # Get value already stored
        if self.sm_state is _IS_CACHED:
            # return cast(ValueT, self._value)
            return self._value  # type: ignore[return-value]

        # Evaluate quantity
        try:
            if self.sm_state is _IS_RESET:
                # Cache the list of owning quantities
                self.owners = tuple(self._weakrefs)

                # Stick to the first owning quantity systematically
                owner = self.owners[0]
                self._owner = owner

                # Initialize quantity if not already done manually
                if not owner._is_initialized:
                    owner.initialize()
                assert owner._is_initialized

            # Get first owning quantity systematically
            # assert self._owner is not None
            owner = self._owner  # type: ignore[assignment]

            # Make sure that the state has been refreshed
            if owner._force_update_state:
                owner.state.get()

            # Refresh quantity
            value = owner.refresh()
        except RecursionError as e:
            raise LookupError(
                "Mutual dependency between quantities is disallowed.") from e

        # Update state machine
        self.sm_state = _IS_CACHED

        # Return value after storing it
        self._value = value
        return value


class InterfaceQuantity(Generic[ValueT], metaclass=ABCMeta):
    """Interface for generic quantities involved observer-controller blocks,
    reward components or termination conditions.

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
        used internally by `QuantityManager` to synchronize cache between them.
        It is advised to use decorator `@dataclass(unsafe_hash=True)` for
        convenience, but it can also be done manually.
    """

    requirements: Dict[str, "InterfaceQuantity"]
    """Intermediary quantities on which this quantity may rely on for its
    evaluation at some point, depending on the optimal computation path at
    runtime. They will be exposed to the user as usual attributes.
    """

    allow_update_graph: ClassVar[bool] = True
    """Whether dynamic computation graph update is allowed. This implies that
    the quantity can be reset at any point in time to re-compute the optimal
    computation path, typically after deletion or addition of some other node
    to its dependent sub-graph. When this happens, the quantity gets reset on
    the spot, even if a simulation is already running. This is not always
    acceptable, hence the capability to disable this feature at class-level.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional["InterfaceQuantity"],
                 requirements: Dict[str, "QuantityCreator"],
                 *,
                 auto_refresh: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param requirements: Intermediary quantities on which this quantity
                             depends for its evaluation, as a dictionary
                             whose keys are tuple gathering their respective
                             class plus any keyword-arguments of its
                             constructor except 'env' and 'parent'.
        :param auto_refresh: Whether this quantity must be refreshed
                             automatically as soon as its shared cache has been
                             cleared if specified, otherwise this does nothing.
        """
        # Backup some of user argument(s)
        self.env = env
        self.parent = parent
        self.auto_refresh = auto_refresh

        # Make sure that all requirement names would be valid as property
        requirement_names = requirements.keys()
        if any(re.match('[^A-Za-z0-9_]', name) for name in requirement_names):
            raise ValueError("The name of all quantity requirements should be "
                             "ASCII alphanumeric characters plus underscore.")

        # Instantiate intermediary quantities if any
        self.requirements: Dict[str, InterfaceQuantity] = {
            name: cls(env, self, **kwargs)
            for name, (cls, kwargs) in requirements.items()}

        # Define proxies for user-specified intermediary quantities.
        # This approach is much faster than hidding quantities behind value
        # getters. In particular, dynamically adding properties, which is hacky
        # but which is the fastest alternative option, still adds 35% overhead
        # on Python 3.11 compared to calling `get` directly. The "official"
        # approaches are even slower, ie implementing custom `__getattribute__`
        # method or worst custom `__getattr__` method.
        for name, quantity in self.requirements.items():
            setattr(self, name, quantity)

        # Update the state explicitly if available but auto-refresh not enabled
        self._force_update_state = False
        if isinstance(self, AbstractQuantity):
            self._force_update_state = not self.state.auto_refresh

        # Shared cache handling
        self._cache: Optional[SharedCache[ValueT]] = None
        self.has_cache = False

        # Track whether the quantity has been called since previous reset
        self._is_active = False

        # Whether the quantity must be re-initialized
        self._is_initialized: bool = False

    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> Any:
            """Get access to intermediary quantities as first-class properties,
            without having to do it through `requirements`.

            .. warning::
                Accessing quantities this way is convenient, but unfortunately
                much slower than do it through dynamically added properties. As
                a result, this approach is only used to fix typing issues.

            :param name: Name of the requested quantity.
            """
            try:
                return self.__getattribute__('requirements')[name].get()
            except KeyError as e:
                raise AttributeError(
                    f"'{type(self)}' object has no attribute '{name}'") from e

    @property
    def cache(self) -> SharedCache[ValueT]:
        """Get shared cache if available, otherwise raises an exception.

        .. warning::
            This method is not meant to be overloaded.
        """
        if not self.has_cache:
            raise RuntimeError(
                "No shared cache has been set for this quantity. Make sure it "
                "is managed by some `QuantityManager` instance.")
        return self._cache  # type: ignore[return-value]

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
            try:
                self._cache.discard(self)
            except ValueError:
                # This may fail if the quantity is already being garbage
                # collected when clearing cache.
                pass

        # Declare this quantity as owner of the cache if specified
        if cache is not None:
            cache.add(self)

        # Update internal cache attribute
        self._cache = cache
        self.has_cache = cache is not None

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
        # Delegate getting value to shared cache if available
        if self._cache is not None:
            # Get value
            value = self._cache.get()

            # This instance is not forceably considered active at this point.
            # Note that it must be done AFTER getting the value, otherwise it
            # would mess up with computation graph tracking at initialization.
            self._is_active = True

            # Return cached value
            return value

        # Evaluate quantity
        try:
            # Initialize quantity
            if not self._is_initialized:
                self.initialize()
                assert self._is_initialized

            # Refresh quantity
            return self.refresh()
        except RecursionError as e:
            raise LookupError(
                "Mutual dependency between quantities is disallowed.") from e

    def reset(self,
              reset_tracking: bool = False,
              *, ignore_other_instances: bool = False) -> None:
        """Consider that the quantity must be re-initialized before being
        evaluated once again.

        If shared cache is available, then it will be cleared first then all
        identical quantities will be jointly reset.

        .. note::
            This method must be called right before performing any agent step,
            otherwise this quantity will not be refreshed if it was evaluated
            previously.

        .. warning::
            This method is not meant to be overloaded.

        :param reset_tracking: Do not consider this quantity as active anymore
                               until the `get` method gets called once again.
                               Optional: False by default.
        :param ignore_other_instances:
            Whether to skip reset of intermediary quantities as well as any
            shared cache co-owner quantity instances.
            Optional: False by default.
        """
        # Make sure that auto-refresh can be honored
        if self.auto_refresh and not self.has_cache:
            raise RuntimeError(
                "Automatic refresh enabled but no shared cache is available. "
                "Please add one before calling this method.")

        # Reset all requirements first
        if not ignore_other_instances:
            for quantity in self.requirements.values():
                quantity.reset(reset_tracking, ignore_other_instances=False)

        # Skip reset if dynamic computation graph update is not allowed
        if self.env.is_simulation_running and not self.allow_update_graph:
            return

        # No longer consider this exact instance as active if requested
        if reset_tracking:
            self._is_active = False

        # No longer consider this exact instance as initialized
        self._is_initialized = False

        # More work must to be done if this quantity has a shared cache that
        # has not been completely reset yet.
        if self.has_cache and self.cache.sm_state is not _IS_RESET:
            # Reset shared cache state machine first, to avoid triggering reset
            # propagation to all identical quantities.
            self.cache.reset(
                ignore_auto_refresh=True, reset_state_machine=True)

            # Reset all identical quantities except itself since already done
            for owner in self.cache.owners:
                if owner is not self:
                    owner.reset(reset_tracking=reset_tracking,
                                ignore_other_instances=True)

            # Reset shared cache afterward with auto-refresh enabled if needed
            if self.env.is_simulation_running:
                self.cache.reset(
                    ignore_auto_refresh=False, reset_state_machine=False)

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
        # The quantity is now considered initialized and active unconditionally
        self._is_initialized = True
        self._is_active = True

    @abstractmethod
    def refresh(self) -> ValueT:
        """Evaluate this quantity based on the agent state at the end of the
        current agent step.
        """


QuantityValueT_co = TypeVar('QuantityValueT_co', covariant=True)
QuantityCreator = Tuple[
    Type[InterfaceQuantity[QuantityValueT_co]], Dict[str, Any]]


class QuantityEvalMode(IntEnum):
    """Specify on which state to evaluate a given quantity.
    """

    TRUE = 0
    """Current state of the environment.
    """

    REFERENCE = 1
    """State of the reference trajectory at the current simulation time.
    """


# Define proxies for fast lookup
_TRUE, _REFERENCE = QuantityEvalMode


@dataclass(unsafe_hash=True)
class AbstractQuantity(InterfaceQuantity, Generic[ValueT]):
    """Base class for generic quantities involved observer-controller blocks,
    reward components or termination conditions.

    .. note::
        A dataset of trajectories made available through `self.trajectories`.
        The latter is synchronized because all quantities as long as shared
        cached is available. At least one trajectory must be added to the
        dataset and selected prior to using `QuantityEvalMode.REFERENCE`
        evaluation mode since the dataset is initially empty by default.

    .. seealso::
        See `InterfaceQuantity` documentation for details.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 requirements: Dict[str, "QuantityCreator"],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE,
                 auto_refresh: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param requirements: Intermediary quantities on which this quantity
                             depends for its evaluation, as a dictionary
                             whose keys are tuple gathering their respective
                             class plus any keyword-arguments of its
                             constructor except 'env' and 'parent'.
        :param mode: Desired mode of evaluation for this quantity. If mode is
                     set to `QuantityEvalMode.TRUE`, then current simulation
                     state will be used in dynamics computations. If mode is
                     set to `QuantityEvalMode.REFERENCE`, then the state at the
                     current simulation time of the selected reference
                     trajectory will be used instead.
        :param auto_refresh: Whether this quantity must be refreshed
                             automatically as soon as its shared cache has been
                             cleared if specified, otherwise this does nothing.
        """
        # Backup user argument(s)
        self.mode = mode

        # Make sure that no user-specified requirement is named 'trajectory'
        requirement_names = requirements.keys()
        if "trajectory" in requirement_names:
            raise ValueError(
                "Key 'trajectory' is reserved and cannot be used for "
                "user-specified requirements.")

        # Make sure that state requirement is valid if any or use default
        quantity = requirements.get("state")
        if quantity is not None:
            cls, kwargs = quantity
            if (not issubclass(cls, StateQuantity) or
                    kwargs.setdefault("mode", mode) != mode):
                raise ValueError(
                    "Key 'state' is reserved and can only be used to specify "
                    "a `StateQuantity` requirement, as a way to give the "
                    "opportunity to overwrite 'update_*' default arguments.")
        else:
            requirements["state"] = (StateQuantity, dict(mode=mode))

        # Call base implementation
        super().__init__(env, parent, requirements, auto_refresh=auto_refresh)

        # Add trajectory quantity proxy
        trajectory = self.state.trajectory
        assert isinstance(trajectory, DatasetTrajectoryQuantity)
        self.trajectory = trajectory

        # Robot for which the quantity must be evaluated
        self.robot = jiminy.Robot()
        self.pinocchio_model = pin.Model()
        self.pinocchio_data = self.pinocchio_model.createData()

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Force initializing state quantity
        self.state.initialize()

        # Refresh robot proxy
        assert isinstance(self.state, StateQuantity)
        self.robot = self.state.robot
        self.pinocchio_model = self.state.pinocchio_model
        self.pinocchio_data = self.state.pinocchio_data


def sync(fun: Callable[..., None]) -> Callable[..., None]:
    """Wrap any `InterfaceQuantity` instance method to forward call to all
    co-owners of the same shared cache.

    This wrapper is useful to keep all identical instances of the same quantity
    in sync.
    """
    @wraps(fun)
    def fun_safe(self: InterfaceQuantity, *args: Any, **kwargs: Any) -> None:
        # Hijack instance for adding private an attribute tracking whether its
        # internal state went out-of-sync between identical instances.
        # Note that a local variable cannot be used because all synched methods
        # must shared the same tracking state variable. Otherwise, one method
        # may be flagged out-of-sync but not the others.
        if not hasattr(self, "__is_synched__"):
            self.__is_synched__ = self.has_cache  # type: ignore[attr-defined]

        # Check if quantity has cache but is already out-of-sync.
        # Raise exception if it now has cache while it was not the case before.
        must_sync = self.has_cache and len(self.cache.owners) > 1
        if not self.__is_synched__ and must_sync:
            raise RuntimeError(
                "This quantity went out-of-sync. Make sure that no synched "
                "method is called priori to setting shared cache.")
        self.__is_synched__ = self.has_cache  # type: ignore[attr-defined]

        # Call instance method on all co-owners of shared cache
        cls = type(self)
        for owner in (self.cache.owners if self.has_cache else (self,)):
            assert isinstance(owner, cls)
            value = fun(  # type: ignore[func-returns-value]
                owner, *args, **kwargs)
            if value is not None:
                raise NotImplementedError(
                    "Only instance methods that returns `None` are supported.")

    return fun_safe


@dataclass(unsafe_hash=True)
class DatasetTrajectoryQuantity(InterfaceQuantity[State]):
    """This class manages a database of trajectories.

    The database is empty by default. Trajectories must be added or discarded
    manually. Only one trajectory can be selected at once. Once a trajectory
    has been selecting, its state at the current simulation can be easily
    retrieved.

    This class supports trajectories for which only part of the attributes of
    the underlying state sequence have been specified. Obviously, missing
    attributes of a trajectory will also be missing from the retrieved state.
    It is the responsibility of the practitioner to make sure that all the
    information that is necessary for its own application is available.

    All instances of this quantity sharing the same cache are synchronized,
    which means that adding, discarding, or selecting a trajectory on any of
    them would propagate on all the others.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        """
        # Call base implementation
        super().__init__(env, parent, requirements={}, auto_refresh=False)

        # Ordered set of named reference trajectories as a dictionary
        self.registry: OrderedDict[str, Trajectory] = OrderedDict()

        # Whether the dataset is locked, ie no traj can be added/discarded
        self._lock = False

        # Name of the trajectory that is currently selected
        self._name = ""

        # Selected trajectory if any
        self._trajectory: Optional[Trajectory] = None

        # Specifies how to deal with query time that are out-of-bounds
        self._mode: Literal['raise', 'wrap', 'clip'] = 'raise'

    @property
    def trajectory(self) -> Trajectory:
        """Trajectory that is currently selected if any, raises an exception
        otherwise.
        """
        # Make sure that a trajectory has been selected
        if self._trajectory is None:
            raise RuntimeError("No trajectory has been selected.")

        # Return selected trajectory
        return self._trajectory

    @property
    def robot(self) -> jiminy.Robot:
        """Robot associated with the selected trajectory.
        """
        return self.trajectory.robot

    @property
    def use_theoretical_model(self) -> bool:
        """Whether the selected trajectory is associated with the theoretical
        dynamical model or extended simulation model of the robot.
        """
        return self.trajectory.use_theoretical_model

    @sync
    def _add(self, name: str, trajectory: Trajectory) -> None:
        """Add a trajectory to local internal registry only without performing
        any validity check.

        .. warning::
            This method is used internally by `add` method. It is not meant to
            be called manually.

        :param name: Desired name of the trajectory.
        :param trajectory: Trajectory instance to register.
        """
        self.registry[name] = trajectory

    def add(self, name: str, trajectory: Trajectory) -> None:
        """Jointly add a trajectory to the local internal registry of all
        instances sharing the same cache as this quantity.

        :param name: Desired name of the trajectory. It must be unique. If a
                     trajectory with the exact same name already exists, then
                     it must be discarded first, so as to prevent silently
                     overwriting it by mistake.
        :param trajectory: Trajectory instance to register.
        """
        # Make sure that the dataset is not locked
        if self._lock:
            raise RuntimeError(
                "Trajectory dataset already locked. Impossible to add any "
                "trajectory.")

        # Make sure that no trajectory with the exact same name already exists
        if name in self.registry:
            raise KeyError(
                "A trajectory with the exact same name already exists. Please "
                "delete it first before adding a new one.")

        # Allocate new dummy robot to avoid altering the simulation one
        if trajectory.robot is self.env.robot:
            trajectory = replace(trajectory, robot=trajectory.robot.copy())

        # Add the same post-processed trajectory to all identical instances.
        # Note that `add` must be splitted in two methods. A first part that
        # applies some trajectory post-processing only once, and a second part
        # that adds the post-processed trajectory to all identical quantities
        # at once. It is absolutely essential to proceed this way, because it
        # guarantees that the underlying trajectories are all references to the
        # same memory, including `pinocchio_data`. This means that calling
        # `update_quantities` will perform the update for all of them at once.
        # Consequently, kinematics and dynamics quantities of all `State`
        # instances will be up-to-date as long as `refresh` is called once for
        # a given evaluation mode.
        self._add(name, trajectory)

    @sync
    def discard(self, name: str) -> None:
        """Jointly remove a trajectory from the local internal registry of all
        instances sharing the same cache as this quantity.

        :param name: Name of the trajectory to discard.
        """
        # Make sure that the dataset is not locked
        if self._lock:
            raise RuntimeError(
                "Trajectory dataset already locked. Impossible to discard any "
                "trajectory.")

        # Un-select trajectory if it corresponds to the discarded one
        if self._name == name:
            self._trajectory = None
            self._name = ""

        # Delete trajectory for global registry
        del self.registry[name]

    @sync
    def clear(self) -> None:
        """Clear the trajectory dataset from the local internal registry of all
        instances sharing the same cache as this quantity.
        """
        # Make sure that the dataset is not locked
        if self._lock:
            raise RuntimeError(
                "Trajectory dataset already locked. Impossible to clear the "
                "dataset.")

        # Un-select trajectory
        self._trajectory = None
        self._name = ""

        # Delete the whole registry
        self.registry.clear()

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate over all the trajectories in the dataset.
        """
        return iter(self.registry.values())

    def __bool__(self) -> bool:
        """Whether the dataset of trajectory is currently empty.
        """
        return bool(self.registry)

    @sync
    def select(self,
               name: str,
               mode: Literal['raise', 'wrap', 'clip'] = 'raise') -> None:
        """Select an existing trajectory from the database shared synchronized
        all managed quantities.

        .. note::
            There is no way to select a different reference trajectory for
            individual quantities at the time being.

        :param name: Name of the trajectory to select.
        :param mode: Specifies how to deal with query time of are out of the
                     time interval of the trajectory. See `Trajectory.get`
                     documentation for details.
        """
        # Make sure that at least one trajectory has been specified
        if not self.registry:
            raise ValueError("Cannot select trajectory on a empty dataset.")

        # Select the desired trajectory for all identical instances
        self._trajectory = self.registry[name]
        self._name = name

        # Backup user-specified mode
        self._mode = mode

        # Un-initialize quantity when the selected trajectory changes
        self.reset(reset_tracking=False)

    def lock(self) -> None:
        """Forbid adding/discarding trajectories to the dataset from now on.
        """
        self._lock = True

    @property
    def name(self) -> str:
        """Name of the trajectory that is currently selected.
        """
        return self._name

    @InterfaceQuantity.cache.setter  # type: ignore[attr-defined]
    def cache(self, cache: Optional[SharedCache[ValueT]]) -> None:
        # Get existing registry if any and making sure not already out-of-sync
        owner: Optional[InterfaceQuantity] = None
        if cache is not None and cache.owners:
            owner = next(iter(cache.owners))
            assert isinstance(owner, DatasetTrajectoryQuantity)
            if self._trajectory:
                raise RuntimeError(
                    "Trajectory dataset not empty. Impossible to add a shared "
                    "cache already having owners.")

        # Call base implementation
        InterfaceQuantity.cache.fset(self, cache)  # type: ignore[attr-defined]

        # Catch-up synchronization
        if owner:
            # Shallow copy the original registry, so that deletion / addition
            # does not propagate to other instances.
            self.registry = owner.registry.copy()
            if owner._trajectory is not None:
                self.select(owner._name, owner._mode)

    def refresh(self) -> State:
        """Compute state of selected trajectory at current simulation time.
        """
        return self.trajectory.get(self.env.stepper_state.t, self._mode)


@dataclass(unsafe_hash=True)
class StateQuantity(InterfaceQuantity[State]):
    """State to consider when evaluating any quantity deriving from
    `AbstractQuantity` using the same evaluation mode as this instance.

    This quantity is refreshed automatically no matter what. This guarantees
    that all low-level kinematics and dynamics quantities that can be computed
    from the current state are up-to-date. More specifically, every quantities
    would be up-to-date if the evaluation mode is `QuantityEvalMode.TRUE`,
    while it would depends on the information available on the selected
    trajectory if the evaluation mode is `QuantityEvalMode.REFERENCE`. See
    `update_quantities` documentation for details.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `QuantityEvalMode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE,
                 update_kinematics: bool = True,
                 update_dynamics: bool = False,
                 update_centroidal: bool = False,
                 update_energy: bool = False,
                 update_jacobian: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param mode: Desired mode of evaluation for this quantity. If mode is
                     set to `QuantityEvalMode.TRUE`, then current simulation
                     state will be used in dynamics computations. If mode is
                     set to `QuantityEvalMode.REFERENCE`, then at the state of
                     some reference trajectory at the current simulation time
                     will be used instead.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        :param update_kinematics: Whether to update body and frame transforms,
                                  spatial velocities and accelerations stored
                                  in `self.pinocchio_data` if necessary to be
                                  consistent with the current state of the
                                  robot. This argument has no effect if mode is
                                  set to `QuantityEvalMode.TRUE` because this
                                  property is already guarantee.
                                  Optional: False by default.
        :param update_dynamics: Whether to update the non-linear effects and
                                the joint internal forces stored in
                                `self.pinocchio_data` if necessary.
                                Optional: False by default.
        :param update_centroidal: Whether to update the centroidal dynamics
                                  (incl. CoM) stored in `self.pinocchio_data`
                                  if necessary.
                                  Optional: True by default.
        :param update_energy: Whether to update the potential and kinematic
                              energy stored in `self.pinocchio_data` if
                              necessary.
                              Optional: False by default.
        :param update_jacobian: Whether to update the joint Jacobian matrices
                                stored in `self.pinocchio_data` if necessary.
                                Optional: False by default.
        """
        # Make sure that the input arguments are valid
        update_kinematics = (
            update_kinematics or update_dynamics or update_centroidal or
            update_energy or update_jacobian)

        # Backup user argument(s)
        self.mode = mode
        self.update_kinematics = update_kinematics
        self.update_dynamics = update_dynamics
        self.update_centroidal = update_centroidal
        self.update_energy = update_energy
        self.update_jacobian = update_jacobian

        # Enable auto-refresh based on the evaluation mode
        # Note that it is necessary to auto-refresh this quantity, as it is the
        # one responsible for making sure that dynamics quantities are always
        # up-to-date when refreshing quantities. The latter are involved one
        # way of the other in the computation of any quantity, which means that
        # pre-computing it does not induce any unnecessary computations as long
        # as the user fetches the value of at least one quantity. Although this
        # assumption is very likely to be true at the step update period, it is
        # not the case at the observer update period. It sounds more efficient
        # refresh to the state the first time any quantity gets computed.
        # However, systematically checking if the state must be refreshed for
        # all quantities adds overhead and may be fairly costly overall. The
        # optimal trade-off is to rely on auto-refresh if the evaluation mode
        # is TRUE, since refreshing the state only consists in copying some
        # data, which is very cheap. On the contrary, it is more efficient to
        # only refresh the state when needed if the evaluation mode is TRAJ.
        # * Update state: 500ns (TRUE) | 5.0us (TRAJ)
        # * Check cache state: 70ns
        auto_refresh = mode is QuantityEvalMode.TRUE

        # Call base implementation.
        super().__init__(
            env,
            parent,
            requirements=dict(trajectory=(DatasetTrajectoryQuantity, {})),
            auto_refresh=auto_refresh)

        # Robot for which the quantity must be evaluated
        self.robot = env.robot
        self.pinocchio_model = env.robot.pinocchio_model
        self.pinocchio_data = env.robot.pinocchio_data

        # State for which the quantity must be evaluated
        self._state = State(t=np.nan, q=np.array([]))

        # Persistent buffer for storing body external forces if necessary
        self._f_external_vec = pin.StdVec_Force()
        self._f_external_list: List[np.ndarray] = []
        self._f_external_batch = np.array([])
        self._f_external_slices: Tuple[np.ndarray, ...] = ()

        # Persistent buffer storing all lambda multipliers for efficiency
        self._constraint_lambda_batch = np.array([])

        # Slices in stacked lambda multiplier flat vector
        self._constraint_lambda_slices: List[np.ndarray] = []

        # Lambda multipliers of all the constraints individually
        self._constraint_lambda_list: List[np.ndarray] = []

        # Whether to update kinematic and dynamic data to be consistent with
        # the current state of the robot, based on the requirement of all the
        # co-owners of shared cache.
        self._update_kinematics = False
        self._update_dynamics = False
        self._update_centroidal = False
        self._update_energy = False
        self._update_jacobian = False

    def initialize(self) -> None:
        # Determine which data must be update based on shared cache co-owners
        owners = self.cache.owners if self.has_cache else (self,)
        self._update_kinematics = False
        self._update_dynamics = False
        self._update_centroidal = False
        self._update_energy = False
        self._update_jacobian = False
        for owner in owners:
            self._update_kinematics |= owner.update_kinematics
            self._update_dynamics |= owner.update_dynamics
            self._update_centroidal |= owner.update_centroidal
            self._update_energy |= owner.update_energy
            self._update_jacobian |= owner.update_jacobian

        # Refresh robot and pinocchio proxies for co-owners of shared cache.
        # Note that automatic refresh is not sufficient to guarantee that
        # `initialize` will be called unconditionally, because it will be
        # skipped if a value is already stored in cache. As a result, it is
        # necessary to synchronize calls to this method between co-owners of
        # the shared cache manually, so that it will be called by the first
        # instance to found the cache empty. Only the necessary bits are
        # synchronized instead of the whole method, to avoid messing up with
        # computation graph tracking.
        for owner in owners:
            assert isinstance(owner, StateQuantity)
            if owner._is_initialized:
                continue
            if owner.mode is QuantityEvalMode.TRUE:
                owner.robot = owner.env.robot
                use_theoretical_model = False
            else:
                owner.robot = owner.trajectory.robot
                use_theoretical_model = owner.trajectory.use_theoretical_model
            if use_theoretical_model:
                owner.pinocchio_model = owner.robot.pinocchio_model_th
                owner.pinocchio_data = owner.robot.pinocchio_data_th
            else:
                owner.pinocchio_model = owner.robot.pinocchio_model
                owner.pinocchio_data = owner.robot.pinocchio_data

        # Call base implementation.
        # The quantity will be considered initialized and active at this point.
        super().initialize()

        # Refresh proxies and allocate memory for storing external forces
        if self.mode is QuantityEvalMode.TRUE:
            self._f_external_vec = self.env.robot_state.f_external
        else:
            self._f_external_vec = pin.StdVec_Force()
            self._f_external_vec.extend([
                pin.Force() for _ in range(self.pinocchio_model.njoints)])
        self._f_external_list = [
            f_ext.vector for f_ext in self._f_external_vec]
        self._f_external_batch = np.zeros((self.pinocchio_model.njoints, 6))
        self._f_external_slices = tuple(self._f_external_batch)

        # Allocate memory for lambda vector
        self._constraint_lambda_batch = np.zeros(
            (len(self.robot.log_constraint_fieldnames),))

        # Refresh mapping from lambda multipliers to corresponding slice
        self._constraint_lambda_list.clear()
        self._constraint_lambda_slices.clear()
        constraint_lookup_pairs = tuple(
            (f"Constraint{registry_type}", registry)
            for registry_type, registry in (
                ("BoundJoints", self.robot.constraints.bounds_joints),
                ("ContactFrames", self.robot.constraints.contact_frames),
                ("CollisionBodies", {
                    name: constraint for constraints in (
                        self.robot.constraints.collision_bodies)
                    for name, constraint in constraints.items()}),
                ("User", self.robot.constraints.user)))
        i = 0
        while i < len(self.robot.log_constraint_fieldnames):
            fieldname = self.robot.log_constraint_fieldnames[i]
            for registry_type, registry in constraint_lookup_pairs:
                if fieldname.startswith(registry_type):
                    break
            constraint_name = fieldname[len(registry_type):-1]
            constraint = registry[constraint_name]
            self._constraint_lambda_list.append(constraint.lambda_c)
            self._constraint_lambda_slices.append(
                self._constraint_lambda_batch[i:(i + constraint.size)])
            i += constraint.size

        # Allocate state for which the quantity must be evaluated if needed
        if self.mode is QuantityEvalMode.TRUE:
            if not self.env.is_simulation_running:
                raise RuntimeError("No simulation running. Impossible to "
                                   "initialize this quantity.")
            self._state = State(
                0.0,
                self.env.robot_state.q,
                self.env.robot_state.v,
                self.env.robot_state.a,
                self.env.robot_state.u,
                self.env.robot_state.command,
                self._f_external_batch,
                self._constraint_lambda_batch)

    def refresh(self) -> State:
        """Compute the current state depending on the mode of evaluation, and
        make sure that kinematics and dynamics quantities are up-to-date.
        """
        if self.mode is _TRUE:
            # Update the current simulation time
            self._state.t = self.env.stepper_state.t

            # Update external forces and constraint multipliers in state buffer
            multi_array_copyto(self._f_external_slices, self._f_external_list)
            multi_array_copyto(
                self._constraint_lambda_slices, self._constraint_lambda_list)
        else:
            self._state = self.trajectory.get()

            # Copy body external forces from stacked buffer to force vector
            has_forces = self._state.f_external is not None
            if has_forces:
                array_copyto(self._f_external_batch, self._state.f_external)
                multi_array_copyto(self._f_external_list,
                                   self._f_external_slices)

            # Update all dynamical quantities that can be given available data
            if self.update_kinematics:
                update_quantities(
                    self.robot,
                    self._state.q,
                    self._state.v,
                    self._state.a,
                    self._f_external_vec if has_forces else None,
                    update_dynamics=self._update_dynamics,
                    update_centroidal=self._update_centroidal,
                    update_energy=self._update_energy,
                    update_jacobian=self._update_jacobian,
                    update_collisions=False,
                    use_theoretical_model=(
                        self.trajectory.use_theoretical_model))

            # Restore lagrangian multipliers of the constraints if available
            if self._state.lambda_c is not None:
                array_copyto(
                    self._constraint_lambda_batch, self._state.lambda_c)
                multi_array_copyto(self._constraint_lambda_list,
                                   self._constraint_lambda_slices)

        # Return state
        return self._state
