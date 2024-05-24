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
from enum import Enum
from weakref import ReferenceType
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import MutableSet
from dataclasses import dataclass, replace
from functools import partial, wraps
from typing import (
    Any, Dict, List, Optional, Tuple, Generic, TypeVar, Type, Iterator,
    Callable, Literal, cast)

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    multi_array_copyto)
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

    owners: WeakMutableCollection["InterfaceQuantity[ValueT]"]
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
        def _callback(self: WeakMutableCollection["InterfaceQuantity"],
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

    def reset(self, *, ignore_auto_refresh: bool = False) -> None:
        """Clear value stored in cache if any.
        """
        # Clear cache
        self._value = None
        self._has_value = False

        # Refresh automatically if any cache owner requested it and not ignored
        if not ignore_auto_refresh:
            for owner in self.owners:
                if owner.auto_refresh:
                    owner.get()
                    break

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


class InterfaceQuantity(ABC, Generic[ValueT]):
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
        used internally by `QuantityManager` to synchronize cache  between
        them. It is advised to use decorator `@dataclass(unsafe_hash=True)` for
        convenience, but it can also be done manually.
    """

    requirements: Dict[str, "InterfaceQuantity"]
    """Intermediary quantities on which this quantity may rely on for its
    evaluation at some point, depending on the optimal computation path at
    runtime. There values will be exposed to the user as usual properties.
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
                             class and all their constructor keyword-arguments
                             except environment 'env' and parent 'parent.
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

        # Shared cache handling
        self._cache: Optional[SharedCache[ValueT]] = None
        self.has_cache = False

        # Track whether the quantity has been called since previous reset
        self._is_active = False

        # Whether the quantity must be re-initialized
        self._is_initialized: bool = False

        # Add getter dynamically for user-specified intermediary quantities.
        # This approach is hacky but much faster than any of other official
        # approach, ie implementing custom a `__getattribute__` method or even
        # worst a custom `__getattr__` method.
        def get_value(name: str, quantity: InterfaceQuantity) -> Any:
            return quantity.requirements[name].get()

        for name in requirement_names:
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
        try:
            return self.__getattribute__('requirements')[name].get()
        except KeyError as e:
            raise AttributeError(
                f"'{type(self)}' object has no attribute '{name}'") from e

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
        if not self.has_cache:
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
        # Get value in cache if available.
        # Note that direct access to internal `_value` attribute is preferred
        # over the public API `get` for speedup. The same cannot be done for
        # `has_value` as it would prevent mocking it during running unit tests
        # or benchmarks.
        if (self.has_cache and
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
        if self.has_cache:
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
        # Make sure that auto-refresh can be honored
        if self.auto_refresh and not self.has_cache:
            raise RuntimeError(
                "Automatic refresh enabled but no shared cache available. "
                "Please add one before calling this method.")

        # No longer consider this exact instance as initialized
        self._is_initialized = False

        # No longer consider this exact instance as active if requested
        if reset_tracking:
            self._is_active = False

        # Reset all requirements first
        for quantity in self.requirements.values():
            quantity.reset(reset_tracking)

        # More work must to be done if shared cache is available and has value
        if self.has_cache:
            # Early return if shared cache has no value
            if not self.cache.has_value:
                return

            # Invalidate cache before looping over all identical properties.
            # Note that auto-refresh must be ignored to avoid infinite loop.
            self.cache.reset(ignore_auto_refresh=True)

            # Reset all identical quantities
            for owner in self.cache.owners:
                owner.reset()

            # Reset shared cache one last time but without ignore auto refresh
            self.cache.reset()

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


QuantityCreator = Tuple[Type[InterfaceQuantity[ValueT]], Dict[str, Any]]


class QuantityEvalMode(Enum):
    """Specify on which state to evaluate a given quantity.
    """

    TRUE = 0
    """Current state of the environment.
    """

    REFERENCE = 1
    """State of the reference trajectory at the current simulation time.
    """


@dataclass(unsafe_hash=True)
class AbstractQuantity(InterfaceQuantity, Generic[ValueT]):
    """Base class for generic quantities involved observer-controller blocks,
    reward components or termination conditions.

    .. note::
        A dataset of trajectories made available through `self.trajectories`.
        The latter is synchronized because all quantities as long as shared
        cached is available. Since the dataset is initially empty by default,
        using `QuantityEvalMode.REFERENCE` evaluation mode requires manually
        adding at least one trajectory to the dataset and selecting it.

    .. seealso::
        See `InterfaceQuantity` documentation for details.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
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
                             class and all their constructor keyword-arguments
                             except environment 'env' and parent 'parent.
        :param mode: Desired mode of evaluation for this quantity. If mode is
                     set to `QuantityEvalMode.TRUE`, then current simulation
                     state will be used in dynamics computations. If mode is
                     set to `QuantityEvalMode.REFERENCE`, then at the state of
                     some reference trajectory at the current simulation time
                     will be used instead.
        :param auto_refresh: Whether this quantity must be refreshed
                             automatically as soon as its shared cache has been
                             cleared if specified, otherwise this does nothing.
        """
        # Backup user argument(s)
        self.mode = mode

        # Make sure that no user-specified requirement is named 'trajectory'
        requirement_names = requirements.keys()
        if any(name in requirement_names for name in ("state", "trajectory")):
            raise ValueError(
                "No user-specified requirement can be named 'state' nor "
                "'trajectory' as these keys are reserved.")

        # Add state quantity as requirement
        requirements["state"] = (StateQuantity, dict(mode=mode))

        # Call base implementation
        super().__init__(env, parent, requirements, auto_refresh=auto_refresh)

        # Add trajectory quantity proxy
        trajectory = self.requirements["state"].requirements["trajectory"]
        assert isinstance(trajectory, DatasetTrajectoryQuantity)
        self.trajectory = trajectory

        # Robot for which the quantity must be evaluated
        self.robot = jiminy.Robot()
        self.pinocchio_model = pin.Model()
        self.pinocchio_data = self.pinocchio_model.createData()

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Refresh robot proxy
        state = self.requirements["state"]
        assert isinstance(state, StateQuantity)
        self.robot = state.robot
        self.pinocchio_model = state.pinocchio_model
        self.pinocchio_data = state.pinocchio_data


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

        # Call instance method on the original caller first
        fun(self, *args, **kwargs)

        # Call instance method on all other co-owners of shared cache if any
        if self.has_cache:
            cls = type(self)
            for owner in self.cache.owners:
                if owner is not self:
                    assert isinstance(owner, cls)
                    value = fun(owner, *args, **kwargs)
                    if value is not None:
                        raise NotImplementedError(
                            "Instance methods that does not return `None` are "
                            "not supported.")

    return fun_safe


@dataclass(unsafe_hash=True)
class DatasetTrajectoryQuantity(InterfaceQuantity[State]):
    """This class manages a database of trajectories.

    The database is empty by default. Trajectories must be added or discarded
    manually. Only one trajectory can be selected at once. Once a trajectory
    has been selecting, its state at the current simulation can be easily
    retrieved.

    This class does not require to only adding trajectories for which all
    attributes of the underlying state sequence have been specified. Missing
    attributes of a trajectory will also be missing from the retrieved state.
    It is the responsible of the user to make sure all cases are properly
    handled if needed.

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
        # Un-select trajectory if it corresponds to the discarded one
        if self._name == name:
            self._trajectory = None
            self._name = ""

        # Delete trajectory for global registry
        del self.registry[name]

    @sync
    def select(self,
               name: str,
               mode: Literal['raise', 'wrap', 'clip'] = 'raise') -> None:
        """Jointly select a trajectory in the internal registry of all
        instances sharing the same cache as this quantity.

        :param name: Name of the trajectory to discard.
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
            self.registry = owner.registry
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
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
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
        """
        # Backup user argument(s)
        self.mode = mode

        # Call base implementation
        super().__init__(env, parent, requirements={}, auto_refresh=True)

        # Create empty trajectory database, manually added as a requirement.
        # Note that it must be done after base initialization, otherwise a
        # getter will be added for it as first-class property.
        self.trajectory = DatasetTrajectoryQuantity(env, self)
        self.requirements["trajectory"] = self.trajectory

        # Robot for which the quantity must be evaluated
        self.robot = env.robot
        self.pinocchio_model = env.robot.pinocchio_model
        self.pinocchio_data = env.robot.pinocchio_data

        # State for which the quantity must be evaluated
        self._f_external_slices: Tuple[np.ndarray, ...] = ()
        self._f_external_list: Tuple[np.ndarray, ...] = ()
        self.state = State(t=np.nan, q=np.array([]))

    def initialize(self) -> None:
        # Refresh robot and pinocchio proxies for co-owners of shared cache.
        # Note that automatic refresh is not sufficient to guarantee that
        # `initialize` will be called unconditionally, because it will be
        # skipped if a value is already stored in cache. As a result, it is
        # necessary to synchronize calls to this method between co-owners of
        # the shared cache manually, so that it will be called by the first
        # instance to found the cache empty. Only the necessary bits are
        # synchronized instead of the whole method, to avoid messing up with
        # computation graph tracking.
        owners = self.cache.owners if self.has_cache else (self,)
        for owner in owners:
            assert isinstance(owner, StateQuantity)
            if owner._is_initialized:
                continue
            if owner.mode == QuantityEvalMode.TRUE:
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
        # Thz quantity will be considered initialized and active at this point.
        super().initialize()

        # State for which the quantity must be evaluated
        if self.mode == QuantityEvalMode.TRUE:
            self._f_external_list = tuple(
                f_ext.vector for f_ext in self.env.robot_state.f_external)
            if self._f_external_list:
                f_external_batch = np.stack(self._f_external_list, axis=0)
            else:
                f_external_batch = np.array([])
            self.state = State(
                self.env.stepper_state.t,
                self.env.robot_state.q,
                self.env.robot_state.v,
                self.env.robot_state.a,
                self.env.robot_state.u,
                self.env.robot_state.command,
                f_external_batch)
            self._f_external_slices = tuple(f_external_batch)

    def refresh(self) -> State:
        """Compute the current state depending on the mode of evaluation, and
        make sure that kinematics and dynamics quantities are up-to-date.
        """
        # Update state at which the quantity must be evaluated
        if self.mode == QuantityEvalMode.TRUE:
            multi_array_copyto(self._f_external_slices, self._f_external_list)
        else:
            self.state = self.trajectory.get()

        # Update all the dynamical quantities that can be given available data
        if self.mode == QuantityEvalMode.REFERENCE:
            update_quantities(
                self.robot,
                self.state.q,
                self.state.v,
                self.state.a,
                update_physics=True,
                update_centroidal=True,
                update_energy=True,
                update_jacobian=False,
                update_collisions=True,
                use_theoretical_model=self.trajectory.use_theoretical_model)

        # Return state
        return self.state
