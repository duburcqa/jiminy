"""This module provides a dedicated quantity manager. Although not necessary
for computing quantities, its usage is strongly recommended for optimal
performance.

It is responsible for optimizing the computation path, which is expected to
significantly increase the step collection throughput. This speedup is achieved
by caching already computed that did not changed since then, computing
redundant intermediary quantities only once per step, and gathering similar
quantities in a large batch to leverage vectorization of math instructions.
"""
from typing import Any, Dict, List, Tuple, Type, cast

from ..bases import (
    QuantityCreator, InterfaceJiminyEnv, InterfaceQuantity, SharedCache,
    DatasetTrajectoryQuantity)


class QuantityManager:
    """This class centralizes the evaluation of all quantities involved in
    reward components or termination conditions evaluation to redundant and
    unnecessary computations.

    It is responsible for making sure all quantities are evaluated on the same
    environment, and internal buffers are re-initialized whenever necessary. It
    also manages a dataset of trajectories synchronized between all managed
    quantities. These trajectories are involves in computation of quantities
    deriving from `AbstractQuantity` for which the mode of evaluation is set to
    `QuantityEvalMode.REFERENCE`. This dataset is initially empty. Trying to
    evaluate quantities involving a reference trajectory without adding and
    selecting one beforehand would raise an exception.

    .. note::
        There is no way to select a different reference trajectory for
        individual quantities at the time being.

    .. note::
        Individual quantities can be accessed either as instance properties or
        items of a dictionary. Choosing one or the other is only a matter of
        taste since both options have been heavily optimized to  minimize
        overhead and should be equally efficient.
    """

    trajectory_dataset: DatasetTrajectoryQuantity
    """Database of reference trajectories synchronized between all managed
    quantities.
    """

    def __init__(self, env: InterfaceJiminyEnv) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        """
        # Backup user argument(s)
        self.env = env

        # List of instantiated quantities to manager
        self._registry: Dict[str, InterfaceQuantity] = {}

        # Initialize shared caches for all managed quantities.
        # Note that keys are not quantities directly but pairs (class, hash).
        # This is necessary because using a quantity as key directly would
        # prevent its garbage collection, hence breaking automatic reset of
        # computation tracking for all quantities sharing its cache.
        # In case of dataclasses, their hash is the same as if it was obtained
        # using `hash(dataclasses.astuple(quantity))`. This is clearly not
        # unique, as all it requires to be the same is being built from the
        # same nested ordered arguments. To get around this issue, we need to
        # store (key, value) pairs in a list.
        self._caches: List[Tuple[
            Tuple[Type[InterfaceQuantity], int], SharedCache]] = []

        # Instantiate trajectory database.
        # Note that this quantity is not added to the global registry to avoid
        # exposing directly to the user. This way, it cannot be deleted.
        self.trajectory_dataset = cast(
            DatasetTrajectoryQuantity, self._build_quantity(
                (DatasetTrajectoryQuantity, {})))

    def reset(self, reset_tracking: bool = False) -> None:
        """Consider that all managed quantity must be re-initialized before
        being able to evaluate them once again.

        .. note::
            The cache is cleared automatically by the quantities themselves.

        .. note::
            This method is supposed to be called before starting a simulation.

        :param reset_tracking: Do not consider any quantity as active anymore.
                               Optional: False by default.
        """
        for quantity in self._registry.values():
            quantity.reset(reset_tracking)

    def clear(self) -> None:
        """Clear internal cache of quantities to force re-evaluating them the
        next time their value is fetched.

        .. note::
            This method is supposed to be called every time the state of the
            environment has changed (ie either the agent or world itself),
            thereby invalidating the value currently stored in cache if any.
        """
        ignore_auto_refresh = not self.env.is_simulation_running
        for _, cache in self._caches:
            cache.reset(ignore_auto_refresh=ignore_auto_refresh)

    def _build_quantity(
            self, quantity_creator: QuantityCreator) -> InterfaceQuantity:
        """Instantiate a quantity sharing caching with all identical quantities
        that has been instantiated previously.

        .. note::
            This method is not meant to be called manually. It is used
            internally for instantiating new quantities sharing cache with all
            identical instances that has been instantiated previously by this
            manager in particular. This method is not responsible for keeping
            track of the new quantity and exposing it to the user by adding it
            to the global registry of the manager afterward.

        :param name: Desired name of the quantity after instantiation. It will
                     raise an exception if another quantity with the exact same
                     name exists.
        :param quantity_creator: Tuple gathering the class of the new quantity
                                 to manage plus any keyword-arguments of its
                                 constructor as a dictionary except 'env' and
                                 'parent'.
        """
        # Instantiate the new quantity
        quantity_cls, quantity_kwargs = quantity_creator
        top_quantity = quantity_cls(self.env, None, **(quantity_kwargs or {}))

        # Get the list of all quantities involved in computations of the top
        # level quantity, sorted from highest level to lowest level.
        quantities_all, quantities_sorted_all = [top_quantity], [top_quantity]
        while quantities_all:
            quantities = quantities_all.pop().requirements.values()
            quantities_all += quantities
            quantities_sorted_all += quantities

        # Set a shared cache entry for all quantities involved in computations.
        # Make sure that the cache associated with requirements precedes their
        # parents in global cache registry. This is essential for automatic
        # refresh, to ensure that cached values of all the intermediary
        # quantities have been cleared before refresh.
        for quantity in quantities_sorted_all[::-1]:
            # Get already available cache entry if any, otherwise create it
            key = (type(quantity), hash(quantity))
            for cache_key, cache in self._caches:
                if key == cache_key:
                    owner, *_ = cache.owners
                    if quantity == owner:
                        break
            else:
                # Partially sort cache entries to reset all quantity instances
                # of the same class at once.
                # The objective is to avoid resetting multiple times the same
                # quantity because of the auto-refresh mechanism.
                cache = SharedCache()
                for i, (cache_key, _) in enumerate(self._caches):
                    if key[0] == cache_key[0]:
                        self._caches.insert(i + 1, (key, cache))
                        break
                else:
                    self._caches.append((key, cache))

            # Set shared cache of the quantity
            quantity.cache = cache

        return top_quantity

    def add(self,
            name: str,
            quantity_creator: QuantityCreator) -> InterfaceQuantity:
        """Instantiate new top-level quantity that will be managed for now on.

        :param name: Desired name of the quantity after instantiation. It will
                     raise an exception if another quantity with the exact same
                     name exists.
        :param quantity_creator: Tuple gathering the class of the new quantity
                                 to manage plus any keyword-arguments of its
                                 constructor as a dictionary except 'env' and
                                 'parent'.
        """
        # Make sure that no quantity with the same name is already managed to
        # avoid silently overriding quantities being managed in user's back.
        if name in self._registry:
            raise KeyError(
                "A quantity with the exact same name already exists. Please "
                "delete it first before adding a new one.")

        # Instantiate new quantity
        quantity: InterfaceQuantity = self._build_quantity(quantity_creator)

        # Add it to the global registry of already managed quantities
        self._registry[name] = quantity

        return quantity

    def discard(self, name: str) -> None:
        """Stop managing a quantity that is no longer relevant.

        .. warning::
            Deleting managed quantities modifies the computation graph, which
            would affect quantities that detect the optimal computation path
            dynamically. Computation tracking of all owners of a shared cache
            will be reset at garbage collection by the cache itself to get the
            opportunity to recover optimality.

        :param name: Name of the managed quantity to be discarded. It will
                     raise an exception if the specified name does not exists.
        """
        # Remove shared cache entries for the quantity and its requirements.
        # Note that done top-down rather than bottom-up, otherwise reset of
        # required quantities no longer having shared cache will be triggered
        # automatically by parent quantities following computation graph
        # tracking reset whenever a shared cache co-owner is removed.
        quantities_all = [self._registry.pop(name)]
        while quantities_all:
            quantity = quantities_all.pop(0)
            cache = quantity.cache
            quantity.cache = None  # type: ignore[assignment]
            if len(cache.owners) == 0:
                for i, (_, _cache) in enumerate(self._caches):
                    if cache is _cache:
                        del self._caches[i]
                        break
            quantities_all += quantity.requirements.values()

    def get(self, name: str) -> Any:
        """Fetch the value of a given quantity.

        The quantity will evaluate if its value is not already cached.

        :param name: Name of the quantity for which to fetch the current value.
        """
        return self._registry[name].get()
