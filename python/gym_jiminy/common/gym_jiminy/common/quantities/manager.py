"""This module provides a dedicated quantity manager. Although not necessary
for computing quantities, its usage is strongly recommended for optimal
performance.

It is responsible for optimizing the computation path, which is expected to
significantly increase the step collection throughput. This speedup is achieved
by caching already computed that did not changed since then, computing
redundant intermediary quantities only once per step, and gathering similar
quantities in a large batch to leverage vectorization of math instructions.
"""
from collections.abc import MutableMapping
from typing import Any, Dict, List, Tuple, Iterator, Type

from ..bases import InterfaceJiminyEnv, AbstractQuantity, SharedCache


QuantityCreator = Tuple[Type[AbstractQuantity], Dict[str, Any]]


class QuantityManager(MutableMapping):
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
    def __init__(self, env: InterfaceJiminyEnv) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        """
        # Backup user argument(s)
        self.env = env

        # List of already instantiated quantities to manager
        self._quantities: Dict[str, AbstractQuantity] = {}

        # Initialize shared caches for all managed quantities.
        # Note that keys are not quantities directly but pairs (class, hash).
        # This is necessary because using a quantity as key directly would
        # prevent its garbage collection and break automatic reset of
        # computation tracking for all quantities sharing its cache.
        self._caches: Dict[
            Tuple[Type[AbstractQuantity], int], SharedCache] = {}

    def reset(self, reset_tracking: bool = False) -> None:
        """Consider that all managed quantity must be re-initialized before
        being able to evaluate them once again.

        .. note::
            The cache is cleared automatically by the quantities themselves.

        :param reset_tracking: Do not consider any quantity as active anymore.
                               Optional: False by default.
        """
        for quantity in self._quantities.values():
            quantity.reset(reset_tracking)

    def clear(self) -> None:
        """Clear internal cache of quantities to force re-evaluating them the
        next time their value is fetched.

        .. note::
            This method is supposed to be called every time the state of the
            environment has changed (ie either the agent or world itself),
            thereby invalidating the value currently stored in cache if any.
        """
        for cache in self._caches.values():
            cache.reset()

    def __getattr__(self, name: str) -> Any:
        """Get access managed quantities as first-class properties, rather than
        dictionary-like values through `__getitem__`.

        .. warning::
            Getting quantities this way is convenient but unfortunately much
            slower than doing it through `__getitem__`. It takes 40ns on
            Python 3.12 and a whooping 180ns on Python 3.11. As a result, this
            approach is mainly intended for ease of use while prototyping and
            is not recommended in production, especially on Python<3.12.

        :param name: Name of the requested quantity.
        """
        return self[name]

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return [*super().__dir__(), *self._quantities.keys()]

    def __setitem__(self,
                    name: str,
                    quantity_creator: QuantityCreator) -> None:
        """Instantiate a new top-level quantity to manage for now on.

        :param name: Desired name of the quantity after instantiation. It will
                     raise an exception if another quantity with the exact same
                     name exists.
        :param quantity_creator: Tuple gathering the class of the new quantity
                                 to manage plus its keyword-arguments except
                                 the environment 'env' as a dictionary.
        """
        # Make sure that no quantity with the same name is already managed to
        # avoid silently overriding quantities being managed in user's back.
        if name in self._quantities:
            raise KeyError(
                "A quantity with the exact same name already exists. Please "
                "delete it first before adding a new one.")

        # Instantiate the new quantity
        quantity_cls, quantity_kwargs = quantity_creator
        top_quantity = quantity_cls(self.env, None, **(quantity_kwargs or {}))

        # Set a shared cache entry for all quantities involved in computations
        quantities_all = [top_quantity]
        while quantities_all:
            quantity = quantities_all.pop()
            key = (type(quantity), hash(quantity))
            quantity.cache = self._caches.setdefault(key, SharedCache())
            quantities_all += quantity.requirements.values()

        # Add it to the map of already managed quantities
        self._quantities[name] = top_quantity

    def __getitem__(self, name: str) -> Any:
        """Get the evaluated value of a given quantity.

        :param name: Name of the quantity for which to fetch the current value.
        """
        return self._quantities[name].get()

    def __delitem__(self, name: str) -> None:
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
        del self._quantities[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over names of managed quantities.
        """
        return iter(self._quantities)

    def __len__(self) -> int:
        """Number of quantities being managed.
        """
        return len(self._quantities)
