"""Implementation of Mahony filter block compatible with gym_jiminy
reinforcement learning pipeline environment design.
"""
from collections import OrderedDict
from typing import Any, Type, TypeVar, cast

import numpy as np
import gymnasium as gym

from jiminy_py import tree

from ..bases import (
    BaseObs, BaseAct, BaseObserverBlock, InterfaceJiminyEnv, InterfaceQuantity)
from ..utils import DataNested, build_copyto


ValueT = TypeVar('ValueT', bound=DataNested)


def get_space(data: DataNested) -> gym.Space[DataNested]:
    """Infer space from a given value.

    .. warning::
        Beware that space inference is lossly. Firstly, one cannot discriminate
        between `gym.spaces.Box` and other non-container spaces, e.g.
        `gym.spaces.Discrete` or `gym.spaces.MultiBinary`. Because of this
        limitation, it is assumed that all `np.ndarray` data has been sampled
        by a `gym.spaces.Box` space. Secondly, it is impossible to determine
        the bounds of the space, so it is assumed to be unbounded.

    :param value: Any value sampled from a given space.
    """
    data_type = type(data)
    if tree.issubclass_mapping(data_type):
        return gym.spaces.Dict(OrderedDict([
            (field, get_space(value))
            for field, value in data.items()]))  # type: ignore[union-attr]
    if tree.issubclass_sequence(data_type):
        return gym.spaces.Tuple([get_space(value) for value in data])
    assert isinstance(data, np.ndarray)
    return gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=data.shape,
            dtype=data.dtype.type)


class QuantityObserver(BaseObserverBlock[ValueT, None, BaseObs, BaseAct]):
    """Add a given pre-defined quantity to the observation of the environment.

    .. warning::
        The observation space of a quantity must be invariant. Yet, nothing
        prevent the shape of the quantity to change dynamically. As a result,
        it is up to user to make sure that does not occur in practice,
        otherwise it will raise an exception.
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 quantity: Type[InterfaceQuantity[ValueT]],
                 *,
                 update_ratio: int = 1,
                 **kwargs: Any) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param quantity: Type of the quantity.
        :param update_ratio: Ratio between the update period of the observer
                             and the one of the subsequent observer. -1 to
                             match the simulation timestep of the environment.
                             Optional: `1` by default.
        :param kwargs: Additional arguments that will be forwarded to the
                       constructor of the quantity.
        """
        # Add the quantity to the environment
        env.quantities[name] = (quantity, kwargs)

        # Define proxy for fast access
        self.data = env.quantities.registry[name]

        # Initialize the observer
        super().__init__(name, env, update_ratio)

        # Try to bind the memory of the quantity to the observation.
        # Note that there is no guarantee that the quantity will be updated
        # in-place without dynamic memory allocation, so it needs to be checked
        # at run-time systematically and copy the value if necessary.
        self.observation = self.data.get()

        # Define specialized copyto operator for efficiency.
        # This is necessary because there is no guarantee that the quantity
        # will be updated in-place without dynamic memory allocation.
        self._copyto_observation = build_copyto(self.observation)

    def __del__(self) -> None:
        try:
            del self.env.quantities[self.name]
        except Exception:   # pylint: disable=broad-except
            # This method must not fail under any circumstances
            pass

    def _initialize_observation_space(self) -> None:
        # Let us infer the observation space from the value of the quantity.
        # Note that it is always valid to fetch the value of a quantity, even
        # if no simulation is running.
        self.observation_space = cast(
            gym.Space[ValueT], get_space(self.data.get()))

    def refresh_observation(self, measurement: BaseObs) -> None:
        # Evaluate the quantity
        value = self.data.get()

        # Update the observation in-place in case of dynamic memory allocation
        if self.observation is not value:
            self._copyto_observation(value)
