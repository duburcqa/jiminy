"""This module implements a block transformation for filtering out some of the
keys of the observation space of an environment that may be arbitrarily nested.
"""
from operator import getitem
from functools import reduce
from collections import OrderedDict
from typing import (
    Sequence, Set, Tuple, Union, Generic, TypeVar, Type, no_type_check)
from typing_extensions import TypeAlias

import gymnasium as gym
from jiminy_py.tree import (
    flatten_with_path, issubclass_mapping, issubclass_sequence)

from ..bases import (NestedObsT,
                     ActT,
                     InterfaceJiminyEnv,
                     BaseTransformObservation)
from ..utils import DataNested, copy


SpaceOrDataT = TypeVar(
    'SpaceOrDataT', bound=Union[DataNested, gym.Space[DataNested]])
FilteredObsT: TypeAlias = NestedObsT


@no_type_check
def _copy_filtered(data: SpaceOrDataT,
                   path_filtered_leaves: Set[Tuple[str, ...]]) -> SpaceOrDataT:
    """Partially shallow copy some nested data structure, so that all leaves
    being filtered in are still references but their corresponding containers
    are copies.

    :param data: Hierarchical data structure to copy without allocation.
    :param path_filtered_leaves: Set gathering the paths of all leaves that
                                 must be kept. Each path is a tuple of keys
                                 to access a given leaf recursively.
    """
    # Special handling if no leaf to filter has been specified
    if not path_filtered_leaves:
        data_type = type(data)
        if issubclass_mapping(data_type) or issubclass_sequence(data_type):
            return data_type()
        return data

    # Shallow copy the whole data structure
    out = copy(data)

    # Convert all parent containers to mutable dictionary
    type_filtered_nodes: Sequence[Type] = []
    out_flat = flatten_with_path(out)
    for key_nested, _ in out_flat:
        if key_nested not in path_filtered_leaves:
            continue
        for i in range(1, len(key_nested) + 1):
            # Extract parent container
            *key_nested_parent, key_leaf = key_nested[:i]
            if key_nested_parent:
                *key_nested_container, key_parent = key_nested_parent
                container = reduce(getitem, key_nested_container, out)
                parent = container[key_parent]
            else:
                parent = out

            # Convert parent container to mutable dictionary
            parent_type = type(parent)
            type_filtered_nodes.append(parent_type)
            if parent_type in (list, dict, OrderedDict):
                continue
            if issubclass(parent_type, gym.Space):
                parent = parent.spaces
            if issubclass_mapping(parent_type):
                parent = dict(parent.items())
            elif issubclass_sequence(parent_type):
                parent = dict(enumerate(parent))
            else:
                raise NotImplementedError(
                    f"Unsupported container type: '{parent_type}'")

            # Re-assign parent data structure
            if key_nested_parent:
                container[key_parent] = parent
            else:
                out = parent

    # Remove unnecessary keys
    for key_nested, _ in out_flat:
        if key_nested in path_filtered_leaves:
            continue
        for i in range(len(key_nested))[::-1]:
            if any(key_nested[:i] == path[:i]
                   for path in path_filtered_leaves):
                break
        *key_nested_parent, key_leaf = key_nested[:(i + 1)]
        try:
            parent = reduce(getitem, key_nested_parent, out)
            del parent[key_leaf]
        except KeyError:
            # Some nested keys may have been deleted previously
            pass

    # Restore original parent container types
    parent_type_it = iter(type_filtered_nodes[::-1])
    for key_nested, _ in out_flat[::-1]:
        if key_nested not in path_filtered_leaves:
            continue
        for i in range(1, len(key_nested) + 1)[::-1]:
            # Extract parent container
            *key_nested_parent, _ = key_nested[:i]
            if key_nested_parent:
                *key_nested_container, key_parent = key_nested_parent
                container = reduce(getitem, key_nested_container, out)
                parent = container[key_parent]
            else:
                parent = out

            # Restore original container type if not already done
            parent_type = next(parent_type_it)
            if isinstance(parent, parent_type):
                continue
            if issubclass_mapping(parent_type):
                parent = parent_type(tuple(parent.items()))
            elif issubclass_sequence(parent_type):
                parent = parent_type(tuple(parent.values()))

            # Re-assign output data structure
            if key_nested_parent:
                container[key_parent] = parent
            else:
                out = parent
    return out


class FilterObservation(
        BaseTransformObservation[FilteredObsT, NestedObsT, ActT],
        Generic[NestedObsT, ActT]):
    """Filter nested observation space.

    This wrapper does nothing but providing an observation only exposing a
    subset of all the leaves of the original observation space. For flattening
    the observation space after filtering, you should wrap the environment with
    `FlattenObservation` as yet another layer.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[NestedObsT, ActT],
                 nested_filter_keys: Sequence[
                    Union[Sequence[Union[str, int]], Union[str, int]]]
                 ) -> None:
        # Make sure that the observation space derives from 'gym.spaces.Dict'
        assert isinstance(env.observation_space, gym.spaces.Dict)

        # Make sure all nested keys are stored in sequence
        assert not isinstance(nested_filter_keys, str)
        self.nested_filter_keys: Sequence[Sequence[Union[str, int]]] = []
        for key_nested in nested_filter_keys:
            if isinstance(key_nested, (str, int)):
                key_nested = (key_nested,)
            self.nested_filter_keys.append(tuple(key_nested))

        # Get all paths associated with leaf values that must be stacked
        self.path_filtered_leaves: Set[Tuple[Union[str, int], ...]] = set()
        for path, _ in flatten_with_path(env.observation_space):
            if any(path[:len(e)] == e for e in self.nested_filter_keys):
                self.path_filtered_leaves.add(path)

        # Make sure that some keys are preserved
        if not self.path_filtered_leaves:
            raise ValueError(
                "At least one observation leaf must be preserved.")

        # Initialize base class
        super().__init__(env)

        # Bind observation of the environment for all filtered keys.
        # Note that all parent containers of the filtered leaves must be
        # constructible from standard `dict` or `tuple` objects, which is the
        # case for all standard `gym.Space`.
        self.observation = _copy_filtered(
            self.env.observation, self.path_filtered_leaves)

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.

        It gathers a subset of all the leaves of the original observation space
        without any further processing.
        """
        self.observation_space = _copy_filtered(
            self.env.observation_space, self.path_filtered_leaves)

    def transform_observation(self) -> None:
        """No-op transform since the transform observation is sharing memory
        with the wrapped one since it is just a partial view.
        """
