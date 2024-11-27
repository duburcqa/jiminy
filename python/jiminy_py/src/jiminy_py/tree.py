"""Generic helpers for manipulating arbitrarily nested data structure with
easy. All these functions are reasonably fast, but they aim at convenience
of use and versatility rather with optimal performance.

.. note::
    This module mirrors the API of the `dm-tree` package that was originally
    developed by Google Deepmind but is not longer actively maintained.
    Internally, the implementation is completely different.

.. warning::
    A node is considered as a leaf unless its type derives from either
    `collection.abc.Mapping`, `collections.abc.Sequence` or
    `collections.abc.Set`.

.. warning::
    Unlike `dm-tree`, all functions preserves key ordering instead of sorting
    them.
"""
from functools import lru_cache
from itertools import chain, starmap
from collections.abc import (Mapping, ValuesView, Sequence, Set)
from typing import (
    Any, Union, Mapping as MappingT, Sequence as SequenceT, Iterable,
    Iterator as Iterator, Tuple, TypeVar, Callable, Type)


ValueT = TypeVar('ValueT')
StructNested = Union[MappingT[str, 'StructNested[ValueT]'],
                     SequenceT['StructNested[ValueT]'],
                     ValueT]


@lru_cache(maxsize=None)
def issubclass_mapping(cls: Type[Any]) -> bool:
    """Determine whether a given class is a mapping, ie its derives from
    'collections.abc.Mapping'.

    `issubclass` is very slow when checking against for abstract collection
    types instead of specific types, because it needs to go through all the
    methods of the abstract interface and make sure it has been implemented.
    This specialization of `issubclass` builtin function leveraging LRU cache
    for speed-up, ensuring constant query time whatever the object.

    :param cls: candidate class.
    """
    return issubclass(cls, Mapping)


@lru_cache(maxsize=None)
def issubclass_sequence(cls: Type[Any]) -> bool:
    """Determine whether a given class is a sequence, ie its derives from
    'itertools.chain', 'collections.abc.Sequence', 'collections.abc.Set', or
    'collections.abc.ValuesView' but not from 'str'.

    Specialization of `issubclass` builtin function leveraging LRU cache for
    speed-up. See `issubclass_mapping` for details.

    :param cls: candidate class.
    """
    return issubclass(cls, (
        chain, Sequence, Set, ValuesView)) and not issubclass(cls, str)


def _flatten_with_path_up_to(
        ref: Any,
        data: Any,
        path: Tuple[Union[str, int], ...]
        ) -> Union[chain, Iterable[Tuple[Tuple[Union[str, int], ...], Any]]]:
    """Internal method flattening a given nested data structure by calling
    itself recursively on each top-level nodes that are not leaves.
    """
    ref_type = type(ref)
    if issubclass_mapping(ref_type):  # type: ignore[arg-type]
        ref_keys, ref_values = zip(*ref.items())
        sub_paths = ((*path, key) for key in ref_keys)
        nodes = zip(ref_values, data.values(), sub_paths)
    elif issubclass_sequence(ref_type):  # type: ignore[arg-type]
        sub_paths = ((*path, i) for i in range(len(ref)))
        nodes = zip(ref, data, sub_paths)
    else:
        return ((path, data),)
    return chain.from_iterable(starmap(_flatten_with_path_up_to, nodes))


def flatten_with_path_up_to(
        data_shallow: StructNested[Any],
        data_nested: StructNested[Any]
        ) -> Tuple[Tuple[Tuple[Union[str, int], ...], Any], ...]:
    """Flatten a given nested data structure as if it was an instance of some
    other, shallower, nested data structure sharing the same top-level layout,
    while keeping track of the original path of each value.

    :param data_shallow: Nested data structure having the same layout than
                         'data_shallow' up to same depth level.
    :param data_nested: Possibly nested data structure.

    :returns: tuple of pairs (value, path) associated with all nodes of the
    partially flattened representation of the provided nested data structure.
    """
    return tuple(_flatten_with_path_up_to(data_shallow, data_nested, ()))


def flatten_with_path(data_nested: StructNested[Any]
                      ) -> Tuple[Tuple[Tuple[Union[str, int], ...], Any], ...]:
    """Flatten a possibly nested data structure into a sequence of leaf nodes,
    while keeping track of their original path.

    :param data_nested: Possibly nested data structure.

    :returns: tuple of pairs (path, value) associated with all leaves of the
    original nested data structure.
    """
    return flatten_with_path_up_to(data_nested, data_nested)


def _flatten_up_to(ref: Any, data: Any) -> Union[chain, Iterable]:
    """Internal method flattening a given nested data structure by calling
    itself recursively on each top-level nodes that are not leaves.
    """
    ref_type = type(ref)
    if issubclass_mapping(ref_type):  # type: ignore[arg-type]
        nodes = zip(ref.values(), data.values())
    elif issubclass_sequence(ref_type):  # type: ignore[arg-type]
        nodes = zip(ref, data)
    else:
        return (data,)
    return chain.from_iterable(starmap(_flatten_up_to, nodes))


def flatten_up_to(data_shallow: Any, data_nested: Any) -> Tuple[Any, ...]:
    """Flatten a given nested data structure as if it was an instance of some
    other, shallower, nested data structure sharing the same top-level layout.

    :param data_shallow: Nested data structure having the same layout than
                         'data_shallow' up to same depth level.
    :param data_nested: Possibly nested data structure.

    :returns: partially flattened representation of the provided nested data
    structure as a tuple.
    """
    # Specialized implementation for speed-up
    return tuple(_flatten_up_to(data_shallow, data_nested))


def _flatten(data: Any) -> Union[chain, Iterable]:
    """Internal method flattening a given nested data structure by calling
    itself recursively on each top-level nodes that are not leaves.
    """
    # Specialized implementation for speed-up
    data_type = type(data)
    if issubclass_mapping(data_type):  # type: ignore[arg-type]
        nodes = data.values()
    elif issubclass_sequence(data_type):  # type: ignore[arg-type]
        nodes = data
    else:
        return (data,)
    return chain.from_iterable(map(_flatten, nodes))


def flatten(data_nested: Any) -> Tuple[Any, ...]:
    """Flatten a possibly nested data structure into a sequence of leaf nodes.

    :param data_nested: Possibly nested data structure.

    :returns: Tuple containing all the leaves of the provided nested data
    structure.
    """
    return tuple(_flatten(data_nested))


def _unflatten_as(data: StructNested[Any],
                  data_leaf_it: Iterator) -> StructNested[Any]:
    """Internal method un-flattening a sequence into a nested data structure by
    calling itself recursively on each top-level nodes that are not leaves and
    fetching the next leaf value otherwise.
    """
    data_type = type(data)
    if issubclass_mapping(data_type):  # type: ignore[arg-type]
        flat_items = [
            (key, _unflatten_as(value, data_leaf_it))
            for key, value in data.items()]  # type: ignore[union-attr]
        try:
            # Initialisation from dict cannot be the default path as
            # `gym.spaces.Dict` would sort keys in this specific scenario,
            # which must be avoided.
            return data_type(flat_items)  # type: ignore[call-arg]
        except (ValueError, RuntimeError):
            # Fallback to initialisation from dict in the rare event of
            # a container type not supporting initialisation from a
            # sequence of key-value pairs.
            return data_type(dict(flat_items))  # type: ignore[call-arg]
    if issubclass_sequence(data_type):  # type: ignore[arg-type]
        return data_type([  # type: ignore[call-arg]
            _unflatten_as(value, data_leaf_it) for value in data])
    return next(data_leaf_it)


def unflatten_as(data_nested: StructNested[Any],
                 data_leaves: Union[Sequence, Iterable, ValuesView]
                 ) -> StructNested[Any]:
    """Un-flatten a given sequence of leaf nodes according to some reference
    nested data structure.

    .. warning::
        All nodes must be constructible from standard `dict` or `tuple`
        objects, depending on whether they derive from `collection.abc.Mapping`
        or `(collections.abc.Sequence, collections.abc.Set)` respectively.

    :param data_nested: Possibly nested data structure.
    :param data_leaves: Flat sequence of leaves to un-flatten.

    :returns: Nested data structure sharing the same layout of the reference.
    """
    return _unflatten_as(data_nested, iter(data_leaves))


def map_structure(fn: Callable, *data_nested: StructNested[Any]
                  ) -> StructNested[Any]:
    """Jointly apply a given transform on all leaves of a set of possibly
    nested data structure.

    .. warning::
        The user is responsible for make sure that the layouts of the provided
        nested data structures are consistent together. It is not asserted
        in this method for the sake of efficiency.

    :param fn: Transform function to be applied on every leaves of the nested
               data structures. This function can perform any out-of-place
               operations without further restriction.
    :param data_nested: Set of possible nested data structures all having the
                        same layout.

    :returns:  New nested data structure having the same layout than the
    original ones for which each leaf are obtained by applying transform
    to all their original counterpart at once.
    """
    if not data_nested:
        return ()
    return unflatten_as(
        data_nested[0], starmap(fn, zip(*map(flatten, data_nested))))
