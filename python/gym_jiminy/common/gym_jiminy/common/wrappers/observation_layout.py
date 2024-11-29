# mypy: disable-error-code="arg-type, index, assignment, union-attr"
"""This module implements a block transformation for adapt the layout of the
observation space of an environment that may be arbitrarily nested.
"""
from operator import getitem
from functools import reduce
from collections import OrderedDict
from typing import (
    Sequence, Tuple, List, Union, Generic, TypeVar, Type, Optional, Dict, cast,
    overload)
from typing_extensions import Unpack

import numpy as np
import gymnasium as gym
from jiminy_py.tree import (
    flatten_with_path, issubclass_mapping, issubclass_sequence)

from ..bases import (Act,
                     NestedObs,
                     InterfaceJiminyEnv,
                     BaseTransformObservation)
from ..utils import DataNested, copy


MaybeNestedObs = TypeVar('MaybeNestedObs', bound=DataNested)
OtherMaybeNestedObs = TypeVar('OtherMaybeNestedObs', bound=DataNested)

Key = Union[str, int]
NestedKey = Tuple[Key, ...]
Slice = Union[Tuple[()], int, Tuple[Optional[int], Optional[int]]]
ArrayBlockSpec = Sequence[Slice]
NestedData = Union[NestedKey, Tuple[Unpack[NestedKey], ArrayBlockSpec]]


@overload
def _adapt_layout(data: DataNested,
                  layout: Sequence[Tuple[NestedKey, NestedData]]
                  ) -> DataNested:
    ...


@overload
def _adapt_layout(data: gym.Space[DataNested],
                  layout: Sequence[Tuple[NestedKey, NestedData]]
                  ) -> gym.Space[DataNested]:
    ...


def _adapt_layout(
        data: Union[DataNested, gym.Space[DataNested]],
        layout: Sequence[Tuple[NestedKey, NestedData]]
        ) -> Union[DataNested, gym.Space[DataNested]]:
    """Extract subtrees and leaves from a given input nested data structure and
    rearrange them an arbitrary output nested data structure according to some
    prescribed layout.

    All the leaves are still sharing memomy with their original counterpart as
    they are references to the same objects. However, all the containers are
    completely independent, even when extracting a complete subtree.

    .. note::
        This method can be used to filter out subtrees of a given nested data
        structure by partially shallow copying it.

    .. warning::
        All the output containers are freshly created rather than copies. This
        means that they must be constructible from plain-old python containers
        (dict, list, tuple, ...) without any extra arguments. Any custom
        attributes dynamically added would be lost if any. This is typically
        the case for all standard `gym.Space`.

    :param data: Hierarchical data structure from which to extract data without
                 memory allocation for leaves.
    :param layout: Sequence of tuples `(nested_key_out, nested_key_in)` mapping
                   the desired path in the output data structure to the
                   original path in input data structure. These tuples are
                   guaranteed to be processed in order. The same path may
                   appear multiple time in the output data structure. If so,
                   the corresponding subtrees while be aggregated in sequence.
                   If the parent node of the first subtree being considered is
                   already a sequence, then subsequent extracted subtrees will
                   be appended to it directly. If not, then a dedicated
                   top-level sequence container while be created first.
    """
    # We need to guarantee that all containers of the output data structure are
    # mutable while building it, and only at the end to convert them back to
    # their respective desired type.
    # Since the output data structure is not known in advance, the only option
    # is to keep track of the desired container type while building the output
    # data structure.

    # Determine if data is a gym.Space or a "classical" nested data structure
    is_space = isinstance(data, gym.Space)

    # Shallow copy the input data structure
    data = copy(data)

    # Convert all parent containers to their mutable counterpart
    container_types_in: Dict[NestedKey, Type] = {}
    for nested_key, _ in flatten_with_path(data):
        for i in range(1, len(nested_key) + 1):
            # Extract parent container
            nested_key_parent = nested_key[:(i - 1)]
            if nested_key_parent:
                *nested_key_container, key_parent = nested_key_parent
                container = reduce(getitem, nested_key_container, data)
                parent = container[key_parent]
            else:
                parent = data

            # Convert parent container to mutable dictionary
            parent_type = type(parent)
            if nested_key_parent not in container_types_in:
                container_types_in[nested_key_parent] = parent_type
            if parent_type in (list, dict, OrderedDict):
                continue
            if issubclass_mapping(parent_type):
                parent = OrderedDict(parent.items())
            elif issubclass_sequence(parent_type):
                parent = list(parent)
            else:
                raise NotImplementedError(
                    f"Unsupported container type: '{parent_type}'")

            # Re-assign parent data structure
            if nested_key_parent:
                container[key_parent] = parent
            else:
                data = parent

    # Build the output data structure sequentially
    container_types_out: Dict[NestedKey, Type] = {}
    out: Optional[DataNested] = None
    for nested_key_out, nested_spec_in in layout:
        # Make sure that requested nested data is a subtree
        if not nested_spec_in:
            raise ValueError("Input nested keys must not be empty.")

        # Split nested keys from block specification if any
        block_spec_in: Optional[ArrayBlockSpec] = None
        if isinstance(nested_spec_in[-1], (tuple, list)):
            nested_key_in = nested_spec_in[:-1]
            block_spec_in = nested_spec_in[-1]
        else:
            nested_key_in = cast(NestedKey, nested_spec_in)

        # Extract the input chunk recursively
        value_in = reduce(getitem, nested_key_in, data)
        if block_spec_in is not None:
            # Convert array block specification to slices
            slices = []
            for start_end in block_spec_in:
                if isinstance(start_end, int):
                    slices.append(start_end)
                elif not start_end:
                    slices.append(slice(None,))
                else:
                    slices.append(slice(*start_end))
            slices = tuple(slices)

            # Extract sub-space or array view depending on the input
            if is_space:
                assert isinstance(value_in, gym.spaces.Box)
                assert isinstance(value_in.dtype, np.dtype) and issubclass(
                    value_in.dtype.type, (np.floating, np.integer))
                low, high = value_in.low[slices], value_in.high[slices]
                value_in = gym.spaces.Box(low=low,
                                          high=high,
                                          shape=low.shape,
                                          dtype=value_in.dtype.type)
            else:
                assert isinstance(value_in, np.ndarray)
                value_in = value_in[slices]

        # Deal with the special case where the output nested key is empty
        if not nested_key_out:
            if out is None:
                # Promote the extracted value as output
                out = value_in
            else:
                # Encapsulate the output in sequence container if it was not
                # the case originally. The whole hierarchy of container types
                # must be shifted one level deeper, while the new top-level
                # container type is the default tuple.
                if not isinstance(out, list):
                    out = [out,]
                    for path, type_ in tuple(container_types_out.items()):
                        container_types_out[(0, *path)] = type_
                        del container_types_out[path]
                    if isinstance(data, gym.Space):
                        container_types_out[()] = gym.spaces.Tuple
                    else:
                        container_types_out[()] = tuple

                # Update out nested key to account for index in sequence
                nested_key_out = (len(out),)
                out.append(value_in)

        # Add extracted input chunks to output
        value_out = out
        depth = len(nested_key_out)
        for i in range(depth):
            # Extract key and subkey
            key = nested_key_out[i]
            subkey = nested_key_out[i + 1] if i + 1 < depth else None

            if isinstance(key, str):
                # Initialize the out container is not done before
                if out is None:
                    value_out = out = {}
                    if is_space:
                        container_types_out[()] = gym.spaces.Dict
                    else:
                        container_types_out[()] = dict
                assert isinstance(value_out, dict)

                # Create new branch if not the final key in path, otherwise
                # add extracted value as leaf. Then, extract child node.
                if key not in value_out or subkey is not None:
                    if key not in value_out:
                        if subkey is None:
                            value_out[key] = value_in
                        else:
                            value_out[key] = {}
                            path = nested_key_out[:(i + 1)]
                            if is_space:
                                container_types_out[path] = gym.spaces.Dict
                            else:
                                container_types_out[path] = dict
                    value_out = value_out[key]
                    continue
            elif isinstance(key, int):
                # Initialize the out container is not done before
                if out is None:
                    value_out = out = []
                    if is_space:
                        container_types_out[()] = gym.spaces.Tuple
                    else:
                        container_types_out[()] = tuple
                assert isinstance(value_out, list)

                # Just extract child node if not the final key in path
                if subkey is not None:
                    value_out = value_out[key]
                    continue

            # The final node expected to be sequence container. Encapsulate the
            # output in sequence container if not already the case, while
            # adapting the hierarchy of output container types accordingly.
            # Then append the extract value.
            assert value_out is not None
            if not isinstance(value_out[key], list):
                value_out[key] = [value_out[key],]
                for path, type_ in tuple(container_types_out.items()):
                    root_path, child_path = path[:depth], path[depth:]
                    if root_path == nested_key_out:
                        path_ = (*nested_key_out, 0, *child_path)
                        container_types_out[path_] = type_
                        del container_types_out[path]
                if is_space:
                    container_types_out[nested_key_out] = gym.spaces.Tuple
                else:
                    container_types_out[nested_key_out] = tuple
            nested_key_out = (*nested_key_out, len(value_out[key]))
            value_out[key].append(value_in)

        # Extract copied out container types
        for path, type_ in container_types_in.items():
            root_path = path[:len(nested_key_in)]
            child_path = path[len(nested_key_in):]
            if root_path == nested_key_in:
                container_types_out[(*nested_key_out, *child_path)] = type_

    # Restore original parent container types
    path_all, _ = zip(*flatten_with_path(out))
    depth = max(map(len, path_all))
    for i in range(depth)[::-1]:
        for nested_key in path_all:
            # Skip if the node is not a container
            if len(nested_key) <= i:
                continue

            # Extract parent container
            nested_key_parent = nested_key[:i]
            if nested_key_parent:
                *nested_key_container, key_parent = nested_key_parent
                container = reduce(getitem, nested_key_container, out)
                parent = container[key_parent]
            else:
                parent = out

            # Restore original container type if not already done
            parent_type = container_types_out[nested_key_parent]
            if isinstance(parent, parent_type):
                continue
            if issubclass_mapping(parent_type):
                parent = parent_type(tuple(parent.items()))
            elif issubclass_sequence(parent_type):
                parent = parent_type(tuple(parent))

            # Re-assign output data structure
            if nested_key_parent:
                container[key_parent] = parent
            else:
                out = parent

    assert out is not None
    return out


class AdaptLayoutObservation(
        BaseTransformObservation[OtherMaybeNestedObs, MaybeNestedObs, Act],
        Generic[OtherMaybeNestedObs, MaybeNestedObs, Act]):
    """Adapt the data structure of the original nested observation space,
    by filtering out some leaves and/or re-ordering others.

    This wrapper does nothing but exposing a subset of all the leaves of the
    original observation space with a completely independent data structure.
    For flattening the observation space after filtering, you should wrap the
    environment with `FlattenObservation` as yet another layer.

    It is possible to operate on subtrees directly without going all the way
    down each leaf. Similarly, one can extract slices (possibly not contiguous)
    of `gym.spaces.Box` spaces. Extra leaves can be added to subtrees of
    original data that has already been re-mapped, knowing that the items of
    the layout are always processed in order. Moreover, the same output key can
    appear multiple times. In such case, all the associated values will be
    stacked as a tuple while preserving their order.

    Let us consider the following nested observation space:

        gym.spaces.Dict(
            x1=gym.spaces.Tuple([
                gym.spaces.Box(float('-inf'), float('inf'), (7, 6, 5, 4)),
            ]),
            x2=gym.spaces.Dict(
                y1=gym.spaces.Dict(
                    z1=gym.spaces.Discrete(5)
                ),
                y2=gym.spaces.Tuple([
                    gym.spaces.Discrete(2),
                    gym.spaces.Box(float('-inf'), float('inf'), (2, 3)),
                    gym.spaces.Discrete(3),
                ])))

    Here is an example that aggregates one leaf to a leaf of a subtree that has
    already been remapped:

        [((), ("x2", "y2")), ((0,), ("x1",))]

        gym.spaces.Tuple([
            gym.spaces.Tuple([
                gym.spaces.Discrete(2),
                gym.spaces.Tuple([
                    gym.spaces.Box(float('-inf'), float('inf'), (7, 6, 5, 4)),
                ])
            ]),
            gym.spaces.Box(float('-inf'), float('inf'), (2, 3)),
            gym.spaces.Discrete(3),
        ]))

    Here is an example that aggregate one leaf and one multi-dimensional slice
    of a Box space (array[:, 1:3]) under two separate keys of a nested dict:

        [(("A", "B1"), ("x2", "y2", 1)),
         (("A", "B2"), ("x1", 0, [(), (1, 4), (1, 3)]))]

        gym.spaces.Dict(
            A=gym.spaces.Dict(
                B1=gym.spaces.Box(float('-inf'), float('inf'), (2, 3)),
                B2=gym.spaces.Box(float('-inf'), float('inf'), (7, 3, 2, 4))
            ))

    Here is an example that aggregate two subtree in the same nested structure:

        [((), ("x2",)), (("y1", "z2"), ("x1",))]

        gym.spaces.Dict(
            y1=gym.spaces.Dict(
                z1=gym.spaces.Discrete(5)
                z2=gym.spaces.Tuple([
                    gym.spaces.Box(float('-inf'), float('inf'), (7, 6, 5, 4)),
                ])
            ),
            y2=gym.spaces.Tuple([
                gym.spaces.Discrete(2),
                gym.spaces.Box(float('-inf'), float('inf'), (2, 3)),
                gym.spaces.Discrete(3),
            ]))
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[MaybeNestedObs, Act],
                 layout: Sequence[Tuple[
                    Union[NestedKey, Key], NestedData]]) -> None:
        """
        :param env: Base or already wrapped jiminy environment.
        :param layout: Sequence of tuples `(nested_key_out, nested_key_in)`
                       mapping the desired path in the output data structure to
                       the original path in input data structure. These tuples
                       are guaranteed to be processed in order. The same path
                       may appear multiple time in the output data structure.
                       If so, the corresponding subtrees while be aggregated in
                       sequence. If the parent node of the first subtree being
                       considered is already a sequence, then all subsequent
                       extracted subtrees will be appended to it directly. If
                       not, then a dedicated top-level sequence container while
                       be created first.
        """
        # Make sure that some keys are preserved
        if not layout:
            raise ValueError(
                "The resulting observation space must not be empty.")

        # Backup user-specified layout while making sure all nested keys are
        # stored in sequence.
        self.layout: Sequence[Tuple[NestedKey, NestedData]] = []
        for nested_key_out, nested_spec_in in layout:
            if isinstance(nested_key_out, (str, int)):
                nested_key_out = (nested_key_out,)
            self.layout.append((nested_key_out, nested_spec_in))

        # Initialize base class
        super().__init__(env)

        # Bind observation of the environment for all extracted keys
        self.observation = _adapt_layout(self.env.observation, self.layout)

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.

        It gathers a subset of all the leaves of the original observation space
        without any further processing.
        """
        self.observation_space = _adapt_layout(
            self.env.observation_space, self.layout)

    def transform_observation(self) -> None:
        """No-op transform since the transform observation is sharing memory
        with the wrapped one since it is just a partial view.
        """


class FilterObservation(
        BaseTransformObservation[NestedObs, NestedObs, Act],
        Generic[NestedObs, Act]):
    """Filter nested observation space.

    This wrapper does nothing but providing an observation only exposing a
    subset of all the leaves of the original observation space. This wrapper is
    a specialization of `AdaptLayoutObservation`, which is more generic. For
    flattening the observation space after filtering, you should wrap the
    environment with `FlattenObservation` as yet another layer.

    Note that it is possible to operate on subtrees directly without going all
    the way down each leaf.

    Beware that the original ordered of leaves within their parent container is
    maintain, whatever the order in which they appear in the specified list of
    filtered nested keys.

    Let us consider the following nested observation space:

        gym.spaces.Dict(
            x1=gym.spaces.Tuple([
                gym.spaces.Box(float('-inf'), float('inf'), (7, 6, 5, 4)),
            ]),
            x2=gym.spaces.Dict(
                y1=gym.spaces.Dict(
                    z1=gym.spaces.Discrete(5)
                ),
                y2=gym.spaces.Tuple([
                    gym.spaces.Discrete(2),
                    gym.spaces.Box(float('-inf'), float('inf'), (2, 3)),
                    gym.spaces.Discrete(3),
                ])))

    Here is an example that filter out part of a sequence container and a
    mapping container:

        [("x2", "y2", 2), ("x2", "y2", 0), ("x1",)]

        gym.spaces.Dict(
            x1=gym.spaces.Tuple([
                gym.spaces.Box(float('-inf'), float('inf'), (7, 6, 5, 4)),
            ]),
            x2=gym.spaces.Dict(
                y2=gym.spaces.Tuple([
                    gym.spaces.Discrete(2),
                    gym.spaces.Discrete(3),
                ])))
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[NestedObs, Act],
                 nested_filter_keys: Sequence[Union[NestedKey, Key]]) -> None:
        # Make sure that the top-most observation space is a container
        space_cls = type(env.observation_space)
        assert issubclass_mapping(space_cls) or issubclass_sequence(space_cls)

        # Make sure all nested keys are stored in sequence
        assert not isinstance(nested_filter_keys, str)
        self.nested_filter_keys: Sequence[NestedKey] = []
        for nested_key in nested_filter_keys:
            if isinstance(nested_key, (str, int)):
                nested_key = (nested_key,)
            self.nested_filter_keys.append(tuple(nested_key))

        # Get all paths associated with leaf values that must be kept.
        # Re-order filtered leaves to match the original nested data structure.
        path_filtered_leaves: List[NestedKey] = []
        for path, _ in flatten_with_path(env.observation_space):
            if any(path[:len(e)] == e for e in self.nested_filter_keys):
                if path not in path_filtered_leaves:
                    path_filtered_leaves.append(path)

        # Backup the layout mapping
        self._layout: Sequence[Tuple[NestedKey, NestedData]] = []
        for nested_key_in in path_filtered_leaves:
            if isinstance(nested_key_in[-1], int):
                nested_key_out = nested_key_in[:-1]
            else:
                nested_key_out = nested_key_in
            self._layout.append((nested_key_out, nested_key_in))

        # Make sure that some keys are preserved
        if not path_filtered_leaves:
            raise ValueError(
                "At least one observation leaf must be preserved.")

        # Initialize base class
        super().__init__(env)

        # Bind observation of the environment for all filtered keys
        self.observation = _adapt_layout(self.env.observation, self._layout)

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.

        It gathers a subset of all the leaves of the original observation space
        without any further processing.
        """
        self.observation_space = _adapt_layout(
            self.env.observation_space, self._layout)

    def transform_observation(self) -> None:
        """No-op transform since the transform observation is sharing memory
        with the wrapped one since it is just a partial view.
        """
