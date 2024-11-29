"""This module implements two block transformations for applying scaling
factors to subtrees of the observation and action spaces of the environment.
"""
from copy import deepcopy
from collections import OrderedDict
from typing import Union, Tuple, List, Sequence, Generic, Optional, TypeAlias

import numpy as np
import numba as nb

import gymnasium as gym

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    array_copyto, multi_array_copyto)
from jiminy_py.tree import flatten_with_path, unflatten_as

from ..bases import (Obs,
                     Act,
                     InterfaceJiminyEnv,
                     BaseTransformObservation,
                     BaseTransformAction)
from ..utils import DataNested, build_reduce

from .observation_layout import NestedKey, NestedData, ArrayBlockSpec


ScaledObs: TypeAlias = Obs
ScaledAct: TypeAlias = Act


@nb.jit(nopython=True, cache=True)
def _array_scale(scale: float, dst: np.ndarray, src: np.ndarray) -> None:
    """Apply a scalar scaling factor to given array out-of-place.

    :param dst: Pre-allocated array into which the result must be stored.
    :param src: Input array.
    :param scale: Scaling factor
    """
    np.multiply(src, scale, dst)  # Faster than `dst[:] = scale * src`


def _split_nested_key_and_slice(
        nested_scale: Sequence[Tuple[NestedData, float]]) -> Tuple[Tuple[
            NestedKey, Optional[Tuple[Union[int, slice], ...]], float], ...]:
    """Split apart nested keys from array view spec, then convert the latter in
    actually sequence of slices if any.

    :param nested_scale: Sequence of tuple (nested_data, scale) where
                         'nested_data' is itself sequence of nested keys, with
                         eventually an array view spec at the end. if an array
                         view is specified, then the nested key must map to a
                         leaf of the corresponding nested space. The array view
                         spec is a sequence of int, empty tuple, or pair of
                         optional int that fully specified independent slices
                         to extract for each dimension of the array associated
                         with the leaf that the nested key is mapping to.
    """
    nested_key_slices_scale_list: List[
        Tuple[NestedKey, Optional[Tuple[Union[int, slice], ...]], float]
        ] = []
    for nested_spec, scale in nested_scale:
        nested_keys: NestedKey
        block_spec: Optional[ArrayBlockSpec] = None
        if nested_spec and isinstance(nested_spec[-1], (tuple, list)):
            # Split nested keys from block specification if any
            nested_keys = tuple(nested_spec[:-1])
            block_spec = nested_spec[-1]

            # Convert array block specification to slices
            slices: List[Union[int, slice]] = []
            for start_end in block_spec:
                if isinstance(start_end, int):
                    slices.append(start_end)
                elif not start_end:
                    slices.append(slice(None,))
                else:
                    slices.append(slice(*start_end))
            slices = tuple(slices)
        else:
            nested_keys = tuple(nested_spec)  # type: ignore[arg-type]
            slices = None
        nested_key_slices_scale_list.append(
            (nested_keys, slices, scale))
    return tuple(nested_key_slices_scale_list)


def _get_rescale_space(
        space: gym.Space[DataNested],
        nested_key_slices_scale_list: Tuple[Tuple[
            NestedKey, Optional[Tuple[Union[int, slice], ...]], float], ...],
        *, is_reversed: bool) -> gym.Space[DataNested]:
    """Apply a sequence of scalar scaling factors on subtrees or leaves of a
    given nested space out-of-place.

    .. warning::
        All leaf space of the space being altered must be `gym.spaces.Box`
        instance with floating point dtype.

    :param space: Space on which to operate.
    :param nested_key_slices_scale_list:
        Sequence of tuple (nested_key, slices, scale) where 'nested_key' is
        itself sequence of nested keys mapping to a node on which to apply a
        scaling factor, 'slices' is an optional sequence of slices that can be
        specified to only operate on a block of a leaf, and 'scale' is the
        value of the scaling factor.
    """
    # Deepcopy the base observation space
    space = deepcopy(space)

    # Apply scaling on the bounds of the leaf spaces
    space_flat: List[gym.Space] = []
    for path, subspace in flatten_with_path(space):
        for nested_key, slices, scale in nested_key_slices_scale_list:
            if is_reversed:
                scale = 1.0 / scale
            if path[:len(nested_key)] == nested_key:
                # Make sure that the space is supported
                if (not isinstance(subspace, gym.spaces.Box) or
                        subspace.dtype is None or
                        not issubclass(subspace.dtype.type, np.floating)):
                    raise RuntimeError(
                        "Rescaled leaf spaces of base observation space "
                        "must be `gym.space.Box` with floating dtype.")

                # Rescale bounds
                low, high = subspace.low, subspace.high
                if slices is None:
                    low *= scale
                    high *= scale
                else:
                    low[slices] *= scale
                    high[slices] *= scale

                # Instantiate rescaled space
                subspace = gym.spaces.Box(low=low,
                                          high=high,
                                          shape=subspace.shape,
                                          dtype=subspace.dtype.type)
        space_flat.append(subspace)
    return unflatten_as(space, space_flat)


class ScaleObservation(BaseTransformObservation[ScaledObs, Obs, Act],
                       Generic[Obs, Act]):
    """Apply (inverse) scaling factors on subtrees or leaves of the observation
    space of a given pipeline environment.

    .. warning::
        All leaf space of the observation space on which a scaling factor is
        applied must be `gym.spaces.Box` instance with floating point dtype.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[Obs, Act],
                 nested_scale: Sequence[Tuple[NestedData, float]]) -> None:
        """
        :param env: Base or already wrapped jiminy environment.
        :param nested_scale: Sequence of tuple (nested_data, scale) where
                             'nested_data' is itself sequence of nested keys,
                             with eventually an array view spec at the end. if
                             an array view is specified, then the nested key
                             must map to a leaf of the corresponding nested
                             space. The array view spec is a sequence of int,
                             empty tuple, or pair of optional int that fully
                             specified independent slices to extract for each
                             dimension of the array associated with the leaf
                             that the nested key is mapping to.
        """
        # Backup user arguments
        self.nested_scale = nested_scale

        # Define all subtrees that must be altered
        self._nested_key_slices_scale_list = (
            _split_nested_key_and_slice(self.nested_scale))

        # Make sure that all nested keys are valid
        base_observation_path_flat = flatten_with_path(env.observation)
        for nested_key, _, _ in self._nested_key_slices_scale_list:
            for path, _ in base_observation_path_flat:
                if path[:len(nested_key)] == nested_key:
                    break
            else:
                raise ValueError(f"Nested key {nested_key} not found in base "
                                 "observation space.")

        # Initialize base class
        super().__init__(env)

        # Build observation binding unaltered leaves to the original data while
        # copying to others. Besides, generate a flattened list of scaling ops
        # to apply sequentially at run-time.
        copy_ops: List[Tuple[np.ndarray, np.ndarray]] = []
        scale_ops_dict: OrderedDict[
            int, Tuple[np.ndarray, np.ndarray, float]] = OrderedDict()
        observation_flat: List[np.ndarray] = []
        for path, src in base_observation_path_flat:
            dst, is_copy, is_scale_full = src, False, False
            for nested_key, slices, scale in (
                    self._nested_key_slices_scale_list):
                if path[:len(nested_key)] == nested_key:
                    if not is_copy:
                        dst = src.copy()
                        is_copy = True
                    if slices is None:
                        # Factorize full scaling to make sure it only happens
                        # once, which incidentally improves efficiency.
                        dst_id = id(dst)
                        scale_op = scale_ops_dict.get(dst_id)
                        if scale_op is not None:
                            scale /= scale_op[2]
                        scale_ops_dict[dst_id] = (dst, src, 1.0 / scale)
                        # Must move first to make sure full scaling always
                        # happens before chunk scaling if any.
                        scale_ops_dict.move_to_end(dst_id, last=False)
                        is_scale_full = True
                    else:
                        scale_op = (dst[slices], dst[slices], 1.0 / scale)
                        scale_ops_dict[id(scale_op[0])] = scale_op
            if is_copy and not is_scale_full:
                copy_ops.append((dst, src))
            observation_flat.append(dst)
        if copy_ops:
            self._copyto_dst, self._copyto_src = map(tuple, zip(*copy_ops))
        else:
            self._copyto_dst, self._copyto_src = (), ()
        self._scale_ops = tuple(scale_ops_dict.values())
        self.observation = unflatten_as(self.env.observation, observation_flat)

    def _initialize_observation_space(self) -> None:
        self.observation_space = _get_rescale_space(
            self.env.observation_space,
            self._nested_key_slices_scale_list,
            is_reversed=True)

    def transform_observation(self) -> None:
        # First, copy the value of some leaves.
        # Guarding function call is faster than calling no-op python bindings.
        if not self._copyto_dst:
            multi_array_copyto(self._copyto_dst, self._copyto_src)

        # Apply scaling factor sequentially on some chunks
        for dst, src, scale in self._scale_ops:
            _array_scale(scale, dst, src)


class ScaleAction(BaseTransformAction[ScaledAct, Obs, Act],
                  Generic[Obs, Act]):
    """Apply scaling factors on subtrees or leaves of the action space of a
    given pipeline environment.

    .. warning::
        All leaf space of the action space on which a scaling factor is applied
        must be `gym.spaces.Box` instance with floating point dtype.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[Obs, Act],
                 nested_scale: Sequence[Tuple[NestedData, float]]) -> None:
        """
        :param env: Base or already wrapped jiminy environment.
        :param nested_scale: Sequence of tuple (nested_data, scale) where
                             'nested_data' is itself sequence of nested keys,
                             with eventually an array view spec at the end. if
                             an array view is specified, then the nested key
                             must map to a leaf of the corresponding nested
                             space. The array view spec is a sequence of int,
                             empty tuple, or pair of optional int that fully
                             specified independent slices to extract for each
                             dimension of the array associated with the leaf
                             that the nested key is mapping to.
        """
        # Backup user arguments
        self.nested_scale = nested_scale

        # Define all subtrees that must be altered
        self._nested_key_slices_scale_list = (
            _split_nested_key_and_slice(self.nested_scale))

        # Make sure that all nested keys are valid
        base_action_path_flat = flatten_with_path(env.action)
        for nested_key, _, _ in self._nested_key_slices_scale_list:
            for path, _ in base_action_path_flat:
                if path[:len(nested_key)] == nested_key:
                    break
            else:
                raise ValueError(
                    f"Nested key {nested_key} not found in base action space.")

        # Initialize base class
        super().__init__(env)

        # Keeping track of the factorized scaling operations per leaf
        slices_scale_list_flat: List[Tuple[
            Tuple[Optional[Tuple[Union[int, slice], ...]], float], ...]] = []
        for path, _ in base_action_path_flat:
            slices_scale_dict: OrderedDict[
                Optional[Tuple[Union[int, slice], ...]], float] = OrderedDict()
            for nested_key, slices, scale in (
                    self._nested_key_slices_scale_list):
                if path[:len(nested_key)] == nested_key:
                    scale *= slices_scale_dict.get(slices, 1.0)
                    slices_scale_dict[slices] = scale
                    if not slices:
                        slices_scale_dict.move_to_end(slices, last=False)
            slices_scale_list_flat.append(tuple(slices_scale_dict.items()))
        nested_slices_scale_list = unflatten_as(
            self.action, slices_scale_list_flat)

        # Define specialized array scaling operator for efficiency
        def _array_scale_chunks(
                slices_scale_list: Tuple[Tuple[
                    Optional[Tuple[Union[int, slice], ...]], float], ...],
                dst: np.ndarray,
                src: np.ndarray) -> None:
            """Apply a series of scalar scaling factors on blocks of a given
            array out-of-place.

            It is assumed that all slices are unique, and the first one is the
            "empty" slices (full array) if any.

            :param dst: Pre-allocated array into which the result must be
                        stored.
            :param src: Input array.
            :param slices_scale_list: Sequence of tuple (slices, scale) where
                                      'slices' is a group of slices to extract
                                      block views from 'dst' and 'src' arrays,
                                      while 'scale' is the scalar scaling
                                      factor to apply on this specific view.
            """
            # Extract first slice if any
            slices = None
            if slices_scale_list:
                slices, _ = slices_scale_list[0]

            # Must first copy src to dst if no scaling factor to apply or only
            # on chunks.
            if not slices_scale_list or slices is not None:
                array_copyto(dst, src)
                src = dst
                if not slices_scale_list:
                    return

            # Apply scaling factor on chunks of full array depending on slices
            for slices, scale in slices_scale_list:
                if slices is None:
                    _array_scale(scale, dst, src)
                    src = dst
                else:
                    _array_scale(scale, dst[slices], src[slices])

        self._scale_action_inv = build_reduce(
            fn=_array_scale_chunks,
            op=None,
            dataset=(nested_slices_scale_list, self.action),
            space=self.action_space,
            arity=1,
            forward_bounds=False)

    def _initialize_action_space(self) -> None:
        self.action_space = _get_rescale_space(
            self.env.action_space,
            self._nested_key_slices_scale_list,
            is_reversed=False)

    def transform_action(self, action: ScaledAct) -> None:
        """Update in-place pre-allocated transformed action buffer with
        the normalized action of the wrapped environment.
        """
        self._scale_action_inv(action)
