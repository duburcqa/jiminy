"""This module implements a block transformation for flattening observation
space of the environment.
"""
from functools import reduce
from typing import Generic, Optional, Iterable, Tuple, TypeAlias, cast

import numpy as np
from numpy import typing as npt

import gymnasium as gym

from jiminy_py import tree

from ..bases import (Obs,
                     Act,
                     InterfaceJiminyEnv,
                     BaseTransformObservation,
                     BaseTransformAction)
from ..utils import get_bounds, build_flatten


FlattenedObs: TypeAlias = npt.NDArray[np.float64]
FlattenedAct: TypeAlias = npt.NDArray[np.float64]


class FlattenObservation(BaseTransformObservation[FlattenedObs, Obs, Act],
                         Generic[Obs, Act]):
    """Flatten the observation space of a pipeline environment. It will appear
    as a simple one-dimension vector.

    .. warning::
        All leaves of the observation space must have type `gym.spaces.Box`.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[Obs, Act],
                 dtype: Optional[npt.DTypeLike] = None) -> None:
        """
        :param env: Environment to wrap.
        :param dtype: Numpy dtype of the flattened observation. If `None`, the
                      most appropriate dtype to avoid lost of information if
                      possible will be picked, following standard coercion
                      rules. See `np.promote_types` for details.
                      Optional: `None` by default.
        """
        # Find most appropriate dtype if not specified
        if dtype is None:
            if env.observation:
                dtype_all = [
                    value.dtype for value in tree.flatten(env.observation)]
                dtype = reduce(np.promote_types, dtype_all)
            else:
                dtype = np.float64

        # Make sure that `gym.space.Box` support the prescribed dtype
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        assert issubclass(dtype.type, (np.floating, np.integer))
        self.dtype = dtype.type

        # Initialize base class
        super().__init__(env)

        # Define specialized operator(s) for efficiency
        self._flatten_observation = build_flatten(
            self.env.observation, self.observation)

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        # Compute bounds of flattened observation space
        min_max_bounds_leaves = cast(Iterable[Tuple[np.ndarray, np.ndarray]], (
            tuple(map(np.ravel, get_bounds(space)))  # type: ignore[arg-type]
            for space in tree.flatten(self.env.observation_space)))
        low, high = (
            np.concatenate(  # pylint: disable=unexpected-keyword-arg
                bound_leaves, dtype=self.dtype)
            for bound_leaves in zip(*min_max_bounds_leaves))

        # Initialize the observation space with proper dtype
        self.observation_space = gym.spaces.Box(low, high, dtype=self.dtype)

    def transform_observation(self) -> None:
        """Update in-place pre-allocated transformed observation buffer with
        the flattened observation of the wrapped environment.
        """
        self._flatten_observation()


class FlattenAction(BaseTransformAction[FlattenedAct, Obs, Act],
                    Generic[Obs, Act]):
    """Flatten the action space of a pipeline environment. It will appear as a
    simple one-dimension vector.

    .. warning::
        All leaves of the action space must have type `gym.spaces.Box`.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[Obs, Act],
                 dtype: Optional[npt.DTypeLike] = None) -> None:
        """
        :param env: Environment to wrap.
        :param dtype: Numpy dtype of the flattened action. If `None`, the most
                      appropriate dtype to avoid lost of information if
                      possible will be picked, following standard coercion
                      rules. See `np.promote_types` for details.
                      Optional: `None` by default.
        """
        # Find most appropriate dtype if not specified
        if dtype is None:
            action_flat = tree.map_structure(
                lambda value: value.dtype, tree.flatten(env.action))
            if action_flat:
                dtype = reduce(np.promote_types, action_flat)
            else:
                dtype = np.float64

        # Make sure that `gym.space.Box` support the prescribed dtype
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        assert issubclass(dtype.type, (np.floating, np.integer))
        self.dtype = dtype.type

        # Initialize base class
        super().__init__(env)

        # Define specialized operator(s) for efficiency
        self._unflatten_to_env_action = build_flatten(
            self.env.action, is_reversed=True)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        # Compute bounds of flattened action space
        min_max_bounds_leaves = cast(Iterable[Tuple[np.ndarray, np.ndarray]], (
            tuple(map(np.ravel, get_bounds(space)))  # type: ignore[arg-type]
            for space in tree.flatten(self.env.action_space)))
        low, high = (
            np.concatenate(  # pylint: disable=unexpected-keyword-arg
                bound_leaves, dtype=self.dtype)
            for bound_leaves in zip(*min_max_bounds_leaves))

        # Initialize the action space with proper dtype
        self.action_space = gym.spaces.Box(low, high, dtype=self.dtype)

    def transform_action(self, action: FlattenedAct) -> None:
        """Update in-place the pre-allocated action buffer of the wrapped
        environment with the un-flattened action.
        """
        self._unflatten_to_env_action(action)
