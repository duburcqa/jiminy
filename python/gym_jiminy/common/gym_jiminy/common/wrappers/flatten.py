""" TODO: Write documentation.
"""
from typing import Generic, Optional
from typing_extensions import TypeAlias

import numpy as np
from numpy import typing as npt

import gymnasium as gym

from ..bases import (ObsT,
                     ActT,
                     JiminyEnvInterface,
                     BaseTransformObservation,
                     BaseTransformAction)
from ..utils import build_reduce, build_flatten


FlattenedObsT: TypeAlias = ObsT
FlattenedActT: TypeAlias = ActT


class FlattenObservation(BaseTransformObservation[FlattenedObsT, ObsT, ActT],
                         Generic[ObsT, ActT]):
    """Flatten the observation space of a pipeline environment. It will appear
    as a simple one-dimension vector.

    .. warning::
        All leaves of the observation space must have type `gym.spaces.Box`.
    """
    def __init__(self,
                 env: JiminyEnvInterface[ObsT, ActT],
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
            dtype = build_reduce(
                lambda value: value.dtype, np.promote_types,
                (env.observation,), None, 0)()

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
        get_leaf_bounds = build_reduce(
            lambda *x: map(np.ravel, x), lambda x, y: x.append(y) or x, (),
            self.env.observation_space, 0, initializer=list)
        low, high = (
            np.concatenate(  # pylint: disable=unexpected-keyword-arg
                leaf_bounds, dtype=self.dtype)
            for leaf_bounds in zip(*get_leaf_bounds()))

        # Initialize the observation space with proper dtype
        self.observation_space = gym.spaces.Box(low, high, dtype=self.dtype)

    def transform_observation(self) -> None:
        """Update in-place pre-allocated transformed observation buffer with
        the flattened observation of the wrapped environment.
        """
        self._flatten_observation()


class FlattenAction(BaseTransformAction[FlattenedActT, ObsT, ActT],
                    Generic[ObsT, ActT]):
    """Flatten the action space of a pipeline environment. It will appear as a
    simple one-dimension vector.

    .. warning::
        All leaves of the action space must have type `gym.spaces.Box`.
    """
    def __init__(self,
                 env: JiminyEnvInterface[ObsT, ActT],
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
            dtype = build_reduce(
                lambda value: value.dtype, np.promote_types,
                (env.observation,), None, 0)()

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
        get_leaf_bounds = build_reduce(
            lambda *x: map(np.ravel, x), lambda x, y: x.append(y) or x, (),
            self.env.action_space, 0, initializer=list)
        low, high = (
            np.concatenate(  # pylint: disable=unexpected-keyword-arg
                leaf_bounds, dtype=self.dtype)
            for leaf_bounds in zip(*get_leaf_bounds()))

        # Initialize the action space with proper dtype
        self.action_space = gym.spaces.Box(low, high, dtype=self.dtype)

        # Initialize the action space with proper dtype
        dtype = low.dtype
        assert dtype is not None and issubclass(dtype.type, np.floating)
        self.action_space = gym.spaces.Box(low, high, dtype=dtype.type)

    def transform_action(self, action: FlattenedActT) -> None:
        """Update in-place the pre-allocated action buffer of the wrapped
        environment with the un-flattened action.
        """
        self._unflatten_to_env_action(action)
