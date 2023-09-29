""" TODO: Write documentation.
"""
from typing import Generic
from typing_extensions import TypeAlias

import numpy as np

import gymnasium as gym

from ..bases import (ObsT,
                     ActT,
                     JiminyEnvInterface,
                     BaseTransformObservation,
                     BaseTransformAction)
from ..utils import build_map, build_normalize


NormalizedObsT: TypeAlias = ObsT
NormalizedActT: TypeAlias = ActT


def _normalize_space(space: gym.spaces.Box) -> gym.spaces.Box:
    """Normalize a space instance deriving from `gym.spaces.Box`.

    This method returns a new space instance identical to the original one
    except for the lower and upper bounds that are -1.0, 1.0 respectively.

    .. note::
        The returned space is a new instance created by calling the
        constructor of the original space with the arguments of
        `gym.spaces.Box` regardless of its actual type. Then, any instance
        attributes of  the original space that are not defined at this point
        for the new instance are copied.
    """
    # Instantiate normalized space
    dtype = space.dtype
    assert dtype is not None and issubclass(dtype.type, np.floating)
    space_ = type(space)(-1.0, 1.0, shape=space.shape, dtype=dtype.type)

    # Preserve instance-specific attributes as is, if any
    space_attrs = vars(space_).keys()
    for key, value in vars(space).items():
        if key not in space_attrs:
            setattr(space_, key, value)

    return space_


class NormalizeObservation(
        BaseTransformObservation[NormalizedObsT, ObsT, ActT],
        Generic[ObsT, ActT]):
    """Normalize (without clipping) the observation space of a pipeline
    environment according to its pre-defined bounds rather than statistics over
    collected data. Unbounded elements if any are left unchanged.

    .. warning::
        All leaves of the observation space must have type `gym.spaces.Box`.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Define specialized operator(s) for efficiency
        self._normalize_observation = build_normalize(
            self.env.observation_space, self.observation, self.env.observation)

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        self.observation_space = build_map(
            _normalize_space, self.env.observation_space, None, 0)()

    def transform_observation(self) -> None:
        """Update in-place pre-allocated transformed observation buffer with
        the normalized observation of the wrapped environment.
        """
        self._normalize_observation()


class NormalizeAction(BaseTransformAction[NormalizedActT, ObsT, ActT],
                      Generic[ObsT, ActT]):
    """Normalize (without clipping) the action space of a pipeline environment
    according to its pre-defined bounds rather than statistics over collected
    data. Unbounded elements if any are left unchanged.

    .. warning::
        All leaves of the action space must have type `gym.spaces.Box`.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Define specialized operator(s) for efficiency
        self._denormalize_to_env_action = build_normalize(
            self.env.action_space, self.env.action, is_reversed=True)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = build_map(
            _normalize_space, self.env.action_space, None, 0)()

    def transform_action(self, action: NormalizedActT) -> None:
        """Update in-place the pre-allocated action buffer of the wrapped
        environment with the de-normalized action.
        """
        self._denormalize_to_env_action(action)
