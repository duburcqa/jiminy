""" TODO: Write documentation.
"""
from typing import Generic
from typing_extensions import TypeAlias

import numpy as np

import gymnasium as gym

from ..bases import (ObsT,
                     ActT,
                     EngineObsType,
                     BasePipelineWrapper,
                     JiminyEnvInterface)
from ..utils import zeros, build_map, build_normalize


NormalizedObsT: TypeAlias = ObsT
NormalizedActT: TypeAlias = ActT


def _normalize_space(space: gym.spaces.Box) -> gym.spaces.Box:
    """TODO: Write documentation.
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


class NormalizeAction(BasePipelineWrapper[ObsT, NormalizedActT, ObsT, ActT],
                      Generic[ObsT, ActT]):
    """Normalize action without clipping.

    .. warning::
        All leaves of the action space must have type `gym.spaces.Box`.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Pre-allocated memory for the action
        self.action: NormalizedActT = zeros(self.action_space)

        # Define specialized operator(s) for efficiency
        self._denormalize_to_env_action = build_normalize(
            self.env.action_space, self.env.action, is_reversed=True)

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to the base implementation, it configures the controller
        and registers its target to the telemetry.
        """
        # Call base implementation
        super()._setup()

        # Copy observe and control update periods
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = build_map(
            _normalize_space, self.env.action_space, None, 0)()

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        self.observation_space = self.env.observation_space

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It simply forwards the observation computed by the wrapped environment
        without any processing.
        """
        self.env.refresh_observation(measurement)

    def compute_command(self, action: ActT) -> np.ndarray:
        """TODO: Write documentation.
        """
        # De-normalization action and store the result in env action directly
        self._denormalize_to_env_action(action)

        # Delegate command computation to base environment
        return self.env.compute_command(self.env.action)


class NormalizeObservation(
        BasePipelineWrapper[NormalizedObsT, ActT, ObsT, ActT],
        Generic[ObsT, ActT]):
    """Normalize observation without clipping.

    .. warning::
        All leaves of the observation space must have type `gym.spaces.Box`.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Pre-allocated memory for the observation
        self.observation: NormalizedObsT = zeros(env.observation_space)

        # Define specialized operator(s) for efficiency
        self._normalize_observation = build_normalize(
            self.env.observation_space, self.observation, self.env.observation)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        self.observation_space = build_map(
            _normalize_space, self.env.observation_space, None, 0)()

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """TODO: Write documentation.
        """
        # Refresh observation of the base environment
        self.env.refresh_observation(measurement)

        # Update normalized observation
        self._normalize_observation()

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)
