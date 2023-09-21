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
from ..utils import zeros, build_reduce, build_flatten


FlatObsT: TypeAlias = ObsT
FlatActT: TypeAlias = ActT


class FlattenAction(BasePipelineWrapper[ObsT, FlatActT, ObsT, ActT],
                    Generic[ObsT, ActT]):
    """Flatten action.

    .. warning::
        All leaves of the action space must have type `gym.spaces.Box`.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Pre-allocated memory for the action
        self.action: FlatActT = zeros(self.action_space)

        # Define specialized operator(s) for efficiency
        self._unflatten_to_env_action = build_flatten(
            self.env.action, is_reversed=True)

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
        # Compute bounds of flattened action space
        low, high = map(np.concatenate, zip(*build_reduce(
            lambda *x: map(np.ravel, x), lambda x, y: x.append(y) or x, (),
            self.env.action_space, 0, initializer=list)()))

        # Initialize the action space with proper dtype
        dtype = low.dtype
        assert dtype is not None and issubclass(dtype.type, np.floating)
        self.action_space = gym.spaces.Box(low, high, dtype=dtype.type)

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
        # Un-flatten action and store the result in env action directly
        self._unflatten_to_env_action(action)

        # Delegate command computation to base environment
        return self.env.compute_command(self.env.action)


class FlattenObservation(BasePipelineWrapper[FlatObsT, ActT, ObsT, ActT],
                         Generic[ObsT, ActT]):
    """Flatten observation.

    .. warning::
        All leaves of the observation space must have type `gym.spaces.Box`.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Pre-allocated memory for the observation
        self.observation: FlatObsT = zeros(self.observation_space)

        # Define specialized operator(s) for efficiency
        self._flatten_observation = build_flatten(
            self.env.observation, self.observation)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        # Compute bounds of flattened action space
        low, high = map(np.concatenate, zip(*build_reduce(
            lambda *x: map(np.ravel, x), lambda x, y: x.append(y) or x, (),
            self.env.observation_space, 0, initializer=list)()))

        # Initialize the action space with proper dtype
        dtype = low.dtype
        assert dtype is not None and issubclass(dtype.type, np.floating)
        self.observation_space = gym.spaces.Box(low, high, dtype=dtype.type)

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """TODO: Write documentation.
        """
        # Refresh observation of the base environment
        self.env.refresh_observation(measurement)

        # Update flattened observation
        self._flatten_observation()

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)
