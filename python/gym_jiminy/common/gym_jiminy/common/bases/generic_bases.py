"""Controller and observer abstract interfaces from reinforcement learning,
specifically design for Jiminy engine, and defined as mixin classes. Any
observer/controller block must inherit and implement those interfaces.
"""
from abc import abstractmethod, abstractproperty, ABC
from collections import OrderedDict
from typing import Dict, Any, TypeVar, TypedDict, Generic, Callable

import numpy as np
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from ..utils import DataNested, is_breakpoint, zeros, fill, copy


# Temporal resolution of simulator steps
DT_EPS: float = jiminy.EngineMultiRobot.telemetry_time_unit


ObsType = TypeVar("ObsType", bound=DataNested)
ActType = TypeVar("ActType", bound=DataNested)
BaseObsType = TypeVar("BaseObsType", bound=DataNested)
BaseActType = TypeVar("BaseActType", bound=DataNested)

SensorsDataType = Dict[str, np.ndarray]
InfoType = Dict[str, Any]

StateType = TypedDict(
    'StateType', {'q': np.ndarray, 'v': np.ndarray})
EngineObsType = TypedDict('EngineObsType', {
    't': np.ndarray, 'agent_state': StateType, 'sensors_data': SensorsDataType
    })

ObserverHandleType = Callable[[
    float, np.ndarray, np.ndarray, Dict[str, np.ndarray]], None]
ControllerHandleType = Callable[[
    float, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray], None]


class ObserverInterface(ABC, Generic[ObsType, BaseObsType]):
    """Observer interface for both observers and environments.
    """
    observe_dt: float = -1
    observation_space: gym.Space[ObsType]
    _observation: ObsType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the observation interface.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

        # Refresh the observation space
        self._initialize_observation_space()

        # Initialize the observation buffer
        self._observation: ObsType = zeros(self.observation_space)

    @abstractmethod
    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        ...

    @abstractmethod
    def refresh_observation(self, measurement: BaseObsType) -> None:
        """Compute observed features based on the current simulation state and
        lower-level measure.

        .. warning:
            When overloading this method, one is expected to use the internal
            buffer `_observation` to store the observation by updating it by
            reference. It may be error prone and tricky to get use to it, but
            it is computationally more efficient as it avoids allocating memory
            multiple times and redundant calculus. Additionally, it enables to
            retrieve the observation later on by calling `get_observation`.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        ...

    def get_observation(self) -> ObsType:
        """Get post-processed observation.

        By default, it does not perform any post-processing. One is responsible
        for clipping the observation if necessary to make sure it does not
        violate the lower and upper bounds. This can be done either by
        overloading this method, or in the case of pipeline design, by adding a
        clipping observation block at the very end.

        .. warning::
            In most cases, it is not necessary to overloaded this method, and
            doing so may lead to unexpected behavior if not done carefully.
        """
        return self._observation


class ControllerInterface(ABC, Generic[ActType, BaseActType]):
    """Controller interface for both controllers and environments.
    """
    control_dt: float = -1
    action_space: gym.Space[ActType]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the control interface.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

        # Refresh the action space
        self._initialize_action_space()

    @abstractmethod
    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        ...

    @abstractmethod
    def compute_command(self, action: ActType) -> BaseActType:
        """Compute the command to send to the subsequent block, based on the
        action and current observation of the environment.

        .. note::
            By design, the observation of the environment has been refreshed
            automatically prior to calling this method.

        :param action: High-level target to achieve.

        :returns: Command to send to the subsequent block. It corresponds to
                  the target features of another lower-level controller if any,
                  the target motors efforts of the environment to ultimately
                  control otherwise.
        """
        ...

    def compute_reward(self,
                       done: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        """Compute the reward related to a specific control block.

        For the corresponding MDP to be stationary, the computation of the
        reward is supposed to involve only the transition from previous to
        current state of the simulation (possibly comprising multiple agents)
        under the ongoing action.

        By default, it returns 0.0 no matter what. It is up to the user to
        provide a dedicated reward function whenever appropriate.

        .. warning::
            Only returning an aggregated scalar reward is supported. Yet, it is
            possible to update 'info' by reference if one wants for keeping
            track of individual reward components or any kind of extra info
            that may be helpful for monitoring or debugging purposes.

        :param done: Whether the episode has reached one of terminal states of
                     the MDP at the current step. This flag can be used to
                     compute a specific terminal reward.
        :param truncated: Whether a truncation condition outside the scope of
                          the MDP has been satisfied at the current step. This
                          flag can be used to adapt the reward.
        :param info: Dictionary of extra information for monitoring.

        :returns: Aggregated reward for the current step.
        """
        # pylint: disable=unused-argument

        return 0.0


# Note that `JiminyEnvInterface` must inherit from `ObserverInterface`
# before `ControllerInterface` to initialize the action space before the
# observation space since the action itself may be part of the observation.
# Similarly, `gym.Env` must be last to make sure all the other initialization
# methods are called first.
class JiminyEnvInterface(
        ObserverInterface[ObsType, EngineObsType],
        ControllerInterface[ActType, np.ndarray],
        gym.Env[ObsType, ActType],
        Generic[ObsType, ActType]):
    """Observer plus controller interface for both generic pipeline blocks,
    including environments.
    """
    simulator: Simulator
    robot: jiminy.Robot
    stepper_state: jiminy.StepperState
    system_state: jiminy.SystemState
    sensors_data: SensorsDataType

    action: ActType

    def _setup(self) -> None:
        """Configure the observer-controller.

        In practice, it only resets the controller and observer update periods.

        .. note::
            This method must be called once, after the environment has been
            reset. This is done automatically when calling `reset` method.
        """
        self.observe_dt = -1
        self.control_dt = -1

        # Set default action.
        # It will be used for the initial step.
        fill(self.action, 0)

    def _observer_handle(self,
                         t: float,
                         q: np.ndarray,
                         v: np.ndarray,
                         sensors_data: SensorsDataType) -> None:
        """Thin wrapper around user-specified `refresh_observation` method.

        .. warning::
            This method is not supposed to be called manually nor overloaded.

        :param t: Current simulation time.
        :param q: Current actual configuration of the robot. Note that it is
                  not the one of the theoretical model even if
                  'use_theoretical_model' is enabled for the backend Python
                  `Simulator`.
        :param v: Current actual velocity vector.
        :param sensors_data: Current sensor data.
        """
        if is_breakpoint(t, self.observe_dt, DT_EPS):
            measurement: EngineObsType = OrderedDict(
                t=np.array((t,)),
                agent_state=OrderedDict(q=q, v=v),
                sensors_data=sensors_data)
            self.refresh_observation(measurement)

    def _controller_handle(self,
                           t: float,
                           q: np.ndarray,
                           v: np.ndarray,
                           sensors_data: SensorsDataType,
                           command: np.ndarray) -> None:
        """Thin wrapper around user-specified `compute_command` method.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
            It must be passed to `set_controller_handle` to send to use the
            controller to send commands directly to the robot.

        :param t: Current simulation time.
        :param q: Current actual configuration of the robot. Note that it is
                  not the one of the theoretical model even if
                  'use_theoretical_model' is enabled for the backend Python
                  `Simulator`.
        :param v: Current actual velocity vector.
        :param sensors_data: Current sensor data.
        :param command: Output argument corresponding to motors torques to
                        apply on the robot. It must be updated by reference
                        using `[:]` or `np.copyto`.
        """
        # pylint: disable=unused-argument

        command[:] = self.compute_command(self.action)

    def get_observation(self) -> ObsType:
        """Get post-processed observation.

        It performs a shallow copy of the observation.

        .. warning::
            This method is not supposed to be overloaded.
        """
        return copy(self._observation)

    @abstractproperty
    def step_dt(self) -> float:
        """Get timestep of a single 'step'.
        """
        ...

    @abstractproperty
    def is_training(self) -> bool:
        """Check whether the environment is in 'train' or 'eval' mode.
        """
        ...

    @abstractmethod
    def train(self) -> None:
        """Sets the environment in training mode.
        """
        ...

    @abstractmethod
    def eval(self) -> None:
        """Sets the environment in evaluation mode.

        This only has an effect on certain environments. It can be used for
        instance to enable clipping or filtering of the action at evaluation
        time specifically. See documentations of a given environment for
        details about their behaviors in training and evaluation modes.
        """
        ...
