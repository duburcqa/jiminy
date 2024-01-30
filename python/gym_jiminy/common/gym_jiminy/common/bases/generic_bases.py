"""Controller and observer abstract interfaces from reinforcement learning,
specifically design for Jiminy engine, and defined as mixin classes. Any
observer/controller block must inherit and implement those interfaces.
"""
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Dict, Any, TypeVar, Generic, no_type_check
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.core import array_copyto  # pylint: disable=no-name-in-module
from jiminy_py.simulator import Simulator
from jiminy_py.viewer.viewer import is_display_available

from ..utils import DataNested, fill


# Temporal resolution of simulator steps
DT_EPS: float = 1e-6  # 'SIMULATION_MIN_TIMESTEP'


ObsT = TypeVar('ObsT', bound=DataNested)
ActT = TypeVar('ActT', bound=DataNested)
BaseObsT = TypeVar('BaseObsT', bound=DataNested)
BaseActT = TypeVar('BaseActT', bound=DataNested)

SensorsDataType = Dict[str, npt.NDArray[np.float64]]
InfoType = Dict[str, Any]


# class EngineObsType(TypedDict):
#     t: np.ndarray
#     state:  DataNested
#     features: DataNested


EngineObsType: TypeAlias = DataNested


class ObserverInterface(ABC, Generic[ObsT, BaseObsT]):
    """Observer interface for both observers and environments.
    """
    observe_dt: float = -1
    observation_space: gym.Space  # [ObsT]
    observation: ObsT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the observer interface.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

        # Refresh the observation space
        self._initialize_observation_space()

    @abstractmethod
    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """

    @abstractmethod
    def refresh_observation(self, measurement: BaseObsT) -> None:
        """Compute observed features based on the current simulation state and
        lower-level measure.

        .. warning:
            When overloading this method, one is expected to use the internal
            buffer `observation` to store the observation by updating it by
            reference. It may be error prone and tricky to get use to it, but
            it is computationally more efficient as it avoids allocating memory
            multiple times and redundant calculus. Additionally, it enables to
            retrieve the observation later on by calling `get_observation`.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """


class ControllerInterface(ABC, Generic[ActT, BaseActT]):
    """Controller interface for both controllers and environments.
    """
    control_dt: float = -1
    action_space: gym.Space  # [ActT]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the controller interface.

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

    @abstractmethod
    def compute_command(self, action: ActT) -> BaseActT:
        """Compute the command to send to the subsequent block, based on the
        action and current observation of the environment.

        .. note::
            By design, the observation of the environment has been refreshed
            automatically prior to calling this method.

        :param action: High-level target to achieve by means of the command.

        :returns: Command to send to the subsequent block. It corresponds to
                  the target features of another lower-level controller if any,
                  the target motors efforts of the environment to ultimately
                  control otherwise.
        """

    def compute_reward(self,
                       terminated: bool,
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

        :param terminated: Whether the episode has reached the terminal state
                           of the MDP at the current step. This flag can be
                           used to compute a specific terminal reward.
        :param truncated: Whether a truncation condition outside the scope of
                          the MDP has been satisfied at the current step. This
                          flag can be used to adapt the reward.
        :param info: Dictionary of extra information for monitoring.

        :returns: Aggregated reward for the current step.
        """
        raise NotImplementedError


# Note that `JiminyEnvInterface` must inherit from `ObserverInterface`
# before `ControllerInterface` to initialize the action space before the
# observation space since the action itself may be part of the observation.
# Similarly, `gym.Env` must be last to make sure all the other initialization
# methods are called first.
class JiminyEnvInterface(
        ObserverInterface[ObsT, EngineObsType],
        ControllerInterface[ActT, np.ndarray],
        gym.Env[ObsT, ActT],
        Generic[ObsT, ActT]):
    """Observer plus controller interface for both generic pipeline blocks,
    including environments.
    """
    metadata: Dict[str, Any] = {
        "render_modes": (
            ['rgb_array'] + (['human'] if is_display_available() else []))
    }

    simulator: Simulator
    robot: jiminy.Robot
    stepper_state: jiminy.StepperState
    system_state: jiminy.SystemState
    sensors_data: SensorsDataType
    is_simulation_running: npt.NDArray[np.bool_]

    action: ActT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Track whether the observation has been refreshed manually since the
        # last called '_controller_handle'. It typically happens at the end of
        # every simulation step to return an observation that is consistent
        # with the updated state of the agent.
        self.__is_observation_refreshed = True

        # Store latest engine measurement for efficiency
        self.__measurement: EngineObsType = OrderedDict(
            t=np.array(0.0),
            states=OrderedDict(
                agent=OrderedDict(q=np.array([]), v=np.array([]))),
            measurements=OrderedDict(self.robot.sensors_data))
        self._sensors_types = tuple(self.robot.sensors_data.keys())

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        """Configure the observer-controller.

        In practice, it only resets the controller and observer update periods.

        .. note::
            This method must be called once, after the environment has been
            reset. This is done automatically when calling `reset` method.
        """
        # Re-initialize observation and control periods to ill-defined value to
        # trigger exception if not set properly later on.
        self.observe_dt = -1
        self.control_dt = -1

        # It is always necessary to refresh the observation at after reset
        self.__is_observation_refreshed = True

        # Reset observation and action buffers
        fill(self.observation, 0)
        fill(self.action, 0)

    @no_type_check
    def _observer_handle(self,
                         t: float,
                         q: np.ndarray,
                         v: np.ndarray,
                         sensors_data: jiminy.sensorsData) -> None:
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
        # Refresh the observation if not already done
        if not self.__is_observation_refreshed:
            measurement = self.__measurement
            measurement["t"][()] = t
            measurement["states"]["agent"]["q"] = q
            measurement["states"]["agent"]["v"] = v
            measurement_sensors = measurement["measurements"]
            sensors_data_it = iter(sensors_data.values())
            for sensor_type in self._sensors_types:
                measurement_sensors[sensor_type] = next(sensors_data_it)
            self.refresh_observation(measurement)

        # Consider observation has been refreshed iif a simulation is running
        self.__is_observation_refreshed = self.is_simulation_running.item()

    def _controller_handle(self,
                           t: float,
                           q: np.ndarray,
                           v: np.ndarray,
                           sensors_data: jiminy.sensorsData,
                           command: np.ndarray) -> None:
        """Thin wrapper around user-specified `refresh_observation` and
        `compute_command` methods.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
            It will be used by the base environment to instantiate a
            `jiminy.ControllerFunctor` that will be responsible for refreshing
            observations and compute commands of all the way through a given
            pipeline in the correct order of the blocks to finally sends
            command motor torques directly to the robot.

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

        :returns: Motors torques to apply on the robot.
        """
        # Refresh the observation
        self._observer_handle(t, q, v, sensors_data)

        # No need to check for breakpoints of the controller because it already
        # matches the update period by design.
        array_copyto(command, self.compute_command(self.action))

        # Always consider that the observation must be refreshed after calling
        # '_controller_handle' as it is never called more often than necessary.
        self.__is_observation_refreshed = False

    @property
    def unwrapped(self) -> "JiminyEnvInterface":
        """Base environment of the pipeline.
        """
        return self

    @property
    @abstractmethod
    def step_dt(self) -> float:
        """Get timestep of a single 'step'.
        """

    @property
    @abstractmethod
    def is_training(self) -> bool:
        """Check whether the environment is in 'train' or 'eval' mode.
        """

    @abstractmethod
    def train(self) -> None:
        """Sets the environment in training mode.
        """

    @abstractmethod
    def eval(self) -> None:
        """Sets the environment in evaluation mode.

        This only has an effect on certain environments. It can be used for
        instance to enable clipping or filtering of the action at evaluation
        time specifically. See documentations of a given environment for
        details about their behaviors in training and evaluation modes.
        """
