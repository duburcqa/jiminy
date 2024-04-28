"""Controller and observer abstract interfaces from reinforcement learning,
specifically design for Jiminy engine, and defined as mixin classes. Any
observer/controller block must inherit and implement those interfaces.
"""
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import (
    Dict, Any, TypeVar, Generic, no_type_check, TypedDict, TYPE_CHECKING)

import numpy as np
import numpy.typing as npt
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator
from jiminy_py.viewer.viewer import is_display_available

from ..utils import DataNested
if TYPE_CHECKING:
    from ..quantities import QuantityManager


# Temporal resolution of simulator steps
DT_EPS: float = 1e-6  # 'SIMULATION_MIN_TIMESTEP'


ObsT = TypeVar('ObsT', bound=DataNested)
ActT = TypeVar('ActT', bound=DataNested)
BaseObsT = TypeVar('BaseObsT', bound=DataNested)
BaseActT = TypeVar('BaseActT', bound=DataNested)

SensorMeasurementStackMap = Dict[str, npt.NDArray[np.float64]]
InfoType = Dict[str, Any]


class EngineObsType(TypedDict):
    """Raw observation provided by Jiminy Core Engine prior to any
    post-processing.
    """
    t: np.ndarray
    """Current simulation time.
    """
    states: Dict[str, DataNested]
    """State of the agent.
    """
    measurements: SensorMeasurementStackMap
    """Sensor measurements. Individual data for each sensor are aggregated by
    types in 2D arrays whose first dimension gathers the measured components
    and second dimension corresponds to individual measurements sorted by
    sensor indices.
    """


class InterfaceObserver(ABC, Generic[ObsT, BaseObsT]):
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


class InterfaceController(ABC, Generic[ActT, BaseActT]):
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
    def compute_command(self, action: ActT, command: BaseActT) -> None:
        """Compute the command to send to the subsequent block, based on the
        action and current observation of the environment.

        .. note::
            By design, the observation of the environment has been refreshed
            automatically prior to calling this method.

        :param action: High-level target to achieve by means of the command.
        :param command: Command to send to the subsequent block. It corresponds
                        to the target features of another lower-level
                        controller if any, the target motors efforts of the
                        environment to ultimately control otherwise. It must be
                        updated in-place.
        """

    def compute_reward(self,
                       terminated: bool,  # pylint: disable=unused-argument
                       truncated: bool,  # pylint: disable=unused-argument
                       info: InfoType  # pylint: disable=unused-argument
                       ) -> float:
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
        return 0.0


# Note that `InterfaceJiminyEnv` must inherit from `InterfaceObserver`
# before `InterfaceController` to initialize the action space before the
# observation space since the action itself may be part of the observation.
# Similarly, `gym.Env` must be last to make sure all the other initialization
# methods are called first.
class InterfaceJiminyEnv(
        InterfaceObserver[ObsT, EngineObsType],
        InterfaceController[ActT, np.ndarray],
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
    robot_state: jiminy.RobotState
    sensor_measurements: SensorMeasurementStackMap
    is_simulation_running: npt.NDArray[np.bool_]

    quantities: "QuantityManager"

    action: ActT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Track whether the observation has been refreshed manually since the
        # last called '_controller_handle'. It typically happens at the end of
        # every simulation step to return an observation that is consistent
        # with the updated state of the agent.
        self.__is_observation_refreshed = True

        # Store latest engine measurement for efficiency
        self.__measurement = EngineObsType(
            t=np.array(0.0),
            states=OrderedDict(
                agent=OrderedDict(q=np.array([]), v=np.array([]))),
            measurements=OrderedDict(self.robot.sensor_measurements))
        self._sensors_types = tuple(self.robot.sensor_measurements.keys())

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

        # The observation must always be refreshed after setup
        self.__is_observation_refreshed = False

    @no_type_check
    def _observer_handle(self,
                         t: float,
                         q: np.ndarray,
                         v: np.ndarray,
                         sensor_measurements: jiminy.SensorMeasurementTree
                         ) -> None:
        """Thin wrapper around user-specified `refresh_observation` method.

        .. warning::
            This method is not supposed to be called manually nor overloaded.

        :param t: Current simulation time.
        :param q: Current extended configuration vector of the robot.
        :param v: Current extended velocity vector of the robot.
        :param sensor_measurements: Current sensor data.
        """
        # Refresh the observation if not already done but only if a simulation
        # is already running. It would be pointless to refresh the observation
        # at this point since the controller will be called multiple times at
        # start. Besides, it would defeat the purpose `_initialize_buffers`,
        # that is supposed to be executed before `refresh_observation` is being
        # called for the first time of an episode.
        if not self.__is_observation_refreshed and self.is_simulation_running:
            measurement = self.__measurement
            measurement["t"][()] = t
            measurement["states"]["agent"]["q"] = q
            measurement["states"]["agent"]["v"] = v
            measurement_sensors = measurement["measurements"]
            sensor_measurements_it = iter(sensor_measurements.values())
            for sensor_type in self._sensors_types:
                measurement_sensors[sensor_type] = next(sensor_measurements_it)
            try:
                self.refresh_observation(measurement)
            except RuntimeError as e:
                raise RuntimeError(
                    "The observation space must be invariant.") from e
            self.__is_observation_refreshed = True

    def _controller_handle(self,
                           t: float,
                           q: np.ndarray,
                           v: np.ndarray,
                           sensor_measurements: jiminy.SensorMeasurementTree,
                           command: np.ndarray) -> None:
        """Thin wrapper around user-specified `refresh_observation` and
        `compute_command` methods.

        .. note::
            The internal cache of managed quantities is cleared right away
            systematically, before anything else.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
            It will be used by the base environment to instantiate a
            `jiminy.FunctionalController` responsible for both refreshing
            observations and compute commands of all the way through a given
            pipeline in the correct order of the blocks to finally sends
            command motor torques directly to the robot.

        :param t: Current simulation time.
        :param q: Current extended configuration vector of the robot.
        :param v: Current actual velocity vector of the robot.
        :param sensor_measurements: Current sensor measurements.
        :param command: Output argument corresponding to motors torques to
                        apply on the robot. It must be updated by reference
                        using `[:]` or `np.copyto`.

        :returns: Motors torques to apply on the robot.
        """
        # Reset the quantity manager.
        # In principle, the internal cache of quantities should be cleared not
        # each time the state of the robot and/or its derivative changes. This
        # is hard to do because there is no way to detect this specifically at
        # the time being. However, `_controller_handle` is never called twice
        # in the exact same state by the engine, so resetting quantities at the
        # beginning of the method should cover most cases. Yet, quantities
        # cannot be used reliably in the definition of profile forces because
        # they are always updated before the controller gets called, no matter
        # if either one or the other is time-continuous. Hacking the internal
        # dynamics to clear quantities does not address this issue either.
        self.quantities.clear()

        # Refresh the observation
        self._observer_handle(t, q, v, sensor_measurements)

        # No need to check for breakpoints of the controller because it already
        # matches the update period by design.
        self.compute_command(self.action, command)

        # Always consider that the observation must be refreshed after calling
        # '_controller_handle' as it is never called more often than necessary.
        self.__is_observation_refreshed = False

    @property
    def unwrapped(self) -> "InterfaceJiminyEnv":
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
