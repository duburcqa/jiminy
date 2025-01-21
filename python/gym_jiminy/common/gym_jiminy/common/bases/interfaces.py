"""Controller and observer abstract interfaces from reinforcement learning,
specifically design for Jiminy engine, and defined as mixin classes. Any
observer/controller block must inherit and implement those interfaces.
"""
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from typing import (
    Dict, Any, Tuple, List, TypeVar, Generic, TypedDict, Optional, Callable,
    Mapping, no_type_check, TYPE_CHECKING)

import numpy as np
import numpy.typing as npt
import gymnasium as gym

import jiminy_py.core as jiminy
from jiminy_py.core import (  # pylint: disable=no-name-in-module
    multi_array_copyto)
from jiminy_py.simulator import Simulator, TabbedFigure
from jiminy_py.viewer.viewer import is_display_available

import pinocchio as pin

from ..utils import FieldNested, DataNested
if TYPE_CHECKING:
    from ..envs.generic import BaseJiminyEnv
    from ..quantities import QuantityManager


# Temporal resolution of simulator steps
DT_EPS: float = 1e-6  # 'SIMULATION_MIN_TIMESTEP'


Obs = TypeVar('Obs', bound=DataNested)
Act = TypeVar('Act', bound=DataNested)
BaseObs = TypeVar('BaseObs', bound=DataNested)
BaseAct = TypeVar('BaseAct', bound=DataNested)

SensorMeasurementStackMap = Dict[str, npt.NDArray[np.float64]]
InfoType = Dict[str, Any]

PolicyCallbackFun = Callable[
    [Obs, Optional[Act], Optional[float], bool, bool, InfoType], Act]


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


class InterfaceObserver(Generic[Obs, BaseObs], metaclass=ABCMeta):
    """Observer interface for both observers and environments.
    """
    observe_dt: float = -1
    observation_space: gym.Space[Obs]
    observation: Obs

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
    def refresh_observation(self, measurement: BaseObs) -> None:
        """Compute observed features based on the current simulation state and
        lower-level measure.

        .. warning:
            When overloading this method, one is expected to use the internal
            buffer `observation` to store the observation by updating it by
            reference. It may be error prone and tricky to get use to it, but
            it is computationally more efficient as it avoids allocating memory
            multiple times and redundant calculus.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """


class InterfaceController(Generic[Act, BaseAct], metaclass=ABCMeta):
    """Controller interface for both controllers and environments.
    """
    control_dt: float = -1
    action_space: gym.Space[Act]

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
    def compute_command(self, action: Act, command: BaseAct) -> None:
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
                       info: InfoType  # pylint: disable=unused-argument
                       ) -> float:
        """Compute the reward related to a specific control block, plus extra
        information that may be helpful for monitoring or debugging purposes.

        For the corresponding MDP to be stationary, the computation of the
        reward is supposed to involve only the transition from previous to
        current state of the simulation (possibly comprising multiple agents)
        under the ongoing action.

        By default, it returns 0.0 without extra information no matter what.
        The user is expected to provide an appropriate reward on its own,
        either by overloading this method or by wrapping the environment with
        `ComposedJiminyEnv` for modular environment pipeline design.

        :param terminated: Whether the episode has reached the terminal state
                           of the MDP at the current step. This flag can be
                           used to compute a specific terminal reward.
        :param info: Dictionary of extra information for monitoring.

        :returns: Aggregated reward for the current step.
        """
        return 0.0


# Note that `InterfaceJiminyEnv` must inherit from `InterfaceObserver` before
# `InterfaceController` to initialize the action space before the observation
# space since the action itself may be part of the observation.
# Similarly, `gym.Env` must be last to make sure all the other initialization
# methods are called first.
class InterfaceJiminyEnv(
        InterfaceObserver[Obs, EngineObsType],  # type: ignore[type-var]
        InterfaceController[Act, np.ndarray],
        gym.Env[Obs, Act],
        Generic[Obs, Act]):
    """Observer plus controller interface for both generic pipeline blocks,
    including environments.
    """

    metadata: Dict[str, Any] = {
        "render_modes": (
            ['rgb_array'] + (['human'] if is_display_available() else []))
    }

    # FIXME: Re-definition in derived class to stop mypy from complaining about
    # incompatible types between the multiple base classes.
    action_space: gym.Space[Act]
    observation_space: gym.Space[Obs]

    action: Act

    simulator: Simulator
    robot: jiminy.Robot
    stepper_state: jiminy.StepperState
    robot_state: jiminy.RobotState
    measurements: EngineObsType
    is_simulation_running: npt.NDArray[np.bool_]

    quantities: "QuantityManager"

    log_fieldnames: Mapping[str, FieldNested]
    """Fielnames associated with all the variables that have been recorded to
    the telemetry by any of the layer of the whole pipeline environment.
    """

    num_steps: npt.NDArray[np.int64]
    """Number of simulation steps that has been performed since last reset of
    the base environment.

    .. note::
        The counter is incremented before updating the observation at the end
        of the step, and consequently, before evaluating the reward and the
        termination conditions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Track whether the observation has been refreshed manually since the
        # last called '_controller_handle'. It typically happens at the end of
        # every simulation step to return an observation that is consistent
        # with the updated state of the agent.
        self.__is_observation_refreshed = True

        # Store latest engine measurement
        self.measurement = EngineObsType(
            t=np.array(0.0),
            states=OrderedDict(
                agent=OrderedDict(
                    q=pin.neutral(self.robot.pinocchio_model),
                    v=np.zeros(self.robot.pinocchio_model.nv))),
            measurements=OrderedDict(zip(
                self.robot.sensor_measurements.keys(),
                map(np.copy, self.robot.sensor_measurements.values()))))

        # Define flattened engine measurement for efficiency
        agent_state = self.measurement['states']['agent']
        assert isinstance(agent_state, dict)
        self._measurement_flat = (self.measurement['t'],
                                  agent_state['q'],
                                  agent_state['v'],
                                  *self.measurement['measurements'].values())

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

    @property
    def log_data(self) -> Dict[str, Any]:
        """Get log data associated with the ongoing simulation if any, the
        previous one otherwise.

        .. seealso::
            See `Simulator.log_data` documentation for details.
        """
        return self.simulator.log_data

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
        # Update engine measurement
        measurement_flat = (t, q, v, *sensor_measurements.values())
        multi_array_copyto(self._measurement_flat, measurement_flat)

        # Early return if no simulation is running
        if not self.is_simulation_running:
            return

        # Reset the quantity manager.
        # In principle, the internal cache of quantities should be cleared each
        # time the state of the robot and/or its derivative changes. This is
        # hard to do because there is no way to detect this specifically at the
        # time being. However, `_observer_handle` is never called twice in the
        # exact same state by the engine, so resetting quantities at the
        # beginning of the method should cover most cases. Yet, quantities
        # cannot be used reliably in the definition of profile forces because
        # they are always updated before the controller gets called, no matter
        # if either one or the other is time-continuous. Hacking the internal
        # dynamics to clear quantities does not address this issue either.
        # self.quantities.clear()

        # Refresh the observation if not already done but only if a simulation
        # is already running. It would be pointless to refresh the observation
        # at this point since the controller will be called multiple times at
        # start. Besides, it would defeat the purpose `_initialize_buffers`,
        # that is supposed to be executed before `refresh_observation` is being
        # called for the first time of an episode.
        if not self.__is_observation_refreshed:
            # Refresh observation
            try:
                self.refresh_observation(self.measurement)
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
        # Refresh the observation
        self._observer_handle(t, q, v, sensor_measurements)

        # No need to check for breakpoints of the controller because it already
        # matches the update period by design.
        self.compute_command(self.action, command)

        # Always consider that the observation must be refreshed after calling
        # '_controller_handle' as it is never called more often than necessary.
        self.__is_observation_refreshed = False

    @abstractmethod
    def _update_pipeline(self,
                         derived: Optional["InterfaceJiminyEnv"]) -> None:
        """Dynamically update which blocks are declared as part of the
        environment pipeline.

        Internally, this method first unregister all blocks of the old
        pipeline, then register all blocks of the new pipeline, and finally
        notify the base environment that the top-most block of the pipeline as
        changed and must be updated accordingly.

        .. warning::
            This method is not supposed to be called manually nor overloaded.

        :param derived: Either the top-most block of the pipeline or None.
                        If None, unregister all blocks of the old pipeline. If
                        not None, first unregister all blocks of the old
                        pipeline, then register all blocks of the new pipeline.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the underlying simulation completely.

        .. note::
            This method has nothing to do with termination and/or truncation of
            episodes. Calling it manually should never be necessary for
            collecting samples during training.

        .. note::
            This method is mainly intended for evaluation analysis and
            debugging. Stopping the episode is only necessary for switching
            between training and evaluation mode, and to log the final state,
            otherwise it will be missing from plots and viewer replay (see
            `InterfaceJiminyEnv.log_data` for details). Moreover, sensor data
            will not be available when calling `replay`. The helper method
            `jiminy_py.viewer.replay.play_logs_data` must be preferred to
            replay an episode that cannot be stopped.

        .. note::
            Resuming a simulation is not supported, which means that calling
            `reset` to start a new simulation is necessary prior to calling
            `step` once again. Falling to do so will trigger an exception.
        """

    @abstractmethod
    def plot(self,
             enable_block_states: bool = False,
             **kwargs: Any) -> TabbedFigure:
        """Plot figures of simulation data over time associated with the
        ongoing episode until now if any, the previous one otherwise.

        :param kwargs: Implementation-specific extra keyword arguments if any.
        """

    @abstractmethod
    def replay(self, **kwargs: Any) -> None:
        """Replay the ongoing episode until now if any, the previous one
        otherwise.

        :param kwargs: Implementation-specific extra keyword arguments if any.
        """

    @abstractmethod
    def evaluate(self,
                 policy_fn: PolicyCallbackFun,
                 seed: Optional[int] = None,
                 horizon: Optional[int] = None,
                 enable_stats: bool = True,
                 enable_replay: Optional[bool] = None,
                 **kwargs: Any) -> Tuple[List[float], List[InfoType]]:
        r"""Evaluate a policy on the environment over a complete episode.

        .. warning::
            It ignores any top-level `gym.Wrapper` that may be used for
            training but are not considered part of the environment pipeline.

        :param policy_fn:
            .. raw:: html

                Policy to evaluate as a callback function. It must have the
                following signature (**rew** = None at reset):

            | policy_fn\(**obs**: Obs,
            |            **action_prev**: Optional[Act],
            |            **reward**: Optional[float],
            |            **terminated**: bool,
            |            **truncated**: bool,
            |            **info**: InfoType
            |            \) -> Act  # **action**
        :param seed: Seed of the environment to be used for the evaluation of
                     the policy.
                     Optional: `None` by default. If not specified, then a
                     strongly random seed will be generated by gym.
        :param horizon: Horizon of the simulation, namely maximum number of
                        env steps before termination. `None` to disable.
                        Optional: Disabled by default.
        :param enable_stats: Whether to print high-level statistics after the
                             simulation.
                             Optional: Enabled by default.
        :param enable_replay: Whether to enable replay of the simulation, and
                              eventually recording if the extra
                              keyword argument `record_video_path` is provided.
                              Optional: Enabled by default if display is
                              available, disabled otherwise.
        :param kwargs: Extra keyword arguments to forward to the `replay`
                       method if replay is requested.
        """

    @abstractmethod
    def play_interactive(self,
                         enable_travelling: Optional[bool] = None,
                         start_paused: bool = True,
                         enable_is_done: bool = True,
                         verbose: bool = True,
                         **kwargs: Any) -> None:
        """Interact evaluation mode where the robot or the world itself are
        actively "controlled" via keyboard inputs, with real-time rendering.

        This method is not available for all pipeline environments. When it
        does, the available interactions and keyboard mapping is completely
        implementation-specific. Please refer to the documentation of the base
        environment being considered for details.

        .. warning::
            It ignores any top-level `gym.Wrapper` that may be used for
            training but are not considered part of the pipeline environment.

        :param enable_travelling: Whether enable travelling, following the
                                  motion of the root frame of the model. This
                                  parameter is ignored if the model has no
                                  freeflyer.
                                  Optional: Enabled by default iif 'panda3d'
                                  viewer backend is used.
        :param start_paused: Whether to start in pause.
                             Optional: Enabled by default.
        :param verbose: Whether to display status messages.
        :param kwargs: Implementation-specific extra keyword arguments if any.
        """

    @abstractmethod
    def has_terminated(self, info: InfoType) -> Tuple[bool, bool]:
        """Determine whether the episode is over, because a terminal state of
        the underlying MDP has been reached or an aborting condition outside
        the scope of the MDP has been triggered.

        .. note::
            This method is called after `refresh_observation`, so that the
            internal buffer 'observation' is up-to-date.

        :param info: Dictionary of extra information for monitoring.

        :returns: terminated and truncated flags.
        """

    @abstractmethod
    def set_wrapper_attr(self,
                         name: str,
                         value: Any,
                         *,
                         force: bool = True) -> None:
        """Assign an attribute to a specified value in the first layer of the
        pipeline environment for which it already exists, from this wrapper to
        the base environment. If the attribute does not exist in any layer, it
        is directly added to this wrapper.

        :param name: Name of the attribute.
        :param value: Desired value of the attribute.
        """

    @abstractmethod
    def train(self, mode: bool = True) -> None:
        """Sets the environment in training or evaluation mode.

        :param mode: Whether to set training (True) or evaluation mode (False).
                     Optional: `True` by default.
        """

    def eval(self) -> None:
        """Sets the environment in evaluation mode.

        This only has an effect on certain environments. It can be used for
        instance to enable clipping or filtering of the action at evaluation
        time specifically. See documentations of a given environment for
        details about their behaviors in training and evaluation modes.
        """
        self.train(False)

    @property
    @abstractmethod
    def training(self) -> bool:
        """Check whether the environment is in training or evaluation mode.
        """

    @training.setter
    def training(self, mode: bool) -> None:
        """Sets the environment in training or evaluation mode.
        """
        self.train(mode)

    @property
    @abstractmethod
    def unwrapped(self) -> "BaseJiminyEnv":
        """The "underlying environment at the basis of the pipeline from which
        this environment is part of.
        """

    @property
    @abstractmethod
    def step_dt(self) -> float:
        """Get timestep of a single 'step'.
        """
