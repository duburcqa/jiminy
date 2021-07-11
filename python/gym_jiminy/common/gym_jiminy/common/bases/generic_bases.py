"""Controller and observer abstract interfaces from reinforcement learning,
specifically design for Jiminy engine, and defined as mixin classes. Any
observer/controller block must inherite and implement those interfaces.
"""
from typing import Optional, Dict, Any

import numpy as np
import gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from ..utils import SpaceDictNested, is_breakpoint


class ObserverInterface:
    """Observer interface for both observers and environments.
    """
    observe_dt: float
    observation_space: Optional[gym.Space]
    _observation: Optional[SpaceDictNested]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the observation interface.

        It only allocates some attributes.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Define some attributes
        self.observe_dt = -1
        self.observation_space = None
        self._observation = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    def get_observation(self) -> SpaceDictNested:
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

    # methods to override:
    # ----------------------------

    def _refresh_observation_space(self) -> None:
        """Configure the observation space.
        """
        raise NotImplementedError

    def refresh_observation(self, *args: Any, **kwargs: Any) -> None:
        """Update the observation based on the current simulation state.

        .. warning:
            When overloading this method, one is expected to use the internal
            buffer `_observation` to store the observation by updating it by
            reference. It may be error prone and tricky to get use to it, but
            it is computionally optimal because it avoids allocating memory and
            redundant calculus. Additionally, it enables to retrieve the
            observation later on by calling `get_observation`.

        :param args: Extra arguments that may be useful to derived
                     implementations.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        raise NotImplementedError


class ControllerInterface:
    """Controller interface for both controllers and environments.
    """
    control_dt: float
    action_space: Optional[gym.Space]
    _action: Optional[SpaceDictNested]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the control interface.

        It only allocates some attributes.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Define some attributes
        self.control_dt = -1
        self.action_space = None
        self._action = None

        self.enable_reward_terminal = (
            self.compute_reward_terminal.  # type: ignore[attr-defined]
            __func__ is not ControllerInterface.compute_reward_terminal)

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    # methods to override:
    # ----------------------------

    def _refresh_action_space(self) -> None:
        """Configure the action space.
        """
        raise NotImplementedError

    def compute_command(self,
                        measure: SpaceDictNested,
                        action: SpaceDictNested) -> SpaceDictNested:
        """Compute the command to send to the subsequent block, based on the
        current target and observation of the environment.

        :param measure: Observation of the environment.
        :param action: High-level target to achieve.

        :returns: Command to send to the subsequent block. It is a target if
                  the latter is another controller, or motors efforts command
                  if it is the environment to ultimately control.
        """
        raise NotImplementedError

    def compute_reward(self,
                       *args: Any,
                       info: Dict[str, Any],
                       **kwargs: Any) -> float:
        """Compute reward at current episode state.

        By default, it always returns 0.0. It must be overloaded to implement
        a proper reward function.

        .. warning::
            For compatibility with openAI gym API, only the total reward is
            returned. Yet, it is possible to update 'info' by reference if
            one wants to log extra info for monitoring.

        :param args: Extra arguments that may be useful for derived
                     environments, for example `Gym.GoalEnv`.
        :param info: Dictionary of extra information for monitoring.
        :param kwargs: Extra keyword arguments. See 'args'.

        :returns: Total reward.
        """
        # pylint: disable=no-self-use,unused-argument

        return 0.0

    def compute_reward_terminal(self, *, info: Dict[str, Any]) -> float:
        """Compute terminal reward at current episode final state.

        .. note::
            Implementation is optional. Not computing terminal reward if not
            overloaded by the user for the sake of efficiency.

        .. warning::
            Similarly to `compute_reward`, 'info' can be updated by reference
            to log extra info for monitoring.

        :param info: Dictionary of extra information for monitoring.

        :returns: Terminal reward.
        """
        raise NotImplementedError


class ObserverControllerInterface(ObserverInterface, ControllerInterface):
    """Observer plus controller interface for both generic pipeline blocks,
    including environments.
    """
    simulator: Optional[Simulator]
    stepper_state: Optional[jiminy.StepperState]
    system_state: Optional[jiminy.SystemState]
    sensors_data: Optional[Dict[str, np.ndarray]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Define some attributes
        self.simulator = None
        self.stepper_state = None
        self.system_state = None
        self.sensors_data = None

        # Define some internal buffers
        self._dt_eps: Optional[float] = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        """Configure the observer-controller.

        .. note::
            This method must be called once, after the environment has been
            reset. This is done automatically when calling `reset` method.
        """
        # Assertion(s) for type checker
        assert isinstance(self.simulator, Simulator)

        # Reset the control and observation update periods
        self.observe_dt = -1
        self.control_dt = -1

        # Get the temporal resolution of simulator steps
        engine_options = self.simulator.engine.get_options()
        self._dt_eps = 1.0 / engine_options["telemetry"]["timeUnit"]

    def _observer_handle(self,
                         t: float,
                         q: np.ndarray,
                         v: np.ndarray,
                         sensors_data: jiminy.sensorsData) -> None:
        """TODO Write documentation.
        """
        # pylint: disable=unused-argument
        if is_breakpoint(t, self.observe_dt, self._dt_eps):
            self.refresh_observation()

    def _controller_handle(self,
                           t: float,
                           q: np.ndarray,
                           v: np.ndarray,
                           sensors_data: jiminy.sensorsData,
                           command: np.ndarray) -> None:
        """This method is the main entry-point to interact with the simulator.

        .. note:
            It is design to apply motors efforts on the robot, but internally,
            it updates the observation before computing the command.

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
        :param sensors_data: Current sensor data. Note that it is the raw data,
                             which means that it is not an actual dictionary
                             but it behaves similarly.
        :param command: Output argument to update by reference using `[:]` or
                        `np.copyto` in order to apply motors torques on the
                        robot.
        """
        raise NotImplementedError
