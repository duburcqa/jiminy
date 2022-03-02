
"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

    - the concept of block that can be connected to a `BaseJiminyEnv`
      environment through any level of indirection
    - a base controller block, along with a concret PD controller
    - a wrapper to combine a controller block and a `BaseJiminyEnv`
      environment, eventually already wrapped, so that it appears as a single,
      unified environment.
"""
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, Any, List, Callable

import numpy as np
import gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from ..utils import (DataNested,
                     is_breakpoint,
                     zeros,
                     fill,
                     copy,
                     set_value)
from ..envs import ObserverHandleType, ControllerHandleType, BaseJiminyEnv

from .block_bases import BaseControllerBlock, BaseObserverBlock
from .generic_bases import ObserverControllerInterface


class BasePipelineWrapper(ObserverControllerInterface, gym.Wrapper):
    """Wrap a `BaseJiminyEnv` Gym environment and a single block, so that it
    appears as a single, unified, environment. Eventually, the environment can
    already be wrapped inside one or several `gym.Wrapper` containers.

    If several successive blocks must be used, just wrap successively each
    block one by one with the resulting intermediary `PipelineWrapper`.

    .. warning::
        This architecture is not designed for trainable blocks, but rather for
        robotic-oriented controllers and observers, such as PID controllers,
        inverse kinematics, Model Predictive Control (MPC), sensor fusion...
        It is recommended to add the controllers and observers into the
        policy itself if they have to be trainable.
    """
    env: Union[gym.Wrapper, BaseJiminyEnv]

    def __init__(self,
                 env: Union[gym.Wrapper, BaseJiminyEnv],
                 **kwargs: Any) -> None:
        """
        :param kwargs: Extra keyword arguments for multiple inheritance.
        """
        # Initialize base wrapper and interfaces through multiple inheritance
        super().__init__(env)  # Do not forward extra arguments, if any

        # Refresh some proxies for fast lookup
        self.simulator: Simulator = self.env.simulator
        self.stepper_state: jiminy.StepperState = self.env.stepper_state
        self.system_state: jiminy.SystemState = self.env.system_state
        self.sensors_data: jiminy.sensorsData = self.env.sensors_data

        # Define some internal buffers
        self._command = zeros(self.env.unwrapped.action_space)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self.env.__dir__()

    def _controller_handle(self,
                           t: float,
                           q: np.ndarray,
                           v: np.ndarray,
                           sensors_data: jiminy.sensorsData,
                           command: np.ndarray) -> None:
        """Thin wrapper around user-specified `compute_command` method.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
        """
        # Assertion(s) for type checker
        assert self._action is not None

        command[:] = self.compute_command(
            self.env.get_observation(), self._action)

    def _get_block_index(self) -> int:
        """Get the index of the block. It corresponds the "deepness" of the
        block, namely how many blocks deriving from the same wrapper type than
        the current one are already wrapped in the environment.
        """
        # Assertion(s) for type checker
        assert self.env is not None

        i = 0
        block = self.env
        while not isinstance(block, BaseJiminyEnv):
            i += isinstance(block, self.__class__)
            block = block.env
        return i

    def get_observation(self) -> DataNested:
        """Get post-processed observation.

        It performs a recursive shallow copy of the observation.

        .. warning::
            This method is not supposed to be called manually nor overloaded.
        """
        # Assertion(s) for type checker
        assert self._observation is not None

        return copy(self._observation)

    def reset(self,
              controller_hook: Optional[Callable[[], Optional[Tuple[
                  Optional[ObserverHandleType],
                  Optional[ControllerHandleType]]]]] = None,
              **kwargs: Any) -> DataNested:
        """Reset the unified environment.

        In practice, it resets the environment and initializes the generic
        pipeline internal buffers through the use of 'controller_hook'.

        :param controller_hook: Used internally for chaining multiple
                                `BasePipelineWrapper`. It is not meant to be
                                defined manually.
                                Optional: None by default.
        :param kwargs: Extra keyword arguments to comply with OpenAI Gym API.
        """
        # pylint: disable=unused-argument,arguments-differ

        # Define chained controller hook
        def register() -> Tuple[ObserverHandleType, ControllerHandleType]:
            """Register the block to the higher-level block.

            This method is used internally to make sure that `_setup` method
            of connected blocks are called in the right order, namely from the
            lowest-level block to the highest-level one, right after reset of
            the low-level simulator and just before performing the first step.
            """
            nonlocal self, controller_hook

            # Initialize the pipeline wrapper
            self._setup()

            # Forward the observer and controller handles provided by the
            # controller hook of higher-level block, if any, or use the
            # ones of this block otherwise.
            observer_handle, controller_handle = None, None
            if controller_hook is not None:
                handles = controller_hook()
                if handles is not None:
                    observer_handle, controller_handle = handles
            if observer_handle is None:
                observer_handle = self._observer_handle
            if controller_handle is None:
                controller_handle = self._controller_handle
            return observer_handle, controller_handle

        # Reset base pipeline
        self.env.reset(register, **kwargs)

        return self.get_observation()

    def step(self,
             action: Optional[DataNested] = None
             ) -> Tuple[DataNested, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Next action to perform. `None` to not update it.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information.
        """
        # Assertion(s) for type checker
        assert self._action is not None

        # Backup the action to perform, if any
        if action is not None:
            set_value(self._action, action)

        # Compute the next learning step
        _, reward, done, info = self.env.step()

        # Compute block's reward and sum it to total
        reward += self.compute_reward(info=info)
        if self.enable_reward_terminal:
            if done and self.env.unwrapped._num_steps_beyond_done == 0:
                reward += self.compute_reward_terminal(info=info)

        return self.get_observation(), reward, done, info

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        """Configure the wrapper.

        By default, it only resets some internal buffers.

        .. note::
            This method must be called once, after the environment has been
            reset. This is done automatically when calling `reset` method.
        """
        # Call base implementation
        super()._setup()

        # Assertion(s) for type checker
        assert self._action is not None

        # Reset some internal buffers
        fill(self._action, 0.0)
        fill(self._command, 0.0)

        # Refresh some proxies for fast lookup
        self.sensors_data = self.env.sensors_data

    def refresh_observation(self) -> None:  # type: ignore[override]
        """Compute the unified observation.

        By default, it forwards the observation computed by the environment.

        :param measure: Observation of the environment.
        """
        # pylint: disable=arguments-differ

        self.env.refresh_observation()

    def compute_command(self,
                        measure: DataNested,
                        action: DataNested) -> DataNested:
        """Compute the motors efforts to apply on the robot.

        By default, it forwards the command computed by the environment.

        :param measure: Observation of the environment.
        :param action: Target to achieve.
        """
        # Assertion(s) for type checker
        assert self._action is not None
        assert self.env._action is not None

        set_value(self._action, action)
        set_value(self.env._action, action)
        return self.env.compute_command(measure, action)


class ObservedJiminyEnv(BasePipelineWrapper):
    """Wrap a `BaseJiminyEnv` Gym environment and a single observer, so that
    it appears as a single, unified, environment. Eventually, the environment
    can already be wrapped inside one or several `gym.Wrapper` containers.

    .. aafig::
        :proportional:
        :textual:

                +---------+           +------------+
        "act_1" |         | "obs_env" |            |
        ------->+  "env"  +---------->+ "observer" +----------->
                |         |   "obs"   |            | "features"
                +---------+           +------------+

    The input observation 'obs_env' of 'observer' must be consistent with the
    observation space 'obs' of the environment. The observation space of the
    outcoming unified environment will be the observation space of the
    highest-level observer, while its action space will be the one of the
    unwrapped environment 'obs'.

    .. warning::
        This design is not suitable for learning the observer, but rather for
        robotic-oriented observers, such as sensor fusion algorithms, Kalman
        filters... It is recommended to add the observer into the policy
        itself if it has to be trainable.
    """  # noqa: E501  # pylint: disable=line-too-long
    env: Union[gym.Wrapper, BaseJiminyEnv]
    observation_space: Optional[gym.Space]

    def __init__(self,
                 env: Union[gym.Wrapper, BaseJiminyEnv],
                 observer: BaseObserverBlock,
                 augment_observation: bool = False,
                 **kwargs: Any):
        """
        :param env: Environment to control. It can be an already controlled
                    environment wrapped in `ObservedJiminyEnv` if one desires
                    to stack several controllers with `BaseJiminyEnv`.
        :param observer: Observer to use to extract higher-level features.
        :param augment_observation: Whether or not to gather the high-level
                                    features computed by the observer with the
                                    raw observation of the environment. This
                                    option is only available if the observation
                                    space is of type `gym.spaces.Dict`.
                                    Optional: Disabled by default.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Backup user arguments
        self.observer = observer
        self.augment_observation = augment_observation

        # Make sure that the unwrapped environment matches the controlled one
        assert env.unwrapped is observer.env

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Assertion(s) for type checker
        assert (self.env.action_space is not None and
                self.env.observation_space is not None)

        # Retrieve the environment observation
        observation = self.env.get_observation()

        # Update the action space
        self.action_space = self.env.action_space

        # Set the controller name, based on the controller index
        self.observer_name = f"observer_{self._get_block_index()}"

        # Assertion(s) for type checker
        assert (self.observer.action_space is not None and
                self.observer.observation_space is not None)

        # Check that the initial observation of the environment is consistent
        # with the action space of the observer.
        assert self.observer.action_space.contains(observation), (
            "The command is not consistent with the action space of the "
            "subsequent block.")

        # Update the observation space
        if self.augment_observation:
            self.observation_space = deepcopy(self.env.observation_space)
            if not isinstance(self.observation_space, gym.spaces.Dict):
                self.observation_space = gym.spaces.Dict(OrderedDict(
                    measures=self.observation_space))
            self.observation_space.spaces.setdefault(
                'features', gym.spaces.Dict()).spaces[
                    self.observer_name] = self.observer.observation_space
        else:
            self.observation_space = self.observer.observation_space

        # Initialize some internal buffers
        self._action = zeros(self.action_space, dtype=np.float64)
        self._observation = zeros(self.observation_space)

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to the base implementation, it configures the observer.
        """
        # Configure the observer
        self.observer._setup()

        # Call base implementation
        super()._setup()

        # Compute the observe and control update periods
        self.observe_dt = self.observer.observe_dt
        self.control_dt = self.env.control_dt

    def refresh_observation(self) -> None:  # type: ignore[override]
        """Compute high-level features based on the current wrapped
        environment's observation.

        It gathers the original observation from the environment with the
        features computed by the observer, if requested, otherwise it forwards
        the features directly without any further processing.

        .. warning::
            Beware it updates and returns '_observation' buffer to deal with
            multiple observers with different update periods. Even so, it is
            safe to call this method multiple times successively.

        :returns: Updated part of the observation only for efficiency.
        """
        # pylint: disable=arguments-differ

        # Get environment observation
        super().refresh_observation()

        # Update observed features if necessary
        t = self.stepper_state.t
        if is_breakpoint(t, self.observe_dt, self._dt_eps):
            obs = self.env.get_observation()
            self.observer.refresh_observation(obs)
            if not self.simulator.is_simulation_running:
                features = self.observer.get_observation()
                if self.augment_observation:
                    # Assertion(s) for type checker
                    assert isinstance(self._observation, dict)
                    # Make sure to store references
                    if isinstance(obs, gym.spaces.Dict):
                        self._observation = obs
                    else:
                        self._observation['measures'] = obs
                    self._observation.setdefault('features', OrderedDict())[
                        self.observer_name] = features
                else:
                    self._observation = features


class ControlledJiminyEnv(BasePipelineWrapper):
    """Wrap a `BaseJiminyEnv` Gym environment and a single controller, so that
    it appears as a single, unified, environment. Eventually, the environment
    can already be wrapped inside one or several `gym.Wrapper` containers.

    If several successive controllers must be used, just wrap successively each
    controller one by one with the resulting `ControlledJiminyEnv`.

    .. aafig::
        :proportional:
        :textual:

                +---------+                    +----------+                      +----------+
        "act_1" |         |            "act_2" |          |              "act_3" |          |
        ------->+  "env"  +------> "+" ------->+ "ctrl_1" +--------> "+" ------->+ "ctrl_2" +--------> "="
                |         | "obs"              |          | "cmd_1"              |          | "cmd_2"
                +---------+                    +----------+                      +----------+

                      +----------------------------------------------------+
                      |                                                    |
                      v                                                    |
                +-----+----+         +----------+         +---------+      |
        "act_3" |          | "act_2" |          | "act_1" |         |      | "obs + cmd_1 + cmd_2"
        ------->+ "ctrl_2" +-------->+ "ctrl_1" +-------->+  "env"  +--o---o---------------------->
                |          | "cmd_2" |          | "cmd_1" |         |  | "obs + cmd_1"
                +----------+         +-----+----+         +---------+  |
                                           ^                           |
                                           |                           |
                                           +---------------------------+

    The output command 'cmd_X' of 'ctrl_X' must be consistent with the action
    space 'act_X' of the subsequent block. The action space of the outcoming
    unified environment will be the action space of the highest-level
    controller 'act_N', while its observation space will be the one of the
    unwrapped environment 'obs'. Alternatively, the later can also gather the
    (stacked) action space of the successive controllers if one is to observe
    the intermediary controllers' targets.

    .. note::
        The environment and each controller has their own update period.

    .. warning::
        This design is not suitable for learning the controllers 'ctrl_X', but
        rather for robotic-oriented controllers, such as PID control, inverse
        kinematics, admittance control, or Model Predictive Control (MPC). It
        is recommended to add the controllers into the policy itself if it has
        to be trainable.
    """  # noqa: E501  # pylint: disable=line-too-long
    observation_space: Optional[gym.Space]

    def __init__(self,
                 env: Union[gym.Wrapper, BaseJiminyEnv],
                 controller: BaseControllerBlock,
                 augment_observation: bool = False,
                 **kwargs: Any):
        """
        .. note::
            As a reminder, `env.step_dt` refers to the learning step period,
            namely the timestep between two successive frames:

                [obs, reward, done, info]

            This definition remains true, independently of whether or not the
            environment is wrapped with a controller using this class. On the
            contrary, `env.control_dt` corresponds to the apparent control
            update period, namely the update period of the higher-level
            controller if multiple are piped together. The same goes for
            `env.observe_dt`.

        :param env: Environment to control. It can be an already controlled
                    environment wrapped in `ControlledJiminyEnv` if one desires
                    to stack several controllers with `BaseJiminyEnv`.
        :param controller: Controller to use to send targets to the subsequent
                           block.
        :param augment_observation: Whether or not to gather the target of the
                                    controller with the observation of the
                                    environment. This option is only available
                                    if the observation space is of type
                                    `gym.spaces.Dict`.
                                    Optional: Disabled by default.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Backup user arguments
        self.controller = controller
        self.augment_observation = augment_observation

        # Make sure that the unwrapped environment matches the controlled one
        assert env.unwrapped is controller.env

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Assertion(s) for type checker
        assert (self.env.action_space is not None and
                self.env.observation_space is not None)

        # Enable terminal reward only if the controller implements it
        self.enable_reward_terminal = self.controller.enable_reward_terminal

        # Set the controller name, based on the controller index
        self.controller_name = f"controller_{self._get_block_index()}"

        # Update the action space
        self.action_space = self.controller.action_space

        # Assertion(s) for type checker
        assert self.action_space is not None

        # Check that 'augment_observation' can be enabled
        assert not self.augment_observation or isinstance(
            self.env.observation_space, gym.spaces.Dict), (
            "'augment_observation' is only available for environments whose "
            "observation space inherits from `gym.spaces.Dict`.")

        # Append the controller's target to the observation if requested
        self.observation_space = deepcopy(self.env.observation_space)
        if self.augment_observation:
            if not isinstance(self.observation_space, gym.spaces.Dict):
                self.observation_space = gym.spaces.Dict(OrderedDict(
                    measures=self.observation_space))
            self.observation_space.spaces.setdefault(
                'targets', gym.spaces.Dict()).spaces[
                    self.controller_name] = self.controller.action_space

        # Initialize some internal buffers
        self._action = zeros(self.action_space, dtype=np.float64)
        self._target = zeros(self.env.action_space, dtype=np.float64)
        self._observation = zeros(self.observation_space)

        # Register the controller target to the telemetry
        self.env.register_variable("action",
                                   self._action,
                                   self.controller.get_fieldnames(),
                                   self.controller_name)

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to the base implementation, it configures the controller
        and registers its target to the telemetry.
        """
        # Configure the controller
        self.controller._setup()

        # Call base implementation
        super()._setup()

        # Reset some additional internal buffers
        fill(self._target, 0.0)

        # Compute the observe and control update periods
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.controller.control_dt

    def compute_command(self,
                        measure: DataNested,
                        action: DataNested
                        ) -> DataNested:
        """Compute the motors efforts to apply on the robot.

        In practice, it updates, whenever it is necessary:

            - the target sent to the subsequent block by the controller
            - the command send to the robot by the environment through the
              subsequent block

        :param measure: Observation of the environment.
        :param action: High-level target to achieve.
        """
        # Assertion(s) for type checker
        assert self._observation is not None
        assert self.env._action is not None

        # Update the target to send to the subsequent block if necessary.
        # Note that `_observation` buffer has already been updated right before
        # calling this method by `_controller_handle`, so it can be used as
        # measure argument without issue.
        t = self.stepper_state.t
        if is_breakpoint(t, self.control_dt, self._dt_eps):
            target = self.controller.compute_command(self._observation, action)
            set_value(self._target, target)

        # Update the command to send to the actuators of the robot.
        # Note that the environment itself is responsible of making sure to
        # update the command of the right period. Ultimately, this is done
        # automatically by the engine, which is calling `_controller_handle` at
        # the right period.
        if self.simulator.is_simulation_running:
            # Do not update command during the first iteration because the
            # action is undefined at this point
            set_value(self.env._action, self._target)
            set_value(self._command, self.env.compute_command(
                self._observation, self._target))

        return self._command

    def refresh_observation(self) -> None:  # type: ignore[override]
        """Compute the unified observation based on the current wrapped
        environment's observation and controller's target.

        It gathers the actual observation from the environment with the target
        of the controller, if requested, otherwise it forwards the observation
        directly without any further processing.

        .. warning::
            Beware it shares the environment observation whenever it is
            possible for the sake of efficiency. Despite that, it is safe to
            call this method multiple times successively.

        :returns: Original environment observation, eventually including
                  controllers targets if requested.
        """
        # Get environment observation
        super().refresh_observation()

        # Add target to observation if requested
        if not self.simulator.is_simulation_running:
            obs = self.env.get_observation()
            if self.augment_observation:
                # Assertion(s) for type checker
                assert isinstance(self._observation, dict)
                # Make sure to store references
                if isinstance(obs, dict):
                    self._observation = copy(obs)
                else:
                    self._observation['measures'] = obs
                self._observation.setdefault('targets', OrderedDict())[
                    self.controller_name] = self._action
            else:
                self._observation = obs

    def compute_reward(self, *args: Any, **kwargs: Any) -> float:
        return self.controller.compute_reward(*args, **kwargs)

    def compute_reward_terminal(self, *args: Any, **kwargs: Any) -> float:
        return self.controller.compute_reward_terminal(*args, **kwargs)
