"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:
- the concept of block thats can be connected to a `BaseJiminyEnv` environment
  through multiple `JiminyEnvInterface` indirections
- a base controller block, along with a concrete PD controller
- a wrapper to combine a controller block and a `BaseJiminyEnv` environment,
  eventually already wrapped, so that it appears as a black-box environment.
"""
from weakref import ref
from copy import deepcopy
from collections import OrderedDict
from itertools import chain
from typing import (
    Dict, Any, Optional, Tuple, Iterable, Generic, TypeVar, SupportsFloat)

import numpy as np
import gymnasium as gym

from ..utils import DataNested, is_breakpoint, zeros, fill, set_value

from .generic_bases import (DT_EPS,
                            ObsType,
                            ActType,
                            BaseObsType,
                            BaseActType,
                            InfoType,
                            ObserverHandleType,
                            ControllerHandleType,
                            EngineObsType,
                            JiminyEnvInterface)
from .block_bases import (
    EnvOrWrapperType, BaseControllerBlock, BaseObserverBlock)


OtherStateType = TypeVar("OtherStateType", bound=DataNested)
NestedObsType = TypeVar("NestedObsType", bound=Dict[str, DataNested])


# Note that `BasePipelineWrapper` must inherit from `gym.Wrapper` before
# `JiminyEnvInterface` as they both inherit from `gym.Env` but `gym.Wrapper`
# implementation is expected to take precedence.
class BasePipelineWrapper(
        gym.Wrapper,  # [ObsType, ActType, BaseObsType, BaseActType],
        JiminyEnvInterface[ObsType, ActType],
        Generic[ObsType, ActType, BaseObsType, BaseActType]):
    """Base class for wrapping a `BaseJiminyEnv` Gym environment so that it
    appears as a single, unified, environment. The environment may have been
    wrapped multiple times already.

    If several successive blocks must be composed, just wrap each of them
    successively one by one.

    .. warning::
        This architecture is not designed for trainable blocks, but rather for
        robotic-oriented controllers and observers, such as PID controllers,
        inverse kinematics, Model Predictive Control (MPC), sensor fusion...
        It is recommended to add the controllers and observers into the
        policy itself if they have to be trainable.
    """
    env: EnvOrWrapperType[BaseObsType, BaseActType]

    def __init__(self,
                 env: EnvOrWrapperType[BaseObsType, BaseActType],
                 **kwargs: Any) -> None:
        """
        :param kwargs: Extra keyword arguments for multiple inheritance.
        """
        # Initialize some proxies for fast lookup
        self.simulator = env.simulator
        self.robot = env.robot
        self.stepper_state = env.stepper_state
        self.system_state = env.system_state
        self.sensors_data = env.sensors_data

        # Manually initialize the base interfaces.
        # This is necessary because `gym.Wrapper` was not designed for multiple
        # inheritance, hence breaking `__init__` chaining.
        gym.Wrapper.__init__(self, env)
        JiminyEnvInterface.__init__(self)  # Do not forward any argument

        # By default, bind the action to the one of the base environment
        self.action = env.action

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), dir(self.env))

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    @property
    def is_training(self) -> bool:
        return self.env.is_training

    def train(self) -> None:
        self.env.train()

    def eval(self) -> None:
        self.env.eval()

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[ObsType, InfoType]:
        """Reset the unified environment.

        In practice, it resets the environment and initializes the generic
        pipeline internal buffers through the use of 'controller_hook'.

        :param controller_hook: Used internally for chaining multiple
                                `BasePipelineWrapper`. It is not meant to be
                                defined manually.
                                Optional: None by default.
        :param kwargs: Extra keyword arguments to comply with OpenAI Gym API.
        """
        # Create weak reference to self.
        # This is necessary to avoid circular reference that would make the
        # corresponding object noncollectable and hence cause a memory leak.
        pipeline_wrapper_ref = ref(self)

        # Define chained controller hook
        def reset_hook() -> Tuple[ObserverHandleType, ControllerHandleType]:
            """Register the block to the higher-level block.

            This method is used internally to make sure that `_setup` method
            of connected blocks are called in the right order, namely from the
            lowest-level block to the highest-level one, right after reset of
            the low-level simulator and just before performing the first step.
            """
            nonlocal pipeline_wrapper_ref, options

            # Extract and initialize the pipeline wrapper
            pipeline_wrapper = pipeline_wrapper_ref()
            assert pipeline_wrapper is not None
            pipeline_wrapper._setup()

            # Forward the observer and controller handles provided by the
            # controller hook of higher-level block, if any, or use the
            # ones of this block otherwise.
            observer_handle, controller_handle = None, None
            if options is not None:
                reset_hook = options.get("reset_hook")
                if reset_hook is not None:
                    assert callable(reset_hook)
                    handles = reset_hook()
                    if handles is not None:
                        observer_handle, controller_handle = handles
            if observer_handle is None:
                observer_handle = pipeline_wrapper._observer_handle
            if controller_handle is None:
                controller_handle = pipeline_wrapper._controller_handle
            return observer_handle, controller_handle

        # Reset the seed of the action and observation spaces
        if seed is not None:
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        # Reset base pipeline
        _, info = self.env.reset(seed=seed, options={"reset_hook": reset_hook})

        return self.get_observation(), info

    def step(self,
             action: Optional[ActType] = None
             ) -> Tuple[ObsType, SupportsFloat, bool, bool, InfoType]:
        """Run a simulation step for a given action.

        :param action: Next action to perform. `None` to not update it.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information.
        """
        # Backup the action to perform, if any
        if action is not None:
            set_value(self.action, action)

        # Compute the next learning step
        _, reward, done, truncated, info = self.env.step(action)

        # Compute block's reward and add it to base one
        reward = float(reward) + self.compute_reward(
            done and self.env.unwrapped._num_steps_beyond_done == 0,
            truncated and self.env.unwrapped._num_steps_beyond_done == 0,
            info)

        return self.get_observation(), reward, done, truncated, info

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

        # Refresh some proxies for fast lookup
        self.robot = self.env.robot
        self.sensors_data = self.env.sensors_data

    def compute_command(self, action: ActType) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        By default, it simply forwards the command computed by the wrapped
        environment without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)


class ObservedJiminyEnv(
        BasePipelineWrapper[NestedObsType, ActType, BaseObsType, ActType],
        Generic[NestedObsType, ActType, BaseObsType]):
    """Wrap a `BaseJiminyEnv` Gym environment and a single observer.

    .. aafig::
        :proportional:
        :textual:

                +---------+                      +--------------+
        "act"   |         |              "mes_1" |              |
        ------->+  "env"  +--------> "+" ------->+ "observer_1" +--------> "="
                |         | "obs_1"              |              | "obs_2"
                +---------+                      +--------------+

                +---------+         +--------------+
        "act"   |         | "mes_1" |              |
        ------->+  "env"  +-------->+ "observer_1" +---------------->
                |         | "obs_1" |              | "obs_1 + obs_2"
                +---------+         +--------------+

    The input 'mes_1' of the 'observer_1' must be consistent with the
    observation 'obs_1' of the environment. The observation space of the
    resulting unified environment will be the observation space of the
    highest-level observer, while its action space will be the one of the
    unwrapped environment 'env'.

    .. note::
        The observation space gathers the original observation of the
        environment with the high-level features computed by the observer
        in `gym.spaces.Dict` instance. If the original observation is already
        a `gym.spaces.Dict` instance, then the features computed by the
        observer are added to if under (nested) key ['features', observer.name]
        along with its internal state ['states', observer.name] if any.
        Otherwise, the original observation is stored under key 'measurement'.

    .. note::
        The environment and the observers all have their own update period.

    .. warning::
        This design is not suitable for learning the observer, but rather for
        robotic-oriented observers, such as sensor fusion algorithms, Kalman
        filters... It is recommended to add the observer into the policy itself
        if it has to be trainable.
    """
    def __init__(self,
                 env: EnvOrWrapperType[BaseObsType, ActType],
                 observer: BaseObserverBlock[
                     ObsType, OtherStateType, BaseObsType, ActType],
                 **kwargs: Any):
        """
        :param env: Environment to control. It can be an already controlled
                    environment wrapped in `ObservedJiminyEnv` if one desires
                    to stack several controllers with `BaseJiminyEnv`.
        :param observer: Observer to use to extract higher-level features.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Make sure that the unwrapped environment matches the observed one
        assert observer.env.unwrapped is env.unwrapped

        # Make sure that the environment pipeline does not already have one
        # block with the same name.
        block_name = observer.name
        env_wrapper: gym.Env = env
        while isinstance(env_wrapper, gym.Wrapper):
            if isinstance(env_wrapper, ObservedJiminyEnv):
                assert block_name != env_wrapper.observer.name
            if isinstance(env_wrapper, ControlledJiminyEnv):
                assert block_name != env_wrapper.controller.name
            env_wrapper = env_wrapper.env

        # Backup user arguments
        self.observer = observer

        # Initialize base wrapper
        super().__init__(env, **kwargs)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        observation_space = OrderedDict()
        base_observation_space = deepcopy(self.env.observation_space)
        if isinstance(base_observation_space, gym.spaces.Dict):
            observation_space.update(base_observation_space)
        else:
            observation_space['measurement'] = base_observation_space
        observation_space.setdefault(
            'states', gym.spaces.Dict())[
                self.observer.name] = self.observer.state_space
        observation_space.setdefault(
            'features', gym.spaces.Dict())[
                self.observer.name] = self.observer.observation_space
        self.observation_space = gym.spaces.Dict(observation_space)

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

    def refresh_observation(self, measurement: EngineObsType) -> None:
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
        # Get environment observation
        self.env.refresh_observation(measurement)

        # Update observed features if necessary
        # The part of the observation corresponding to the original observation
        # of the environment is not copied by shared with it for efficiency.
        # Because of this, it is only necessary to update its memory reference
        # at reset, just before the simulation starts.
        t = self.stepper_state.t
        if is_breakpoint(t, self.observe_dt, DT_EPS):
            base_observation = self.env.get_observation()
            self.observer.refresh_observation(base_observation)
            if not self.simulator.is_simulation_running:
                self._observation = OrderedDict()
                if isinstance(base_observation, dict):
                    # Store references of the values but not the dict itself,
                    # otherwise it will share all extra keys added later on.
                    self._observation.update(base_observation)
                    if base_features := base_observation.get('features'):
                        assert isinstance(self._observation['features'], dict)
                        self._observation['features'].update(base_features)
                    if base_states := base_observation.get('states'):
                        assert isinstance(self._observation['states'], dict)
                        self._observation['states'].update(base_states)
                else:
                    self._observation['measurement'] = base_observation
                if (state := self.observer.get_state()) is not None:
                    self._observation.setdefault('states', OrderedDict())[
                        self.observer.name] = state
                self._observation.setdefault('features', OrderedDict())[
                    self.observer.name] = self.observer.get_observation()


class ControlledJiminyEnv(
        BasePipelineWrapper[NestedObsType, ActType, BaseObsType, BaseActType],
        Generic[NestedObsType, ActType, BaseObsType, BaseActType]):
    """Wrap a `BaseJiminyEnv` Gym environment and a single controller.

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
    space 'act_X' of the subsequent block. The action space of the resulting
    unified environment will be the action space of the highest-level
    controller 'act_N', while its observation space will be the one of the
    unwrapped environment 'obs'. Alternatively, the later can also gather the
    (stacked) action space of the successive controllers if one is to observe
    the intermediary controllers' targets.

    .. note::
        The environment and the controllers all have their own update period.

    .. warning::
        This design is not suitable for learning the controllers 'ctrl_X', but
        rather for robotic-oriented pre-defined and possibly model-based
        controllers, such as PID control, inverse kinematics, admittance
        control, or Model Predictive Control (MPC). It is recommended to add
        the controllers into the policy itself if it has to be trainable.
    """  # noqa: E501  # pylint: disable=line-too-long
    def __init__(self,
                 env: EnvOrWrapperType[BaseObsType, BaseActType],
                 controller: BaseControllerBlock[
                     ActType, OtherStateType, BaseObsType, BaseActType],
                 augment_observation: bool = False,
                 **kwargs: Any):
        """
        .. note::
            As a reminder, `env.step_dt` refers to the learning step period,
            namely the timestep between two successive frames:

                [obs, reward, done, info]

            This definition remains true, independently of whether the
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
        :param augment_observation: Whether to gather the target state of the
                                    controller with the observation of the
                                    environment. Regardless, the internal state
                                    of the controller will be added if any.
                                    Optional: Disabled by default.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Make sure that the unwrapped environment matches the observed one
        assert controller.env.unwrapped is env.unwrapped

        # Backup user arguments
        self.controller = controller
        self.augment_observation = augment_observation

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Initialize the action buffer
        self.action: ActType = zeros(self.action_space)

        # Register the controller's target to the telemetry
        self.env.register_variable("action",
                                   self.action,
                                   self.controller.get_fieldnames(),
                                   self.controller.name)

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.controller.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        # Append the controller's target to the observation if requested
        observation_space = OrderedDict()
        base_observation_space = deepcopy(self.env.observation_space)
        if isinstance(base_observation_space, gym.spaces.Dict):
            observation_space.update(base_observation_space)
        else:
            observation_space['measurement'] = base_observation_space
        observation_space.setdefault(
            'states', gym.spaces.Dict())[
                self.controller.name] = self.controller.state_space
        if self.augment_observation:
            observation_space.setdefault(
                'actions', gym.spaces.Dict())[
                    self.controller.name] = self.controller.action_space
        self.observation_space = gym.spaces.Dict(observation_space)

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
        fill(self.action, 0)

        # Compute the observe and control update periods
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.controller.control_dt

    def compute_command(self, action: ActType) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        In practice, it updates whenever necessary:

            - the target sent to the subsequent block by the controller
            - the command send to the robot by the environment through the
              subsequent block

        :param action: High-level target to achieve by means of the command.
        """
        # Update the target to send to the subsequent block if necessary.
        # Note that `_observation` buffer has already been updated right before
        # calling this method by `_controller_handle`, so it can be used as
        # measure argument without issue.
        if is_breakpoint(self.stepper_state.t, self.control_dt, DT_EPS):
            target = self.controller.compute_command(action)
            set_value(self.env.action, target)

        # Update the command to send to the actuators of the robot.
        # Note that the environment itself is responsible of making sure to
        # update the command at the right period. Ultimately, this is done
        # automatically by the engine, which is calling `_controller_handle` at
        # the right period.
        return self.env.compute_command(self.env.action)

    def refresh_observation(self, measurement: EngineObsType) -> None:
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
        self.env.refresh_observation(measurement)

        # Add target and internal state to observation if requested
        if not self.simulator.is_simulation_running:
            self._observation = OrderedDict()
            base_observation = self.env.get_observation()
            if isinstance(base_observation, dict):
                # Store references of the values but not the dict itself,
                # otherwise it will share all extra keys added later on.
                self._observation.update(base_observation)
                if base_actions := base_observation.get('actions'):
                    assert isinstance(self._observation['actions'], dict)
                    self._observation['actions'].update(base_actions)
                if base_states := base_observation.get('states'):
                    assert isinstance(self._observation['states'], dict)
                    self._observation['states'].update(base_states)
            else:
                self._observation['measurement'] = base_observation
            if (state := self.controller.get_state()) is not None:
                self._observation.setdefault('states', OrderedDict())[
                    self.controller.name] = state
            if self.augment_observation:
                self._observation.setdefault('actions', OrderedDict())[
                    self.controller.name] = self.action

    def compute_reward(self,
                       done: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        return self.controller.compute_reward(done, truncated, info)
