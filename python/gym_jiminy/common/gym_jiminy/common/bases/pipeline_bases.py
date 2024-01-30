"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

* the concept of block thats can be connected to a `BaseJiminyEnv` environment
  through multiple `JiminyEnvInterface` indirections
* a base controller block, along with a concrete PD controller
* a wrapper to combine a controller block and a `BaseJiminyEnv` environment,
  eventually already wrapped, so that it appears as a black-box environment.
"""
import math
from weakref import ref
from copy import deepcopy
from abc import abstractmethod
from collections import OrderedDict
from itertools import chain
from typing import (
    Dict, Any, List, Optional, Tuple, Union, Iterable, Generic, TypeVar,
    SupportsFloat, Callable, cast)

import numpy as np
import gymnasium as gym
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import EnvSpec

from ..utils import DataNested, is_breakpoint, zeros, build_copyto, copy

from .generic_bases import (DT_EPS,
                            ObsT,
                            ActT,
                            BaseObsT,
                            BaseActT,
                            InfoType,
                            EngineObsType,
                            JiminyEnvInterface)
from .block_bases import BaseControllerBlock, BaseObserverBlock


OtherObsT = TypeVar('OtherObsT', bound=DataNested)
OtherStateT = TypeVar('OtherStateT', bound=DataNested)
NestedObsT = TypeVar('NestedObsT', bound=Dict[str, DataNested])
TransformedObsT = TypeVar('TransformedObsT', bound=DataNested)
TransformedActT = TypeVar('TransformedActT', bound=DataNested)


class BasePipelineWrapper(
        JiminyEnvInterface[ObsT, ActT],
        Generic[ObsT, ActT, BaseObsT, BaseActT]):
    """Base class for wrapping a `BaseJiminyEnv` Gym environment so that it
    appears as a single, unified, environment. The environment may have been
    wrapped multiple times already.

    If several successive blocks must be composed, just wrap each of them
    successively one by one.

    .. warning::
        Hot-plug of additional blocks is supported, but the environment need to
        be reset after changing the pipeline.

    .. warning::
        This architecture is not designed for trainable blocks, but rather for
        robotic-oriented controllers and observers, such as PID controllers,
        inverse kinematics, Model Predictive Control (MPC), sensor fusion...
        It is recommended to add the controllers and observers into the
        policy itself if they have to be trainable.
    """
    env: JiminyEnvInterface[BaseObsT, BaseActT]

    def __init__(self,
                 env: JiminyEnvInterface[BaseObsT, BaseActT],
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
        self.is_simulation_running = env.is_simulation_running

        # Backup the parent environment
        self.env = env

        # Call base implementation
        super().__init__()  # Do not forward any argument

        # Define specialized operator(s) for efficiency.
        # Note that it cannot be done at this point because the action
        # may be overwritten by derived classes afterward.
        self._copyto_action: Callable[[ActT], None] = lambda action: None

    def __getattr__(self, name: str) -> Any:
        """Convenient fallback attribute getter.

        It enables to get access to the attribute and methods of the wrapped
        environment directly without having to do it through `env`.
        """
        return getattr(self.__getattribute__('env'), name)

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), dir(self.env))

    @property
    def spec(self) -> Optional[EnvSpec]:
        """Random number generator of the base environment.
        """
        return self.env.spec

    @spec.setter
    def spec(self, spec: EnvSpec) -> None:
        self.env.spec = spec

    @property
    def np_random(self) -> np.random.Generator:
        """Random number generator of the base environment.
        """
        return self.env.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator) -> None:
        self.env.np_random = value

    @property
    def unwrapped(self) -> JiminyEnvInterface:
        """Base environment of the pipeline.
        """
        return self.env.unwrapped

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

    def reset(self,  # type: ignore[override]
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[DataNested, InfoType]:
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

        # Extra reset_hook from options if provided
        derived_reset_hook: Optional[Callable[[], JiminyEnvInterface]] = (
            options or {}).get("reset_hook")

        # Define chained controller hook
        def reset_hook() -> Optional[JiminyEnvInterface]:
            """Register the block to the higher-level block.

            This method is used internally to make sure that `_setup` method
            of connected blocks are called in the right order, namely from the
            lowest-level block to the highest-level one, right after reset of
            the low-level simulator and just before performing the first step.
            """
            nonlocal pipeline_wrapper_ref, derived_reset_hook

            # Extract and initialize the pipeline wrapper
            pipeline_wrapper = pipeline_wrapper_ref()
            assert pipeline_wrapper is not None
            pipeline_wrapper._setup()

            # Forward the environment provided by the reset hook of higher-
            # level block if any, or use this wrapper otherwise.
            if derived_reset_hook is None:
                env_derived: JiminyEnvInterface = pipeline_wrapper
            else:
                assert callable(derived_reset_hook)
                env_derived = derived_reset_hook() or pipeline_wrapper

            return env_derived

        # Reset the seed of the action and observation spaces
        if seed is not None:
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        # Reset base pipeline
        return self.env.reset(seed=seed, options={"reset_hook": reset_hook})

    def step(self,  # type: ignore[override]
             action: ActT
             ) -> Tuple[DataNested, SupportsFloat, bool, bool, InfoType]:
        """Run a simulation step for a given action.

        :param action: Next action to perform. `None` to not update it.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information.
        """
        if action is not self.action:
            # Backup the action to perform for top-most layer of the pipeline
            self._copyto_action(action)

            # Make sure that the pipeline has not change since last reset
            env_derived = (
                self.unwrapped._env_derived)  # type: ignore[attr-defined]
            if env_derived is not self:
                raise RuntimeError(
                    "Pipeline environment has changed. Please call 'reset' "
                    "before 'step'.")

        # Compute the next learning step.
        # Note that forwarding 'self.env.action' enables skipping action update
        # since it is only relevant for the most derived block. For the others,
        # it will be done in 'compute_command' instead because it is unknown at
        # this point and needs to be updated only if necessary.
        obs, reward, terminated, truncated, info = self.env.step(
            self.env.action)

        # Compute block's reward and add it to base one as long as it is worth
        # doing so, namely it is not 'nan' already.
        # Note that the reward would be 'nan' if the episode is over and the
        # user keeps doing more steps nonetheless.
        reward = float(reward)
        if not math.isnan(reward):
            try:
                reward += self.compute_reward(terminated, truncated, info)
            except NotImplementedError:
                pass

        return obs, reward, terminated, truncated, info

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
        self.system_state = self.env.system_state
        self.sensors_data = self.env.sensors_data

        # Initialize specialized operator(s) for efficiency
        self._copyto_action = build_copyto(self.action)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Render the unified environment.

        By default, it does nothing but forwarding the request to the base
        environment. This behavior can be overwritten by the user.
        """
        return self.env.render()

    def close(self) -> None:
        """Closes the wrapper and its base environment.

        By default, it does nothing but forwarding the request to the base
        environment. This behavior can be overwritten by the user.
        """
        self.env.close()


class ObservedJiminyEnv(
        BasePipelineWrapper[NestedObsT, ActT, BaseObsT, ActT],
        Generic[NestedObsT, ActT, BaseObsT]):
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
                 env: JiminyEnvInterface[BaseObsT, ActT],
                 observer: BaseObserverBlock[
                     OtherObsT, OtherStateT, BaseObsT, ActT],
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

        # Backup user arguments
        self.observer = observer

        # Make sure that the environment is either some `ObservedJiminyEnv` or
        # `ControlledJiminyEnv` block, or the base environment directly.
        if isinstance(env, BasePipelineWrapper) and not isinstance(
                env, (ObservedJiminyEnv, ControlledJiminyEnv)):
            raise TypeError(
                "Observers can only be added on top of another observer, "
                "controller, or a base environment itself.")

        # Make sure that there is no other block with the exact same name
        block_name = observer.name
        env_unwrapped: JiminyEnvInterface = env
        while isinstance(env_unwrapped, BasePipelineWrapper):
            if isinstance(env_unwrapped, ObservedJiminyEnv):
                assert block_name != env_unwrapped.observer.name
            elif isinstance(env_unwrapped, ControlledJiminyEnv):
                assert block_name != env_unwrapped.controller.name
            env_unwrapped = env_unwrapped.env

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Bind action of the base environment
        assert self.action_space.contains(env.action)
        self.action = env.action

        # Initialize the observation.
        # One part is bound to the environment while the other is bound to the
        # observer. In this way, no memory at all must be allocated.
        observation: Dict[str, DataNested] = OrderedDict()
        base_observation = self.env.observation
        if isinstance(base_observation, dict):
            # Bind values but not dict itself, otherwise the base observation
            # would be altered when adding extra keys.
            observation.update(base_observation)
            if base_features := base_observation.get('features'):
                assert isinstance(observation['features'], dict)
                observation['features'] = copy(base_features)
            if base_states := base_observation.get('states'):
                assert isinstance(observation['states'], dict)
                observation['states'] = copy(base_states)
        else:
            observation['measurement'] = base_observation
        if (state := self.observer.get_state()) is not None:
            observation.setdefault(
                'states', OrderedDict())[  # type: ignore[index]
                    self.observer.name] = state
        observation.setdefault(
            'features', OrderedDict())[  # type: ignore[index]
                self.observer.name] = self.observer.observation
        self.observation = cast(NestedObsT, observation)

        # Register the observer's internal state and feature to the telemetry
        if state is not None:
            try:
                self.env.register_variable(  # type: ignore[attr-defined]
                    'state', state, None, self.observer.name)
            except ValueError:
                pass
        self.env.register_variable('feature',  # type: ignore[attr-defined]
                                   self.observer.observation,
                                   self.observer.fieldnames,
                                   self.observer.name)

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

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        observation_space: Dict[str, gym.Space[Any]] = OrderedDict()
        base_observation_space = deepcopy(self.env.observation_space)
        if isinstance(base_observation_space, gym.spaces.Dict):
            observation_space.update(base_observation_space)
        else:
            observation_space['measurement'] = base_observation_space
        if self.observer.state_space is not None:
            observation_space.setdefault(  # type: ignore[index]
                'states', gym.spaces.Dict())[
                    self.observer.name] = self.observer.state_space
        observation_space.setdefault(  # type: ignore[index]
            'features', gym.spaces.Dict())[
                self.observer.name] = self.observer.observation_space
        self.observation_space = gym.spaces.Dict(observation_space)

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It gathers the original observation from the environment with the
        features computed by the observer.

        .. note::
            Internally, it can deal with multiple observers with different
            update periods. Besides, it is safe to call this method multiple
            times successively.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        # Get environment observation
        self.env.refresh_observation(measurement)

        # Update observed features if necessary
        if is_breakpoint(self.stepper_state.t, self.observe_dt, DT_EPS):
            self.observer.refresh_observation(self.env.observation)

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)


class ControlledJiminyEnv(
        BasePipelineWrapper[NestedObsT, ActT, BaseObsT, BaseActT],
        Generic[NestedObsT, ActT, BaseObsT, BaseActT]):
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
                 env: JiminyEnvInterface[BaseObsT, BaseActT],
                 controller: BaseControllerBlock[
                     ActT, OtherStateT, BaseObsT, BaseActT],
                 augment_observation: bool = False,
                 **kwargs: Any):
        """
        .. note::
            As a reminder, `env.step_dt` refers to the learning step period,
            namely the timestep between two successive frames:

                [observation, reward, terminated, truncated, info]

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

        # Make sure that the environment is either some `ObservedJiminyEnv` or
        # `ControlledJiminyEnv` block, or the base environment directly.
        if isinstance(env, BasePipelineWrapper) and not isinstance(
                env, (ObservedJiminyEnv, ControlledJiminyEnv)):
            raise TypeError(
                "Controllers can only be added on top of another observer, "
                "controller, or a base environment itself.")

        # Make sure that the pipeline does not have a block with the same name
        block_name = controller.name
        env_unwrapped: JiminyEnvInterface = env
        while isinstance(env_unwrapped, BasePipelineWrapper):
            if isinstance(env_unwrapped, ObservedJiminyEnv):
                assert block_name != env_unwrapped.observer.name
            elif isinstance(env_unwrapped, ControlledJiminyEnv):
                assert block_name != env_unwrapped.controller.name
            env_unwrapped = env_unwrapped.env

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Define specialized operator(s) for efficiency
        self._copyto_env_action = build_copyto(self.env.action)

        # Allocate action buffer
        self.action: ActT = zeros(self.action_space)

        # Initialize the observation
        observation: Dict[str, DataNested] = OrderedDict()
        base_observation = self.env.observation
        if isinstance(base_observation, dict):
            observation.update(base_observation)
            if base_actions := base_observation.get('actions'):
                assert isinstance(observation['actions'], dict)
                observation['actions'] = copy(base_actions)
            if base_states := base_observation.get('states'):
                assert isinstance(observation['states'], dict)
                observation['states'] = copy(base_states)
        else:
            observation['measurement'] = base_observation
        if (state := self.controller.get_state()) is not None:
            observation.setdefault(
                'states', OrderedDict())[  # type: ignore[index]
                    self.controller.name] = state
        if self.augment_observation:
            observation.setdefault(
                'actions', OrderedDict())[  # type: ignore[index]
                    self.controller.name] = self.action
        self.observation = cast(NestedObsT, observation)

        # Register the controller's internal state and target to the telemetry
        if state is not None:
            try:
                self.env.register_variable(  # type: ignore[attr-defined]
                    'state', state, None, self.controller.name)
            except ValueError:
                pass
        self.env.register_variable('action',  # type: ignore[attr-defined]
                                   self.action,
                                   self.controller.fieldnames,
                                   self.controller.name)

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to the base implementation, it configures the controller
        and registers its target to the telemetry.
        """
        # Configure the controller
        self.controller._setup()

        # Call base implementation
        super()._setup()

        # Compute the observe and control update periods
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.controller.control_dt

    def _initialize_action_space(self) -> None:
        """Configure the action space.
        """
        self.action_space = self.controller.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.

        It gathers the original observation from the environment plus the
        internal state to of the controller, and optionally the target computed
        by the controller if requested.
        """
        # Append the controller's target to the observation if requested
        observation_space: Dict[str, gym.Space[Any]] = OrderedDict()
        base_observation_space = deepcopy(self.env.observation_space)
        if isinstance(base_observation_space, gym.spaces.Dict):
            observation_space.update(base_observation_space)
        else:
            observation_space['measurement'] = base_observation_space
        if self.controller.state_space is not None:
            observation_space.setdefault(  # type: ignore[index]
                'states', gym.spaces.Dict())[
                    self.controller.name] = self.controller.state_space
        if self.augment_observation:
            observation_space.setdefault(  # type: ignore[index]
                'actions', gym.spaces.Dict())[
                    self.controller.name] = self.controller.action_space
        self.observation_space = gym.spaces.Dict(observation_space)

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute the unified observation based on the current wrapped
        environment's observation and controller's target.

        It gathers the actual observation from the environment with the target
        of the controller, if requested, otherwise it forwards the observation
        directly without any further processing.

        .. warning::
            Beware it shares the environment observation whenever possible
            for the sake of efficiency. Despite that, it is safe to call this
            method multiple times successively.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        self.env.refresh_observation(measurement)

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        In practice, it updates whenever necessary:

            - the target sent to the subsequent block by the controller
            - the command send to the robot by the environment through the
              subsequent block

        :param action: High-level target to achieve by means of the command.
        """
        # Update the target to send to the subsequent block if necessary.
        # Note that `observation` buffer has already been updated right before
        # calling this method by `_controller_handle`, so it can be used as
        # measure argument without issue.
        if is_breakpoint(self.stepper_state.t, self.control_dt, DT_EPS):
            target = self.controller.compute_command(action)
            self._copyto_env_action(target)

        # Update the command to send to the actuators of the robot.
        # Note that the environment itself is responsible of making sure to
        # update the command at the right period. Ultimately, this is done
        # automatically by the engine, which is calling `_controller_handle` at
        # the right period.
        return self.env.compute_command(self.env.action)

    def compute_reward(self,
                       terminated: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        return self.controller.compute_reward(terminated, truncated, info)


class BaseTransformObservation(
        BasePipelineWrapper[TransformedObsT, ActT, ObsT, ActT],
        Generic[TransformedObsT, ObsT, ActT]):
    """Apply some transform on the observation of the wrapped environment.

    The observation transform is only applied once per step, as post-processing
    right before returning. It is meant to change the way a whole pipeline
    environment is exposed to the outside rather than changing its internal
    machinery. Incidentally, the transformed observation is not to be involved
    in the computations of any subsequent pipeline layer.

    .. note::
        The user is expected to define the observation transform and its
        corresponding space by overloading both `_initialize_action_space` and
        `transform_action`. The transform will be applied at the end of every
        environment step.

    .. note::
        This wrapper derives from `BasePipelineWrapper`, and such as, it is
        considered as internal unlike `gym.Wrapper`. This means that it will be
        taken into account calling `evaluate` or `play_interactive` on the
        wrapped environment.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Initialize some proxies for fast lookup
        self._step_dt = self.env.step_dt

        # Pre-allocated memory for the observation
        self.observation: TransformedObsT = zeros(self.observation_space)

        # Bind action of the base environment
        assert self.action_space.contains(self.env.action)
        self.action = self.env.action

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to calling the base implementation, it sets the observe
        and control update period.
        """
        # Call base implementation
        super()._setup()

        # Refresh some proxies for fast lookup
        self._step_dt = self.env.step_dt

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

    def _initialize_action_space(self) -> None:
        """Configure the action space.

        It simply copy the action space of the wrapped environment.
        """
        self.action_space = self.env.action_space

    def compute_command(self, action: ActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        """
        return self.env.compute_command(action)

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It calls `transform_observation` at `step_dt` update period, right
        after having refreshed the base observation.

        .. warning::
            The method `transform_observation` must have been overwritten by
            the user prior to calling this method.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        # Refresh observation of the base environment
        self.env.refresh_observation(measurement)

        # Transform observation at the end of the step only
        if is_breakpoint(self.stepper_state.t, self._step_dt, DT_EPS):
            self.transform_observation()

    @abstractmethod
    def transform_observation(self) -> None:
        """Compute the transformed observation from the original wrapped
        environment observation.

        .. note::
            The environment observation `self.env.observation` has been updated
            prior to calling this method and therefore can be safely accessed.

        .. note::
            For the sake of efficiency, this method should directly update
            in-place the pre-allocated transformed observation buffer
            `self.observation` instead of returning a temporary.
        """


class BaseTransformAction(
        BasePipelineWrapper[ObsT, TransformedActT, ObsT, ActT],
        Generic[TransformedActT, ObsT, ActT]):
    """Apply some transform on the action of the wrapped environment.

    The action transform is only applied once per step, as pre-processing
    right at the beginning. It is meant to change the way a whole pipeline
    environment is exposed to the outside rather than changing its internal
    machinery.

    .. note::
        The user is expected to define the observation transform and its
        corresponding space by overloading both `_initialize_action_space` and
        `transform_action`. The transform will be applied at the beginning of
        every environment step.

    .. note::
        This wrapper derives from `BasePipelineWrapper`, and such as, it is
        considered as internal unlike `gym.Wrapper`. This means that it will be
        taken into account calling `evaluate` or `play_interactive` on the
        wrapped environment.
    """
    def __init__(self, env: JiminyEnvInterface[ObsT, ActT]) -> None:
        # Initialize base class
        super().__init__(env)

        # Initialize some proxies for fast lookup
        self._step_dt = self.env.step_dt

        # Pre-allocated memory for the action
        self.action: TransformedActT = zeros(self.action_space)

        # Bind observation of the base environment
        assert self.observation_space.contains(self.env.observation)
        self.observation = self.env.observation

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to calling the base implementation, it sets the observe
        and control update period.
        """
        # Call base implementation
        super()._setup()

        # Refresh some proxies for fast lookup
        self._step_dt = self.env.step_dt

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.

        It simply copy the observation space of the wrapped environment.
        """
        self.observation_space = self.env.observation_space

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It simply forwards the observation computed by the wrapped environment
        without any processing.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        self.env.refresh_observation(measurement)

    def compute_command(self, action: TransformedActT) -> np.ndarray:
        """Compute the motors efforts to apply on the robot.

        It calls `transform_action` at `step_dt` update period, which will
        update the environment action. Then, it delegates computation of the
        command to the base environment.

        .. warning::
            The method `transform_action` must have been overwritten by the
            user prior to calling this method.

        :param action: High-level target to achieve by means of the command.
        """
        # Transform action at the beginning of the step only
        if is_breakpoint(self.stepper_state.t, self._step_dt, DT_EPS):
            self.transform_action(action)

        # Delegate command computation to wrapped environment
        return self.env.compute_command(self.env.action)

    @abstractmethod
    def transform_action(self, action: TransformedActT) -> None:
        """Compute the transformed action from the provided wrapped environment
        action.

        .. note::
            For the sake of efficiency, this method should directly update
            in-place the pre-allocated action buffer of the wrapped environment
            `self.env.action` instead of returning a temporary.
        """
