"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

* the concept of block thats can be connected to a `BaseJiminyEnv` environment
  through multiple `InterfaceJiminyEnv` indirections
* a base controller block, along with a concrete PD controller
* a wrapper to combine a controller block and a `BaseJiminyEnv` environment,
  eventually already wrapped, so that it appears as a black-box environment.
"""
import sys
import math
import logging
from weakref import ref
from copy import deepcopy
from abc import abstractmethod
from collections import OrderedDict
from typing import (
    Dict, Any, List, Sequence, Optional, Tuple, Union, Generic, TypeVar,
    Type, Mapping, SupportsFloat, Callable, cast, overload, TYPE_CHECKING)

import numpy as np

import gymnasium as gym
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import EnvSpec

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    is_breakpoint, array_copyto)
from jiminy_py.dynamics import Trajectory
from jiminy_py.tree import issubclass_mapping

from .interfaces import (DT_EPS,
                         Obs,
                         Act,
                         BaseObs,
                         BaseAct,
                         InfoType,
                         EngineObsType,
                         InterfaceJiminyEnv)
from .compositions import AbstractReward, AbstractTerminationCondition
from .blocks import BaseControllerBlock, BaseObserverBlock

from ..utils import (DataNested,
                     zeros,
                     build_copyto,
                     copy,
                     get_robot_state_space)
if TYPE_CHECKING:
    from ..envs.generic import BaseJiminyEnv


OtherObs = TypeVar('OtherObs', bound=DataNested)
OtherState = TypeVar('OtherState', bound=DataNested)
NestedObs = TypeVar('NestedObs', bound=Dict[str, DataNested])
TransformedObs = TypeVar('TransformedObs', bound=DataNested)
TransformedAct = TypeVar('TransformedAct', bound=DataNested)

NestedSpaceOrData = Union[DataNested, gym.Space[DataNested]]


LOGGER = logging.getLogger(__name__)


@overload
def _merge_base_env_with_block(
        block_name: str,
        base_observation: DataNested,
        block_state: Optional[DataNested],
        block_feature: Optional[DataNested],
        block_action: Optional[DataNested]
        ) -> DataNested:
    ...


@overload
def _merge_base_env_with_block(
        block_name: str,
        base_observation: gym.Space[DataNested],
        block_state: Optional[gym.Space[DataNested]],
        block_feature: Optional[gym.Space[DataNested]],
        block_action: Optional[gym.Space[DataNested]]
        ) -> gym.Space[DataNested]:
    ...


def _merge_base_env_with_block(
        block_name: str,
        base_observation: NestedSpaceOrData,
        block_state: Optional[NestedSpaceOrData],
        block_feature: Optional[NestedSpaceOrData],
        block_action: Optional[NestedSpaceOrData],
        ) -> NestedSpaceOrData:
    """Merge the observation space of a base environment with the state,
    feature and action spaces of a given block.

    This method supports specifying both spaces or values for all the input
    arguments at once. In both cases, the base observation is shallow copy
    first to avoid altering it while sharing memory with the original leaves.

    If the base observation space is a mapping, then the state, feature and
    action of the block are added under nested keys ("states", block_name),
    ("feature", block_name), and ("action", block_name). Otherwise, the base
    observation is first stored under nested key ("measurement",) of a new
    mapping, while block spaces are stored under the same hierarchy as before.

    :param block_name: Name of the block. It will be used as parent key of the
                       state, feature and action spaces.
    :param base_observation: Observation space or value of the base
                             environment.
    :param block_state: State space or value of the block. `None` if it does
                        not exist for the block at hand.
    :param block_feature: Feature space or value of the block. `None` if it
                          does not exist for the block at hand.
    :param block_action: Action space or value of the block. `None` if it does
                         not exist for the block at hand.
    """
    observation: Dict[str, NestedSpaceOrData] = OrderedDict()

    # Get the right map container
    if isinstance(base_observation, gym.Space):
        mapping_cls: Type[Mapping] = gym.spaces.Dict
    else:
        mapping_cls = OrderedDict

    # Deal with the base observation
    base_observation = copy(base_observation)
    if issubclass_mapping(type(base_observation)):
        observation.update(base_observation)  # type: ignore[arg-type]
    else:
        observation['measurement'] = base_observation

    # Deal with the block state, feature and action
    for group_name, block_group in (
            ('states', block_state),
            ('features', block_feature),
            ('actions', block_action)):
        if block_group is not None:
            base_group = observation.setdefault(
                group_name, mapping_cls())
            assert issubclass_mapping(type(base_group))
            base_group[block_name] = block_group  # type: ignore[index]

    return mapping_cls(observation)  # type: ignore[call-arg]


class BasePipelineWrapper(
        InterfaceJiminyEnv[Obs, Act],
        Generic[Obs, Act, BaseObs, BaseAct]):
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
    env: InterfaceJiminyEnv[BaseObs, BaseAct]

    def __init__(self,
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 **kwargs: Any) -> None:
        """
        :param env: Base or already wrapped jiminy environment.
        :param kwargs: Extra keyword arguments for multiple inheritance.
        """
        # Initialize some proxies for fast lookup
        self.simulator = env.simulator
        self.stepper_state = env.stepper_state
        self.robot = env.robot
        self.robot_state = env.robot_state
        self.sensor_measurements = env.sensor_measurements
        self.is_simulation_running = env.is_simulation_running
        self.num_steps = env.num_steps

        # Backup the parent environment
        self.env = env

        # Whether the block is registered in a environment pipeline
        self._is_registered = False

        # Call base implementation
        super().__init__()  # Do not forward any argument

        # Enable direct forwarding by default for efficiency
        if BasePipelineWrapper.has_terminated is type(self).has_terminated:
            self.has_terminated = (  # type: ignore[method-assign]
                self.env.has_terminated)

        # Define specialized operator(s) for efficiency.
        # Note that it cannot be done at this point because the action
        # may be overwritten by derived classes afterward.
        self._copyto_action: Callable[[Act], None] = lambda action: None

    def __getattr__(self, name: str) -> Any:
        """Convenient fallback attribute getter.

        It enables to get access to the attribute and methods of the wrapped
        environment directly without having to do it through `env`.

        .. warning::
            This fallback incurs a significant runtime overhead. As such, it
            must only be used for debug and manual analysis between episodes.
            Calling this method in script mode while a simulation is already
            running would trigger a warning to avoid relying on it by mistake.
        """
        try:
            # Make sure that no simulaton is running
            if (self.__getattribute__('is_simulation_running') and
                    self.env.is_training and not hasattr(sys, 'ps1')):
                # `hasattr(sys, 'ps1')` is used to detect whether the method
                # was called from an interpreter or within a script. For
                # details, see: https://stackoverflow.com/a/64523765/4820605
                LOGGER.warning(
                    "Relying on fallback attribute getter is inefficient and "
                    "strongly discouraged at runtime.")

            # Ensure that the block that has been declared as top-most layer of
            # the environment pipeline is consistent with the callee of this
            # method, i.e. `self.env.derived` is a parent of `self`. If not,
            # set the callee as parent if no simulation is running. Otherwise,
            # raise an exception.
            if not self.__getattribute__('_is_registered'):
                if self.is_simulation_running:
                    raise RuntimeError(
                        "This block is not registered as part of the "
                        "environment pipeline. Please stop the simulation "
                        "before adding new blocks.")
                self.update_pipeline(self)
        except AttributeError:
            # The block is not fully initialized at this point. Skipping check.
            pass

        return getattr(self.__getattribute__('env'), name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return [*super().__dir__(), *dir(self.env)]

    @property
    def render_mode(self) -> Optional[str]:
        """Rendering mode of the base environment.
        """
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, render_mode: str) -> None:
        self.env.render_mode = render_mode

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
    def unwrapped(self) -> "BaseJiminyEnv":
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

    def update_pipeline(self, derived: Optional[InterfaceJiminyEnv]) -> None:
        if derived is None:
            self._is_registered = False
            self.env.update_pipeline(None)
        else:
            self.unwrapped.update_pipeline(None)
            assert not self._is_registered
            self.env.update_pipeline(derived)
            self._is_registered = True

    def reset(self,  # type: ignore[override]
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[DataNested, InfoType]:
        """Reset the unified environment.

        In practice, it resets the environment and initializes the generic
        pipeline internal buffers through the use of 'controller_hook'.

        :param seed: Random seed, as a positive integer.
                     Optional: `None` by default. If `None`, then the internal
                     random generator of the environment will be kept as-is,
                     without updating its seed.
        :param options: Additional information to specify how the environment
                        is reset. The field 'reset_hook' is reserved for
                        chaining multiple `BasePipelineWrapper`. It is not
                        meant to be defined manually.
                        Optional: None by default.
        """
        # Create weak reference to self.
        # This is necessary to avoid circular reference that would make the
        # corresponding object noncollectable and hence cause a memory leak.
        pipeline_wrapper_ref = ref(self)

        # Extra reset_hook from options if provided
        derived_reset_hook: Optional[Callable[[], InterfaceJiminyEnv]] = (
            options or {}).get("reset_hook")

        # Define chained controller hook
        def reset_hook() -> Optional[InterfaceJiminyEnv]:
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
                env_derived: InterfaceJiminyEnv = pipeline_wrapper
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
             action: Act
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
            env_derived = self.unwrapped.derived
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
            reward += self.compute_reward(terminated, info)

        return obs, reward, terminated, truncated, info

    def stop(self) -> None:
        self.env.stop()

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
        self.robot_state = self.env.robot_state
        self.sensor_measurements = self.env.sensor_measurements

        # Initialize specialized operator(s) for efficiency
        self._copyto_action = build_copyto(self.action)

    def has_terminated(self, info: InfoType) -> Tuple[bool, bool]:
        """Determine whether the episode is over, because a terminal state of
        the underlying MDP has been reached or an aborting condition outside
        the scope of the MDP has been triggered.

        By default, it does nothing but forwarding the request to the base
        environment. This behavior can be overwritten by the user.

        .. note::
            This method is called after `refresh_observation`, so that the
            internal buffer 'observation' is up-to-date.

        :param info: Dictionary of extra information for monitoring.

        :returns: terminated and truncated flags.
        """
        return self.env.has_terminated(info)

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


class ComposedJiminyEnv(BasePipelineWrapper[Obs, Act, Obs, Act],
                        Generic[Obs, Act]):
    """Extend an environment, eventually already wrapped, by plugging ad-hoc
    reward components and termination conditions, including their accompanying
    trajectory database if any.

    This wrappers optionally adds the current state of the reference trajectory
    to the observation space under nested key ('states', 'reference'), while
    leaving its action space unchanged. Transformation of the observation and
    action space is done via additional observation and/or control blocks.

    .. note::
        This wrapper derives from `BasePipelineWrapper`, and such as, it is
        considered as part of the environment pipeline unlike `gym.Wrapper`.
        This means that it will be taken into account when calling `evaluate`
        or `play_interactive` on the wrapped environment.

    .. warning::
        Setting 'augment_observation=True' enforces several restriction on the
        trajectory database to make sure that the observation space remains
        invariant. First, the database is locked, so that no trajectory can be
        added nor removed anymore. Then, the robot model must be the same for
        all the trajectories.

    .. warning::
        This class is final, ie not meant to be derived.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv[Obs, Act],
                 *,
                 reward: Optional[AbstractReward] = None,
                 terminations: Sequence[AbstractTerminationCondition] = (),
                 trajectories: Optional[Dict[str, Trajectory]] = None,
                 augment_observation: bool = False) -> None:
        """
        :param env: Environment to extend, eventually already wrapped.
        :param reward: Reward object deriving from `AbstractReward`. It will be
                       evaluated at each step of the environment and summed up
                       with the one returned by the wrapped environment. This
                       reward must be already instantiated and associated with
                       the provided environment. `None` for not considering any
                       reward.
                       Optional: `None` by default.
        :param terminations: Sequence of termination condition objects deriving
                             from `AbstractTerminationCondition`. They will be
                             checked along with the one enforced by the wrapped
                             environment. If provided, these termination
                             conditions must be already instantiated and
                             associated with the environment at hand.
                             Optional: Empty sequence by default.
        :param trajectories: Ordered set of named trajectories as a dictionary.
                             The first trajectory being specified, in any, will
                             be selected as reference by default. `None` to
                             skip the whole process.
                             Optional: `None` by default.
        :param augment_observation: Whether to add the current state of the
                                    reference trajectory to the observation
                                    of the environment if any.
                                    Optional: False by default.
        """
        # Make sure that the unwrapped environment of compositions matches
        assert reward is None or env.unwrapped is reward.env.unwrapped
        assert all(env.unwrapped is termination.env.unwrapped
                   for termination in terminations)

        # Backup user argument(s)
        self.augment_observation = augment_observation
        self.reward = reward
        self.terminations = tuple(terminations)

        # Keep track of the "global" trajectory database
        self._trajectory_dataset = env.quantities.trajectory_dataset
        self._trajectory_optional_fields: Tuple[str, ...] = ()

        # Handling of reference trajectories if any
        if trajectories:
            # Add reference trajectories to managed quantities
            for name, trajectory in trajectories.items():
                self._trajectory_dataset.add(name, trajectory)

            # Select the first trajectory with 'raise' mode by default
            if not self._trajectory_dataset.name:
                name = next(iter(trajectories.keys()))
                self._trajectory_dataset.select(name, "raise")

        # Enforces some restrictions on the trajectory database if necessary
        if self.augment_observation:
            # Lock the dataset at this point
            self._trajectory_dataset.lock()

            # Make sure that the robot model is identical for all trajectories
            traj = self._trajectory_dataset.trajectory
            for traj_ in self._trajectory_dataset:
                if traj.robot.pinocchio_model != traj_.robot.pinocchio_model:
                    raise ValueError(
                        "The robot model must be identical for all "
                        "trajectories in the dataset.")

            # Determine the state information that are common to all
            # trajectories, not just the one being selected.
            self._trajectory_optional_fields = traj.optional_fields
            for traj_ in self._trajectory_dataset:
                self._trajectory_optional_fields = tuple(
                    field for field in traj_.optional_fields
                    if field in self._trajectory_optional_fields)

        # Initialize base class
        super().__init__(env)

        # Enable direct forwarding by default for efficiency
        if ComposedJiminyEnv.compute_command is type(self).compute_command:
            self.compute_command = (  # type: ignore[method-assign]
                self.env.compute_command)

        # Bind action of the base environment
        assert self.action_space.contains(env.action)
        self.action = env.action

        # Allocate memory for the trajectory state if necessary
        self._trajectory_state: Optional[Dict[str, np.ndarray]] = None
        if self.augment_observation and self._trajectory_dataset:
            self._trajectory_state = zeros(self.observation_space[
                "states"]["reference"])  # type: ignore[index]

        # Initialize the observation
        self.observation = cast(Obs, _merge_base_env_with_block(
            "reference",
            self.env.observation,
            self._trajectory_state,
            None,
            None))

    def _initialize_action_space(self) -> None:
        """Configure the action space.

        It simply copy the action space of the wrapped environment.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        # Define the trajectory space if necessary
        trajectory_space: Optional[gym.spaces.Dict] = None
        if self.augment_observation and self._trajectory_dataset:
            state_space: Dict[str, gym.Space] = OrderedDict()
            traj = self._trajectory_dataset.trajectory
            robot_state_space = get_robot_state_space(traj.robot)
            state_space["q"] = robot_state_space["q"]
            if "v" in self._trajectory_optional_fields:
                state_space["v"] = robot_state_space["v"]
            if "a" in self._trajectory_optional_fields:
                state_space["a"] = deepcopy(robot_state_space["v"])
            if "u" in self._trajectory_optional_fields:
                state_space["u"] = deepcopy(robot_state_space["v"])
            if "command" in self._trajectory_optional_fields:
                command_limit = np.array([
                    motor.effort_limit for motor in traj.robot.motors])
                state_space["command"] = gym.spaces.Box(
                    low=-command_limit, high=command_limit, dtype=np.float64)
            if "f_external" in self._trajectory_optional_fields:
                state_space["f_external"] = gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=(traj.robot.pinocchio_model.njoints, 6),
                    dtype=np.float64)
            if "lambda_c" in self._trajectory_optional_fields:
                length_lambda_c = len(traj.robot.log_constraint_fieldnames)
                state_space["lambda_c"] = gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=(length_lambda_c,),
                    dtype=np.float64)
            trajectory_space = gym.spaces.Dict(state_space)

        # Aggregate the reference trajectory space with the base observation
        self.observation_space = cast(
            gym.Space[Obs], _merge_base_env_with_block(
                "reference",
                self.env.observation_space,
                trajectory_space,
                None,
                None))

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to calling the base implementation, it sets the observe
        and control update period.
        """
        # Call base implementation
        super()._setup()

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It simply forwards the observation computed by the wrapped environment
        without any processing.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        # Refresh environment observation
        self.env.refresh_observation(measurement)

        # Update trajectory reference state if necessary
        if self._trajectory_state is not None:
            trajectory_state = self._trajectory_dataset.get()
            array_copyto(self._trajectory_state["q"], trajectory_state.q)
            for field in self._trajectory_optional_fields:
                array_copyto(self._trajectory_state[field],
                             getattr(trajectory_state, field))

    def has_terminated(self, info: InfoType) -> Tuple[bool, bool]:
        """Determine whether the practitioner is instructed to stop the ongoing
        episode on the spot because a termination condition has been triggered,
        either coming from the based environment or from the ad-hoc termination
        conditions that has been plugged on top of it.

        At each step of the wrapped environment, all its termination conditions
        will be evaluated sequentially until one of them eventually gets
        triggered. If this happens, evaluation is skipped for the remaining
        ones and the reward is evaluated straight away. Ultimately, the
        practitioner is instructed to stop the ongoing episode, but it is his
        own responsibility to honor this request. The first condition being
        evaluated is the one of the underlying environment, then comes the ones
        of this composition layer, following the original sequence ordering.

        .. note::
            This method is called after `refresh_observation`, so that the
            internal buffer 'observation' is up-to-date.

        .. seealso::
            See `InterfaceJiminyEnv.has_terminated` documentation for details.

        :param info: Dictionary of extra information for monitoring.

        :returns: terminated and truncated flags.
        """
        # Call unwrapped environment implementation
        terminated, truncated = self.env.has_terminated(info)

        # Evaluate conditions one-by-one as long as none has been triggered
        for termination in self.terminations:
            if terminated or truncated:
                break
            terminated, truncated = termination(info)

        return terminated, truncated

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to updated in-place.
        """
        self.env.compute_command(action, command)

    def compute_reward(self, terminated: bool, info: InfoType) -> float:
        """Compute the sum of the ad-hoc reward components that have been
        plugged on top of the wrapped environment.

        .. seealso::
            See `InterfaceController.compute_reward` documentation for details.

        :param terminated: Whether the episode has reached the terminal state
                           of the MDP at the current step. This flag can be
                           used to compute a specific terminal reward.
        :param info: Dictionary of extra information for monitoring.

        :returns: Aggregated reward for the current step.
        """
        # Early return if no composed reward is defined
        if self.reward is None:
            return 0.0

        # Evaluated and return composed reward
        return self.reward(terminated, info)


class ObservedJiminyEnv(
        BasePipelineWrapper[NestedObs, Act, BaseObs, Act],
        Generic[NestedObs, Act, BaseObs]):
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
                 env: InterfaceJiminyEnv[BaseObs, Act],
                 observer: BaseObserverBlock[
                     OtherObs, OtherState, BaseObs, Act],
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
        if isinstance(env, BasePipelineWrapper) and not isinstance(env, (
                ObservedJiminyEnv, ControlledJiminyEnv, ComposedJiminyEnv)):
            raise TypeError(
                "Observers can only be added on top of another observer, "
                "controller, or a base environment itself.")

        # Make sure that there is no other block with the exact same name
        block_name = observer.name
        env_unwrapped: InterfaceJiminyEnv = env
        while isinstance(env_unwrapped, BasePipelineWrapper):
            if isinstance(env_unwrapped, ObservedJiminyEnv):
                assert block_name != env_unwrapped.observer.name
            elif isinstance(env_unwrapped, ControlledJiminyEnv):
                assert block_name != env_unwrapped.controller.name
            env_unwrapped = env_unwrapped.env

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Enable direct forwarding by default for efficiency
        if ObservedJiminyEnv.compute_command is type(self).compute_command:
            self.compute_command = (  # type: ignore[method-assign]
                self.env.compute_command)

        # Bind action of the base environment
        assert self.action_space.contains(env.action)
        self.action = env.action

        # Initialize the observation
        state = self.observer.get_state()
        self.observation = cast(NestedObs, _merge_base_env_with_block(
            self.observer.name,
            self.env.observation,
            state,
            self.observer.observation,
            None))

        # Register the observer's internal state and feature to the telemetry
        if state is not None:
            try:
                self.unwrapped.register_variable(
                    'state', state, None, self.observer.name)
            except ValueError:
                pass
        self.unwrapped.register_variable('feature',
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
        self.observation_space = cast(
            gym.Space[NestedObs], _merge_base_env_with_block(
                self.observer.name,
                self.env.observation_space,
                self.observer.state_space,
                self.observer.observation_space,
                None))

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
        # Refresh environment observation
        self.env.refresh_observation(measurement)

        # Update observed features if necessary
        if is_breakpoint(self.stepper_state, self.observe_dt, DT_EPS):
            self.observer.refresh_observation(self.env.observation)

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to updated in-place.
        """
        self.env.compute_command(action, command)


class ControlledJiminyEnv(
        BasePipelineWrapper[NestedObs, Act, BaseObs, BaseAct],
        Generic[NestedObs, Act, BaseObs, BaseAct]):
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
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 controller: BaseControllerBlock[
                     Act, OtherState, BaseObs, BaseAct],
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
        :param augment_observation: Whether to add the target state of the
                                    controller to the observation of the
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
        if isinstance(env, BasePipelineWrapper) and not isinstance(env, (
                ObservedJiminyEnv, ControlledJiminyEnv, ComposedJiminyEnv)):
            raise TypeError(
                "Controllers can only be added on top of another observer, "
                "controller, or a base environment itself.")

        # Make sure that the pipeline does not have a block with the same name
        block_name = controller.name
        env_unwrapped: InterfaceJiminyEnv = env
        while isinstance(env_unwrapped, BasePipelineWrapper):
            if isinstance(env_unwrapped, ObservedJiminyEnv):
                assert block_name != env_unwrapped.observer.name
            elif isinstance(env_unwrapped, ControlledJiminyEnv):
                assert block_name != env_unwrapped.controller.name
            env_unwrapped = env_unwrapped.env

        # Initialize base wrapper
        super().__init__(env, **kwargs)

        # Enable direct forwarding by default for efficiency
        if (ControlledJiminyEnv.refresh_observation is
                type(self).refresh_observation):
            self.refresh_observation = (  # type: ignore[method-assign]
                self.env.refresh_observation)

        # Allocate action buffer
        self.action: Act = zeros(self.action_space)

        # Initialize the observation
        state = self.controller.get_state()
        self.observation = cast(NestedObs, _merge_base_env_with_block(
            self.controller.name,
            self.env.observation,
            state,
            None,
            self.action if self.augment_observation else None))

        # Register the controller's internal state and target to the telemetry
        if state is not None:
            try:
                self.unwrapped.register_variable(
                    'state', state, None, self.controller.name)
            except ValueError:
                pass
        self.unwrapped.register_variable('action',
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
        self.observation_space = cast(
            gym.Space[NestedObs], _merge_base_env_with_block(
                self.controller.name,
                self.env.observation_space,
                self.controller.state_space,
                None,
                (self.controller.action_space
                    if self.augment_observation else None)))

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

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        In practice, it updates whenever necessary:

            - the target sent to the subsequent block by the controller
            - the command send to the robot by the environment through the
              subsequent block

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to update in-place.
        """
        # Update the target to send to the subsequent block if necessary.
        # Note that `observation` buffer has already been updated right before
        # calling this method by `_controller_handle`, so it can be used as
        # measure argument without issue.
        if is_breakpoint(self.stepper_state, self.control_dt, DT_EPS):
            self.controller.compute_command(action, self.env.action)

        # Update the command to send to the actuators of the robot.
        # Note that the environment itself is responsible of making sure to
        # update the command at the right period. Ultimately, this is done
        # automatically by the engine, which is calling `_controller_handle` at
        # the right period.
        self.env.compute_command(self.env.action, command)

    def compute_reward(self, terminated: bool, info: InfoType) -> float:
        return self.controller.compute_reward(terminated, info)


class BaseTransformObservation(
        BasePipelineWrapper[TransformedObs, Act, Obs, Act],
        Generic[TransformedObs, Obs, Act]):
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
        considered internal unlike `gym.Wrapper`. This means that it will be
        taken into account when calling `evaluate` or `play_interactive` on the
        wrapped environment.
    """
    def __init__(self, env: InterfaceJiminyEnv[Obs, Act]) -> None:
        # Initialize base class
        super().__init__(env)

        # Enable direct forwarding by default for efficiency
        if (BaseTransformObservation.compute_command is
                type(self).compute_command):
            self.compute_command = (  # type: ignore[method-assign]
                self.env.compute_command)

        # Define base env proxies for fast access
        self._step_dt = self.env.step_dt

        # Pre-allocated memory for the observation
        self.observation = zeros(self.observation_space)

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

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

    def _initialize_action_space(self) -> None:
        """Configure the action space.

        It simply copy the action space of the wrapped environment.
        """
        self.action_space = self.env.action_space

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to update in-place.
        """
        self.env.compute_command(action, command)

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
        if is_breakpoint(self.stepper_state, self._step_dt, DT_EPS):
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
        BasePipelineWrapper[Obs, TransformedAct, Obs, Act],
        Generic[TransformedAct, Obs, Act]):
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
        considered internal unlike `gym.Wrapper`. This means that it will be
        taken into account when calling `evaluate` or `play_interactive` on the
        wrapped environment.
    """
    def __init__(self, env: InterfaceJiminyEnv[Obs, Act]) -> None:
        # Initialize base class
        super().__init__(env)

        # Enable direct forwarding by default for efficiency
        if (BaseTransformAction.refresh_observation is
                type(self).refresh_observation):
            self.refresh_observation = (  # type: ignore[method-assign]
                self.env.refresh_observation)

        # Initialize some proxies for fast access
        self._step_dt = self.env.step_dt

        # Pre-allocated memory for the action
        self.action: TransformedAct = zeros(self.action_space)

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

    def compute_command(self,
                        action: TransformedAct,
                        command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        It calls `transform_action` at `step_dt` update period, which will
        update the environment action. Then, it delegates computation of the
        command to the base environment.

        .. warning::
            The method `transform_action` must have been overwritten by the
            user prior to calling this method.

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to update in-place.
        """
        # Transform action at the beginning of the step only
        if is_breakpoint(self.stepper_state, self._step_dt, DT_EPS):
            self.transform_action(action)

        # Delegate command computation to wrapped environment
        self.env.compute_command(self.env.action, command)

    @abstractmethod
    def transform_action(self, action: TransformedAct) -> None:
        """Compute the transformed action from the provided wrapped environment
        action.

        .. note::
            For the sake of efficiency, this method should directly update
            in-place the pre-allocated action buffer of the wrapped environment
            `self.env.action` instead of returning a temporary.
        """
