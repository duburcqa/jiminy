
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
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, Any, Type, List

import numpy as np
import gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from .utils import (
    _is_breakpoint, zeros, set_value, register_variables,
    SpaceDictRecursive, FieldDictRecursive)
from .generic_bases import ControlInterface, ObserveInterface
from .env_bases import BaseJiminyEnv
from .block_bases import BlockInterface, BaseControllerBlock, BaseObserverBlock


class BasePipelineWrapper(gym.Wrapper, ControlInterface, ObserveInterface):
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
                 augment_observation: bool = False) -> None:
        """
        :param augment_observation: Whether or not to augment the observation
                                    of the environment with information
                                    provided by the wrapped block. What it
                                    means in practice depends on the type of
                                    the wrapped block.
                                    Optional: disable by default.
        """
        # Initialize base wrapper and interfaces through multiple inheritance
        super().__init__(env)

        # Backup some user arguments
        self.augment_observation = augment_observation

    def _get_block_index(self):
        """Get the index of the block. It corresponds the "deepness" of the
        block, namely how many blocks deriving from the same wrapper type than
        the current one are already wrapped in the environment.
        """
        # Assertion(s) for type checker
        assert self.env is not None

        i = 0
        env = self.env
        while isinstance(self.env, self.__class__):
            i += 1
            env = env.env
        return i


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

                      +-----------------------------------+
                      |                                   |
                      v                                   |
                +-----+----+         +----------+         |   +---------+
        "act_3" |          | "act_2" |          | "act_1" |   |         |
        ------->+ "ctrl_2" +-------->+ "ctrl_1" +---------o-->+  "env"  +--o------------------------->
                |          | "cmd_2" |          | "cmd_1"     |         |  | "obs or {obs + [cmd_1...] + [cmd_2...]}"
                +----------+         +-----+----+             +---------+  |
                                           ^                               |
                                           |                               |
                                           +-------------------------------+

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
                 augment_observation: bool = False):
        """
        .. note::
            As a reminder, `env.dt` refers to the learning step period,
            namely the timestep between two successive samples:

                [obs, reward, done, info]

            This definition remains true, independently of whether or not the
            environment is wrapped with a controller using this class. On the
            contrary, `env.control_dt` corresponds to the apparent control
            update period, namely the update period of the higher-level
            controller if multiple are piped together.

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
                                    Optional: disable by default.
        """
        # Initialize base wrapper
        super().__init__(env, augment_observation)

        # Backup user arguments
        self.controller = controller
        self._target: Optional[SpaceDictRecursive] = None
        self._command: Optional[np.ndarray] = None
        self._observation_env: Optional[SpaceDictRecursive] = None
        self._dt_eps: Optional[float] = None
        self._controller_name: Optional[str] = None

        # Reset the unified environment
        self.reset()

    def compute_command(self,
                        action: SpaceDictRecursive
                        ) -> SpaceDictRecursive:
        """Compute the motors efforts to apply on the robot.

        In practice, it updates, whenever it is necessary:

            - the target sent to the subsequent block by the controller
            - the command send to the robot by the environment through the
              subsequent block

        .. warning::
            This method is not meant to be overloaded.

        :param action: Next high-level action to perform.
        """
        # Assertion(s) for type checker
        assert self.simulator is not None and self._command is not None

        # Get the current time
        t = self.simulator.stepper_state.t

        # Update the target to send to the subsequent block if necessary
        if _is_breakpoint(t, self.control_dt, self._dt_eps):
            set_value(self._target, self.controller.compute_command(action))

        # Update the command to send to the actuators of the robot if necessary
        if _is_breakpoint(t, self.env.control_dt, self._dt_eps):
            self._command[:] = self.env.compute_command(self._target)

        return self._command

    def fetch_obs(self) -> SpaceDictRecursive:
        """Compute the unified observation based on the current wrapped
        environment's observation and controller's target.

        It gathers the actual observation from the environment with the target
        of the controller, if requested, otherwise it forwards the observation
        directly without any further processing.

        .. warning::
            Beware it updates and returns the internal buffer of environment
            observation '_observation_env' whenever it is possible for the sake
            of efficiency. Even so, it is always safe to call this method
            multiple times successively.

        :returns: Updated environment's observation with the controller's
                  target appended.
        """
        obs = self._observation_env
        if self.augment_observation:
            # Assertion(s) for type checker
            assert isinstance(obs, dict)

            obs.setdefault('targets', OrderedDict())[
                self._controller_name] = self._action
        return obs

    def reset(self, **kwargs: Any) -> SpaceDictRecursive:
        """Reset the unified environment.

        In practice, it resets first the wrapped environment, next comes the
        controller, the observation space, and finally the low-level simulator
        controller.

        :param kwargs: Extra keyword arguments to match the standard OpenAI gym
                       Wrapper API.
        """
        # pylint: disable=unused-argument

        # Assertion(s) for type checker
        assert self.simulator is not None

        # Reset the environment
        self._observation_env = self.env.reset()

        # Assertion(s) for type checker
        assert (self.env.control_dt is not None and
                self.env.action_space is not None and
                self.env.observation_space is not None)

        # Reset the controller
        self.controller.reset(self.env)
        self.control_dt = self.controller.control_dt

        # Assertion(s) for type checker
        assert self.control_dt is not None

        # Make sure the controller period is lower than environment timestep
        assert self.control_dt <= self.env.unwrapped.dt, (
            "The control update period must be lower than or equal to the "
            "environment simulation timestep.")

        # Set the controller name, based on the controller index
        self._controller_name = f"controller_{self._get_block_index()}"

        # Update the action space
        self.action_space = self.controller.action_space

        # Assertion(s) for type checker
        assert self.action_space is not None

        # Initialize the controller's input action and output target
        self._action = zeros(self.action_space)
        self._target = self.controller.compute_command(self._action)

        # Check that the initial action of the controller is consistent with
        # the action space of the environment.
        assert self.env.action_space.contains(self._target), (
            "The command is not consistent with the action space of the "
            "subsequent block.")

        # Initialize the command to apply on the robot
        self._command = self.env.compute_command(self._target)

        # Backup the temporal resolution of simulator steps
        engine_options = self.simulator.engine.get_options()
        self._dt_eps = 1.0 / engine_options["telemetry"]["timeUnit"]

        # Enforce the low-level controller.
        # Note that altering the original controller of the wrapped environment
        # is possible since it is systematically re-initialized at reset. So
        # one can restore a valid state for the environment after unwrapping it
        # simply calling `reset` method.
        self.simulator.controller.set_controller_handle(self._send_command)

        # Register the controller target to the telemetry.
        # It may be useful later for computing the terminal reward or debug.
        register_variables(
            self.simulator.controller, self.controller.get_fieldnames(),
            self._action, self._controller_name)

        # Check that 'augment_observation' can be enabled
        assert not self.augment_observation or isinstance(
            self.env.observation_space, gym.spaces.Dict), (
            "'augment_observation' is only available for environments whose "
            "observation space inherits from `gym.spaces.Dict`.")

        # Append the controller's target to the observation if requested
        self.observation_space = self.env.observation_space
        if self.augment_observation:
            self.observation_space.spaces.setdefault(
                'targets', gym.spaces.Dict())[self._controller_name] = \
                    self.controller.action_space

        # Compute the unified observation
        self._observation = self.fetch_obs()

        return self.get_obs()

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Next action to perform. `None` to not update it.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information
        """
        # Backup the action to perform, if any
        if action is not None:
            set_value(self._action, action)

        # Compute the next learning step
        self._observation_env, reward, done, info = self.env.step()

        # Compute controller's rewards and sum it to total reward
        reward += self.controller.compute_reward(info=info)
        if self.controller.enable_reward_terminal:
            if done and self.env.unwrapped._num_steps_beyond_done == 0:
                reward += self.controller.compute_reward_terminal(info=info)

        # Compute the unified observation
        self._observation = self.fetch_obs()

        return self.get_obs(), reward, done, info


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
                 augment_observation: bool = False):
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
                                    Optional: disable by default.
        """
        # Initialize base wrapper
        super().__init__(env, augment_observation)

        # Backup user arguments
        self.observer = observer

        # Reset the unified environment
        self.reset()

    def compute_command(self,
                        action: SpaceDictRecursive
                        ) -> SpaceDictRecursive:
        """Compute the motors efforts to apply on the robot.

        In practice, it forwards the command computed by the environment
        itself.

        .. warning::
            This method is not meant to be overloaded.

        :param action: Next high-level action to perform.
        """
        return self.env.compute_command(action)


    def fetch_obs(self) -> SpaceDictRecursive:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It gathers the original observation from the environment with the
        features computed by the observer, if requested, otherwise it forwards
        the features directly without any further processing.

        .. warning::
            Beware it updates and returns the internal buffer of environment
            observation '_observation_env' whenever it is possible for the sake
            of efficiency. Even so, it is always safe to call this method
            multiple times successively.

        :returns: Updated environment's observation with the controller's
                  target appended.
        """
        obs_features = self.observer.fetch_obs(self._observation_env)
        if self.augment_observation:
            obs = self._observation_env

            # Assertion(s) for type checker
            assert isinstance(obs, dict)

            obs.setdefault('features', OrderedDict())[
                self._observer_name] = obs_features
        else:
            obs = obs_features
        return obs

    def reset(self, **kwargs: Any) -> SpaceDictRecursive:
        """Reset the unified environment.

        In practice, it resets first the wrapped environment, next comes the
        controller, the observation space, and finally the low-level simulator
        controller.

        :param kwargs: Extra keyword arguments to match the standard OpenAI gym
                       Wrapper API.
        """
        # pylint: disable=unused-argument

        # Reset the environment
        self._observation_env = self.env.reset()

        # Assertion(s) for type checker
        assert (self.env.action_space is not None and
                self.env.observation_space is not None)

        # Update the action space
        self.action_space = self.env.action_space

        # Initialize the environment's action and command
        self._action = zeros(self.action_space)
        self._command = self.env.compute_command(self._action)

        # Reset the observer
        self.observer.reset(self.env)

        # Set the controller name, based on the controller index
        self._observer_name = f"observer_{self._get_block_index()}"

        # Assertion(s) for type checker
        assert (self.observer.action_space is not None and
                self.observer.observation_space is not None)

        # Check that the initial observation of the environment is consistent
        # with the action space of the observer.
        assert self.observer.action_space.contains(self._observation_env), (
            "The command is not consistent with the action space of the "
            "subsequent block.")

        # Update the observation space
        if self.augment_observation:
            self.observation_space = self.env.observation_space
            self.observation_space.spaces.setdefault(
                'features', gym.spaces.Dict())[self._observer_name] = \
                    self.observer.observation_space
        else:
            self.observation_space = self.observer.observation_space

        # Compute the unified observation
        self._observation = self.fetch_obs()
        return self.get_obs()

    def step(self,
             action: Optional[np.ndarray] = None
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        """Run a simulation step for a given action.

        :param action: Next action to perform. `None` to not update it.

        :returns: Next observation, reward, status of the episode (done or
                  not), and a dictionary of extra information
        """
        # Compute the next learning step
        self._observation_env, reward, done, info = self.env.step(action)

        # Compute the unified observation
        self._observation = self.fetch_obs()

        return self.get_obs(), reward, done, info


def build_pipeline(env_config: Tuple[
                       Type[BaseJiminyEnv],
                       Dict[str, Any]],
                   controllers_config: Optional[List[Tuple[
                       Type[BaseControllerBlock],
                       Dict[str, Any],
                       Dict[str, Any]]]] = None,
                   observers_config: Optional[List[Tuple[
                       Type[BaseObserverBlock],
                       Dict[str, Any],
                       Dict[str, Any]]]] = None,
                   ) -> Type[BasePipelineWrapper]:
    """Wrap together an environment inheriting from `BaseJiminyEnv` with any
    number of controllers and observers as a unified pipeline environment class
    inheriting from `BasePipelineWrapper`.

    Each controller and observers are wrapped individually, successively. The
    controllers are wrapped first, using `ControlledJiminyEnv`. Then comes the
    observers, using `ObservedJiminyEnv`, so that intermediary controllers
    targets are always available if requested.

    :param env_config:
        Configuration of the environment, as a tuple:

          - [0] Environment class type.
          - [1] Environment constructor default arguments.

    :param controllers_config:
        Configuration of the controllers, as a list. The list is ordered from
        the lowest level controller to the highest, each element corresponding
        to the configuration of a individual controller, as a tuple:

          - [0] Controller class type.
          - [1] Controller constructor default arguments.
          - [2] `ControlledJiminyEnv` constructor default arguments.

    :param observers_config:
        Configuration of the observers, as a list. The list is ordered from
        the lowest level observer to the highest, each element corresponding
        to the configuration of a individual observer, as a tuple:

          - [0] Observer class type.
          - [1] Observer constructor default arguments.
          - [2] `ObservedJiminyEnv` constructor default arguments.
    """
    # pylint: disable-all

    def _build_wrapper(env_class: Type[Union[gym.Wrapper, BaseJiminyEnv]],
                       env_kwargs_default: Optional[Dict[str, Any]],
                       block_class: Type[BlockInterface],
                       block_kwargs_default: Dict[str, Any],
                       wrapper_class: Type[BasePipelineWrapper],
                       wrapper_kwargs_default: Dict[str, Any]
                       ) -> Type[ControlledJiminyEnv]:
        """Generate a class inheriting from 'wrapper_class' and wrapping a
        given type of environment and block together.

        .. warning::
            Beware of the collision between the keywords arguments of the
            wrapped environment and block. It would be impossible to
            overwrite their default values independently.

        :param env_class: Type of environment to wrap.
        :param env_kwargs_default: Keyword arguments to forward to the
                                   constructor of the wrapped environment. Note
                                   that it will only overwrite the default
                                   value, and it will still be possible to set
                                   different values by explicitly defining them
                                   when calling the constructor of the
                                   generated wrapper.
        :param block_class: Type of block to connect to the environment.
        :param block_kwargs_default: Keyword arguments to forward to the
                                     constructor of the wrapped block.
                                     See 'env_kwargs_default'.
        :param wrapper_class: Type of wrapper to use to gather the environment
                              and the block.
        :param wrapper_kwargs_default: Keyword arguments to forward to the
                                       constructor of the wrapper.
                                       See 'env_kwargs_default'.
        """
        # pylint: disable-all

        wrapped_env_class = type(
            f"{block_class.__name__}{env_class.__name__}",  # Class name
            (wrapper_class,),  # Bases
            {})  # methods (__init__ cannot be implemented this way, cf below)

        # Implementation of __init__ method must be done after declaration of
        # the class, because the required closure for calling `super()` is not
        # available when creating a class dynamically.
        def __init__(self: wrapper_class, **kwargs: Any) -> None:
            """
            :param kwargs: Keyword arguments to forward to both the wrapped
                           environment and the controller. It will overwrite
                           default values.
            """
            if env_kwargs_default is not None:
                env_kwargs = {**env_kwargs_default, **kwargs}
            else:
                env_kwargs = kwargs
            env = env_class(**env_kwargs)
            block = block_class(**{**block_kwargs_default, **kwargs})
            super(wrapped_env_class, self).__init__(  # type: ignore[arg-type]
                env, block, **{**wrapper_kwargs_default, **kwargs})

        wrapped_env_class.__init__ = __init__  # type: ignore[misc]

        return wrapped_env_class

    env_class, env_kwargs = env_config
    pipeline_class = env_class
    for (ctrl_class, ctrl_kwargs, wrapper_kwargs) in controllers_config:
        pipeline_class = _build_wrapper(
            pipeline_class, env_kwargs, ctrl_class, ctrl_kwargs,
            ControlledJiminyEnv, wrapper_kwargs)
        env_kwargs = None
    for (obs_class, obs_kwargs, wrapper_kwargs) in observers_config:
        pipeline_class = _build_wrapper(
            pipeline_class, env_kwargs, obs_class, obs_kwargs,
            ObservedJiminyEnv, wrapper_kwargs)
        env_kwargs = None
    return pipeline_class
