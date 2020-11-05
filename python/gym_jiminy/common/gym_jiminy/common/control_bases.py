"""This method gathers base implementations for pipeline control design.

It implements:

    - the concept of block that can be connected to a `BaseJiminyEnv`
      environment through any level of indirection
    - a base controller block, along with a concret PD controller
    - a wrapper to combine a controller block and a `BaseJiminyEnv`
      environment, eventually already wrapped, so that it appears as a single,
      unified environment.
"""
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, Any, Type

import numpy as np
import gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from .utils import (
    _is_breakpoint, zeros, set_value, register_variables,
    SpaceDictRecursive, FieldDictRecursive)
from .generic_bases import ControlInterface, ObserveInterface
from .env_bases import BaseJiminyEnv


class BlockInterface:
    r"""Base class for blocks used for pipeline control design.

    Block can be either observers and controllers. A block can be connected to
    any number of subsequent blocks, or directly to a `BaseJiminyEnv`
    environment.
    """
    env: Optional[BaseJiminyEnv]
    observation_space: Optional[gym.Space]
    action_space: Optional[gym.Space]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the block interface.

        It only allocates some attributes.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Define some attributes
        self.env = None
        self.observation_space = None
        self.action_space = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    @property
    def robot(self) -> jiminy.Robot:
        """Get low-level Jiminy robot of the associated environment.
        """
        if self.env is None:
            raise RuntimeError("Associated environment undefined.")
        return self.env.robot

    @property
    def simulator(self) -> Simulator:
        """Get low-level simulator of the associated environment.
        """
        if self.env is None:
            raise RuntimeError("Associated environment undefined.")
        if self.env.simulator is None:
            raise RuntimeError("Associated environment not initialized.")
        return self.env.simulator

    @property
    def system_state(self) -> jiminy.SystemState:
        """Get low-level engine system state of the associated environment.
        """
        return self.simulator.engine.system_state

    def reset(self, env: Union[gym.Wrapper, BaseJiminyEnv]) -> None:
        """Reset the block for a given environment, eventually already wrapped.

        .. note::
            The environment itself is not necessarily directly connected to
            this block since it may actually be connected to another block
            instead.

        .. warning::
            This method that must not be overloaded. `_setup` is the
            unique entry-point to customize the block's initialization.

        :param env: Environment.
        """
        # Make sure the environment actually derive for BaseJiminyEnv
        assert isinstance(env.unwrapped, BaseJiminyEnv), (
            "env.unwrapped must derived from `BaseJiminyEnv`.")

        # Backup the unwrapped environment
        self.env = env.unwrapped

        # Configure the block
        self._setup()

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._refresh_action_space()

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        """Configure the block.

        .. note::
            Note that the environment `env` has already been fully initialized
            at this point, so that each of its internal buffers is up-to-date,
            but the simulation is not running yet. As a result, it is still
            possible to update the configuration of the simulator, and for
            example, to register some extra variables to monitor the internal
            state of the block.
        """

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the block.

        .. note::
            The observation space refers to the output of system once connected
            with another block. For example, for a controller, it is the
            action from the next block.

        .. note::
            This method is called right after `_setup`, so that both the
            environment and this block should be already initialized.
        """
        raise NotImplementedError

    def _refresh_action_space(self) -> None:
        """Configure the action of the block.

        .. note::
            The action space refers to the input of the block. It does not have
            to be an actual action. For example, for an observer, it is the
            observation from the previous block.

        .. note::
            This method is called right after `_setup`, so that both the
            environment and this block should be already initialized.
        """
        raise NotImplementedError


class BaseControllerBlock(BlockInterface, ControlInterface):
    r"""Base class to implement controller that can be used compute targets to
    apply to the robot of a `BaseJiminyEnv` environment, through any number of
    lower-level controllers.

    .. aafig::
        :proportional:
        :textual:

                   +----------+
        "act_ctrl" |          |
         --------->+  "ctrl"  +--------->
                   |          | "cmd_ctrl / act_env"
                   +----------+

    Formally, a controller is defined as a block mapping any action space
    'act_ctrl' to the action space of the subsequent controller 'cmd_ctrl',
    if any, and ultimately to the one of the associated environment 'act_env',
    ie the motors efforts to apply on the robot.

    The update period of the controller must be higher than the control update
    period of the environment, but both can be infinite, ie time-continuous.
    """
    def __init__(self, update_ratio: int = 1, **kwargs: Any) -> None:
        """
        .. note::
            The space in which the command must be contained is completely
            determined by the action space of the next block (another
            controller or the environment to ultimately control). Thus, it does
            not have to be defined explicitely.

            On the contrary, the action space of the controller 'action_ctrl'
            is free and it is up to the user to define it.

        :param update_ratio: Ratio between the update period of the high-level
                             controller and the one of the subsequent
                             lower-level controller.
        :param kwargs: Extra keyword arguments that may be useful for dervied
                       controller with multiple inheritance, and to allow
                       automatic pipeline wrapper generation.
        """
        # pylint: disable=unused-argument

        # Initialize the block and control interface
        super().__init__()

        # Backup some user arguments
        self.update_ratio = update_ratio

    def _refresh_observation_space(self) -> None:
        """Configure the observation space of the controller.

        It does nothing but to return the observation space of the environment
        since it is only affecting the action space.

        .. warning::
            This method that must not be overloaded. If one need to overload
            it, when using `BaseObserverBlock` or `BlockInterface` directly
            is probably the way to go.
        """
        assert self.env is not None
        self.observation_space = self.env.action_space

    def reset(self, env: Union[gym.Wrapper, BaseJiminyEnv]) -> None:
        """Reset the controller for a given environment.

        In addition to the base implementation, it also set 'control_dt'
        dynamically, based on the environment 'control_dt'.

        :param env: Environment to control, eventually already wrapped.
        """
        super().reset(env)
        self.control_dt = self.env.control_dt * self.update_ratio

    # methods to override:
    # ----------------------------

    def get_fieldnames(self) -> FieldDictRecursive:
        """Get mapping between each scalar element of the action space of the
        controller and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        action space, the difference being numerical arrays replaced by lists
        of string.

        .. note::
            This method is not supposed to be called before `reset`, so that
            the controller should be already initialized at this point.
        """
        raise NotImplementedError


BaseControllerBlock._setup.__doc__ = \
    """Configure the controller.

    It includes:

        - refreshing the action space of the controller
        - allocating memory of the controller's internal state and
          initializing it

    .. note::
        Note that the environment to ultimately control `env` has already
        been fully initialized at this point, so that each of its internal
        buffers is up-to-date, but the simulation is not running yet. As a
        result, it is still possible to update the configuration of the
        simulator, and for example, to register some extra variables to
        monitor the internal state of the controller.
    """

BaseControllerBlock._refresh_action_space.__doc__ = \
    """Configure the action space of the controller.

    .. note::
        This method is called right after `_setup`, so that both the
        environment to control and the controller itself should be already
        initialized.
    """

BaseControllerBlock.compute_command.__doc__ = \
    """Compute the action to perform by the subsequent block, namely a
    lower-level controller, if any, or the environment to ultimately
    control, based on a given high-level action.

    .. note::
        The controller is supposed to be already fully configured whenever
        this method might be called. Thus it can only be called manually
        after `reset`.  This method has to deal with the initialization of
        the internal state, but `_setup` method does so.

    :param action: Action to perform.
    """


class BaseObserverBlock(BlockInterface, ObserveInterface):
    r"""Base class to implement observe that can be used compute observation
    features of a `BaseJiminyEnv` environment, through any number of
    lower-level observer.

    .. aafig::
        :proportional:
        :textual:

                  +------------+
        "obs_env" |            |
         -------->+ "observer" +--------->
                  |            | "features"
                  +------------+

    Formally, an observer is a defined as a block mapping the observation space
    of the preceding observer, if any, and directly the one of the environment
    'obs_env', to any observation space 'features'. It is more generic than
    estimating the state of the robot.

    The update period of the observer is the same than the simulation timestep
    of the environment for now.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param kwargs: Extra keyword arguments that may be useful for dervied
                       observer with multiple inheritance, and to allow
                       automatic pipeline wrapper generation.
        """
        # pylint: disable=unused-argument

        # Initialize the block and observe interface
        super().__init__(*args, **kwargs)

    def _refresh_action_space(self) -> None:
        """Configure the action space of the observer.

        It does nothing but to return the action space of the environment
        since it is only affecting the observation space.

        .. warning::
            This method that must not be overloaded. If one need to overload
            it, when using `BaseControllerBlock` or `BlockInterface` directly
            is probably the way to go.
        """
        assert self.env is not None
        self.action_space = self.env.observation_space


class ControlledJiminyEnv(gym.Wrapper, ControlInterface, ObserveInterface):
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
        The environment and each controller has its own step period. The global
        step period of the pipe will be their GCD (Greatest Common Divisor).

    .. warning::
        This design is not suitable for learning the controllers 'ctrl_X', but
        rather for robotic-oriented controllers, such as PID control, inverse
        kinematics, admittance control, or Model Predictive Control (MPC). It
        is recommended to add the controllers into the policy itself if it has
        to be trainable.
    """  # noqa: E501  # pylint: disable=line-too-long
    env: Union[gym.Wrapper, BaseJiminyEnv]
    observation_space: Optional[gym.Space]

    def __init__(self,
                 env: Union[gym.Wrapper, BaseJiminyEnv],
                 controller: BaseControllerBlock,
                 observe_target: bool = False):
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
        :param observe_target: Whether or not to gather the target of the
                               controller with the observation of the
                               environment. This option is only available if
                               the observation space is of type
                               `gym.spaces.Dict` or `gym.spaces.Box`.
                               Optional: disable by default.
        """
        # Initialize base wrapper and interfaces through multiple inheritance
        super().__init__(env)

        # Backup user arguments
        self.controller = controller
        self.observe_target = observe_target
        self.debug: Optional[bool] = None
        self._target: Optional[SpaceDictRecursive] = None
        self._command: Optional[np.ndarray] = None
        self._observation_env: Optional[SpaceDictRecursive] = None
        self._dt_eps: Optional[float] = None
        self._ctrl_name: Optional[str] = None

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
        if self.observe_target:
            # Assertion(s) for type checker
            assert isinstance(obs, dict)

            obs.setdefault('targets', OrderedDict())[
                self._ctrl_name] = self._action
        return obs

    def reset(self, **kwargs: Any) -> SpaceDictRecursive:
        """Reset the unified environment.

        In practice, it resets first the wrapped environment, next comes the
        controller, the observation space, and finally the low-level simulator
        controller.
        """
        # pylint: disable=unused-argument

        # Assertion(s) for type checker
        assert self.simulator is not None

        # Reset the environment
        self._observation_env = self.env.reset()
        self.debug = self.env.debug

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

        # Set the controller name.
        # It corresponds to the number of subsequent control blocks.
        i = 0
        env = self.env
        while isinstance(self.env, ControlledJiminyEnv):
            i += 1
            env = env.env
        self._ctrl_name = f"ctrl_{i}"

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
            self._action, self._ctrl_name)

        # Check that 'observe_target' can be enabled
        assert not self.observe_target or isinstance(
            self.env.observation_space, gym.spaces.Dict), (
            "'observe_target' is only available for environments whose "
            "observation space inherits from `gym.spaces.Dict`.")

        # Append the controller's target to the observation if requested
        self.observation_space = self.env.observation_space
        if self.observe_target:
            self.observation_space.spaces['targets'] = \
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


def build_controlled_env(env_class: Type[Union[gym.Wrapper, BaseJiminyEnv]],
                         controller_class: Type[BaseControllerBlock],
                         observe_target: bool = False,
                         **kwargs_default: Any
                         ) -> Type[ControlledJiminyEnv]:
    """Generate a class inheriting from `ControlledJiminyEnv` and wrapping a
    given type of environment and controller together.

    .. warning::
        The arguments are passed to the environment and controller constructors
        by keyword only. So make sure there is no collision between their
        keywords arguments.

    :param env_class: Type of environment to control.
    :param controller_class: Type of controller to use to send targets to the
                             subsequent block.
    :param observe_target: Whether or not to gather the target of the
                           controller with the observation of the environment.
                           This option is only available if the observation
                           space is of type `gym.spaces.Dict` or
                           `gym.spaces.Box`.
                           Optional: disable by default.
    :param kwargs_default: Keyword arguments to forward systematically to the
                           the constructor of both the wrapped environment and
                           the controller. Note that it will only overwrite the
                           default value, and it will still be possible to set
                           different values by explicitly defining them when
                           calling the constructor of the generated wrapper.
    """
    # pylint: disable-all

    controlled_env_class = type(
        f"{controller_class.__name__}{env_class.__name__}",  # Class name
        (ControlledJiminyEnv,),  # Bases
        {})  # methods (__init__ cannot be implemented this way, cf below)

    # Implementation of __init__ method must be done after declaration of the
    # class, because the required closure for calling `super()` is not
    # available when creating a class dynamically.
    def __init__(self: ControlledJiminyEnv, **kwargs: Any) -> None:
        """
        :param kwargs: Keyword arguments to forward to both the wrapped
                       environment and the controller. It will overwrite
                       default values defined when creating this class, if any.
        """
        kwargs = {**kwargs_default, **kwargs}
        env = env_class(**kwargs)
        controller = controller_class(**kwargs)
        super(controlled_env_class, self).__init__(  # type: ignore[arg-type]
            env, controller, observe_target)

    controlled_env_class.__init__ = __init__  # type: ignore[misc]

    return controlled_env_class
