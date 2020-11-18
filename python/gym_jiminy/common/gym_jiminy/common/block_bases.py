"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

    - the concept of block that can be connected to a `BaseJiminyEnv`
      environment through any level of indirection
    - the base controller block
    - the base observer block
"""
from typing import Optional, Any, Union

import gym

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

from .utils import FieldDictRecursive, SpaceDictRecursive
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

    def __init__(self,
                 update_ratio: int = 1,
                 **kwargs: Any) -> None:
        """Initialize the block interface.

        It only allocates some attributes.

        :param update_ratio: Ratio between the update period of the high-level
                             controller and the one of the subsequent
                             lower-level controller.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Define some attributes
        self.env = None
        self.observation_space = None
        self.action_space = None

        # Backup some user arguments
        self.update_ratio = update_ratio

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(**kwargs)  # type: ignore[call-arg]

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
        # Backup the unwrapped environment
        if isinstance(env, gym.Wrapper):
            # Make sure the environment actually derive for BaseJiminyEnv
            assert isinstance(env.unwrapped, BaseJiminyEnv), (
                "env.unwrapped must derived from `BaseJiminyEnv`.")
            self.env = env.unwrapped
        else:
            self.env = env

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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        .. note::
            The space in which the command must be contained is completely
            determined by the action space of the next block (another
            controller or the environment to ultimately control). Thus, it does
            not have to be defined explicitely.

            On the contrary, the action space of the controller 'action_ctrl'
            is free and it is up to the user to define it.

        :param args: Extra arguments that may be useful for mixing multiple
                     inheritance through multiple inheritance, and to allow
                     automatic pipeline wrapper generation.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # pylint: disable=unused-argument

        # Initialize the block and control interface
        super().__init__(*args, **kwargs)

    def _refresh_observation_space(self) -> None:
        """Configure the observation space of the controller.

        It does nothing but to return the observation space of the environment
        since it is only affecting the action space.

        .. warning::
            This method that must not be overloaded. If one need to overload
            it, then using `BaseObserverBlock` or `BlockInterface` directly
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

        # Assertion(s) for type checker
        assert self.env is not None and self.env.control_dt is not None

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

    :param measure: Observation of the environment.
    :param action: Target to achieve.

    :returns: Action to perform
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
            it, then using `BaseControllerBlock` or `BlockInterface` directly
            is probably the way to go.
        """
        assert self.env is not None
        self.action_space = self.env.observation_space

    def reset(self, env: Union[gym.Wrapper, BaseJiminyEnv]) -> None:
        """Reset the observer for a given environment.

        In addition to the base implementation, it also set 'observe_dt'
        dynamically, based on the environment 'observe_dt'.

        :param env: Environment to observe, eventually already wrapped.
        """
        super().reset(env)

        # Assertion(s) for type checker
        assert self.env is not None and self.env.observe_dt is not None

        self.observe_dt = self.env.observe_dt * self.update_ratio

    def compute_observation(self,  # type: ignore[override]
                            measure: SpaceDictRecursive
                            ) -> SpaceDictRecursive:
        """Compute observed features based on the current simulation state and
        lower-level measure.

        :param measure: Measure from the environment to process to get
                        high-level observation.
        """
        # pylint: disable=arguments-differ

        raise NotImplementedError
