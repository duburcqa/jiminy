"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

    - the concept of block that can be connected to a `BaseJiminyEnv`
      environment through any level of indirection
    - the base controller block
    - the base observer block
"""
from typing import Optional, Any, List

import gym

from ..utils import FieldDictNested, SpaceDictNested, get_fieldnames
from ..envs import BaseJiminyEnv

from .generic_bases import ControllerInterface, ObserverInterface


class BlockInterface:
    r"""Base class for blocks used for pipeline control design.

    Block can be either observers and controllers. A block can be connected to
    any number of subsequent blocks, or directly to a `BaseJiminyEnv`
    environment.
    """
    observation_space: Optional[gym.Space]
    action_space: Optional[gym.Space]

    def __init__(self,
                 env: BaseJiminyEnv,
                 update_ratio: int = 1,
                 **kwargs: Any) -> None:
        """Initialize the block interface.

        It only allocates some attributes.

        :param env: Environment to ultimately control, ie completely unwrapped.
        :param update_ratio: Ratio between the update period of the high-level
                             controller and the one of the subsequent
                             lower-level controller.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Backup some user arguments
        self.env = env
        self.update_ratio = update_ratio

        # Define some attributes
        self.observation_space = None
        self.action_space = None

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(**kwargs)  # type: ignore[call-arg]

        # Refresh the observation and action spaces
        self._refresh_observation_space()
        self._refresh_action_space()

        # Assertion(s) for type checker
        assert (self.observation_space is not None and
                self.action_space is not None)  # type: ignore[unreachable]

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Jiminy engine directly, without having to do it through `env`.
        """
        return getattr(self.env, name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self.env.__dir__()  # type: ignore[operator]

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        """Reset the internal state of the block.

        .. note::
            The environment itself is not necessarily directly connected to
            this block since it may actually be connected through another block
            instead.

        .. note::
            It is possible to update the configuration of the simulator, for
            example to register some extra variables to monitor the internal
            state of the block.
        """

    def _refresh_observation_space(self) -> None:
        """Configure the observation of the block.

        .. note::
            The observation space refers to the output of system once connected
            with another block. For example, for a controller, it is the
            action from the next block.
        """
        raise NotImplementedError

    def _refresh_action_space(self) -> None:
        """Configure the action of the block.

        .. note::
            The action space refers to the input of the block. It does not have
            to be an actual action. For example, for an observer, it is the
            observation from the previous block.
        """
        raise NotImplementedError


class BaseObserverBlock(ObserverInterface, BlockInterface):
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
        self.action_space = self.env.observation_space

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        # Compute the update period
        self.observe_dt = self.env.observe_dt * self.update_ratio

        # Make sure the controller period is lower than environment timestep
        assert self.observe_dt <= self.env.step_dt, (
            "The observer update period must be lower than or equal to the "
            "environment simulation timestep.")

    def refresh_observation(self,  # type: ignore[override]
                            measure: SpaceDictNested) -> None:
        """Compute observed features based on the current simulation state and
        lower-level measure.

        :param measure: Measure from the environment to process to get
                        high-level observation.
        """
        # pylint: disable=arguments-differ

        raise NotImplementedError


class BaseControllerBlock(ControllerInterface, BlockInterface):
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
        self.observation_space = self.env.action_space

    # methods to override:
    # ----------------------------

    def _setup(self) -> None:
        # Compute the update period
        self.control_dt = self.env.control_dt * self.update_ratio

        # Make sure the controller period is lower than environment timestep
        assert self.control_dt <= self.env.step_dt, (
            "The controller update period must be lower than or equal to the "
            "environment simulation timestep.")

    def get_fieldnames(self) -> FieldDictNested:
        """Get mapping between each scalar element of the action space of the
        controller and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        action space, the difference being numerical arrays replaced by lists
        of string.

        By default, generic fieldnames using 'Action' prefix and index as
        suffix for `np.ndarray`.

        .. note::
            This method is not supposed to be called before `reset`, so that
            the controller should be already initialized at this point.
        """
        return get_fieldnames(self.action_space)


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
