"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

    - the concept of blocks that can be connected to a gym environment
    - the base controller block
    - the base observer block
"""
from abc import abstractmethod, ABC
from typing import Any, Union, Generic, TypeVar, cast

import gymnasium as gym

from ..utils import FieldNested, DataNested, get_fieldnames, fill, zeros

from .generic_bases import (ObsT,
                            ActT,
                            BaseObsT,
                            BaseActT,
                            ControllerInterface,
                            ObserverInterface,
                            JiminyEnvInterface)


BlockStateT = TypeVar('BlockStateT', bound=Union[DataNested, None])


class BlockInterface(ABC, Generic[BlockStateT, BaseObsT, BaseActT]):
    """Base class for blocks used for pipeline control design. Blocks can be
    either observers and controllers.

    .. warning::
        A block may be stateful. In such a case, `_initialize_state_space`
        and `get_state` must be overloaded accordingly. The internal state will
        be added automatically to the observation space of the environment.
    """
    env: JiminyEnvInterface[BaseObsT, BaseActT]
    name: str
    update_ratio: int
    state_space: gym.Space[BlockStateT]

    # Type of the block, ie 'observer' or 'controller'.
    type: str = ""

    def __init__(self,
                 name: str,
                 env: JiminyEnvInterface[BaseObsT, BaseActT],
                 update_ratio: int = 1,
                 **kwargs: Any) -> None:
        """Initialize the block interface.

        It defines some proxies for fast access, then it initializes the
        internal state space of the block and allocates memory for it.

        ..warning::
            All blocks (observers and controllers) must be an unique name
            within a given pipeline. In practice, it will be impossible to plug
            a given block to an existing pipeline if the later already has one
            block of the same type and name. The user is responsible to take
            care it never happens.

        :param name: Name of the block.
        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the high-level
                             controller and the one of the subsequent
                             lower-level controller.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Make sure that the provided environment is valid
        assert isinstance(env.unwrapped, JiminyEnvInterface)

        # Backup some user argument(s)
        self.env = env
        self.name = name
        self.update_ratio = update_ratio

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(**kwargs)

        # Refresh the observation space
        self._initialize_state_space()

    @abstractmethod
    def _setup(self) -> None:
        """Configure the internal state of the block.

        .. note::
            The environment itself is not necessarily directly connected to
            this block since it may actually be connected through another block
            instead.

        .. note::
            The environment to ultimately control is already fully initialized
            at this point, so that all its internal buffers is up-to-date, but
            no simulation is running yet. As a result, it is still possible to
            update the configuration of the simulator, and for example, to
            register some extra variables to monitor the internal state of the
            block.
        """

    def _initialize_state_space(self) -> None:
        """Configure the internal state space of the controller.
        """
        self.state_space = cast(gym.Space[BlockStateT], None)

    def get_state(self) -> BlockStateT:
        """Get the internal state space of the controller.
        """
        return cast(BlockStateT, None)

    @property
    @abstractmethod
    def fieldnames(self) -> FieldNested:
        """Blocks fieldnames for logging.
        """


class BaseObserverBlock(ObserverInterface[ObsT, BaseObsT],
                        BlockInterface[BlockStateT, BaseObsT, BaseActT],
                        Generic[ObsT, BlockStateT, BaseObsT, BaseActT]):
    """Base class to implement observe that can be used compute observation
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
        """Initialize the observer interface.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

        # Allocate observation buffer
        self.observation: ObsT = zeros(self.observation_space)

    def _setup(self) -> None:
        # Compute the update period
        self.observe_dt = self.env.observe_dt * self.update_ratio

        # Set default observation
        fill(self.observation, 0)

        # Make sure the controller period is lower than environment timestep
        assert self.observe_dt <= self.env.step_dt, (
            "The observer update period must be lower than or equal to the "
            "environment simulation timestep.")

    @property
    def get_fieldnames(self) -> FieldNested:
        """Get mapping between each scalar element of the observation space of
        the observer block and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        observation space, but having lists of string as leaves. Generic
        fieldnames are used by default.
        """
        return get_fieldnames(self.observation_space)


class BaseControllerBlock(
        ControllerInterface[ActT, BaseActT],
        BlockInterface[BlockStateT, BaseObsT, BaseActT],
        Generic[ActT, BlockStateT, BaseObsT, BaseActT]):
    """Base class to implement controller that can be used compute targets to
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
    'act_ctrl' to the action space of the subsequent controller 'cmd_ctrl', if
    any, and ultimately to the one of the associated environment 'act_env', ie
    the motors efforts to apply on the robot.

    The update period of the controller must be higher than the control update
    period of the environment, but both can be infinite, i.e. time-continuous.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the controller interface.

        .. note::
            No buffer is pre-allocated for the action since it is already done
            by the parent environment.

        :param args: Extra arguments that may be useful for mixing
                     multiple inheritance through multiple inheritance.
        :param kwargs: Extra keyword arguments. See 'args'.
        """
        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        # Compute the update period
        self.control_dt = self.env.control_dt * self.update_ratio

        # Make sure the controller period is lower than environment timestep
        assert self.control_dt <= self.env.step_dt, (
            "The controller update period must be lower than or equal to the "
            "environment simulation timestep.")

    @property
    def get_fieldnames(self) -> FieldNested:
        """Get mapping between each scalar element of the action space of
        the controller block and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        action space, but having lists of string as leaves. Generic fieldnames
        are used by default.
        """
        return get_fieldnames(self.action_space)


BaseControllerBlock.compute_command.__doc__ = \
    """Compute the action to perform by the subsequent block, namely a
    lower-level controller, if any, or the environment to ultimately
    control, based on a given high-level action.

    .. note::
        The controller is supposed to be already fully configured whenever
        this method might be called. Thus it can only be called manually
        after `reset`. This method has to deal with the initialization of
        the internal state, but `_setup` method does so.

    .. note::
        The user is expected to fetch by itself the observation of the
        environment if necessary to carry out its computations by calling
        `self.env.observation`. Beware it will NOT contain any information
        provided by higher-level blocks in the pipeline.

    :param target: Target to achieve by means of the output action.

    :returns: Action to perform.
    """
