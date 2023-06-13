"""This method gathers base implementations for blocks to be used in pipeline
control design.

It implements:

    - the concept of block that can be connected to a gym environment
    - the base controller block
    - the base observer block
"""
from itertools import chain
from abc import abstractmethod, ABC
from typing import Any, Iterable, Generic

import gymnasium as gym

from ..utils import FieldNested, get_fieldnames, fill
from ..envs import BaseJiminyEnv

from .generic_bases import (ObsType,
                            ActType,
                            BaseObsType,
                            BaseActType,
                            ControllerInterface,
                            ObserverInterface)


class BlockInterface(ABC):
    """Base class for blocks used for pipeline control design. Blocks can be
    either observers and controllers.
    """
    env: BaseJiminyEnv[ObsType, ActType]

    def __init__(self,
                 env: gym.Env[ObsType, ActType],
                 update_ratio: int = 1,
                 **kwargs: Any) -> None:
        """Initialize the block interface.

        It only allocates some attributes.

        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the high-level
                             controller and the one of the subsequent
                             lower-level controller.
        :param kwargs: Extra keyword arguments that may be useful for mixing
                       multiple inheritance through multiple inheritance.
        """
        # Make sure that the base environment inherits from `BaseJiminyEnv`
        assert isinstance(env.unwrapped, BaseJiminyEnv)

        # Backup some user arguments
        self.env = env.unwrapped
        self.update_ratio = update_ratio

        # Call super to allow mixing interfaces through multiple inheritance
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Jiminy engine directly, without having to do it through `env`.
        """
        return getattr(self.__getattribute__('env'), name)

    def __dir__(self) -> Iterable[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return chain(super().__dir__(), dir(self.env))

    @abstractmethod
    def _setup(self):
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
        ...


class BaseObserverBlock(ObserverInterface[ObsType, BaseObsType],
                        BlockInterface,
                        Generic[ObsType, BaseObsType]):
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
    def _setup(self) -> None:
        # Compute the update period
        self.observe_dt = self.env.observe_dt * self.update_ratio

        # Set default observation
        fill(self._observation, 0.0)

        # Make sure the controller period is lower than environment timestep
        assert self.observe_dt <= self.env.step_dt, (
            "The observer update period must be lower than or equal to the "
            "environment simulation timestep.")


class BaseControllerBlock(
        ControllerInterface[ObsType, ActType, BaseActType],
        BlockInterface,
        Generic[ObsType, ActType, BaseActType]):
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
    period of the environment, but both can be infinite, ie time-continuous.
    """
    def _setup(self) -> None:
        # Compute the update period
        self.control_dt = self.env.control_dt * self.update_ratio

        # Set default action
        fill(self._action, 0.0)

        # Make sure the controller period is lower than environment timestep
        assert self.control_dt <= self.env.step_dt, (
            "The controller update period must be lower than or equal to the "
            "environment simulation timestep.")

    def get_fieldnames(self) -> FieldNested:
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

    @abstractmethod
    def compute_command(self,
                        observation: ObsType,
                        target: BaseActType) -> ActType:
        """Compute the action to perform by the subsequent block, namely a
        lower-level controller, if any, or the environment to ultimately
        control, based on a given high-level action.

        .. note::
            The controller is supposed to be already fully configured whenever
            this method might be called. Thus it can only be called manually
            after `reset`. This method has to deal with the initialization of
            the internal state, but `_setup` method does so.

        :param observation: Observation of the environment.
        :param action: Target to achieve.

        :returns: Action to perform.
        """
        ...
