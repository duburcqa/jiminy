"""This module promotes reward components as first-class objects.

Defining rewards this way allows for standardization of usual metrics. Overall,
it greatly reduces code duplication and bugs.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Callable, Optional, Tuple, Type

import numpy as np

from ..bases import InterfaceJiminyEnv, QuantityCreator, InfoType


ValueT = TypeVar('ValueT')


class AbstractReward(ABC):
    """Abstract class from which all reward component must derived.

    This goal of the agent is to maximize the expectation of the cumulative sum
    of discounted reward over complete episodes. This holds true no matter if
    its sign is always negative (aka. reward), always positive (aka. cost) or
    indefinite (aka. objective).

    Defining cost is allowed by not recommended. Although it encourages the
    agent to achieve the task at hands as quickly as possible if success if the
    only termination condition, it has the side-effect to give the opportunity
    to the agent to maximize the return by killing itself whenever this is an
    option, which is rarely the desired behavior. No restriction is enforced as
    it may be limiting in some relevant cases, so it is up to the user to make
    sure that its design makes sense overall.
    """

    def __init__(self, env: InterfaceJiminyEnv) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        """
        self.env = env

    @property
    @abstractmethod
    def name(self) -> str:
        """Name uniquely identifying a given reward component.
        """

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """Whether the reward is terminal, ie only evaluated at the end of an
        episode if a termination condition has been triggered.

        .. note::
            Truncation is not consider the same as termination. The reward to
            not be evaluated in such a case, which means that it will never be
            for such episodes.
        """

    @property
    @abstractmethod
    def is_normalized(self) -> bool:
        """Whether the reward is guaranteed to be normalized, ie it is in range
        [0.0, 1.0].
        """

    @abstractmethod
    def __call__(self, terminated: bool, info: InfoType) -> float:
        """Evaluate the reward for the current state of the environment.
        """


class BaseQuantityReward(AbstractReward):
    """Base class that makes easy easy to derive reward components from generic
    quantities.

    All this class does is applying some user-specified post-processing to the
    value of a given multi-variate quantity to return a floating-point scalar
    value, eventually normalized between 0.0 and 1.0 if desired.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 quantity: QuantityCreator[ValueT],
                 transform_fun: Optional[Callable[[ValueT], float]],
                 is_normalized: bool,
                 is_terminal: bool) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the reward. This name will be used as key
                     for storing current value of the reward in 'info', and to
                     add the underlying quantity to the set of already managed
                     quantities by the environment. As a result, it must be
                     unique otherwise an exception will be raised.
        :param quantity: Tuple gathering the class of the underlying quantity
                         to use as reward after some post-processing, plus all
                         its constructor keyword-arguments except environment
                         'env' and parent 'parent.
        :param transform_fun: Transform function responsible for aggregating a
                              multi-variate quantity as floating-point scalar
                              value to maximize. Typical examples are `np.min`,
                              `np.max`, `lambda x: np.linalg.norm(x, order=N)`.
                              This function is also responsible for rescaling
                              the transformed quantity in range [0.0, 1.0] if
                              the reward is advertised as normalized. The
                              Radial Basis Function (RBF) kernel is the most
                              common choice to derive a reward to maximize from
                              errors based on distance metrics (See
                              `radial_basis_function` for details.). `None` to
                              skip transform entirely if not necessary.
        :param is_terminal: Whether the reward is terminal. A terminal reward
                            will only be evaluated at most once, at the end of
                            each episode for which a termination condition has
                            been triggered. On the contrary, a non-terminal
                            reward will be evaluated systematically except at
                            the end of the episode. The value 0.0 is returned
                            and 'info' is not filled when reward evaluation is
                            skipped.
        """
        # Backup user argument(s)
        self._name = name
        self._transform_fun = transform_fun
        self._is_normalized = is_normalized
        self._is_terminal = is_terminal

        # Call base implementation
        super().__init__(env)

        # Add quantity to the set of quantities managed by the environment
        self.env.quantities[self.name] = quantity

    @property
    def name(self) -> str:
        """Name uniquely identifying every reward. It will be used to add the
        underlying quantity to the ones already managed by the environment.

        .. warning::
            It must be prefixed by "reward_" as a risk mitigation for name
            collision with some other user-defined quantity.
        """
        return self._name

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    def __call__(self, terminated: bool, info: InfoType) -> float:
        # Early return depending on whether the reward and state are terminal
        if terminated ^ self.is_terminal:
            return 0.0

        # Evaluate raw quantity
        value = self.env.quantities[self.name]

        # Apply some post-processing if requested
        if self._transform_fun is not None:
            value = self._transform_fun(value)
        assert np.ndim(value) == 0
        if self._is_normalized and (value < 0.0 or value > 1.0):
            raise ValueError(
                "Reward not normalized in range [0.0, 1.0] as it ought to be.")

        # Store its value as info
        info[self.name] = value

        # Returning the reward
        return value


RewardCreator = Tuple[Type[AbstractReward], Dict[str, Any]]
