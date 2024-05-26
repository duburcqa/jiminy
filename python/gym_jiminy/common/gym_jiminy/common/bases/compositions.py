"""This module promotes reward components as first-class objects.

Defining rewards this way allows for standardization of usual metrics. Overall,
it greatly reduces code duplication and bugs.
"""
from abc import ABC, abstractmethod
from typing import Sequence, Callable, Optional, Tuple, TypeVar

import numpy as np

from .interfaces import InfoType, InterfaceJiminyEnv
from .quantities import QuantityCreator


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

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the reward.
        """
        self.env = env
        self._name = name

    @property
    def name(self) -> str:
        """Name uniquely identifying a given reward component.

        This name will be used as key for storing reward-specific monitoring
        information in 'info' if key is missing, otherwise it will raise an
        exception.
        """
        return self._name

    @property
    def is_terminal(self) -> Optional[bool]:
        """Whether the reward is terminal, non-terminal, or indefinite.

        A reward is said to be "terminal" if only evaluated for the terminal
        state of the MDP, "non-terminal" if evaluated for all states except the
        terminal one, or indefinite if systematically evaluated no matter what.

        All rewards are supposed to be indefinite unless stated otherwise by
        overloading this method. The responsibility of evaluating the reward
        only when necessary is delegated to `compute`. This allows for complex
        evaluation logics beyond terminal or non-terminal without restriction.

        .. note::
            Truncation is not consider the same as termination. The reward to
            not be evaluated in such a case, which means that it will never be
            for such episodes.
        """
        return None

    @property
    @abstractmethod
    def is_normalized(self) -> bool:
        """Whether the reward is guaranteed to be normalized, ie it is in range
        [0.0, 1.0].
        """

    @abstractmethod
    def compute(self, terminated: bool, info: InfoType) -> Optional[float]:
        """Compute the reward.

        .. note::
            Return value can be set to `None` to indicate that evaluation was
            skipped for some reason, and therefore the reward must not be taken
            into account when computing the total reward. This is useful when
            the reward is undefined or simply inappropriate in the current
            state of the environment.

        .. warning::
            It is the responsibility of the practitioner overloading this
            method to honor flags 'is_terminated' (if not indefinite) and
            'is_normalized'. Failing this, an exception will be raised.

        :returns: Scalar value if the reward was evaluated, `None` otherwise.
        """

    def __call__(self, terminated: bool, info: InfoType) -> float:
        """Return the reward associated with the current environment step.

        For the corresponding MDP to be stationary, the computation of the
        reward is supposed to involve only the transition from previous to
        current state of the environment under the ongoing action.

        .. note::
            This method is a lightweight wrapper around `compute` to skip
            evaluation depending on whether the current state and the reward
            are terminal. If the reward was truly evaluated, then 'info' is
            updated to store either reward-specific 'info' if any or its value
            otherwise. If not, then 'info' is left as-is and 0.0 is returned.

        .. warning::
            This method is not meant to be overloaded.

        :param terminated: Whether the episode has reached a terminal state of
                           the MDP at the current step.
        :param info: Dictionary of extra information for monitoring. It will be
                     updated in-place for storing current value of the reward
                     in 'info' if it was truly evaluated.
        """
        # Evaluate the reward and store extra information
        reward_info: InfoType = {}
        value = self.compute(terminated, reward_info)

        # Early return if None, which means that the reward was not evaluated
        if value is None:
            return 0.0

        # Make sure that terminal flag is honored
        if self.is_terminal is not None and self.is_terminal ^ terminated:
            raise ValueError("Flag 'is_terminal' not honored.")

        # Make sure that the reward is scalar
        assert np.ndim(value) == 0

        # Make sure that the reward is normalized
        if self.is_normalized and (value < 0.0 or value > 1.0):
            raise ValueError(
                "Reward not normalized in range [0.0, 1.0] as it ought to be.")

        # Store its value as info
        if self.name is info.keys():
            raise KeyError(
                f"Key '{self.name}' already reserved in 'info'. Impossible to "
                "store value of reward component.")
        if reward_info:
            info[self.name] = reward_info
        else:
            info[self.name] = value

        # Returning the reward
        return value


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
                 transform_fn: Optional[Callable[[ValueT], float]],
                 is_normalized: bool,
                 is_terminal: Optional[bool]) -> None:
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
        :param transform_fn: Transform function responsible for aggregating a
                             multi-variate quantity as floating-point scalar
                             value to maximize. Typical examples are `np.min`,
                             `np.max`, `lambda x: np.linalg.norm(x, order=N)`.
                             This function is also responsible for rescaling
                             the transformed quantity in range [0.0, 1.0] if
                             the reward is advertised as normalized. The Radial
                             Basis Function (RBF) kernel is the most common
                             choice to derive a reward to maximize from errors
                             based on distance metrics (See
                             `radial_basis_function` for details.). `None` to
                             skip transform entirely if not necessary.
        :param is_normalized: Whether the reward is guaranteed to be normalized
                              after applying transform function `transform_fn`.
        :param is_terminal: Whether the reward is terminal, non-terminal or
                            indefinite. A terminal reward will be evaluated at
                            most once, at the end of each episode for which a
                            termination condition has been triggered. On the
                            contrary, a non-terminal reward will be evaluated
                            systematically except at the end of the episode.
                            Finally, a indefinite reward will be evaluated
                            systematically. The value 0.0 is returned and no
                            'info' will be stored when reward evaluation is
                            skipped.
        """
        # Backup user argument(s)
        self._transform_fn = transform_fn
        self._is_normalized = is_normalized
        self._is_terminal = is_terminal

        # Call base implementation
        super().__init__(env, name)

        # Add quantity to the set of quantities managed by the environment
        self.env.quantities[self.name] = quantity

        # Keep track of the underlying quantity
        self.quantity = self.env.quantities.registry[self.name]

    def __del__(self) -> None:
        try:
            del self.env.quantities[self.name]
        except Exception:   # pylint: disable=broad-except
            # This method must not fail under any circumstances
            pass

    @property
    def is_terminal(self) -> Optional[bool]:
        return self._is_terminal

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    def compute(self, terminated: bool, info: InfoType) -> Optional[float]:
        """Compute the reward if necessary depending on whether the reward and
        state are terminal. If so, then first evaluate the underlying quantity,
        next apply post-processing if requested.

        .. warning::
            This method is not meant to be overloaded.

        :returns: Scalar value if the reward was evaluated, `None` otherwise.
        """
        # Early return depending on whether the reward and state are terminal
        if self.is_terminal is not None and self.is_terminal ^ terminated:
            return None

        # Evaluate raw quantity
        value = self.env.quantities[self.name]

        # Early return if quantity is None
        if value is None:
            return None

        # Apply some post-processing if requested
        if self._transform_fn is not None:
            value = self._transform_fn(value)

        # Return the reward
        return value


BaseQuantityReward.name.__doc__ = \
    """Name uniquely identifying every reward.

    It will be used as key not only for storing reward-specific monitoring
    information in 'info', but also for adding the underlying quantity to
    the ones already managed by the environment.
    """


class BaseMixtureReward(AbstractReward):
    """Base class for aggregating multiple independent reward components in a
    single one.
    """

    components: Tuple[AbstractReward, ...]
    """List of all the reward components that must be aggregated together.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 components: Sequence[AbstractReward],
                 reduce_fn: Callable[
                    [Sequence[Optional[float]]], Optional[float]],
                 is_normalized: bool) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the total reward.
        :param components: Sequence of reward components to aggregate.
        :param reduce_fn: Transform function responsible for aggregating all
                          the reward components that were evaluated. Typical
                          examples are cumulative product and weighted sum.
        :param is_normalized: Whether the reward is guaranteed to be normalized
                              after applying reduction function `reduce_fn`.
        """
        # Make sure that at least one reward component has been specified
        if not components:
            raise ValueError(
                "At least one reward component must be specified.")

        # Make sure that all reward components share the same environment
        for reward in components:
            if env is not reward.env:
                raise ValueError(
                    "All reward components must share the same environment.")

        # Backup some user argument(s)
        self.components = tuple(components)
        self._reduce_fn = reduce_fn
        self._is_normalized = is_normalized

        # Call base implementation
        super().__init__(env, name)

        # Determine whether the reward mixture is terminal
        is_terminal = {reward.is_terminal for reward in self.components}
        self._is_terminal: Optional[bool] = None
        if len(is_terminal) == 1:
            self._is_terminal = next(iter(is_terminal))

    @property
    def is_terminal(self) -> Optional[bool]:
        """Whether the reward is terminal, ie only evaluated at the end of an
        episode if a termination condition has been triggered.

        The cumulative reward is considered terminal if and only if all its
        individual reward components are terminal.
        """
        return self._is_terminal

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    def compute(self, terminated: bool, info: InfoType) -> Optional[float]:
        """Evaluate each individual reward component for the current state of
        the environment, then aggregate them in one.
        """
        # Early return depending on whether the reward and state are terminal
        if self.is_terminal is not None and self.is_terminal ^ terminated:
            return None

        # Compute all reward components
        values = []
        for reward in self.components:
            # Evaluate reward
            reward_info: InfoType = {}
            value: Optional[float] = reward(terminated, reward_info)

            # Clear reward value if the reward was never truly evaluated
            if not reward_info:
                value = None

            # Append reward value and information
            info.update(reward_info)
            values.append(value)

        # Aggregate all reward components in one
        reward_total = self._reduce_fn(values)

        return reward_total
