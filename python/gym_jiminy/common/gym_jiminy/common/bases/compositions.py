"""This module promotes reward components and termination conditions as
first-class objects. Those building blocks that can be plugged onto an existing
pipeline by composition to keep everything modular, from the task definition to
the low-level observers and controllers.

This modular approach allows for standardization of usual metrics. Overall, it
greatly reduces code duplication and bugs.
"""
from abc import abstractmethod, ABCMeta
from enum import IntEnum
from typing import Tuple, Sequence, Callable, Union, Optional, Generic, TypeVar

import numpy as np

from ..utils.spaces import _array_contains

from .interfaces import InfoType, InterfaceJiminyEnv
from .quantities import QuantityCreator


ValueT = TypeVar('ValueT')

Number = Union[float, int, bool, complex]
ArrayOrScalar = Union[np.ndarray, np.number, Number]
ArrayLikeOrScalar = Union[ArrayOrScalar, Sequence[Union[Number, np.number]]]


class AbstractReward(metaclass=ABCMeta):
    """Abstract class from which all reward component must derived.

    This goal of the agent is to maximize the expectation of the cumulative sum
    of discounted reward over complete episodes. This holds true no matter if
    its sign is always negative (aka. reward), always positive (aka. cost) or
    indefinite (aka. objective).

    Defining cost is allowed by not recommended. Although it encourages the
    agent to achieve the task at hand as quickly as possible if success is the
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
        and debugging information in 'info' if key does not already exists,
        otherwise it will raise an exception.
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

        :param terminated: Whether the episode has reached a terminal state of
                           the MDP at the current step.
        :param info: Dictionary of extra information for monitoring. It will be
                     updated in-place for storing current value of the reward
                     in 'info' if it was truly evaluated.

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
            updated to store either custom debugging information if any or its
            value otherwise. If the reward is not evaluated, then 'info' is
            left as-is and 0.0 is returned.

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
        if bool(self.is_terminal) ^ terminated:
            raise ValueError("Flag 'is_terminal' not honored.")

        # Make sure that the reward is scalar
        assert np.ndim(value) == 0

        # Make sure that the reward is normalized
        if self.is_normalized and (value < 0.0 or value > 1.0):
            raise ValueError(
                "Reward not normalized in range [0.0, 1.0] as it ought to be.")

        # Store its value as info
        if self.name in info.keys():
            raise KeyError(
                f"Key '{self.name}' already reserved in 'info'. Impossible to "
                "store value of reward component.")
        if reward_info:
            info[self.name] = reward_info
        else:
            info[self.name] = value

        # Returning the reward
        return value


class QuantityReward(AbstractReward, Generic[ValueT]):
    """Convenience class making it easy to derive reward components from
    generic quantities.

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
                         to use as reward after some post-processing, plus any
                         keyword-arguments of its constructor except 'env',
                         and 'parent'.
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
        self.data = self.env.quantities.registry[self.name]

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
        if bool(self.is_terminal) ^ terminated:
            return None

        # Evaluate raw quantity
        value = self.data.get()

        # Early return if quantity is None
        if value is None:
            return None

        # Apply some post-processing if requested
        if self._transform_fn is not None:
            value = self._transform_fn(value)

        # Return the reward
        return value


QuantityReward.name.__doc__ = \
    """Name uniquely identifying every reward.

    It will be used as key not only for storing reward-specific monitoring
    and debugging information in 'info', but also for adding the underlying
    quantity to the ones already managed by the environment.
    """


class MixtureReward(AbstractReward):
    """Base class for aggregating multiple independent reward components as a
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
                    [Tuple[Optional[float], ...]], Optional[float]],
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
        if bool(self.is_terminal) ^ terminated:
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
        reward_total = self._reduce_fn(tuple(values))

        return reward_total


class EpisodeState(IntEnum):
    """Specify the current state of the ongoing episode.
    """

    CONTINUED = 0
    """No termination condition has been triggered this step.
    """

    TERMINATED = 1
    """The terminal state has been reached.
    """

    TRUNCATED = 2
    """A truncation condition has been triggered.
    """


class AbstractTerminationCondition(metaclass=ABCMeta):
    """Abstract class from which all termination conditions must derived.

    Request the ongoing episode to stop immediately as soon as a termination
    condition is triggered.

    There are two cases: truncating the episode or reaching the terminal state.
    In the former case, the agent is instructed to stop collecting samples from
    the ongoing episode and move to the next one, without considering this as a
    failure. As such, the reward-to-go that has not been observed will be
    estimated via a value function estimator. This is  already what happens
    when collecting sample batches in the infinite horizon RL framework, except
    that the episode is not resumed to collect the rest of the episode in the
    following sample batched. In the case of a termination condition, the agent
    is just as much instructed to move to the next  episode, but also to
    consider that it was an actual failure. This means that, unlike truncation
    conditions, the reward-to-go is known to be exactly zero. This is usually
    dramatic for the agent in the perspective of an infinite horizon reward,
    even more as the maximum discounted reward grows larger as the discount
    factor gets closer to one. As a result, the agent will avoid at all cost
    triggering terminal conditions, to the point of becoming risk averse by
    taking extra security margins lowering the average reward if necessary.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 grace_period: float = 0.0,
                 *,
                 is_truncation: bool = False,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the termination condition. This name will
                     be used as key for storing the current episode state from
                     the perspective of this specific condition in 'info', and
                     to add the underlying quantity to the set of already
                     managed quantities by the environment. As a result, it
                     must be unique otherwise an exception will be raised.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_truncation: Whether the episode should be considered
                              terminated or truncated whenever the termination
                              condition is triggered.
                              Optional: False by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        self.env = env
        self._name = name
        self.grace_period = grace_period
        self.is_truncation = is_truncation
        self.is_training_only = is_training_only

    @property
    def name(self) -> str:
        """Name uniquely identifying a given termination condition.

        This name will be used as key for storing termination
        condition-specific monitoring information in 'info' if key does not
        already exists, otherwise it will raise an exception.
        """
        return self._name

    @abstractmethod
    def compute(self, info: InfoType) -> bool:
        """Evaluate the termination condition at hand.

        :param info: Dictionary of extra information for monitoring. It will be
                     updated in-place for storing terminated and truncated
                     flags in 'info' as a tri-states `EpisodeState` value.
        """

    def __call__(self, info: InfoType) -> Tuple[bool, bool]:
        """Return whether the termination condition has been triggered.

        For the corresponding MDP to be stationary, the condition to trigger
        termination is supposed to involve only the transition from previous to
        current state of the environment under the ongoing action.

        .. note::
            This method is a lightweight wrapper around `compute` to return two
            boolean flags 'terminated', 'truncated' complying with Gym API.
            'info' will be updated to store either custom debug information if
            any, a tri-states episode state  `EpisodeState` otherwise.

        .. warning::
            This method is not meant to be overloaded.

        :param info: Dictionary of extra information for monitoring. It will be
                     updated in-place for storing terminated and truncated
                     flags in 'info' as a tri-states `EpisodeState` value.

        :returns: terminated and truncated flags.
        """
        # Skip termination condition in eval mode or during grace period
        termination_info: InfoType = {}
        if (self.is_training_only and not self.env.is_training) or (
                self.env.stepper_state.t < self.grace_period):
            # Always continue
            is_terminated, is_truncated = False, False
        else:
            # Evaluate the reward and store extra information
            is_done = self.compute(termination_info)
            is_terminated = is_done and not self.is_truncation
            is_truncated = is_done and self.is_truncation

        # Store episode state as info
        if self.name in info.keys():
            raise KeyError(
                f"Key '{self.name}' already reserved in 'info'. Impossible to "
                "store value of termination condition.")
        if termination_info:
            info[self.name] = termination_info
        else:
            if is_terminated:
                episode_state = EpisodeState.TERMINATED
            elif is_truncated:
                episode_state = EpisodeState.TRUNCATED
            else:
                episode_state = EpisodeState.CONTINUED
            info[self.name] = episode_state

        # Returning terminated and truncated flags
        return is_terminated, is_truncated


class QuantityTermination(AbstractTerminationCondition, Generic[ValueT]):
    """Convenience class making it easy to derive termination conditions from
    generic quantities.

    All this class does is checking that, all elements of a given quantity are
    within bounds. If so, then the episode continues, otherwise it is either
    truncated or terminated according to 'is_truncation' constructor argument.
    This only applies after the end of a grace period. Before that, the
    episode continues no matter what.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 quantity: QuantityCreator[Optional[ArrayOrScalar]],
                 low: Optional[ArrayLikeOrScalar],
                 high: Optional[ArrayLikeOrScalar],
                 grace_period: float = 0.0,
                 *,
                 is_truncation: bool = False,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the termination condition. This name will
                     be used as key for storing the current episode state from
                     the perspective of this specific condition in 'info', and
                     to add the underlying quantity to the set of already
                     managed quantities by the environment. As a result, it
                     must be unique otherwise an exception will be raised.
        :param quantity: Tuple gathering the class of the underlying quantity
                         to use as termination condition, plus any
                         keyword-arguments of its constructor except 'env',
                         and 'parent'.
        :param low: Lower bound below which termination is triggered.
        :param high: Upper bound above which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_truncation: Whether the episode should be considered
                              terminated or truncated whenever the termination
                              condition is triggered.
                              Optional: False by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Backup user argument(s)
        self.low = np.asarray(low) if isinstance(low, Sequence) else low
        self.high = np.asarray(high) if isinstance(high, Sequence) else high

        # Call base implementation
        super().__init__(
            env,
            name,
            grace_period,
            is_truncation=is_truncation,
            is_training_only=is_training_only)

        # Add quantity to the set of quantities managed by the environment
        self.env.quantities[self.name] = quantity

        # Keep track of the underlying quantity
        self.data = self.env.quantities.registry[self.name]

    def __del__(self) -> None:
        try:
            del self.env.quantities[self.name]
        except Exception:   # pylint: disable=broad-except
            # This method must not fail under any circumstances
            pass

    def compute(self, info: InfoType) -> bool:
        """Evaluate the termination condition.

        The underlying quantity is first evaluated. The episode continues if
        all the elements of its value are within bounds, otherwise the episode
        is either truncated or terminated according to 'is_truncation'.

        .. warning::
            This method is not meant to be overloaded.
        """
        # Evaluate the quantity
        value = self.data.get()

        # Check if the quantity is out-of-bounds.
        # Note that it may be `None` if the quantity is ill-defined for the
        # current simulation state, which triggers termination unconditionally.
        return value is None or not _array_contains(value, self.low, self.high)


QuantityTermination.name.__doc__ = \
    """Name uniquely identifying every termination condition.

    It will be used as key not only for storing termination condition-specific
    monitoring and debugging information in 'info', but also for adding the
    underlying quantity to the ones already managed by the environment.
    """
