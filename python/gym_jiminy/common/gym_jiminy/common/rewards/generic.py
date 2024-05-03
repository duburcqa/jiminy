"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
import math
import logging
from typing import Sequence, Tuple

import numpy as np
import numba as nb

from ..bases import (
    InterfaceJiminyEnv, AbstractReward, RewardCreator, InfoType)


# Reward value at cutoff threshold
RBF_CUTOFF_ESP = 1.0e-2


LOGGER = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True, fastmath=True)
def radial_basis_function(error: float,
                          cutoff: float,
                          order: int = 2) -> float:
    r"""Radial basis function (RBF) kernel (aka squared-exponential kernel).

    The RBF kernel is defined as:

    .. math::

        f(x) = \exp{\frac{dist(x, x_ref)}{2 \sigma^2}}

    where :math:`dist(x, x_ref)` is some distance metric of the error between
    the observed (:math:`x`) and desired (:math:`x_ref`) values of a
    multi-variate quantity. The L2-norm (Euclidean norm) was used when it was
    first introduced as a non-linear kernel for Support Vector Machine (SVM)
    algorithm. Such restriction does not make sense in the context of reward
    normalization. The scaling parameter :math:`sigma` is derived from the
    user-specified cutoff. The latter is defined as the distance from which the
    attenuation reaches 99%.

    :param error: Multi-variate error on some tangent space.
    :param cutoff: Cut-off threshold to consider.
    :param order: Order of Lp-Norm that will be used as distance metric.
    """
    distance = np.linalg.norm(error, order)
    return math.pow(RBF_CUTOFF_ESP, math.pow(distance / cutoff, 2))


class AdditiveMixtureReward(AbstractReward):
    """Weighted sum of multiple independent reward components.

    Aggregate of reward components using the addition operator is appropriate
    when improving the behavior for any of them without the others is equally
    beneficial, and unbalanced performance for each reward component is
    considered acceptable rather than detrimental. It especially makes sense
    for reward that are not competing with each other (improving one tends to
    impede some other). In the latter case, the multiplicative operator is more
    appropriate. See `MultiplicativeMixtureReward` documentation for details.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 rewards: Sequence[Tuple[float, RewardCreator]]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param rewards: Sequence of pairs (weight, reward specification), where
                        the reward specification gathers its class and all its
                        constructor keyword-arguments except environment 'env'.
        """
        # Backup user argument(s)
        self._name = name

        # Call base implementation
        super().__init__(env)

        # List of pair (weight, instantiated reward components to aggregate)
        self._weight_reward_pairs: Sequence[Tuple[float, AbstractReward]]
        self._weight_reward_pairs = tuple(
            (weight, reward_cls(self.env, **(reward_kwargs or {})))
            for weight, (reward_cls, reward_kwargs) in rewards)

        # Determine whether the cumulative reward is normalized
        weight_total = 0.0
        for weight, reward_fun in self._weight_reward_pairs:
            if not reward_fun.is_normalized:
                LOGGER.warning(
                    "Reward '%s' is not normalized. Aggregating rewards that "
                    "are not normalized using the addition operator is not "
                    "recommended.", reward_fun.name)
                self._is_normalized = False
                break
            weight_total += weight
        else:
            self._is_normalized = abs(weight_total - 1.0) < 1e-4

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_normalized(self) -> bool:
        """Whether the reward is guaranteed to be normalized, ie it is in range
        [0.0, 1.0].

        The cumulative reward is considered normalized if all its individual
        reward components are normalized and their weights sums up to 1.0.
        """
        return self._is_normalized

    def __call__(self, terminated: bool, info: InfoType) -> float:
        """Evaluate each individual reward component for the current state of
        the environment, then compute their weighted sum to aggregate them.
        """
        reward_total = 0.0
        for weight, reward_fun in self._weight_reward_pairs:
            reward = reward_fun(terminated, info)
            reward_total += weight * reward
        return reward_total


class MultiplicativeMixtureReward(AbstractReward):
    """Product of multiple independent reward components.

    Aggregate of reward components using multiplication operator is appropriate
    when maintaining balanced performance between all reward components is
    essential, and having poor performance for any of them is unacceptable.
    This type of aggregation is especially useful when reward components are
    competing with each other (improving one tends to impede some other) but
    not mutually exclusive.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 rewards: Sequence[RewardCreator],
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param rewards: Sequence of reward specifications, each of which
                        gathering their respective class and all their
                        constructor keyword-arguments except environment 'env'.
        """
        # Backup user argument(s)
        self._name = name

        # Call base implementation
        super().__init__(env)

        # Make sure that at least one reward component has been specified
        if len(rewards) < 1:
            raise ValueError(
                "At least one reward component must be specified.")

        # List of instantiated reward components to aggregate
        self._rewards: Sequence[AbstractReward] = tuple(
            reward_cls(self.env, **(reward_kwargs or {}))
            for reward_cls, reward_kwargs in rewards)

        # Determine whether the cumulative reward is normalized
        self._is_normalized = all(
            reward_fun.is_normalized for reward_fun in self._rewards)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_normalized(self) -> bool:
        """Whether the reward is guaranteed to be normalized, ie it is in range
        [0.0, 1.0].

        The cumulative reward is considered normalized if all its individual
        reward components are normalized.
        """
        return self._is_normalized

    def __call__(self, terminated: bool, info: InfoType) -> float:
        """Evaluate each individual reward component for the current state of
        the environment, then compute their cumulative product to aggregate
        them.
        """
        reward_total = 1.0
        for reward_fun in self._rewards:
            reward_component = reward_fun(terminated, info)
            reward_total *= reward_component
        return reward_total
