"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
import math
import logging
from typing import Sequence, Optional, Union

import numpy as np
import numba as nb

from ..bases import InterfaceJiminyEnv, AbstractReward, MixtureReward


# Reward value at cutoff threshold
CUTOFF_ESP = 1.0e-2


ArrayOrScalar = Union[np.ndarray, float]

LOGGER = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True, fastmath=True)
def radial_basis_function(error: ArrayOrScalar,
                          cutoff: float,
                          order: int = 2) -> float:
    r"""Radial basis function (RBF) kernel (aka squared-exponential kernel).

    The RBF kernel is defined as:

    .. math::

        f(x) = \exp{\frac{dist(x, x_ref)^2}{2 \sigma^2}}

    where :math:`dist(x, x_ref)` is some distance metric of the error between
    the observed (:math:`x`) and desired (:math:`x_ref`) values of a
    multi-variate quantity. The L2-norm (Euclidean norm) was used when it was
    first introduced as a non-linear kernel for Support Vector Machine (SVM)
    algorithm. Such restriction does not make sense in the context of reward
    normalization. The scaling parameter :math:`sigma` is derived from the
    user-specified cutoff. The latter is defined as the distance from which the
    attenuation reaches 99%.

    :param error: Multi-variate error on some tangent space as a 1D array.
    :param cutoff: Cut-off threshold to consider.
    :param order: Order of Lp-Norm that will be used as distance metric.
    """
    error_ = np.asarray(error)
    is_contiguous = error_.flags.f_contiguous or error_.flags.c_contiguous
    if is_contiguous or order != 2:
        if error_.ndim > 1 and not is_contiguous:
            error_ = np.ascontiguousarray(error_)
        if error_.flags.c_contiguous:
            error1d = np.asarray(error_).ravel()
        else:
            error1d = np.asarray(error_.T).ravel()
        if order == 2:
            squared_dist_rel = np.dot(error1d, error1d) / math.pow(cutoff, 2)
        else:
            squared_dist_rel = math.pow(
                np.linalg.norm(error1d, order) / cutoff, 2)
    else:
        squared_dist_rel = np.sum(np.square(error_)) / math.pow(cutoff, 2)
    return math.pow(CUTOFF_ESP, squared_dist_rel)


class AdditiveMixtureReward(MixtureReward):
    """Weighted sum of multiple independent reward components.

    Aggregation of reward components using the addition operator is suitable
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
                 components: Sequence[AbstractReward],
                 weights: Optional[Sequence[float]] = None) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the total reward.
        :param components: Sequence of reward components to aggregate.
        :param weights: Sequence of weights associated with each reward
                        components, with the same ordering as 'components'.
                        Optional: 1.0 for all reward components by default.
        """
        # Handling of default arguments
        if weights is None:
            weights = (1.0,) * len(components)

        # Make sure that the weight sequence is consistent with the components
        if len(weights) != len(components):
            raise ValueError(
                "Exactly one weight per reward component must be specified.")

        # Determine whether the cumulative reward is normalized
        weight_total = 0.0
        for weight, reward in zip(weights, components):
            if not reward.is_normalized:
                LOGGER.warning(
                    "Reward '%s' is not normalized. Aggregating rewards that "
                    "are not normalized using the addition operator is not "
                    "recommended.", reward.name)
                is_normalized = False
                break
            weight_total += weight
        else:
            is_normalized = abs(weight_total - 1.0) < 1e-4

        # Backup user-arguments
        self.weights = weights

        # Call base implementation
        super().__init__(env, name, components, self._reduce, is_normalized)

    def _reduce(self, values: Sequence[Optional[float]]) -> Optional[float]:
        """Compute the weighted sum of all the reward components that has been
        evaluated, filtering out the others.

        This method returns `None` if no reward component has been evaluated.

        :param values: Sequence of scalar value for reward components that has
                       been evaluated, `None` otherwise, with the same ordering
                       as 'components'.

        :returns: Scalar value if at least one of the reward component has been
                  evaluated, `None` otherwise.
        """
        # TODO: x2 speedup can be expected with `nb.jit`
        total, any_value = 0.0, False
        for weight, value in zip(self.weights, values):
            if value is not None:
                total += weight * value
                any_value = True
        return total if any_value else None


AdditiveMixtureReward.is_normalized.__doc__ = \
    """Whether the reward is guaranteed to be normalized, ie it is in range
    [0.0, 1.0].

    The cumulative reward is considered normalized if all its individual
    reward components are normalized and their weights sums up to 1.0.
    """


class MultiplicativeMixtureReward(MixtureReward):
    """Product of multiple independent reward components.

    Aggregation of reward components using multiplication operator is suitable
    when maintaining balanced performance between all reward components is
    essential, and having poor performance for any of them is unacceptable.
    This type of aggregation is especially useful when reward components are
    competing with each other (improving one tends to impede some other) but
    not mutually exclusive.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 components: Sequence[AbstractReward]
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the reward.
        :param components: Sequence of reward components to aggregate.
        """
        # Determine whether the cumulative reward is normalized
        is_normalized = all(reward.is_normalized for reward in components)

        # Call base implementation
        super().__init__(env, name, components, self._reduce, is_normalized)

    def _reduce(self, values: Sequence[Optional[float]]) -> Optional[float]:
        """Compute the product of all the reward components that has been
        evaluated, filtering out the others.

        This method returns `None` if no reward component has been evaluated.

        :param values: Sequence of scalar value for reward components that has
                       been evaluated, `None` otherwise, with the same ordering
                       as 'components'.

        :returns: Scalar value if at least one of the reward component has been
                  evaluated, `None` otherwise.
        """
        # TODO: x2 speedup can be expected with `nb.jit`
        total, any_value = 1.0, False
        for value in values:
            if value is not None:
                total *= value
                any_value = True
        return total if any_value else None


MultiplicativeMixtureReward.is_normalized.__doc__ = \
    """Whether the reward is guaranteed to be normalized, ie it is in range
    [0.0, 1.0].

    The cumulative reward is considered normalized if all its individual
    reward components are normalized.
    """
