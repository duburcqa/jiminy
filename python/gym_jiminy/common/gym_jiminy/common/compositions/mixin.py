"""Generic reward components that may be relevant for any kind of robot,
regardless its topology (multiple or single branch, fixed or floating base...)
and the application (locomotion, grasping...).
"""
import math
import logging
from functools import partial
from typing import Sequence, Tuple, Optional, Union, Literal

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
    multi-variate quantity. The L^2-norm (Euclidean norm) was used when it was
    first introduced as a non-linear kernel for Support Vector Machine (SVM)
    algorithm. Such restriction does not make sense in the context of reward
    normalization. The scaling parameter :math:`sigma` is derived from the
    user-specified cutoff. The latter is defined as the distance from which the
    attenuation reaches 99%.

    :param error: Multi-variate error on some tangent space as a 1D array.
    :param cutoff: Cut-off threshold to consider.
    :param order: Order of L^p-norm that will be used as distance metric.
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
    """Weighted L^p-norm of multiple independent reward components.

    Aggregating the reward components using L^p-norm progressively transition
    from promoting versatility for 0 < p < 1, to overall competency for p = 1,
    and ultimately specialization for p > 1. In particular, the L^1-norm is
    appropriate when improving the behavior for any of them without the others
    is equally beneficial, and unbalanced performance for each reward component
    is considered acceptable rather than detrimental. It especially makes sense
    for reward that are not competing with each other (improving one tends to
    impede some other). In the latter case, the multiplicative operator is more
    appropriate. See `MultiplicativeMixtureReward` documentation for details.

    .. note::
        Combining `AdditiveMixtureReward` for L^inf-norm with `SurviveReward`
        ensures a minimum reward that the agent would obtain no matter what,
        to encourage surviving in last resort. This is usually useful to
        bootstrap learning at the very beginning.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 components: Sequence[AbstractReward],
                 order: Union[int, float, Literal['inf']] = 1,
                 weights: Optional[Sequence[float]] = None) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the total reward.
        :param components: Sequence of reward components to aggregate.
        :param order: Order of L^p-norm used to add up the reward components.
        :param weights: Optional sequence of weights associated with each
                        reward component, with same ordering as 'components'.
                        Optional: Same weights that preserves normalization by
                        default, `(1.0 / len(components),) * len(components)`.
        """
        # Handling of default arguments
        if weights is None:
            weights = (1.0 / len(components),) * len(components)

        # Make sure that the order is strictly positive
        if not isinstance(order, str) and order <= 0.0:
            raise ValueError("'order' must be strictly positive or 'inf'.")

        # Make sure that the weight sequence is consistent with the components
        if len(weights) != len(components):
            raise ValueError(
                "Exactly one weight per reward component must be specified.")

        # Filter out components whose weight are zero
        weights, components = zip(*(
            (weight, reward)
            for weight, reward in zip(weights, components)
            if weight > 0.0))

        # Determine whether the cumulative reward is normalized
        scale = 0.0
        for weight, reward in zip(weights, components):
            if not reward.is_normalized:
                LOGGER.warning(
                    "Reward '%s' is not normalized. Aggregating rewards that "
                    "are not normalized using the addition operator is not "
                    "recommended.", reward.name)
                is_normalized = False
                break
            if order == 'inf':
                scale = max(scale, weight)
            else:
                scale += weight
        else:
            is_normalized = abs(1.0 - scale) < 1e-4

        # Backup user-arguments
        self.order = order
        self.weights = tuple(weights)

        # Jit-able method computing the weighted sum of reward components
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def weighted_norm(weights: Tuple[float, ...],
                          order: Union[int, float, Literal['inf']],
                          values: Tuple[Optional[float], ...]
                          ) -> Optional[float]:
            """Compute the weighted L^p-norm of all the reward components that
            has been evaluated, filtering out the others.

            This method returns `None` if no reward component has been
            evaluated.

            :param weights: Sequence of weights for each reward component, with
                            same ordering as 'components'.
            :param order: Order of the L^p-norm.
            :param values: Sequence of scalar value for reward components that
                           has been evaluated, `None` otherwise, with the same
                           ordering as 'components'.

            :returns: Scalar value if at least one of the reward component has
                      been evaluated, `None` otherwise.
            """
            total, any_value = 0.0, False
            for value, weight in zip(values, weights):
                if value is not None:
                    if isinstance(order, str):
                        if any_value:
                            total = max(total, weight * value)
                        else:
                            total = weight * value
                    else:
                        total += weight * math.pow(value, order)
                    any_value = True
            if any_value:
                if isinstance(order, str):
                    return total
                return math.pow(total, 1.0 / order)
            return None

        # Call base implementation
        super().__init__(
            env,
            name,
            components,
            partial(weighted_norm, self.weights, self.order),
            is_normalized)


AdditiveMixtureReward.is_normalized.__doc__ = \
    """Whether the reward is guaranteed to be normalized, ie it is in range
    [0.0, 1.0].

    The cumulative reward is considered normalized if all its individual
    reward components are normalized and their weights sums up to 1.0.
    """


class MultiplicativeMixtureReward(MixtureReward):
    """Geometric mean of independent reward components, to promote versatility.

    Aggregating the reward components using the geometric mean is appropriate
    when maintaining balanced performance between all reward components is
    essential, and having poor performance for any of them is unacceptable.
    This type of aggregation is especially useful when reward components are
    competing with each other (improving one tends to impede some other) but
    not mutually exclusive.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 name: str,
                 components: Sequence[AbstractReward]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param name: Desired name of the reward.
        :param components: Sequence of reward components to aggregate.
        """
        # Determine whether the cumulative reward is normalized
        is_normalized = all(reward.is_normalized for reward in components)

        # Jit-able method computing the product of reward components
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def geometric_mean(
                values: Tuple[Optional[float], ...]) -> Optional[float]:
            """Compute the product of all the reward components that has
            been evaluated, filtering out the others.

            This method returns `None` if no reward component has been
            evaluated.

            :param values: Sequence of scalar value for reward components that
                           has been evaluated, `None` otherwise, with the same
                           ordering as 'components'.

            :returns: Scalar value if at least one of the reward component has
                      been evaluated, `None` otherwise.
            """
            total, any_value, n_values = 1.0, False, 0
            for value in values:
                if value is not None:
                    total *= value
                    any_value = True
                    n_values += 1
            return math.pow(total, 1.0 / n_values) if any_value else None

        # Call base implementation
        super().__init__(env, name, components, geometric_mean, is_normalized)


MultiplicativeMixtureReward.is_normalized.__doc__ = \
    """Whether the reward is guaranteed to be normalized, ie it is in range
    [0.0, 1.0].

    The cumulative reward is considered normalized if all its individual
    reward components are normalized.
    """
