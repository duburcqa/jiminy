"""Rewards mainly relevant for locomotion tasks on floating-base robots.
"""
import math
from functools import partial

import numba as nb

from gym_jiminy.common.compositions import CUTOFF_ESP
from gym_jiminy.common.bases import (
    InterfaceJiminyEnv, QuantityEvalMode, QuantityReward)

from ..quantities import StabilityMarginProjectedSupportPolygon


@nb.jit(nopython=True, cache=True)
def tanh_normalization(value: float,
                       cutoff_low: float,
                       cutoff_high: float) -> float:
    """Normalize a given quantity between 0.0 and 1.0.

    The extremum 0.0 and 1.0 correspond to the upper and lower cutoff
    respectively, if the lower cutoff is smaller than the upper cutoff. The
    other way around otherwise. These extremum are reached asymptotically,
    which is that the gradient is never zero but rather vanishes exponentially.
    The gradient will be steeper if the cutoff range is tighter and the other
    way around.

    :param value: Value of the scalar floating-point quantity. The quantity may
                  be bounded or unbounded, and signed or not, without
                  restrictions.
    :param cutoff: Cut-off threshold to consider.
    :param order: Order of L^p-norm that will be used as distance metric.
    """
    value_rel = (
        cutoff_high + cutoff_low - 2 * value) / (cutoff_high - cutoff_low)
    return 1.0 / (1.0 + math.pow(CUTOFF_ESP / (1.0 - CUTOFF_ESP), value_rel))


class MaximizeRobusntess(QuantityReward):
    """Encourage the agent to maintain itself in postures as robust as possible
    to external disturbances.

    The signed distance is transformed in a normalized reward to maximize by
    applying rescaled tanh. The reward is smaller than CUTOFF_ESP if the ZMP is
    outside the projected support polygon and further away from the border than
    the upper cutoff. Conversely, the reward is larger than 1.0 - CUTOFF_ESP if
    the ZMP is inside the projected support polygon and further away from the
    border than the lower cutoff.

    The agent may opt from one of the two very different strategies to maximize
    this reward:
      * Foot placement: reshaping the projected support polygon by moving the
        feet (aka the candidate contact points in the direction of the ZMP
        without actually moving the ZMP itself.
      * Torso/Ankle control: Modulating the linear and angular momentum of its
        upper-body to move the ZMP closer to the Chebyshev center of the
        projected support polygon while holding the feet at the exact same
        location.

    These two strategies are complementary rather than mutually exclusive.
    Usually, ankle control is preferred for small disturbances. Foot placement
    comes to place when ankle control is no longer sufficient to keep balance.
    Indeed, the first strategy is only capable of recovering 0-step capturable
    disturbances, while the second one is only limited to inf-step capturable
    disturbances, which includes and dramatically extends 0-step capturability.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float,
                 cutoff_outer: float = 0.0) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold when the ZMP lies inside the support
                       polygon. The reward will be larger than
                       '1.0 - CUTOFF_ESP' if the distance from the border is
                       larger than 'cutoff_inner'.
        :param cutoff_outer: Cutoff threshold when the ZMP lies outside the
                             support polygon. The reward will be smaller than
                             'CUTOFF_ESP' if the ZMP is further away from the
                             border of the support polygon than 'cutoff_outer'.
        """
        # Backup some user argument(s)
        self.cutoff_inner = cutoff
        self.cutoff_outer = cutoff_outer

        # The cutoff thresholds must be positive
        if self.cutoff_inner < 0.0 or self.cutoff_outer < 0.0:
            raise ValueError(
                "The inner and outer cutoff must both be positive.")

        # Call base implementation
        super().__init__(
            env,
            "reward_robustness",
            (StabilityMarginProjectedSupportPolygon, dict(
                mode=QuantityEvalMode.TRUE
            )),
            partial(tanh_normalization,
                    cutoff_low=self.cutoff_inner,
                    cutoff_high=-self.cutoff_outer),
            is_normalized=True,
            is_terminal=False)
