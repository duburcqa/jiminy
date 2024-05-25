# pylint: disable=missing-module-docstring

from .mixin import (radial_basis_function,
                    AdditiveMixtureReward,
                    MultiplicativeMixtureReward)
from .generic import (TrackingActuatedJointPositionsReward,
                      SurviveReward)
from .locomotion import (TrackingBaseHeightReward,
                         TrackingOdometryVelocityReward)

__all__ = [
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "TrackingActuatedJointPositionsReward",
    "TrackingBaseHeightReward",
    "TrackingOdometryVelocityReward",
    "SurviveReward"
]
