# pylint: disable=missing-module-docstring

from .mixin import (radial_basis_function,
                    AdditiveMixtureReward,
                    MultiplicativeMixtureReward)
from .generic import (BaseTrackingReward,
                      TrackingActuatedJointPositionsReward,
                      SurviveReward)
from .locomotion import (TrackingBaseHeightReward,
                         TrackingBaseOdometryVelocityReward,
                         TrackingCapturePointReward,
                         TrackingFootPositionsReward,
                         TrackingFootOrientationsReward,
                         MinimizeAngularMomentumReward)

__all__ = [
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "BaseTrackingReward",
    "TrackingActuatedJointPositionsReward",
    "TrackingBaseHeightReward",
    "TrackingBaseOdometryVelocityReward",
    "TrackingCapturePointReward",
    "TrackingFootPositionsReward",
    "TrackingFootOrientationsReward",
    "MinimizeAngularMomentumReward",
    "SurviveReward"
]
