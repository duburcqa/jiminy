# pylint: disable=missing-module-docstring

from .mixin import (radial_basis_function,
                    AdditiveMixtureReward,
                    MultiplicativeMixtureReward)
from .generic import (BaseTrackingReward,
                      TrackingActuatedJointPositionsReward,
                      SurviveReward)
from .locomotion import (TrackingBaseHeightReward,
                         TrackingOdometryVelocityReward,
                         TrackingCapturePointReward,
                         TrackingFootPositionsReward,
                         TrackingFootOrientationsReward)

__all__ = [
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "BaseTrackingReward",
    "TrackingActuatedJointPositionsReward",
    "TrackingBaseHeightReward",
    "TrackingOdometryVelocityReward",
    "TrackingCapturePointReward",
    "TrackingFootPositionsReward",
    "TrackingFootOrientationsReward",
    "SurviveReward"
]
