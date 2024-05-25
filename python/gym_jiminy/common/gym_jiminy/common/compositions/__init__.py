# pylint: disable=missing-module-docstring

from .mixin import (radial_basis_function,
                    AdditiveMixtureReward,
                    MultiplicativeMixtureReward)
from .generic import (TrackingMechanicalJointPositionsReward,
                      SurviveReward)
from .locomotion import TrackingOdometryVelocityReward

__all__ = [
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "TrackingMechanicalJointPositionsReward",
    "TrackingOdometryVelocityReward",
    "SurviveReward"
]
