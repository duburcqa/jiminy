# pylint: disable=missing-module-docstring

from .generic import (radial_basis_function,
                      AdditiveMixtureReward,
                      MultiplicativeMixtureReward,
                      SurviveReward)
from .locomotion import OdometryVelocityReward

__all__ = [
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "SurviveReward",
    "OdometryVelocityReward"
]
