# pylint: disable=missing-module-docstring

from .generic import (radial_basis_function,
                      AdditiveMixtureReward,
                      MultiplicativeMixtureReward)
from .locomotion import OdometryVelocityReward

__all__ = [
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "OdometryVelocityReward"
]
