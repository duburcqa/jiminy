# pylint: disable=missing-module-docstring

from .mixin import (CUTOFF_ESP,
                    radial_basis_function,
                    AdditiveMixtureReward,
                    MultiplicativeMixtureReward)
from .generic import (SurviveReward,
                      TrackingQuantityReward,
                      TrackingActuatedJointPositionsReward,
                      MinimizeMechanicalPowerConsumption,
                      DriftTrackingQuantityTermination,
                      ShiftTrackingQuantityTermination,
                      MechanicalSafetyTermination,
                      MechanicalPowerConsumptionTermination,
                      ShiftTrackingMotorPositionsTermination)
from .locomotion import (TrackingBaseHeightReward,
                         TrackingBaseOdometryVelocityReward,
                         TrackingCapturePointReward,
                         TrackingFootPositionsReward,
                         TrackingFootOrientationsReward,
                         TrackingFootForceDistributionReward,
                         DriftTrackingBaseOdometryPositionTermination,
                         DriftTrackingBaseOdometryOrientationTermination,
                         ShiftTrackingFootOdometryPositionsTermination,
                         ShiftTrackingFootOdometryOrientationsTermination,
                         MinimizeAngularMomentumReward,
                         MinimizeFrictionReward,
                         BaseRollPitchTermination,
                         FallingTermination,
                         FootCollisionTermination,
                         FlyingTermination,
                         ImpactForceTermination)

__all__ = [
    "CUTOFF_ESP",
    "radial_basis_function",
    "AdditiveMixtureReward",
    "MultiplicativeMixtureReward",
    "SurviveReward",
    "MinimizeFrictionReward",
    "MinimizeAngularMomentumReward",
    "MinimizeMechanicalPowerConsumption",
    "TrackingQuantityReward",
    "TrackingActuatedJointPositionsReward",
    "TrackingBaseHeightReward",
    "TrackingBaseOdometryVelocityReward",
    "TrackingCapturePointReward",
    "TrackingFootPositionsReward",
    "TrackingFootOrientationsReward",
    "TrackingFootForceDistributionReward",
    "DriftTrackingQuantityTermination",
    "DriftTrackingBaseOdometryPositionTermination",
    "DriftTrackingBaseOdometryOrientationTermination",
    "ShiftTrackingQuantityTermination",
    "ShiftTrackingMotorPositionsTermination",
    "ShiftTrackingFootOdometryPositionsTermination",
    "ShiftTrackingFootOdometryOrientationsTermination",
    "MechanicalSafetyTermination",
    "MechanicalPowerConsumptionTermination",
    "FlyingTermination",
    "BaseRollPitchTermination",
    "FallingTermination",
    "FootCollisionTermination",
    "ImpactForceTermination"
]
