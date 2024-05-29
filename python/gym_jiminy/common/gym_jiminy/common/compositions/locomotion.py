"""Rewards mainly relevant for locomotion tasks on floating-base robots.
"""
from functools import partial
from typing import Union, Sequence, Literal

import numpy as np
import pinocchio as pin

from ..bases import InterfaceJiminyEnv, StateQuantity
from ..quantities import (
    MaskedQuantity, UnaryOpQuantity, AverageOdometryVelocity,
    MultiFootRelativeXYZQuat, CapturePoint)
from ..quantities.locomotion import sanitize_foot_frame_names
from ..utils import quat_difference

from .generic import BaseTrackingReward


class TrackingBaseHeightReward(BaseTrackingReward):
    """Reward the agent for tracking the height of the floating base of the
    robot wrt some reference trajectory.

    .. seealso::
        See `BaseTrackingReward` documentation for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_base_height",
            lambda mode: (MaskedQuantity, dict(
                quantity=(UnaryOpQuantity, dict(
                    quantity=(StateQuantity, dict(mode=mode)),
                    op=lambda state: state.q)),
                key=(2,))),
            cutoff)


class TrackingOdometryVelocityReward(BaseTrackingReward):
    """Reward the agent for tracking the odometry velocity wrt some reference
    trajectory.

    .. seealso::
        See `BaseTrackingReward` documentation for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_odometry_velocity",
            lambda mode: (AverageOdometryVelocity, dict(mode=mode)),
            cutoff)


class TrackingCapturePointReward(BaseTrackingReward):
    """Reward the agent for tracking the capture point wrt some reference
    trajectory.

    .. seealso::
        See `BaseTrackingReward` documentation for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_capture_point",
            lambda mode: (CapturePoint, dict(
                reference_frame=pin.LOCAL,
                mode=mode)),
            cutoff)


class TrackingFootPositionsReward(BaseTrackingReward):
    """Reward the agent for tracking the relative position of the feet wrt each
    other.

    .. seealso::
        See `BaseTrackingReward` documentation for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float,
                 *,
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto'
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Sanitize frame names corresponding to the feet of the robot
        frame_names = tuple(sanitize_foot_frame_names(env, frame_names))

        # Buffer storing the difference before current and reference poses
        self._spatial_velocities = np.zeros((6, len(frame_names)))

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_foot_positions",
            lambda mode: (MaskedQuantity, dict(
                quantity=(MultiFootRelativeXYZQuat, dict(
                    frame_names=frame_names,
                    mode=mode)),
                key=(0, 1, 2))),
            cutoff)


class TrackingFootOrientationsReward(BaseTrackingReward):
    """Reward the agent for tracking the relative orientation of the feet wrt
    each other.

    .. seealso::
        See `BaseTrackingReward` documentation for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float,
                 *,
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto'
                 ) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        """
        # Backup some user argument(s)
        self.cutoff = cutoff

        # Sanitize frame names corresponding to the feet of the robot
        frame_names = tuple(sanitize_foot_frame_names(env, frame_names))

        # Buffer storing the difference before current and reference poses
        # FIXME: Is it worth it to create a temporary ?
        self._diff = np.zeros((3, len(frame_names) - 1))

        # Define buffered quaternion difference operator for efficiency
        def quat_difference_buffered(out: np.ndarray,
                                     q1: np.ndarray,
                                     q2: np.ndarray) -> np.ndarray:
            """Wrapper around `quat_difference` passing buffer in and out
            instead of allocating fresh memory for efficiency.
            """
            quat_difference(q1, q2, out)
            return out

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_foot_orientations",
            lambda mode: (MaskedQuantity, dict(
                quantity=(MultiFootRelativeXYZQuat, dict(
                    frame_names=frame_names,
                    mode=mode)),
                axis=0,
                key=(3, 4, 5, 6))),
            cutoff,
            op=partial(quat_difference_buffered, self._diff))
