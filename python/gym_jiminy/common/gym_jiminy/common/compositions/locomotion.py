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


class TrackingFootPoseReward(BaseTrackingReward):
    """Reward the agent for tracking the relative pose of the feet wrt to each
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

        # Define "vectorized" difference operator on SE3 Lie group
        se3_diff_vector = (
            pin.liegroups.SE3().difference)  # pylint: disable=no-member

        def pose_difference(xyzquats_left: np.ndarray,
                            xyzquats_right: np.ndarray,
                            out: np.ndarray) -> np.ndarray:
            """Compute the pair-wise difference between batches of pose vectors
            (X, Y, Z, QuatX, QuatY, QuatZ, QuatW), ie `quat_left - quat_right`.

            Internally, this method is not vectorized for now as it loops over
            all pairs sequentially and applies the operator
            `pinocchio.liegroups.SE3.difference`.

            :param xyzquats_left: Left-hand side of the SE3 difference, as a
                                  N-dimensional array whose first dimension
                                  gathers the 7 pose coordinates (x, y, z, qx,
                                  qy, qz, qw).
            :param xyzquats_right: Right-hand side of the SE3 difference, as a
                                   N-dimensional array whose first dimension
                                   gathers the 7 pose coordinates (x, y, z, qx,
                                   qy, qz, qw).
            :param out: A pre-allocated array into which the result is stored.
            """
            nonlocal se3_diff_vector

            # FIXME: Implement vectorized `log6` operator defined here:
            # https://github.com/stack-of-tasks/pinocchio/blob/master/include/pinocchio/spatial/log.hxx  # noqa: E501  # pylint: disable=line-too-long
            for xyzquat_left, xyzquat_right, out_ in zip(
                    xyzquats_left.T, xyzquats_right.T, out.T):
                out_[:] = se3_diff_vector(xyzquat_left, xyzquat_right)
            return out

        # Buffer storing the difference before current and reference poses
        self._spatial_velocities = np.zeros((6, len(frame_names)))

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_foot_pose",
            lambda mode: (MultiFootRelativeXYZQuat, dict(
                frame_names=frame_names,
                mode=mode)),
            cutoff,
            op=partial(pose_difference, out=self._spatial_velocities))
