"""Rewards mainly relevant for locomotion tasks on floating-base robots.
"""
from functools import partial
from typing import Optional, Union, Sequence, Literal, Callable, cast

import numpy as np
import pinocchio as pin

from ..bases import (
    InterfaceJiminyEnv, StateQuantity, QuantityEvalMode, QuantityReward)
from ..quantities import (
    OrientationType, MaskedQuantity, UnaryOpQuantity, FrameOrientation,
    BaseRelativeHeight, BaseOdometryAverageVelocity, CapturePoint,
    MultiFootRelativeXYZQuat, MultiContactNormalizedForceTangential,
    MultiFootNormalizedForceVertical, MultiFootCollisionDetection,
    AverageBaseMomentum)
from ..quantities.locomotion import sanitize_foot_frame_names
from ..utils import quat_difference

from .generic import (
    ArrayLikeOrScalar, TrackingQuantityReward, QuantityTermination)
from .mixin import radial_basis_function


class TrackingBaseHeightReward(TrackingQuantityReward):
    """Reward the agent for tracking the height of the floating base of the
    robot wrt some reference trajectory.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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
                keys=(2,))),
            cutoff)


class TrackingBaseOdometryVelocityReward(TrackingQuantityReward):
    """Reward the agent for tracking the odometry velocity wrt some reference
    trajectory.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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
            lambda mode: (BaseOdometryAverageVelocity, dict(mode=mode)),
            cutoff)


class TrackingCapturePointReward(TrackingQuantityReward):
    """Reward the agent for tracking the capture point wrt some reference
    trajectory.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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


class TrackingFootPositionsReward(TrackingQuantityReward):
    """Reward the agent for tracking the relative position of the feet wrt each
    other.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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
                keys=(0, 1, 2))),
            cutoff)


class TrackingFootOrientationsReward(TrackingQuantityReward):
    """Reward the agent for tracking the relative orientation of the feet wrt
    each other.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_foot_orientations",
            lambda mode: (MaskedQuantity, dict(
                quantity=(MultiFootRelativeXYZQuat, dict(
                    frame_names=frame_names,
                    mode=mode)),
                axis=0,
                keys=(3, 4, 5, 6))),
            cutoff,
            op=cast(Callable[
                [np.ndarray, np.ndarray], np.ndarray], quat_difference))


class TrackingFootForceDistributionReward(TrackingQuantityReward):
    """Reward the agent for tracking the relative vertical force in world frame
    applied on each foot.

    .. note::
        The force is normalized by the weight of the robot rather than the
        total force applied on all feet. This is important as it not only takes
        into account the force distribution between the feet, but also the
        overall ground contact interact force. This way, building up momentum
        before jumping will be distinguished for standing still. Moreover, it
        ensures that the reward is always properly defined, even if the robot
        has no contact with the ground at all, which typically arises during
        the flying phase of running.

    .. seealso::
        See `TrackingQuantityReward` documentation for technical details.
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

        # Call base implementation
        super().__init__(
            env,
            "reward_tracking_foot_force_distribution",
            lambda mode: (MultiFootNormalizedForceVertical, dict(
                frame_names=frame_names,
                mode=mode)),
            cutoff)


class MinimizeAngularMomentumReward(QuantityReward):
    """Reward the agent for minimizing the angular momentum in world plane.

    The angular momentum along x- and y-axes in local odometry frame is
    transform in a normalized reward to maximize by applying RBF kernel on the
    error. See `TrackingQuantityReward` documentation for technical details.
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
            "reward_momentum",
            (AverageBaseMomentum, dict(mode=QuantityEvalMode.TRUE)),
            partial(radial_basis_function, cutoff=self.cutoff, order=2),
            is_normalized=True,
            is_terminal=False)


class MinimizeFrictionReward(QuantityReward):
    """Reward the agent for minimizing the tangential forces at all the contact
    points and collision bodies, and to avoid jerky intermittent contact state.

    The L2-norm is used to aggregate all the local tangential forces. While the
    L1-norm would be more natural in this specific cases, using the L2-norm is
    preferable as it promotes space-time regularity, ie balancing the  force
    distribution evenly between all the candidate contact points and avoiding
    jerky contact forces over time (high-frequency vibrations),  phenomena to
    which the L1-norm is completely insensitive.
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
            "reward_friction",
            (MultiContactNormalizedForceTangential, dict()),
            partial(radial_basis_function, cutoff=self.cutoff, order=2),
            is_normalized=True,
            is_terminal=False)


class BaseRollPitchTermination(QuantityTermination):
    """Encourages the agent to keep the floating base straight, ie its torso in
    case of a humanoid robot, by prohibiting excessive roll and pitch angles.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 low: Optional[ArrayLikeOrScalar],
                 high: Optional[ArrayLikeOrScalar],
                 grace_period: float = 0.0,
                 *,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param low: Lower bound below which termination is triggered.
        :param high: Upper bound above which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Call base implementation
        super().__init__(
            env,
            "termination_base_roll_pitch",
            (MaskedQuantity, dict(  # type: ignore[arg-type]
                quantity=(FrameOrientation, dict(
                    frame_name="root_joint",
                    type=OrientationType.EULER)),
                axis=0,
                keys=(0, 1))),
            low,
            high,
            grace_period,
            is_truncation=False,
            is_training_only=is_training_only)


class BaseHeightTermination(QuantityTermination):
    """Terminate the episode immediately if the floating base of the robot
    gets too close from the ground.

    It is assumed that the state is no longer recoverable when its condition
    is triggered. As such, the episode is terminated on the spot as the
    situation is hopeless. Generally speaking, aborting an epsiode in
    anticipation of catastrophic failure is beneficial. Assuming the condition
    is on point, doing this improves the signal to noice ratio when estimating
    the gradient by avoiding cluterring the training batches with irrelevant
    information.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 thr: float,
                 grace_period: float = 0.0,
                 *,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param low: Lower bound below which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Call base implementation
        super().__init__(
            env,
            "termination_base_height",
            (BaseRelativeHeight, {}),  # type: ignore[arg-type]
            thr,
            None,
            grace_period,
            is_truncation=False,
            is_training_only=is_training_only)


class FootCollisionTermination(QuantityTermination):
    """Terminate the episode immediately if some of the feet of the robot are
    getting too close from each other.

    Self-collision must be avoided at all cost, as it can damage the hardware.
    Considering this condition as a dramatically failure urges the agent to do
    his best in this matter, to the point of becoming risk averse.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 security_margin: float = 0.0,
                 grace_period: float = 0.0,
                 *,
                 is_training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param low: Lower bound below which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param is_training_only: Whether the termination condition should be
                                 completely by-passed if the environment is in
                                 evaluation mode.
                                 Optional: False by default.
        """
        # Call base implementation
        super().__init__(
            env,
            "termination_foot_collision",
            (MultiFootCollisionDetection, dict(  # type: ignore[arg-type]
                security_margin=security_margin)),
            False,
            False,
            grace_period,
            is_truncation=False,
            is_training_only=is_training_only)
