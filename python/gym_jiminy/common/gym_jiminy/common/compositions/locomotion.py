"""Rewards mainly relevant for locomotion tasks on floating-base robots.
"""
import math
from functools import partial
from dataclasses import dataclass
from typing import (
    Optional, Union, Sequence, Literal, Callable, Tuple, List, cast)

import numpy as np
import numba as nb

import jiminy_py.core as jiminy
import pinocchio as pin

from ..bases import (
    InterfaceJiminyEnv, InterfaceQuantity, QuantityEvalMode, AbstractQuantity,
    QuantityReward)
from ..bases.compositions import ArrayOrScalar, ArrayLikeOrScalar
from ..quantities import (
    OrientationType, MaskedQuantity, UnaryOpQuantity, ConcatenatedQuantity,
    FrameOrientation, BaseRelativeHeight, BaseOdometryPose,
    DeltaBaseOdometryPosition, DeltaBaseOdometryOrientation,
    BaseOdometryAverageVelocity, CapturePoint, MultiFramePosition,
    MultiFootRelativeXYZQuat, MultiContactNormalizedSpatialForce,
    MultiFootNormalizedForceVertical, MultiFootCollisionDetection,
    AverageBaseMomentum)
from ..utils import quat_difference, quat_to_yaw

from .generic import (
    TrackingQuantityReward, QuantityTermination,
    DriftTrackingQuantityTermination, ShiftTrackingQuantityTermination)
from ..quantities.locomotion import angle_difference
from .mixin import radial_basis_function


class TrackingBaseHeightReward(TrackingQuantityReward):
    """Reward the agent for tracking the height of the floating base of the
    robot relative to lowest contact point wrt some reference trajectory.

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
        super().__init__(
            env,
            "reward_tracking_base_height",
            lambda mode: (BaseRelativeHeight, dict(mode=mode)),
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
        super().__init__(
            env,
            "reward_tracking_odometry_velocity",
            lambda mode: (BaseOdometryAverageVelocity, dict(mode=mode)),
            cutoff)


@nb.jit(nopython=True, cache=True, fastmath=True, inline='always')
def l2_norm(vec: np.ndarray) -> np.ndarray:
    """Compute the L2-norm of a vector.

    :param array: Input array.
    """
    assert vec.ndim == 1
    return np.sqrt(np.sum(np.square(vec)))


class DriftTrackingBaseOdometryPoseReward(TrackingQuantityReward):
    """Reward the agent for tracking the drift of the odometry pose over a
    horizon wrt some reference trajectory.

    .. seealso::
        See `DeltaBaseOdometryPosition`, `DeltaBaseOdometryOrientation` and
        `TrackingQuantityReward` documentations for technical details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 cutoff: float,
                 horizon: float) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param cutoff: Cutoff threshold for the RBF kernel transform.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the drift.
        """
        # Note that it is essential to operate on the Cartesian distance rather
        # than the absolute position in world plan in order to decouple drift
        # in position from drift in orientation. Otherwise, any drift on
        # orientation would cause the drift in absolute position to diverge.
        super().__init__(
            env,
            "reward_tracking_odometry_pose",
            lambda mode: (ConcatenatedQuantity, dict(
                quantities=(
                    (UnaryOpQuantity, dict(
                        quantity=(DeltaBaseOdometryPosition, dict(
                            horizon=horizon,
                            mode=mode)),
                        op=l2_norm)),
                    (DeltaBaseOdometryOrientation, dict(
                        horizon=horizon,
                        mode=mode))))),
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
        super().__init__(
            env,
            "reward_tracking_capture_point",
            lambda mode: (CapturePoint, dict(
                reference_frame=pin.ReferenceFrame.LOCAL,
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
        super().__init__(
            env,
            "reward_tracking_foot_positions",
            lambda mode: (MaskedQuantity, dict(
                quantity=(MultiFootRelativeXYZQuat, dict(
                    frame_names=frame_names,
                    mode=mode)),
                axis=0,
                keys=(0, 1, 2))),
            cutoff)


class TrackingFootOrientationsReward(TrackingQuantityReward):
    """Reward the agent for tracking the relative orientation of the feet wrt
    each other.

    The error corresponds to the sum of squared total angles of the differences
    between the true and reference relative orientations for each foot.

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

    The L^2-norm is used to aggregate all the local tangential forces. While
    the L^1-norm would be more natural in this specific cases, using the L-2
    norm is preferable as it promotes space-time regularity, ie balancing the
    force distribution evenly between all the candidate contact points and
    avoiding jerky contact forces over time (high-frequency vibrations),
    phenomena to which the L^1-norm is completely insensitive.
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
            (MaskedQuantity, dict(
                quantity=(MultiContactNormalizedSpatialForce, dict()),
                axis=0,
                keys=(0, 1))),
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
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param low: Lower bound below which termination is triggered.
        :param high: Upper bound above which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
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
            training_only=training_only)


class FallingTermination(QuantityTermination):
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
                 min_base_height: float,
                 grace_period: float = 0.0,
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param min_base_height: Minimum height of the floating base of the
                                robot below which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_base_height",
            (BaseRelativeHeight, dict(  # type: ignore[arg-type]
                mode=QuantityEvalMode.TRUE)),
            min_base_height,
            None,
            grace_period,
            is_truncation=False,
            training_only=training_only)


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
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto',
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param security_margin:
            Minimum signed distance below which termination is triggered. This
            can be interpreted as inflating or deflating the geometry objects
            by the safety margin depending on whether it is positive or
            negative. See `MultiFootCollisionDetection` for details.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_foot_collision",
            (MultiFootCollisionDetection, dict(  # type: ignore[arg-type]
                frame_names=frame_names,
                security_margin=security_margin)),
            False,
            False,
            grace_period,
            is_truncation=False,
            training_only=training_only)


@nb.jit(nopython=True, cache=True, fastmath=True)
def depth_approx(positions: np.ndarray,
                 heights: np.ndarray,
                 normals: np.ndarray) -> np.ndarray:
    """Approximate signed distance from the ground profile (positive if above
    the ground, negative otherwise) of a set of the query points.

    Internally, it uses a first order approximation assuming zero local
    curvature around each query point.

    :param positions: Position of all the query points from which to compute
                      from the ground profile, as a 2D array whose first
                      dimension gathers the 3 position coordinates (X, Y, Z)
                      while the second correponds to the N individual query
                      points.
    :param heights: Vertical height wrt the ground profile of the N individual
                    query points in world frame as 1D array.
    :param normals: Normal of the ground profile for the projection in world
                    plane of all the query points, as a 2D array whose first
                    dimension gathers the 3 position coordinates (X, Y, Z)
                    while the second correponds to the N individual query
                    points.
    """
    return (positions[2] - heights) * normals[2]


@dataclass(unsafe_hash=True)
class _MultiContactGroundDistanceAndNormal(
        InterfaceQuantity[Tuple[np.ndarray, np.ndarray]]):
    """Signed distance (positive if above the ground, negative otherwise) and
    normal from the ground profile of all the candidate contact points.

    .. note::
        Internally, it does not compute the exact shortest distance from the
        ground profile because it would be computionally too demanding for now.
        As a surrogate, it relies on a first order approximation assuming zero
        local curvature around all the contact points individually.

    .. warning::
        The set of contact points must not change over episodes. In addition,
        collision bodies are not supported for now.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity]) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        """
        # Get the name of all the contact points
        contact_frame_names = env.robot.contact_frame_names

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                positions=(MultiFramePosition, dict(
                    frame_names=contact_frame_names,
                    mode=QuantityEvalMode.TRUE
                ))),
            auto_refresh=False)

        # Reference to the heightmap function for the ongoing epsiode
        self._heightmap = jiminy.HeightmapFunction(lambda: None)

        # Allocate memory for the height and normal of all the contact points
        self._heights = np.zeros((len(contact_frame_names),))
        self._normals = np.zeros((3, len(contact_frame_names)), order="F")

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Refresh the heighmap function
        engine_options = self.env.unwrapped.engine.get_options()
        self._heightmap = engine_options["world"]["groundProfile"]

    def refresh(self) -> Tuple[np.ndarray, np.ndarray]:
        # Query the height and normal to the ground profile for the position in
        # world plane of all the contact points.
        positions = self.positions.get()
        jiminy.query_heightmap(self._heightmap,
                               positions[:2],
                               self._heights,
                               self._normals)

        # Make sure the ground normal has unit length
        # self._normals /= np.linalg.norm(self._normals, axis=0)

        # First-order distance estimation assuming no curvature
        depth = depth_approx(positions, self._heights, self._normals)

        return depth, self._normals


class FlyingTermination(QuantityTermination):
    """Discourage the agent of jumping by terminating the episode immediately
    if the robot is flying too high above the ground.

    This kind of behavior is unsually undesirable because it may be frightning
    for people nearby, damage the hardware, be difficult to predict and be
    hardly repeatable. Moreover, such dynamic motions tend to transfer poorly
    to reality because the simulation to real gap is worsening.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 max_height: float,
                 grace_period: float = 0.0,
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param max_height: Maximum height of the lowest contact points wrt the
                           groupd above which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_flying",
            (UnaryOpQuantity, dict(
                quantity=(_MultiContactGroundDistanceAndNormal, {}),
                op=lambda depths_and_normals: depths_and_normals[0].min()
            )),
            None,
            max_height,
            grace_period,
            is_truncation=False,
            training_only=training_only)


@nb.jit(nopython=True, cache=True, fastmath=True)
def compute_linear_velocity_local_frame(frame_pos_rel: np.ndarray,
                                        joint_rot_mat: np.ndarray,
                                        joint_vel_linear: np.ndarray,
                                        joint_vel_ang: np.ndarray
                                        ) -> np.ndarray:
    """Compute the linear velocity of a given frame in local-world-aligned
    reference frame.

    :param frame_pos_rel: Relative position of the frame wrt its parent joint.
    :param joint_rot_mat: Orientation of the joint in world refence frame.
    :param joint_vel_linear: Linear velocity of the joint in local reference
                             frame.
    :param joint_vel_ang: Angular velocity of the joint in local reference
                          frame.
    """
    return joint_rot_mat @ (
        joint_vel_linear + np.cross(joint_vel_ang, frame_pos_rel))


@nb.jit(nopython=True, cache=True, fastmath=True)
def compute_velocity_tangential(velocity: np.ndarray,
                                normal: np.ndarray) -> float:
    """Compute the norm of the velocity projected in the plan orthogonal to a
    given normal direction vector.

    .. warning::
        The normal direction vector used assumed to be normalized. It is up to
        the pratitioner to make sure this holds true.

    :param velocity: Linear velocity in world-aligned reference frame.
    :param normal: Normal direction vector in world reference frame.
    """
    return math.sqrt(
        np.sum(np.square(velocity), 0) - np.sum(velocity * normal, 0) ** 2)


@nb.jit(nopython=True, cache=True, fastmath=True)
def _compute_max_velocity_tangential(
        local_linear_velocities_args: Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...],
        depths: np.ndarray,
        normals: np.ndarray,
        height_thr: float) -> float:
    """Compute the maximum norm of the tangential velocity wrt local curvature
    of the ground profile of all the frames that are considered in contact.

    :param local_linear_velocities_args:
        Sequence of tuples (`frame_pos_rel`, `joint_rot_mat`,
        `joint_vel_linear`, `joint_vel_ang`) that fully specify the linear
        velocity of each frame. These arguments will be passed to
        `compute_linear_velocity_local_frame`.
    :param depths: Signed distance of each frames from the ground as a vector.
    :param normals: Normal direction vector that fully specify the local
                    curvature of the ground profile at the location of each
                    frame as a 2D array whose first dimension gathers the
                    position components (X, Y, Z) and the second corresponds
                    to individual frames.
    :param height_thr: Distance threshold below which frames are considered in
                       contact with the ground.
    """
    # Compute the maximum tangential velocity sequentially
    vel_tangential_max = 0.0
    for local_linear_velocity_args, depth, normal in zip(
            local_linear_velocities_args, depths, normals.T):
        # Early return if the contact point is not close to the ground
        if depth > height_thr:
            continue

        # Get the linear velocity of the contact point
        velocity = compute_linear_velocity_local_frame(
            *local_linear_velocity_args)

        # Compute the norm of the tangential velocity
        vel_tangential = compute_velocity_tangential(velocity, normal)

        # Update the maximum tangential velocity
        vel_tangential_max = max(vel_tangential_max, vel_tangential)

    return vel_tangential_max


@dataclass(unsafe_hash=True)
class _MultiContactMaxVelocityTangential(AbstractQuantity[float]):
    """Maximum norm of the tangential velocity wrt local curvature of the
    ground profile of all the candidate contact frames that are close enough
    from the ground.

    .. note::
        The maximum norm of the tangential velocity is considered to be 0.0 if
        none of the candidate contact frames are close enough from the ground.
    """

    height_thr: float
    """Height threshold above which a candidate contact point is deemed too far
    from the ground and is discarded from the set of frames being considered
    when looking for the maximum norm of the tangential velocity.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 height_thr: float) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param height_thr: Height threshold above which a candidate contact
                           point is ignored for being too far from the ground.
        """
        # Backup some user-argument(s)
        self.height_thr = height_thr

        # Get the name of all the contact points
        self.contact_frame_names = env.robot.contact_frame_names

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                depths_and_normals=(
                    _MultiContactGroundDistanceAndNormal, {})),
            auto_refresh=False)

        # Define proxies for fast access
        self._contact_frame_local_linear_velocities_args: Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...] = ()

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Refresh proxies
        contact_frame_local_linear_velocities_args: List[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for contact_frame_index in jiminy.get_frame_indices(
                self.pinocchio_model, self.contact_frame_names):
            frame = self.pinocchio_model.frames[contact_frame_index]
            joint_pose = self.pinocchio_data.oMi[frame.parent]
            joint_velocity_spatial = self.pinocchio_data.v[frame.parent]
            contact_frame_local_linear_velocities_args.append((
                frame.placement.translation,
                joint_pose.rotation,
                joint_velocity_spatial.linear,
                joint_velocity_spatial.angular))
        self._contact_frame_local_linear_velocities_args = tuple(
            contact_frame_local_linear_velocities_args)

    def refresh(self) -> float:
        # Get the distance and normal of all the contact points from the ground
        depths, normals = self.depths_and_normals.get()

        # Compute the maximum tangential velocity
        return _compute_max_velocity_tangential(
            self._contact_frame_local_linear_velocities_args,
            depths,
            normals,
            self.height_thr)


class SlippageTermination(QuantityTermination):
    """Discourage the agent of sliding on the ground purposedly by terminating
    the episode immediately if some of the active contact points are slipping
    on the ground.

    This kind of behavior is usually undesirable because they are hardly
    repeatable and tend to transfer poorly to reality. Moreover, it may cause
    a sense of poorly controlled motion to people nearby.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 height_thr: float,
                 max_velocity: float,
                 grace_period: float = 0.0,
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param height_thr: Height threshold below which a candidate contact
                           point is closed enough from the ground for its
                           tangential velocity to be considered.
        :param max_velocity: Maximum norm of the tangential velocity wrt ground
                             of the contact points that are close enough above
                             which termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_slippage",
            (_MultiContactMaxVelocityTangential, dict(
                height_thr=height_thr)),
            None,
            max_velocity,
            grace_period,
            is_truncation=False,
            training_only=training_only)


class ImpactForceTermination(QuantityTermination):
    """Terminate the episode immediately in case of violent impact on the
    ground.

    Similarly to the jumping behavior, this kind of behavior is usually
    undesirable. See `FlyingTermination` documentation for details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 max_force_rel: float,
                 grace_period: float = 0.0,
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param max_force_rel: Maximum vertical force applied on any of the
                              contact points or collision bodies above which
                              termination is triggered.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_impact_force",
            (MaskedQuantity, dict(  # type: ignore[arg-type]
                quantity=(MultiContactNormalizedSpatialForce, dict()),
                axis=0,
                keys=(2,))),
            None,
            max_force_rel,
            grace_period,
            is_truncation=False,
            training_only=training_only)


class DriftTrackingBaseOdometryPositionTermination(
        DriftTrackingQuantityTermination):
    """Terminate the episode if the current base odometry position is drifting
    too much over wrt some reference trajectory that is being tracked.

    It is generally important to make sure that the robot is not deviating too
    much from some reference trajectory. It sounds appealing to make sure that
    the absolute error between the current and reference trajectory is bounded
    at all time. However, such a condition is very restrictive, especially for
    robots dealing with external disturbances or evolving on an uneven terrain.
    Moreover, when it comes to infinite-horizon trajectories in particular, eg
    periodic motions, avoiding drifting away over time involves being able to
    sense the absolute position of the robot in world frame via exteroceptive
    navigation sensors such as depth cameras or LIDARs. This kind of advanced
    sensor may not be able, thereby making the objective out of reach. Still,
    in the case of legged locomotion, what really matters is tracking
    accurately a nominal limit cycle as long as doing so does not compromise
    local stability. If it does, then the agent expected to make every effort
    to recover balance as fast as possible before going back to the nominal
    limit cycle, without trying to catch up with the ensuing drift since the
    exact absolute odometry pose in world frame is of little interest. See
    `BaseOdometryPose` and `DriftTrackingQuantityTermination` documentations
    for details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 max_position_err: float,
                 horizon: float,
                 grace_period: float = 0.0,
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param max_position_err:
            Maximum drift error in translation (X, Y) in world plane above
            which termination is triggered.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the drift.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_tracking_base_odom_position",
            lambda mode: (  # type: ignore[arg-type, return-value]
                MaskedQuantity, dict(
                    quantity=(BaseOdometryPose, dict(
                        mode=mode)),
                    axis=0,
                    keys=(0, 1))),
            max_position_err,
            horizon,
            grace_period,
            is_truncation=False,
            training_only=training_only)


class DriftTrackingBaseOdometryOrientationTermination(
        DriftTrackingQuantityTermination):
    """Terminate the episode if the current base odometry orientation is
    drifting too much over wrt some reference trajectory that is being tracked.

    See `BaseOdometryPose` and `DriftTrackingBaseOdometryPositionTermination`
    documentations for details.

    .. note::
        It takes into account the  number of turns of the yaw angle of the
        floating base over the whole span of the history.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 max_orientation_err: float,
                 horizon: float,
                 grace_period: float = 0.0,
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param max_orientation_err:
            Maximum drift error in orientation (yaw,) in world plane above
            which termination is triggered.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the drift.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_tracking_base_odom_orientation",
            lambda mode: (  # type: ignore[arg-type, return-value]
                MaskedQuantity, dict(
                    quantity=(BaseOdometryPose, dict(
                        mode=mode)),
                    axis=0,
                    keys=(2,))),
            max_orientation_err,
            horizon,
            grace_period,
            op=angle_difference,
            bounds_only=False,
            is_truncation=False,
            training_only=training_only)


class ShiftTrackingFootOdometryPositionsTermination(
        ShiftTrackingQuantityTermination):
    """Terminate the episode if the selected reference trajectory is not
    tracked with expected accuracy regarding the relative foot odometry
    positions, whatever the timestep being considered over some fixed-size
    sliding window.

    See `MultiFootRelativeXYZQuat` and `ShiftTrackingMotorPositionsTermination`
    documentation for details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 max_position_err: float,
                 horizon: float,
                 grace_period: float = 0.0,
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto',
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param max_position_err:
            Maximum shift error in translation (X, Y) in world plane above
            which termination is triggered.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the shift.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        super().__init__(
            env,
            "termination_tracking_foot_odom_positions",
            lambda mode: (  # type: ignore[arg-type, return-value]
                MaskedQuantity, dict(
                    quantity=(MultiFootRelativeXYZQuat, dict(
                        frame_names=frame_names,
                        mode=mode)),
                    axis=0,
                    keys=(0, 1))),
            max_position_err,
            horizon,
            grace_period,
            is_truncation=False,
            training_only=training_only)


@nb.jit(nopython=True, cache=True, fastmath=True, inline='always')
def angle_distance(angle_left: ArrayOrScalar,
                   angle_right: ArrayOrScalar) -> ArrayOrScalar:
    """Compute the element-wise distance between two batches of angles.

    The distance is defined as the smallest angle in absolute value between
    right and left angles (ignoring multi-turns).

    .. seealso::
        See `angle_difference` documentation for details.

    :param angle_left: Left-hand side angles.
    :param angle_right: Right-hand side angles.
    """
    delta = angle_left - angle_right
    delta -= np.floor(delta / (2 * np.pi)) * (2 * np.pi)
    return np.pi - np.abs(delta - np.pi)


class ShiftTrackingFootOdometryOrientationsTermination(
        ShiftTrackingQuantityTermination):
    """Terminate the episode if the selected reference trajectory is not
    tracked with expected accuracy regarding the relative foot odometry
    orientations, whatever the timestep being considered over some fixed-size
    sliding window.

    See `MultiFootRelativeXYZQuat` and `ShiftTrackingMotorPositionsTermination`
    documentation for details.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 max_orientation_err: float,
                 horizon: float,
                 grace_period: float = 0.0,
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto',
                 *,
                 training_only: bool = False) -> None:
        """
        :param env: Base or wrapped jiminy environment.
            Maximum shift error in orientation (yaw,) in world plane above
            which termination is triggered.
        :param horizon: Horizon over which values of the quantity will be
                        stacked before computing the shift.
        :param grace_period: Grace period effective only at the very beginning
                             of the episode, during which the latter is bound
                             to continue whatever happens.
                             Optional: 0.0 by default.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        :param training_only: Whether the termination condition should be
                              completely by-passed if the environment is in
                              evaluation mode.
                              Optional: False by default.
        """
        # Call base implementation
        super().__init__(
            env,
            "termination_tracking_foot_odom_orientations",
            lambda mode: (UnaryOpQuantity, dict(
                quantity=(MaskedQuantity, dict(
                    quantity=(MultiFootRelativeXYZQuat, dict(
                        frame_names=frame_names,
                        mode=mode)),
                    axis=0,
                    keys=(3, 4, 5, 6))),
                op=quat_to_yaw)),
            max_orientation_err,
            horizon,
            grace_period,
            op=angle_distance,
            is_truncation=False,
            training_only=training_only)
