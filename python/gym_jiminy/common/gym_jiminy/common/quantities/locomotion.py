"""Quantities mainly relevant for locomotion tasks on floating-base robots.
"""
import math
from typing import Optional, Tuple, Sequence, Literal, Union
from dataclasses import dataclass

import numpy as np
import numba as nb

import jiminy_py.core as jiminy
from jiminy_py.core import array_copyto  # pylint: disable=no-name-in-module
from jiminy_py.dynamics import update_quantities
import pinocchio as pin

from ..bases import (
    InterfaceJiminyEnv, InterfaceQuantity, AbstractQuantity, QuantityEvalMode)
from ..utils import (
    fill, matrix_to_quat, matrix_to_yaw, quat_to_yaw, quat_to_matrix)

from ..quantities import (
    Orientation, MaskedQuantity, AverageFrameSpatialVelocity,
    MultiFramePosition, MultiFrameOrientation, MultiFrameMeanXYZQuat)


def sanitize_foot_frame_names(
        env: InterfaceJiminyEnv,
        frame_names: Union[Sequence[str], Literal['auto']] = 'auto'
        ) -> Sequence[str]:
    """Try to detect automatically one frame name per foot of a given legged
    robot if 'auto' mode is enabled. Otherwise, make sure that the specified
    sequence of frame names is non-empty, and all of them corresponds to
    end-effectors, ie having one of the leaf joints of the kinematic tree of
    the robot as parent.

    :param env: Base or wrapped jiminy environment.
    :param frame_names: Name of the frames corresponding to some feet of the
                        robot. 'auto' to automatically detect them from the set
                        of contact and force sensors of the robot.
    """
    # Make sure that the robot has a freeflyer
    if not env.robot.has_freeflyer:
        raise ValueError("Only legged robot with floating base are supported.")

    # Determine the leaf joints of the kinematic tree
    pinocchio_model = env.robot.pinocchio_model_th
    parents = pinocchio_model.parents
    leaf_joint_indices = set(range(len(parents))) - set(parents)
    leaf_frame_names = tuple(
        frame.name for frame in pinocchio_model.frames
        if frame.parent in leaf_joint_indices)

    if frame_names == 'auto':
        # Determine the most likely set of frames corresponding to the feet
        foot_frame_names = set()
        for sensor_class in (jiminy.ContactSensor, jiminy.ForceSensor):
            for sensor in env.robot.sensors.get(sensor_class.type, ()):
                assert isinstance(sensor, ((
                    jiminy.ContactSensor, jiminy.ForceSensor)))
                # Skip sensors not attached to a leaf joint
                if sensor.frame_name in leaf_frame_names:
                    # The joint name is used as frame name. This avoids
                    # considering multiple fixed frames wrt to the same
                    # joint. They would be completely redundant, slowing
                    # down computations for no reason.
                    frame = pinocchio_model.frames[sensor.frame_index]
                    joint_name = pinocchio_model.names[frame.parent]
                    foot_frame_names.add(joint_name)
        frame_names = tuple(foot_frame_names)

    # Make sure that at least one frame has been found
    if not frame_names:
        raise ValueError("At least one frame must be specified.")

    # Make sure that the frame names are end-effectors
    if any(name not in leaf_frame_names for name in frame_names):
        raise ValueError("All frames must correspond to end-effectors.")

    return sorted(frame_names)


@dataclass(unsafe_hash=True)
class BaseOdometryPose(AbstractQuantity[np.ndarray]):
    """Odometry pose of the floating base of the robot at the end of the agent
    step.

    The odometry pose fully specifies the position and heading of the robot in
    2D world plane. As such, it comprises the linear translation (X, Y) and
    the rotation around Z axis (namely rate of change of Yaw Euler angle).
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Call base implementation
        super().__init__(
            env, parent, requirements={}, mode=mode, auto_refresh=False)

        # Translation (X, Y) and rotation matrix of the floating base
        self.xy, self.rot_mat = np.array([]), np.array([])

        # Buffer to store the odometry pose
        self.data = np.zeros((3,))

        # Split odometry pose in translation (X, Y) and yaw angle
        self.xy_view = self.data[:2]
        self.yaw_view = self.data[-1:].reshape(())

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Make sure that the robot has a floating base
        if not self.env.robot.has_freeflyer:
            raise RuntimeError(
                "Robot has no floating base. Cannot compute this quantity.")

        # Refresh proxies
        base_pose = self.pinocchio_data.oMf[1]
        self.xy, self.rot_mat = base_pose.translation[:2], base_pose.rotation

    def refresh(self) -> np.ndarray:
        # Copy translation part
        array_copyto(self.xy_view, self.xy)

        # Compute Yaw angle
        matrix_to_yaw(self.rot_mat, self.yaw_view)

        # Return buffer
        return self.data


@dataclass(unsafe_hash=True)
class MultiFootMeanXYZQuat(InterfaceQuantity[np.ndarray]):
    """Average position and orientation of the feet of a legged robot at the
    end of the agent step.

    The average foot pose may be more appropriate than the floating base pose
    to characterize the position and orientation the robot in the world,
    especially when it comes to assessing the tracking error of the foot
    trajectories. It has the advantage to make foot tracking independent from
    floating base tracking, giving the opportunity to the robot to locally
    recover stability by moving its upper body without impeding foot tracking.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames corresponding to some feet of the robot.

    These frames must be part of the end-effectors, ie being associated with a
    leaf joint in the kinematic tree of the robot.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto',
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_names: Name of the frames corresponding to some feet of
                            the robot. 'auto' to automatically detect them from
                            the set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        :param mode: Desired mode of evaluation for this quantity.
        """
        # Backup some user argument(s)
        self.frame_names = tuple(sanitize_foot_frame_names(env, frame_names))
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(MultiFrameMeanXYZQuat, dict(
                    frame_names=self.frame_names,
                    mode=mode))),
            auto_refresh=False)

    def refresh(self) -> np.ndarray:
        return self.data


@dataclass(unsafe_hash=True)
class MultiFootMeanOdometryPose(InterfaceQuantity[np.ndarray]):
    """Odometry pose of the average position and orientation of the feet of a
    legged robot at the end of the agent step.

    Using the average foot odometry pose may be more appropriate than the
    floating base odometry pose to characterize the position and heading the
    robot in the world plane. See `MultiFootMeanXYZQuat` documentation for
    details.

    The odometry pose fully specifies the position and orientation of the robot
    in 2D world plane. See `BaseOdometryPose` documentation for details.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames corresponding to the feet of the robot.

    These frames must be part of the end-effectors, ie being associated with a
    leaf joint in the kinematic tree of the robot.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto',
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_names = tuple(sanitize_foot_frame_names(env, frame_names))
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(MultiFootMeanXYZQuat, dict(
                    frame_names=self.frame_names,
                    mode=mode))),
            auto_refresh=False)

        # Pre-allocate memory for the odometry pose (X, Y, Yaw)
        self._odom_pose = np.zeros((3,))

        # Split odometry pose in translation (X, Y) and yaw angle
        self._xy_view = self._odom_pose[:2]
        self._yaw_view = self._odom_pose[-1:].reshape(())

    def refresh(self) -> np.ndarray:
        # Copy translation part
        array_copyto(self._xy_view, self.data[:2])

        # Compute Yaw angle
        quat_to_yaw(self.data[-4:], self._yaw_view)

        return self._odom_pose


@dataclass(unsafe_hash=True)
class MultiFootRelativeXYZQuat(InterfaceQuantity[np.ndarray]):
    """Relative position and orientation of the feet of a legged robot wrt
    themselves at the end of the agent step.

    The reference frame used to compute the relative pose of the frames is the
    mean foot pose. See `MultiFootMeanXYZQuat` documentation for details.

    Note that there is always one of the relative frame pose that is redundant
    wrt the others. Notably, in particular case where there is only two frames,
    it is one is the opposite of the other. As a result, the last relative pose
    is always dropped from the returned value, based on the same ordering as
    'self.frame_names'. As for `MultiFrameXYZQuat`, the data associated with
    each frame are returned as a 2D contiguous array. The first dimension
    gathers the 7 components (X, Y, Z, QuatX, QuatY, QuatZ, QuaW), while the
    last one corresponds to individual relative frames poses.
    """

    frame_names: Tuple[str, ...]
    """Name of the frames corresponding to the feet of the robot.

    These frames must be part of the end-effectors, ie being associated with a
    leaf joint in the kinematic tree of the robot.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 frame_names: Union[Sequence[str], Literal['auto']] = 'auto',
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param frame_names: Name of the frames corresponding to the feet of the
                            robot. 'auto' to automatically detect them from the
                            set of contact and force sensors of the robot.
                            Optional: 'auto' by default.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.frame_names = tuple(sanitize_foot_frame_names(env, frame_names))
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                xyzquat_mean=(MultiFootMeanXYZQuat, dict(
                    frame_names=self.frame_names,
                    mode=mode)),
                positions=(MultiFramePosition, dict(
                    frame_names=self.frame_names,
                    mode=mode)),
                rotations=(MultiFrameOrientation, dict(
                    frame_names=self.frame_names,
                    type=Orientation.MATRIX,
                    mode=mode))),
            auto_refresh=False)

        # Buffer to store the inverse mean quaternion
        self._rot_mean = np.zeros((3, 3))

        # Buffer to store relative rotation matrices
        self._foot_rots_rel = np.zeros((3, 3, len(self.frame_names) - 1))

        # Pre-allocate memory for the relative poses of all feet
        self._foot_poses_rel = np.zeros((7, len(self.frame_names) - 1))

        # Define jit-able method translating multiple positions to local frame
        @nb.jit(nopython=True, cache=True, fastmath=True)
        def translate_positions(positions: np.ndarray,
                                position_ref: np.ndarray,
                                rotation_ref: np.ndarray,
                                out: np.ndarray) -> None:
            """Translate a batch of position vectors from world to local frame.

            :param positions: Batch of positions vectors as a 2D array whose
                              first dimension gathers the 3 spatial coordinates
                              (X, Y, Z) while the second corresponds to the
                              independent points.
            :param position_ref: Position of the reference frame in world.
            :param rotation_ref: Orientation of the reference frame in world as
                                 a 2D rotation matrix.
            :param out: Pre-allocated array into which the result is stored.
            """
            out[:] = rotation_ref.T @ (positions - position_ref[:, np.newaxis])

        self._translate_positions_fun = translate_positions

        # Split foot poses in position and orientation vectors
        self._foot_position_view = self._foot_poses_rel[:3]
        self._foot_quat_view = self._foot_poses_rel[-4:]

    def refresh(self) -> np.ndarray:
        # Extract mean frame position and quaternion
        xyzquat_mean = self.xyzquat_mean
        position_mean, quat_mean = xyzquat_mean[:3], xyzquat_mean[-4:]

        # Compute the mean rotation matrix
        quat_to_matrix(quat_mean, self._rot_mean)

        # Compute the relative frame position of each foot.
        # Note that the translation and orientation are treated independently
        # (double geodesic), to be consistent with the method that was used to
        # compute the mean foot pose. This way, the norm of the relative
        # position is not affected by the orientation of the feet.
        self._translate_positions_fun(self.positions[:, :-1],
                                      position_mean,
                                      self._rot_mean,
                                      self._foot_position_view)

        # Compute relative frame orientations.
        # Note that using quaternion to compose rotations or rotate position
        # vectors if significantly more costly than using rotation matrices.
        # Because of this, it is more efficient overall to perform all the
        # processing steps using rotation matrices and only convert them to
        # quaternions as the very last step.
        np.einsum('ij,jkn',
                  self._rot_mean.T,
                  self.rotations[..., :-1],
                  out=self._foot_rots_rel)

        # Convert relative orientations from rotation matrices to quaternions
        matrix_to_quat(self._foot_rots_rel, self._foot_quat_view)

        return self._foot_poses_rel


@dataclass(unsafe_hash=True)
class AverageOdometryVelocity(InterfaceQuantity[np.ndarray]):
    """Average odometry velocity of the floating base of the robot in
    local-world-aligned frame at the end of the agent step.

    The odometry pose fully specifies the position and orientation of the robot
    in 2D world plane. See `BaseOdometryPose` documentation for details.

    The average spatial velocity is obtained by finite difference. See
    `AverageFrameSpatialVelocity` documentation for details.
    """

    mode: QuantityEvalMode
    """Specify on which state to evaluate this quantity. See `Mode`
    documentation for details about each mode.

    .. warning::
        Mode `REFERENCE` requires a reference trajectory to be selected
        manually prior to evaluating this quantity for the first time.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.mode = mode

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                data=(MaskedQuantity, dict(
                    quantity=(AverageFrameSpatialVelocity, dict(
                        frame_name="root_joint",
                        reference_frame=pin.LOCAL_WORLD_ALIGNED,
                        mode=mode)),
                    key=(0, 1, 5)))),
            auto_refresh=False)

    def refresh(self) -> np.ndarray:
        return self.data


@dataclass(unsafe_hash=True)
class CenterOfMass(AbstractQuantity[np.ndarray]):
    """Position, Velocity or Acceleration of the center of mass of the robot as
    a whole.
    """

    kinematic_level: pin.KinematicLevel
    """Kinematic level to compute, ie position, velocity or acceleration.
    """

    def __init__(
            self,
            env: InterfaceJiminyEnv,
            parent: Optional[InterfaceQuantity],
            *,
            kinematic_level: pin.KinematicLevel = pin.POSITION,
            mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :para kinematic_level: Desired kinematic level, ie position, velocity
                               or acceleration.
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Backup some user argument(s)
        self.kinematic_level = kinematic_level

        # Call base implementation
        super().__init__(
            env, parent, requirements={}, mode=mode, auto_refresh=False)

        # Pre-allocate memory for the CoM quantity
        self._com_data: np.ndarray = np.array([])

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Make sure that the state is consistent with required kinematic level
        state = self.state
        if ((self.kinematic_level == pin.ACCELERATION and state.a is None) or
                (self.kinematic_level >= pin.VELOCITY and state.v is None)):
            raise RuntimeError(
                "State data inconsistent with required kinematic level")

        # Refresh CoM quantity proxy based on kinematic level
        if self.kinematic_level == pin.POSITION:
            self._com_data = self.pinocchio_data.com[0]
        elif self.kinematic_level == pin.VELOCITY:
            self._com_data = self.pinocchio_data.vcom[0]
        else:
            self._com_data = self.pinocchio_data.acom[0]

    def refresh(self) -> np.ndarray:
        # Jiminy does not compute the CoM acceleration automatically
        if (self.mode == QuantityEvalMode.TRUE and
                self.kinematic_level == pin.ACCELERATION):
            pin.centerOfMass(self.pinocchio_model,
                             self.pinocchio_data,
                             pin.ACCELERATION)

        # Return proxy directly without copy
        return self._com_data


@dataclass(unsafe_hash=True)
class ZeroMomentPoint(AbstractQuantity[np.ndarray]):
    """Zero-Tilting Moment Point (ZMP), also called Center of Pressure (CoP).

    This quantity only makes sense for legged robots. Such a robot will keep
    balance if the ZMP [1] is maintained inside the support polygon [2].

    .. seealso::
        For academic reference about its relation with the notion of stability,
        see: [1] https://scaron.info/robotics/zero-tilting-moment-point.html
             [2] https://scaron.info/robotics/zmp-support-area.html
    """

    reference_frame: pin.ReferenceFrame
    """Whether to compute the ZMP in local frame specified by the odometry pose
    of floating base of the robot or the frame located on the position of the
    floating base with axes kept aligned with world frame.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 reference_frame: pin.ReferenceFrame = pin.LOCAL,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param reference_frame: Whether to compute the ZMP in local odometry
                                frame (aka 'pin.LOCAL') or aligned with world
                                axes (aka 'pin.LOCAL_WORLD_ALIGNED').
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure at requested reference frame is supported
        if reference_frame not in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED):
            raise ValueError("Reference frame must be either 'pin.LOCAL' or "
                             "'pin.LOCAL_WORLD_ALIGNED'.")

        # Backup some user argument(s)
        self.reference_frame = reference_frame

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                com=(CenterOfMass, dict(
                    kinematic_level=pin.POSITION,
                    mode=mode)),
                odom_pose=(BaseOdometryPose, dict(mode=mode))
            ),
            mode=mode,
            auto_refresh=False)

        # Weight of the robot
        self._robot_weight: float = -1

        # Proxy for the derivative of the spatial centroidal momentum
        self.dhg: Tuple[np.ndarray, np.ndarray] = (np.array([]),) * 2

        # Pre-allocate memory for the ZMP
        self._zmp = np.zeros(2)

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Make sure that the state is consistent with required kinematic level
        if (self.state.v is None or self.state.a is None):
            raise RuntimeError(
                "State data inconsistent with required kinematic level")

        # Compute the weight of the robot
        gravity = abs(self.pinocchio_model.gravity.linear[2])
        robot_mass = self.pinocchio_data.mass[0]
        self._robot_weight = robot_mass * gravity

        # Refresh derivative of the spatial centroidal momentum
        self.dhg = ((dhg := self.pinocchio_data.dhg).linear, dhg.angular)

        # Re-initialized pre-allocated memory buffer
        fill(self._zmp, 0)

    def refresh(self) -> np.ndarray:
        # Extract intermediary quantities for convenience
        (dhg_linear, dhg_angular), com = self.dhg, self.com

        # Compute the vertical force applied by the robot
        f_z = dhg_linear[2] + self._robot_weight

        # Compute the ZMP in world frame
        self._zmp[:] = com[:2] * (self._robot_weight / f_z)
        if abs(f_z) > np.finfo(np.float32).eps:
            self._zmp[0] -= (dhg_angular[1] + dhg_linear[0] * com[2]) / f_z
            self._zmp[1] += (dhg_angular[0] - dhg_linear[1] * com[2]) / f_z

        # Translate the ZMP from world to local odometry frame if requested
        if self.reference_frame == pin.LOCAL:
            base_yaw = self.odom_pose[2]
            cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)
            self._zmp[:] = (+ cos_yaw * self._zmp[0] + sin_yaw * self._zmp[1],
                            - sin_yaw * self._zmp[0] + cos_yaw * self._zmp[1])

        return self._zmp


@dataclass(unsafe_hash=True)
class CapturePoint(AbstractQuantity[np.ndarray]):
    """Divergent Component of Motion (DCM), also called Capture Point (CP).

    This quantity only makes sense for legged robots, and in particular bipedal
    robots for which the inverted pendulum is a relevant approximate dynamic
    model. It is involved in various dynamic stability metrics (usually only on
    flat ground), such as N-steps capturability. The capture point is defined
    as "where should a bipedal robot should step right now to eliminate linear
    momentum and come asymptotically to a stop" [1].

    .. seealso::
        For academic reference about its relation with the notion of stability,
        see: [1] https://scaron.info/robotics/capture-point.html
    """

    reference_frame: pin.ReferenceFrame
    """Whether to compute the DCM in local frame specified by the odometry pose
    of floating base of the robot or the frame located on the position of the
    floating base with axes kept aligned with world frame.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv,
                 parent: Optional[InterfaceQuantity],
                 *,
                 reference_frame: pin.ReferenceFrame = pin.LOCAL,
                 mode: QuantityEvalMode = QuantityEvalMode.TRUE) -> None:
        """
        :param env: Base or wrapped jiminy environment.
        :param parent: Higher-level quantity from which this quantity is a
                       requirement if any, `None` otherwise.
        :param reference_frame: Whether to compute the DCM in local odometry
                                frame (aka 'pin.LOCAL') or aligned with world
                                axes (aka 'pin.LOCAL_WORLD_ALIGNED').
        :param mode: Desired mode of evaluation for this quantity.
                     Optional: 'QuantityEvalMode.TRUE' by default.
        """
        # Make sure at requested reference frame is supported
        if reference_frame not in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED):
            raise ValueError("Reference frame must be either 'pin.LOCAL' or "
                             "'pin.LOCAL_WORLD_ALIGNED'.")

        # Backup some user argument(s)
        self.reference_frame = reference_frame

        # Call base implementation
        super().__init__(
            env,
            parent,
            requirements=dict(
                com_position=(CenterOfMass, dict(
                    kinematic_level=pin.POSITION,
                    mode=mode)),
                com_velocity=(CenterOfMass, dict(
                    kinematic_level=pin.VELOCITY,
                    mode=mode)),
                odom_pose=(BaseOdometryPose, dict(mode=mode))
            ),
            mode=mode,
            auto_refresh=False)

        # Natural frequency of linear pendulum approximate model of the robot
        self.omega: float = float("nan")

        # Pre-allocate memory for the DCM
        self._dcm = np.zeros(2)

    def initialize(self) -> None:
        # Call base implementation
        super().initialize()

        # Make sure that the state is consistent with required kinematic level
        if self.state.v is None:
            raise RuntimeError(
                "State data inconsistent with required kinematic level")

        # Compute the natural frequency of linear pendulum approximate model.
        # Note that the height of the robot is defined as the position of the
        # center of mass of the robot in neutral configuration.
        update_quantities(
            self.robot,
            pin.neutral(self.robot.pinocchio_model_th),
            update_physics=True,
            update_centroidal=True,
            update_energy=False,
            update_jacobian=False,
            update_collisions=False,
            use_theoretical_model=True)
        min_height = min(
            oMf.translation[2] for oMf in self.robot.pinocchio_data_th.oMf)
        gravity = abs(self.pinocchio_model.gravity.linear[2])
        robot_height = self.robot.pinocchio_data_th.com[0][2] - min_height
        self.omega = math.sqrt(gravity / robot_height)

        # Re-initialized pre-allocated memory buffer
        fill(self._dcm, 0)

    def refresh(self) -> np.ndarray:
        # Compute the DCM in world frame
        com_position, com_velocity = self.com_position, self.com_velocity
        self._dcm[:] = com_position[:2] + com_velocity[:2] / self.omega

        # Translate the ZMP from world to local odometry frame if requested
        if self.reference_frame == pin.LOCAL:
            base_yaw = self.odom_pose[2]
            cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)
            self._dcm[:] = (+ cos_yaw * self._dcm[0] + sin_yaw * self._dcm[1],
                            - sin_yaw * self._dcm[0] + cos_yaw * self._dcm[1])

        return self._dcm
