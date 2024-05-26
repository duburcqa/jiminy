"""Quantities mainly relevant for locomotion tasks on floating-base robots.
"""
import math
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from jiminy_py.core import array_copyto  # pylint: disable=no-name-in-module
from jiminy_py.dynamics import update_quantities
import pinocchio as pin

from ..bases import (
    InterfaceJiminyEnv, InterfaceQuantity, AbstractQuantity, QuantityEvalMode)
from ..utils import fill, matrix_to_yaw

from ..quantities import MaskedQuantity, AverageFrameSpatialVelocity


@dataclass(unsafe_hash=True)
class OdometryPose(AbstractQuantity[np.ndarray]):
    """Odometry pose agent step.

    The odometry pose fully specifies the position and orientation of the robot
    in 2D world plane. As such, it comprises the linear translation (X, Y) and
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
class AverageOdometryVelocity(InterfaceQuantity[np.ndarray]):
    """Average odometry velocity in local-world-aligned frame at the end of the
    agent step.

    The odometry pose fully specifies the position and orientation of the robot
    in 2D world plane. See `OdometryPose` documentation for details.

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
    """Whether the ZMP must be computed in local odometry frame or aligned with
    world axes.
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
                odom_pose=(OdometryPose, dict(mode=mode))
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
    """Whether the DCM must be computed in local odometry frame or aligned with
    world axes.
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
                odom_pose=(OdometryPose, dict(mode=mode))
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
