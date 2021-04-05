import logging
import numpy as np
from typing import Optional, Tuple, Callable

import hppfcl
import pinocchio as pin
from pinocchio.rpy import (rpyToMatrix,
                           matrixToRpy,
                           computeRpyJacobian,
                           computeRpyJacobianInverse)

from . import core as jiminy
from .viewer import TrajectoryDataType


logger = logging.getLogger(__name__)


# #####################################################################
# ######################### Generic math ##############################
# #####################################################################

def SE3ToXYZRPY(M):
    """Convert Pinocchio SE3 object to [X,Y,Z,Roll,Pitch,Yaw] vector.
    """
    return np.concatenate((M.translation, matrixToRpy(M.rotation)))


def XYZRPYToSE3(xyzrpy):
    """Convert [X,Y,Z,Roll,Pitch,Yaw] vector to Pinocchio SE3 object.
    """
    return pin.SE3(rpyToMatrix(xyzrpy[3:]), xyzrpy[:3])


def XYZRPYToXYZQuat(xyzrpy):
    """Convert [X,Y,Z,Roll,Pitch,Yaw] to [X,Y,Z,Qx,Qy,Qz,Qw].
    """
    return pin.SE3ToXYZQUAT(XYZRPYToSE3(xyzrpy))


def XYZQuatToXYZRPY(xyzquat):
    """Convert [X,Y,Z,Qx,Qy,Qz,Qw] to [X,Y,Z,Roll,Pitch,Yaw].
    """
    return np.concatenate((
        xyzquat[:3], matrixToRpy(pin.Quaternion(xyzquat[3:]).matrix())))


def velocityXYZRPYToXYZQuat(xyzrpy: np.ndarray,
                            dxyzrpy: np.ndarray) -> np.ndarray:
    """Convert the derivative of [X,Y,Z,Roll,Pitch,Yaw] to [X,Y,Z,Qx,Qy,Qz,Qw].

    .. warning::
        Linear velocity in XYZRPY must be local-world-aligned frame, while
        returned linear velocity in XYZQuat is in local frame.
    """
    rpy = xyzrpy[3:]
    R = rpyToMatrix(rpy)
    J_rpy = computeRpyJacobian(rpy)
    return np.concatenate((R.T @ dxyzrpy[:3], J_rpy @ dxyzrpy[3:]))


def velocityXYZQuatToXYZRPY(xyzquat: np.ndarray,
                            v: np.ndarray) -> np.ndarray:
    """Convert the derivative of [X,Y,Z,Roll,Pitch,Yaw] to [X,Y,Z,Qx,Qy,Qz,Qw].

    .. warning::
        Linear velocity in XYZRPY must be local-world-aligned frame, while
        returned linear velocity in XYZQuat is in local frame.
    """
    quat = pin.Quaternion(xyzquat[3:])
    rpy = matrixToRpy(quat.matrix())
    J_rpy_inv = computeRpyJacobianInverse(rpy)
    return np.concatenate((quat * v[:3], J_rpy_inv @ v[3:]))


# #####################################################################
# ################### Kinematic and dynamics ##########################
# #####################################################################

def update_quantities(robot: jiminy.Robot,
                      position: np.ndarray,
                      velocity: Optional[np.ndarray] = None,
                      acceleration: Optional[np.ndarray] = None,
                      update_physics: bool = True,
                      update_com: bool = True,
                      update_energy: bool = True,
                      update_jacobian: bool = False,
                      update_collisions: bool = True,
                      use_theoretical_model: bool = True) -> None:
    """Compute all quantities using position, velocity and acceleration
    configurations.

    Run multiple algorithms to compute all quantities which can be known with
    the model position, velocity and acceleration.

    This includes:
    - body spatial transforms,
    - body spatial velocities,
    - body spatial drifts,
    - body transform acceleration,
    - body transform jacobians,
    - center-of-mass position,
    - center-of-mass velocity,
    - center-of-mass drift,
    - center-of-mass acceleration,
    (- center-of-mass jacobian : No Python binding available so far),
    - articular inertia matrix,
    - non-linear effects (Coriolis + gravity)
    - collisions and distances

    .. note::
        Computation results are stored internally in the robot, and can
        be retrieved with associated getters.

    .. warning::
        This function modifies the internal robot data.

    :param robot: Jiminy robot.
    :param position: Robot position vector.
    :param velocity: Robot velocity vector.
    :param acceleration: Robot acceleration vector.
    :param update_physics: Whether or not to compute the non-linear effects and
                           internal/external forces.
                           Optional: True by default.
    :param update_com: Whether or not to compute the COM of the robot AND each
                       link individually. The global COM is the first index.
                       Optional: False by default.
    :param update_energy: Whether or not to compute the energy of the robot.
                          Optional: False by default
    :param update_jacobian: Whether or not to compute the jacobians.
                            Optional: False by default.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the robot's state.
                                  Optional: True by default.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    if (update_physics and update_com and
            update_energy and update_jacobian and
            velocity is not None and acceleration is None):
        pin.computeAllTerms(pnc_model, pnc_data, position, velocity)
    else:
        if velocity is None:
            pin.forwardKinematics(pnc_model, pnc_data, position)
        elif acceleration is None:
            pin.forwardKinematics(pnc_model, pnc_data, position, velocity)
        else:
            pin.forwardKinematics(
                pnc_model, pnc_data, position, velocity, acceleration)

        if update_com:
            if velocity is None:
                pin.centerOfMass(pnc_model, pnc_data, position)
            elif acceleration is None:
                pin.centerOfMass(pnc_model, pnc_data, position, velocity)
            else:
                pin.centerOfMass(
                    pnc_model, pnc_data, position, velocity, acceleration)

        if update_jacobian:
            if update_com:
                pin.jacobianCenterOfMass(pnc_model, pnc_data)
            pin.computeJointJacobians(pnc_model, pnc_data)

        if update_physics:
            if velocity is not None:
                pin.nonLinearEffects(pnc_model, pnc_data, position, velocity)
            pin.crba(pnc_model, pnc_data, position)

        if update_energy:
            if velocity is not None:
                pin.computeKineticEnergy(pnc_model, pnc_data)
            pin.computePotentialEnergy(pnc_model, pnc_data)

    pin.updateFramePlacements(pnc_model, pnc_data)

    if update_collisions:
        pin.updateGeometryPlacements(
            pnc_model, pnc_data, robot.collision_model, robot.collision_data)
        pin.computeCollisions(
            robot.collision_model, robot.collision_data,
            stop_at_first_collision=False)
        pin.computeDistances(robot.collision_model, robot.collision_data)
        for dist_req in robot.collision_data.distanceResults:
            if np.linalg.norm(dist_req.normal) < 1e-6:
                pin.computeDistances(
                    robot.collision_model, robot.collision_data)
                break


def get_body_index_and_fixedness(
        robot: jiminy.Robot,
        body_name: str,
        use_theoretical_model: bool = True) -> Tuple[int, bool]:
    """Retrieve the body index and fixedness from its name.

    :param robot: Jiminy robot.
    :param body_name: Name of the body.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the robot's state.
                                  Optional: True by default.

    :returns: [0] Index of the body. [1] Whether or not it is a fixed body.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
    else:
        pnc_model = robot.pinocchio_model

    frame_id = pnc_model.getFrameId(body_name)
    parent_frame_id = pnc_model.frames[frame_id].previousFrame
    parent_frame_type = pnc_model.frames[parent_frame_id].type
    is_body_fixed = (parent_frame_type == pin.FrameType.FIXED_JOINT)
    if is_body_fixed:
        body_id = frame_id
    else:
        body_id = pnc_model.frames[frame_id].parent

    return body_id, is_body_fixed


def get_body_world_transform(robot: jiminy.Robot,
                             body_name: str,
                             use_theoretical_model: bool = True,
                             copy: bool = True) -> pin.SE3:
    """Get the transform from world frame to body frame for a given body.

    .. warning::
        It is assumed that `update_quantities` has been called beforehand.

    :param robot: Jiminy robot.
    :param body_name: Name of the body.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the robot's state.
                                  Optional: True by default.
    :param copy: Whether to return the internal buffers (which could be
                 altered) or copy them.
                 Optional: True by default. It is less efficient but safer.

    :returns: Body transform.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    transform = pnc_data.oMf[pnc_model.getFrameId(body_name)]
    if copy:
        transform = transform.copy()
    return transform


def get_body_world_velocity(robot: jiminy.Robot,
                            body_name: str,
                            use_theoretical_model: bool = True) -> pin.SE3:
    """Get the spatial velocity wrt world in body frame for a given body.

    .. warning::
        It is assumed that `update_quantities` has been called.

    :param robot: Jiminy robot.
    :param body_name: Name of the body.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the robot's state.
                                  Optional: True by default.

    :returns: Spatial velocity.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    body_id, body_is_fixed = get_body_index_and_fixedness(
        robot, body_name, use_theoretical_model)
    if body_is_fixed:
        last_moving_parent_id = pnc_model.frames[body_id].parent
        parent_transform_in_world = pnc_data.oMi[last_moving_parent_id]
        parent_velocity_in_parent_frame = pnc_data.v[last_moving_parent_id]
        spatial_velocity = parent_velocity_in_parent_frame.se3Action(
            parent_transform_in_world)
    else:
        transform = pnc_data.oMi[body_id]
        velocity_in_body_frame = pnc_data.v[body_id]
        spatial_velocity = velocity_in_body_frame.se3Action(transform)

    return spatial_velocity


def get_body_world_acceleration(robot: jiminy.Robot,
                                body_name: str,
                                use_theoretical_model: bool = True) -> pin.SE3:
    """Get the body spatial acceleration in world frame.

    The moment of this tensor (i.e linear part) is NOT the linear acceleration
    of the center of the body frame, expressed in the world frame.

    .. warning::
        It is assumed that `update_quantities` has been called.

    :param robot: Jiminy robot.
    :param body_name: Name of the body.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the robot's state.
                                  Optional: True by default.

    :returns: Spatial acceleration.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    body_id, body_is_fixed = get_body_index_and_fixedness(
        robot, body_name, use_theoretical_model)

    if body_is_fixed:
        last_moving_parent_id = pnc_model.frames[body_id].parent
        parent_transform_in_world = pnc_data.oMi[last_moving_parent_id]
        parent_acceleration_in_parent_frame = pnc_data.a[last_moving_parent_id]
        spatial_acceleration = parent_acceleration_in_parent_frame.se3Action(
            parent_transform_in_world)
    else:
        transform = pnc_data.oMi[body_id]
        acceleration_in_body_frame = pnc_data.a[body_id]
        spatial_acceleration = acceleration_in_body_frame.se3Action(transform)

    return spatial_acceleration


def compute_transform_contact(
        robot: jiminy.Robot,
        ground_profile: Optional[Callable[
            [np.ndarray], Tuple[float, np.ndarray]]] = None) -> pin.SE3:
    """Compute the transform the apply to the freeflyer to touch the ground
    with up to 3 contact points.

    This method can be used in conjunction with
    `compute_freeflyer_state_from_fixed_body` to ensures no contact points are
    going through the ground and up to three are in contact.

    .. warning::
        It is assumed that `update_quantities` has been called.

    :param robot: Jiminy robot.
    :param ground_profile: Ground profile callback.

    :returns: The transform the apply in order to touch the ground.
              If the robot has no contact point, then the identity is returned.
    """
    # Compute the transform in the world of the contact points
    contact_frames_transform = []
    for frame_idx in robot.contact_frames_idx:
        transform = robot.pinocchio_data.oMf[frame_idx]
        contact_frames_transform.append(transform)

    # Compute the transform of the ground at these points
    if ground_profile is not None:
        contact_ground_transform = []
        ground_pos = np.zeros(3)
        for frame_transform in contact_frames_transform:
            ground_pos[2], normal = ground_profile(frame_transform.translation)
            ground_rot = pin.Quaternion.FromTwoVectors(
                np.array([0.0, 0.0, 1.0]), normal).matrix()
            contact_ground_transform.append(pin.SE3(ground_rot, ground_pos))
    else:
        contact_ground_transform = [
            pin.SE3.Identity() for _ in contact_frames_transform]

    # Compute the position and normal of the contact points wrt their
    # respective ground transform.
    contact_frames_pos_rel = []
    for frame_transform, ground_transform in \
            zip(contact_frames_transform, contact_ground_transform):
        transform_rel = ground_transform.actInv(frame_transform)
        contact_frames_pos_rel.append(transform_rel.translation)

    # Order the contact points by depth
    contact_frames_pos_rel = [contact_frames_pos_rel[i] for i in np.argsort(
        [pos[2] for pos in contact_frames_pos_rel])]

    # Compute the contact plane normal
    if len(contact_frames_pos_rel) > 2:
        contact_edge_ref = \
            contact_frames_pos_rel[0] - contact_frames_pos_rel[1]
        contact_edge_ref /= np.linalg.norm(contact_edge_ref)
        for i in range(2, len(contact_frames_pos_rel)):
            contact_edge_alt = \
                contact_frames_pos_rel[0] - contact_frames_pos_rel[i]
            contact_edge_alt /= np.linalg.norm(contact_edge_alt)
            normal_offset = np.cross(contact_edge_ref, contact_edge_alt)
            if np.linalg.norm(normal_offset) > 0.2:  # At least 11 degrees
                break
        if normal_offset[2] < 0.0:
            normal_offset *= -1.0
    else:
        normal_offset = np.array([0.0, 0.0, 1.0])

    # Make sure that the normal is valid, otherwise use the default one
    if np.linalg.norm(normal_offset) < 0.2:
        normal_offset = np.array([0.0, 0.0, 1.0])

    # Compute the translation and rotation to apply the touch the ground
    rot_offset = pin.Quaternion.FromTwoVectors(
        normal_offset, np.array([0.0, 0.0, 1.0])).matrix()
    if contact_frames_pos_rel:
        pos_offset = np.array([
            0.0, 0.0, -(rot_offset.T @ contact_frames_pos_rel[0])[2]])
    else:
        pos_offset = np.zeros(3)
    transform_offset = pin.SE3(rot_offset, pos_offset)

    # Take into account the collision bodies
    # TODO: Take into account the ground profile
    min_distance = np.inf
    deepest_idx = None
    for i, dist_req in enumerate(robot.collision_data.distanceResults):
        if np.linalg.norm(dist_req.normal) > 1e-6:
            body_idx = robot.collision_model.collisionPairs[0].first
            body_geom = robot.collision_model.geometryObjects[body_idx]
            if dist_req.normal[2] > 0.0 and \
                    isinstance(body_geom.geometry, hppfcl.Box):
                ground_idx = robot.collision_model.collisionPairs[0].second
                ground_geom = robot.collision_model.geometryObjects[ground_idx]
                box_size = 2.0 * ground_geom.geometry.halfSide
                body_size = 2.0 * body_geom.geometry.halfSide
                distance = - body_size[2] - box_size[2] - dist_req.min_distance
            else:
                distance = dist_req.min_distance
            if distance < min_distance:
                min_distance = distance
                deepest_idx = i
        else:
            logger.warning("Collision computation failed for some reason. "
                           "Skipping this collision pair.")
    if deepest_idx is not None and (
            not contact_frames_pos_rel or
            transform_offset.translation[2] < -min_distance):
        transform_offset.translation[2] = -min_distance
        if not contact_frames_pos_rel:
            geom_idx = robot.collision_model.collisionPairs[deepest_idx].first
            geom = robot.collision_model.geometryObjects[geom_idx]
            if isinstance(geom.geometry, hppfcl.Box):
                dist_rslt = robot.collision_data.distanceResults[deepest_idx]
                collision_position = dist_rslt.getNearestPoint1()
                transform_offset.rotation = \
                    robot.collision_data.oMg[geom_idx].rotation.T
                transform_offset.translation[2] += (
                    collision_position -
                    transform_offset.rotation @ collision_position)[2]

    return transform_offset


def compute_freeflyer_state_from_fixed_body(
        robot: jiminy.Robot,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        fixed_body_name: Optional[str] = None,
        ground_profile: Optional[Callable[
            [np.ndarray], Tuple[float, np.ndarray]]] = None,
        use_theoretical_model: Optional[bool] = None) -> str:
    """Fill rootjoint data from articular data when a body is fixed and
    aligned with world frame.

    This method computes the position of freeflyer in the fixed body frame.

    If **fixed_body_name** is omitted, it will default to either:

        - the set of three contact points
        - a single collision body

    In such a way that no contact points nor collision bodies are going through
    the ground and at least contact points or a collision body are touching it.

    `None` is returned if their is no contact frame or if the robot does not
    have a freeflyer.

    .. warning::
        This function modifies the internal robot data.

    .. warning::
        The method fills in freeflyer data instead of returning an updated copy
        for efficiency.

    :param robot: Jiminy robot.
    :param position:
        Must contain current articular data. The freeflyer data can contain any
        value, it will be ignored and replaced.
    :param velocity: See position.
    :param acceleration: See position.
    :param fixed_body_name: Name of the body frame that is considered fixed
                            parallel to world frame.
                            Optional: It will be infered from the set of
                            contact points and collision bodies.
    :param ground_profile: Ground profile callback.
    :param use_theoretical_model:
        Whether the state corresponds to the theoretical model when updating
        and fetching the robot's state. Must be False if `fixed_body_name` is
        not speficied.
        Optional: True by default if `fixed_body_name` is specified, False
        otherwise.

    :returns: Name of the contact frame, if any.
    """
    # Early return if no freeflyer
    if not robot.has_freeflyer:
        return None

    # Handling of default arguments
    if use_theoretical_model is None:
        use_theoretical_model = fixed_body_name is not None

    position[:6].fill(0.0)
    position[6] = 1.0
    if velocity is not None:
        velocity[:6].fill(0.0)
    if acceleration is not None:
        acceleration[:6].fill(0.0)
    update_quantities(
        robot, position, velocity, acceleration, update_physics=False,
        use_theoretical_model=use_theoretical_model)

    if fixed_body_name is None:
        if use_theoretical_model:
            raise RuntimeError(
                "Cannot infer contact transform for theoretical model.")
        w_M_ff = compute_transform_contact(robot, ground_profile)
    else:
        ff_M_fixed_body = get_body_world_transform(
            robot, fixed_body_name, use_theoretical_model, copy=False)
        if ground_profile is not None:
            ground_translation = np.zeros(3)
            ground_translation[2], normal = ground_profile(
                ff_M_fixed_body.translation)
            ground_rotation = pin.Quaternion.FromTwoVectors(
                np.array([0.0, 0.0, 1.0]), normal).matrix()
            w_M_ground = pin.SE3(ground_rotation, ground_translation)
        else:
            w_M_ground = pin.SE3.Identity()
        w_M_ff = w_M_ground.act(ff_M_fixed_body.inverse())
    position[:7] = pin.SE3ToXYZQUAT(w_M_ff)

    if fixed_body_name is not None:
        if velocity is not None:
            ff_v_fixed_body = get_body_world_velocity(
                robot, fixed_body_name, use_theoretical_model)
            base_link_velocity = - ff_v_fixed_body
            velocity[:6] = base_link_velocity.vector

        if acceleration is not None:
            ff_a_fixedBody = get_body_world_acceleration(
                robot, fixed_body_name, use_theoretical_model)
            base_link_acceleration = - ff_a_fixedBody
            acceleration[:6] = base_link_acceleration.vector

    return fixed_body_name


def compute_efforts_from_fixed_body(
        robot: jiminy.Robot,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        fixed_body_name: str,
        use_theoretical_model: bool = True) -> Tuple[np.ndarray, pin.Force]:
    """Compute the efforts using RNEA method.

    .. warning::
        This function modifies the internal robot data.

    :param robot: Jiminy robot
    :param position: Robot configuration vector.
    :param velocity: Robot velocity vector.
    :param acceleration: Robot acceleration vector.
    :param fixed_body_name: Name of the body frame.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the robot's state.
                                  Optional: True by default.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    # Apply a first run of rnea without explicit external forces
    pin.computeJointJacobians(pnc_model, pnc_data, position)
    pin.rnea(pnc_model, pnc_data, position, velocity, acceleration)

    # Initialize vector of exterior forces to zero
    f_ext = pin.StdVec_Force()
    f_ext.extend(len(pnc_model.names) * (pin.Force.Zero(),))

    # Compute the force at the contact frame
    support_foot_idx = pnc_model.frames[
        pnc_model.getBodyId(fixed_body_name)].parent
    f_ext[support_foot_idx] = pnc_data.oMi[support_foot_idx] \
        .actInv(pnc_data.oMi[1]).act(pnc_data.f[1])

    # Recompute the efforts with RNEA and the correct external forces
    tau = pin.rnea(
        pnc_model, pnc_data, position, velocity, acceleration, f_ext)
    f_ext = f_ext[support_foot_idx]

    return tau, f_ext


# #####################################################################
# ################### State sequence wrappers #########################
# #####################################################################

def retrieve_freeflyer(trajectory_data: TrajectoryDataType,
                       freeflyer_continuity: bool = True) -> None:
    """Retrieves the freeflyer positions and velocities.

    .. warning::
        This function modifies the internal robot data.

    :param trajectory_data: Sequence of States for which to retrieve the
                            freeflyer.
    :param freeflyer_continuity: Whether or not to enforce the continuity
                                 in position of the freeflyer.
                                 Optional: True by default.
    """
    robot = trajectory_data['robot']

    contact_frame_prev = None
    w_M_ff_offset = pin.SE3.Identity()
    w_M_ff_prev = None
    for s in trajectory_data['evolution_robot']:
        # Compute freeflyer using contact frame as reference frame
        s.contact_frame = compute_freeflyer_state_from_fixed_body(
            robot, s.q, s.v, s.a, s.contact_frame, None)

        # Move freeflyer to ensure continuity over time, if requested
        if freeflyer_continuity:
            # Extract the current freeflyer transform
            w_M_ff = pin.XYZQUATToSE3(s.q[:7])

            # Update the internal buffer of the freeflyer transform
            if contact_frame_prev is not None \
                    and contact_frame_prev != s.contact_frame:
                w_M_ff_offset = w_M_ff_offset * w_M_ff_prev * w_M_ff.inverse()
            contact_frame_prev = s.contact_frame
            w_M_ff_prev = w_M_ff

            # Add the appropriate offset to the freeflyer
            w_M_ff = w_M_ff_offset * w_M_ff
            s.q[:7] = pin.SE3ToXYZQUAT(w_M_ff)


def compute_efforts(trajectory_data: TrajectoryDataType) -> None:
    """Compute the efforts in the trajectory using RNEA method.

    :param trajectory_data: Sequence of States for which to compute the
                            efforts.
    """
    robot = trajectory_data['robot']
    use_theoretical_model = trajectory_data['use_theoretical_model']

    for s in trajectory_data['evolution_robot']:
        s.tau, s.f_ext = compute_efforts_from_fixed_body(
            robot, s.q, s.v, s.a, s.contact_frame, use_theoretical_model)
