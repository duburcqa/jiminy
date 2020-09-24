## @file jiminy_py/dynamics.py
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any

import pinocchio as pin
from pinocchio.rpy import rpyToMatrix, matrixToRpy

from . import core as jiminy


######################################################################
########################## Generic math ##############################
######################################################################

def se3ToXYZRPY(M):
    p = np.zeros((6,))
    p[:3] = M.translation
    p[3:] = matrixToRpy(M.rotation)
    return p

def XYZRPYToSe3(xyzrpy):
    return pin.SE3(rpyToMatrix(xyzrpy[3:]), xyzrpy[:3])

######################################################################
#################### Kinematic and dynamics ##########################
######################################################################

def update_quantities(robot: jiminy.Robot,
                      position: np.ndarray,
                      velocity: Optional[np.ndarray] = None,
                      acceleration: Optional[np.ndarray] = None,
                      update_physics: bool = True,
                      update_com: bool = False,
                      update_energy: bool = False,
                      update_jacobian: bool = False,
                      use_theoretical_model: bool = True) -> None:
    """
    @brief Compute all quantities using position, velocity and acceleration
           configurations.

    @details Run multiple algorithms to compute all quantities which can be
             known with the model position, velocity and acceleration.

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

             Computation results are stored internally in the robot, and can
             be retrieved with associated getters.

    @remark This function modifies the internal robot data.

    @param robot  Jiminy robot.
    @param position  Robot position vector.
    @param velocity  Robot velocity vector.
    @param acceleration  Robot acceleration vector.
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    if (update_physics and update_com and \
        update_energy and update_jacobian and \
        velocity is not None):
        pin.computeAllTerms(pnc_model, pnc_data, position, velocity)
    else:
        if update_physics:
            if velocity is not None:
                pin.nonLinearEffects(pnc_model, pnc_data, position, velocity)
            pin.crba(pnc_model, pnc_data, position)

        if update_jacobian:
            # if update_com:
            #     pin.getJacobianComFromCrba(pnc_model, pnc_data)
            pin.computeJointJacobians(pnc_model, pnc_data)

        if update_com:
            if velocity is None:
                pin.centerOfMass(pnc_model, pnc_data, position)
            elif acceleration is None:
                pin.centerOfMass(pnc_model, pnc_data, position, velocity)
            else:
                pin.centerOfMass(
                    pnc_model, pnc_data, position, velocity, acceleration)
        else:
            if velocity is None:
                pin.forwardKinematics(pnc_model, pnc_data, position)
            elif acceleration is None:
                pin.forwardKinematics(pnc_model, pnc_data, position, velocity)
            else:
                pin.forwardKinematics(
                    pnc_model, pnc_data, position, velocity, acceleration)
            pin.framesForwardKinematics(pnc_model, pnc_data, position)

        pin.computeCollisions(robot.collision_model, robot.collision_data,
            stop_at_first_collision=False)
        pin.computeDistances(robot.collision_model, robot.collision_data)

        if update_energy:
            if velocity is not None:
                pin.kineticEnergy(
                    pnc_model, pnc_data, position, velocity, False)
            pin.potentialEnergy(pnc_model, pnc_data, position, False)

def get_body_index_and_fixedness(
        robot: jiminy.Robot,
        body_name: str,
        use_theoretical_model: bool = True) -> Tuple[int, bool]:
    """
    @brief Retrieve the body index and fixedness from its name.

    @param robot  Jiminy robot.
    @param body_name  Name of the body.
    @return [0] Index of the body.
            [1] Whether or not it is a fixed body.
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
    """
    @brief Get the transform from world frame to body frame for a given body.

    @remark It is assumed that `update_quantities` has been called beforehand.

    @param robot  Jiminy robot.
    @param body_name  Name of the body.

    @return Body transform.
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
    """
    @brief Get the spatial velocity wrt world in body frame for a given body.

    @remark It is assumed that `update_quantities` has been called.

    @param robot  Jiminy robot.
    @param body_name  Name of the body.

    @return Spatial velocity.
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
    """
    @brief Get the body spatial acceleration in world frame.

    @details The moment of this tensor (i.e linear part) is NOT the linear
             acceleration of the center of the body frame, expressed in the
             world frame.

    @remark It is assumed that `update_quantities` has been called.

    @param body_name  Name of the body.

    @return Spatial acceleration.
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

def compute_transform_contact(robot: jiminy.Robot,
                              ground_profile: Optional[Callable] = None) -> pin.SE3:
    """
    @brief Compute the transform the apply to the freeflyer to touch the ground
           with up to 3 contact points.

    @details This method can be used in conjunction with
             `compute_freeflyer_state_from_fixed_body` to ensures no contact
             points are going through the ground and up to 3 are in contact.

             Note that collision bodies are NOT supported for now.

             If the robot has no contact point, then the identity is returned.

    @remark It is assumed that `update_quantities` has been called.

    @param robot  Jiminy robot.
    @param ground_profile  Ground profile callback.

    @return The transform the apply in order to touch the ground.
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
            pin.SE3.Identity() for _ in range(len(contact_frames_transform))]

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
        contact_edge_ref = contact_frames_pos_rel[0] - contact_frames_pos_rel[1]
        contact_edge_ref /= np.linalg.norm(contact_edge_ref)
        for i in range(2, len(contact_frames_pos_rel)):
            contact_edge_alt = contact_frames_pos_rel[0] - contact_frames_pos_rel[i]
            contact_edge_alt /= np.linalg.norm(contact_edge_alt)
            normal_offset = np.cross(contact_edge_ref, contact_edge_alt)
            if np.linalg.norm(normal_offset) > 0.2:  # At least 11 degrees of angle
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
    for dist_req in robot.collision_data.distanceResults:
        min_distance = min(min_distance, dist_req.min_distance)
    transform_offset.translation[2] = max(
        transform_offset.translation[2], -min_distance)

    return transform_offset

def compute_freeflyer_state_from_fixed_body(
        robot: jiminy.Robot,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        fixed_body_name: Optional[str] = None,
        ground_profile: Optional[Callable] = None) -> str:
    """
    @brief Fill rootjoint data from articular data when a body is fixed
           parallel to world.

    @details If 'fixed_body_name' is omitted, it will default to the contact
             point that ensures no contact points are going through the ground
             and a single one is touching it.

             The hypothesis is that 'fixed_body_name' is fixed parallel to
             world frame. So this method computes the position of freeflyer
             rootjoint in the fixed body frame.

             Note that collision bodies are not supported for now. Yet, the
             vertical position of the freeflyer is updated in order to make
             sure the minimum collision distance is zero.

             `None` is returned if their is no contact frame or if the robot
             has no freeflyer.

    @remark This function modifies the internal robot data.

    @param robot Jiminy robot.
    @param[inout] position  Must contain current articular data. The rootjoint
                            data can contain any value, it will be ignored and
                            replaced. The method fills in rootjoint data.
    @param[inout] velocity  See position.
    @param[inout] acceleration  See position.
    @param fixed_body_name  Name of the body frame that is considered fixed
                            parallel to world frame.
    @param ground_profile  Ground profile callback.

    @return Name of the contact frame, if any.
    """
    if not robot.has_freeflyer:
        return None

    position[:6].fill(0.0)
    position[6] = 1.0
    if velocity is not None:
        velocity[:6].fill(0.0)
    if acceleration is not None:
        acceleration[:6].fill(0.0)
    update_quantities(robot, position, velocity, acceleration,
        update_physics=False, use_theoretical_model=False)

    if fixed_body_name is None:
        w_M_ff = compute_transform_contact(robot, ground_profile)
    else:
        ff_M_fixed_body = get_body_world_transform(
            robot, fixed_body_name, use_theoretical_model=False, copy=False)
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
    position[:7] = pin.se3ToXYZQUAT(w_M_ff)

    if fixed_body_name is None:
        if velocity is not None:
            ff_v_fixed_body = get_body_world_velocity(
                robot, fixed_body_name, use_theoretical_model=False)
            base_link_velocity = - ff_v_fixed_body
            velocity[:6] = base_link_velocity.vector

        if acceleration is not None:
            ff_a_fixedBody = get_body_world_acceleration(
                robot, fixed_body_name, use_theoretical_model=False)
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
    """
    @brief Compute the efforts using RNEA method.

    @remark This function modifies the internal robot data.

    @param robot  Jiminy robot
    @param position  Robot configuration vector.
    @param velocity  Robot velocity vector.
    @param acceleration  Robot acceleration vector.
    @param fixed_body_name  Name of the body frame.
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

######################################################################
#################### State sequence wrappers #########################
######################################################################

def retrieve_freeflyer(trajectory_data: Dict[str, Any],
                       freeflyer_continuity: bool = True) -> None:
    """
    @brief Retrieves the freeflyer positions and velocities.

    @details The reference frame is the support foot.

    @remark This function modifies the internal robot data.

    @param trajectory_data  Sequence of States for which to retrieve the
                            freeflyer.
    """
    robot = trajectory_data['robot']
    use_theoretical_model = trajectory_data['use_theoretical_model']

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
            w_M_ff = pin.XYZQUATToSe3(s.q[:7])

            # Update the internal buffer of the freeflyer transform
            if contact_frame_prev is not None \
                    and contact_frame_prev != s.contact_frame:
                w_M_ff_offset = w_M_ff_offset * w_M_ff_prev * w_M_ff.inverse()
            contact_frame_prev = s.contact_frame
            w_M_ff_prev = w_M_ff

            # Add the appropriate offset to the freeflyer
            w_M_ff = w_M_ff_offset * w_M_ff
            s.q[:7] = pin.se3ToXYZQUAT(w_M_ff)

def compute_efforts(trajectory_data: Dict[str, Any]) -> None:
    """
    @brief Compute the efforts in the trajectory using RNEA method.

    @param trajectory_data  Sequence of States for which to compute the
                            efforts.
    """
    robot = trajectory_data['robot']
    use_theoretical_model = trajectory_data['use_theoretical_model']

    for s in trajectory_data['evolution_robot']:
        s.tau, s.f_ext = compute_efforts_from_fixed_body(
            robot, s.q, s.v, s.a, s.contact_frame, use_theoretical_model)
