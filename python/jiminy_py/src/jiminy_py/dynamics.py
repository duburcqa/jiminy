#!/usr/bin/env python

## @file jiminy_py/dynamics.py

import numpy as np

import pinocchio as pin
from pinocchio.rpy import rpyToMatrix, matrixToRpy

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

def update_quantities(robot,
                      position,
                      velocity=None,
                      acceleration=None,
                      update_physics=True,
                      update_com=False,
                      update_energy=False,
                      update_jacobian=False,
                      use_theoretical_model=True):
    """
    @brief Compute all quantities using position, velocity and acceleration
           configurations.

    @details Run multiple algorithms to compute all quantities
             which can be known with the model position, velocity
             and acceleration configuration.

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

             Computation results are stored in internal
             data and can be retrieved with associated getters.

    @note This function modifies internal data.

    @param robot            The jiminy robot
    @param position         Joint position vector
    @param velocity         Joint velocity vector
    @param acceleration     Joint acceleration vector
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
                pin.centerOfMass(pnc_model, pnc_data, position, velocity, acceleration)
        else:
            if velocity is None:
                pin.forwardKinematics(pnc_model, pnc_data, position)
            elif acceleration is None:
                pin.forwardKinematics(pnc_model, pnc_data, position, velocity)
            else:
                pin.forwardKinematics(pnc_model, pnc_data, position, velocity, acceleration)
            pin.framesForwardKinematics(pnc_model, pnc_data, position)

        if update_energy:
            if velocity is not None:
                pin.kineticEnergy(pnc_model, pnc_data, position, velocity, False)
            pin.potentialEnergy(pnc_model, pnc_data, position, False)

def get_body_index_and_fixedness(robot, body_name, use_theoretical_model=True):
    """
    @brie Retrieve a body index in model and its fixedness from the body name.

    @details The method searchs bodyNames array and fixedBodyNames array
             for the provided bodyName.

    @param  robot           The jiminy robot
    @param	body_name       The name of the body for which index and fixedness are needed
    @return	body_id         The index of the body whose name is bodyName in model
    @return is_body_fixed   Whether or not the body is considered as a fixed-body by the model
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

def get_body_world_transform(robot, body_name, use_theoretical_model=True, copy=True):
    """
    @brief Get the transform from world frame to body frame for a given body.

    @details It is assumed that update_quantities has been called.

    @param  robot           The jiminy robot
    @param  body_name       The name of the body
    @return transform       Object representing the transform

    @pre body_name must be the name of a body existing in model.
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

def get_body_world_velocity(robot, body_name, use_theoretical_model=True):
    """
    @brief Get the velocity wrt world in body frame for a given body.

    @details It is assumed that update_quantities has been called.

    @return body transform in world frame.

    @param  robot               The jiminy robot
    @param  body_name           The name of the body
    @return spatial_velocity    se3 object representing the velocity

    @pre body_name must be the name of a body existing in model.
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
        spatial_velocity = parent_velocity_in_parent_frame.se3Action(parent_transform_in_world)
    else:
        transform = pnc_data.oMi[body_id]
        velocity_in_body_frame = pnc_data.v[body_id]
        spatial_velocity = velocity_in_body_frame.se3Action(transform)

    return spatial_velocity

def get_body_world_acceleration(robot, body_name, use_theoretical_model=True):
    """
    @brief Get the body spatial acceleration in world frame.

    @details It is assumed that update_quantities has been called.

    @note The moment of this tensor (i.e linear part) is \b NOT the linear
          acceleration of the center of the body frame, expressed in the world
          frame. Use getBodyWorldLinearAcceleration for that.

    @param  body_name       The name of the body
    @return acceleration    Body spatial acceleration.

    @pre body_name must be the name of a body existing in model.
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
        spatial_acceleration = parent_acceleration_in_parent_frame.se3Action(parent_transform_in_world)
    else:
        transform = pnc_data.oMi[body_id]
        acceleration_in_body_frame = pnc_data.a[body_id]
        spatial_acceleration = acceleration_in_body_frame.se3Action(transform)

    return spatial_acceleration

def _compute_closest_contact_frame(robot, ground_profile=None, use_theoretical_model=True):
    """
    @brief   Compute the closest contact point to the ground, in their respective local
             frame and wrt the ground position and orientation.

    @details This method can be used in conjunction with compute_freeflyer_state_from_fixed_body
             to determine the right fixed_body_name that ensures no contact points are going
             through the ground and a single one is touching it.

             It is assumed that update_quantities has been called.

    @param robot     The jiminy robot
    @param ground_profile   The ground profile callback
    @param position   Joint position vector
    """

    if use_theoretical_model:
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_data = robot.pinocchio_data

    # Compute the transform in the world of the contact points
    contact_frames_transform = []
    for frame_idx in robot.contact_frames_idx:
        transform = pnc_data.oMf[frame_idx]
        contact_frames_transform.append(transform)

    # Compute the transform of the ground at these points
    if ground_profile is not None:
        contact_ground_transform = []
        ground_pos = np.zeros(3)
        for frame_transform in contact_frames_transform:
            ground_pos[2], normal = ground_profile(frame_transform.translation)
            ground_rot = pin.Quaternion.FromTwoVectors(np.array([0.0, 0.0, 1.0]), normal).matrix()
            contact_ground_transform.append(pin.SE3(ground_rot, ground_pos))
    else:
        contact_ground_transform = len(contact_frames_transform) * [pin.SE3.Identity()]

    # Compute the position and normal of the contact points wrt their respective ground transform
    contact_frames_pos_rel = []
    contact_frames_normal_rel = []
    for frame_transform, ground_transform in \
            zip(contact_frames_transform, contact_ground_transform):
        transform_rel = ground_transform.actInv(frame_transform)
        contact_frames_pos_rel.append(transform_rel.translation)
        contact_frames_normal_rel.append(transform_rel.rotation[:, 2])

    # Compute the closest contact points to the ground in their respective local frame
    for i in range(len(robot.contact_frames_idx)):
        height_frame = contact_frames_pos_rel[i] @ contact_frames_normal_rel[i]
        is_closest = True
        for j in range(i+1, len(robot.contact_frames_idx)):
            height = contact_frames_pos_rel[j] @ contact_frames_normal_rel[i]
            if (height_frame > height + 1e-6): # Add a small 1um tol since "closest" is meaningless at this point
                is_closest = False
                break
        if is_closest:
            break

    return robot.contact_frames_names[i]

def compute_freeflyer_state_from_fixed_body(robot, position, velocity=None, acceleration=None,
                                            fixed_body_name=None, ground_profile=None,
                                            use_theoretical_model=True):
    """
    @brief   Fill rootjoint data from articular data when a body is fixed parallel to world.

    @details If 'fixed_body_name' is omitted, it will default to the contact point that ensures
             no contact points are going through the ground and a single one is touching it.

    @remark  The hypothesis is that 'fixed_body_name' is fixed parallel to world frame.
             So this method computes the position of freeflyer rootjoint in the fixed body frame.

    @note This function modifies internal data.

    @param robot            The jiminy robot
    @param[inout] position  Must contain current articular data. The rootjoint data can
                            contain any value, it will be ignored and replaced.
                            The method fills in rootjoint data.
    @param[inout] velocity  See position
    @param[inout] acceleration  See position
    @param fixed_body_name  The name of the body frame that is considered fixed parallel to world frame
    @param ground_profile   The ground profile callback
    """
    if not robot.has_freeflyer:
        raise RuntimeError("The robot does not have a freeflyer.")

    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    position[:6].fill(0.0)
    position[6] = 1.0
    if velocity is not None and acceleration is not None:
        velocity[:6].fill(0.0)
        acceleration[:6].fill(0.0)
        pin.forwardKinematics(pnc_model, pnc_data, position, velocity, acceleration)
    elif velocity is not None:
        velocity[:6].fill(0.0)
        pin.forwardKinematics(pnc_model, pnc_data, position, velocity)
    else:
        pin.forwardKinematics(pnc_model, pnc_data, position)
    pin.framesForwardKinematics(pnc_model, pnc_data, position)

    if fixed_body_name is None:
        fixed_body_name = _compute_closest_contact_frame(
            robot, ground_profile, use_theoretical_model)

    ff_M_fixed_body = get_body_world_transform(
        robot, fixed_body_name, use_theoretical_model, copy=False)

    if ground_profile is not None:
        ground_translation = np.zeros(3)
        ground_translation[2], normal = ground_profile(ff_M_fixed_body.translation)
        ground_rotation = pin.Quaternion.FromTwoVectors(np.array([0.0, 0.0, 1.0]), normal).matrix()
        w_M_ground = pin.SE3(ground_rotation, ground_translation)
    else:
        w_M_ground = pin.SE3.Identity()

    w_M_ff = w_M_ground.act(ff_M_fixed_body.inverse())
    position[:7] = pin.se3ToXYZQUAT(w_M_ff)

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

def compute_efforts_from_fixed_body(robot, position, velocity, acceleration,
                                    fixed_body_name, use_theoretical_model=True):
    """
    @brief   Compute the efforts using RNEA method.

    @note This function modifies internal data.

    @param robot            The jiminy robot
    @param position         Joint position vector
    @param velocity         See position
    @param acceleration     See position
    @param fixed_body_name  The name of the body frame on which to apply the external forces
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
    support_foot_idx = pnc_model.frames[pnc_model.getBodyId(fixed_body_name)].parent
    f_ext[support_foot_idx] = pnc_data.oMi[support_foot_idx] \
        .actInv(pnc_data.oMi[1]).act(pnc_data.f[1])

    # Recompute the efforts with RNEA and the correct external forces
    tau = pin.rnea(pnc_model, pnc_data, position, velocity, acceleration, f_ext)
    f_ext = f_ext[support_foot_idx]

    return tau, f_ext

######################################################################
#################### State sequence wrappers #########################
######################################################################

def retrieve_freeflyer(trajectory_data, freeflyer_continuity=True):
    """
    @brief   Retrieves the freeflyer positions and velocities.
             The reference frame is the support foot.

    @param   trajectory_data Sequence of States for which to retrieve the freeflyer.
    """
    robot = trajectory_data['robot']
    use_theoretical_model = trajectory_data['use_theoretical_model']

    contact_frame_prev = None
    w_M_ff_offset = pin.SE3.Identity()
    w_M_ff_prev = None
    for s in trajectory_data['evolution_robot']:
        # Compute freeflyer using contact frame as reference frame
        s.contact_frame = compute_freeflyer_state_from_fixed_body(
            robot, s.q, s.v, s.a, s.contact_frame,
            None, use_theoretical_model)

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

def compute_efforts(trajectory_data):
    """
    @brief   Compute the efforts in the trajectory using RNEA method.

    @param   trajectory_data Sequence of States for which to compute the efforts.
    """
    robot = trajectory_data['robot']
    use_theoretical_model = trajectory_data['use_theoretical_model']

    for s in trajectory_data['evolution_robot']:
        s.tau, s.f_ext = compute_efforts_from_fixed_body(
            robot, s.q, s.v, s.a, s.contact_frame, use_theoretical_model)
