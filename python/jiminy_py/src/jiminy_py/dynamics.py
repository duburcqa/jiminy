#!/usr/bin/env python

## @file jiminy_py/dynamics.py

import numpy as np

import pinocchio as pnc
from pinocchio import FrameType, Quaternion, SE3, XYZQUATToSe3
from pinocchio.rpy import rpyToMatrix, matrixToRpy


def se3ToXYZRPY(M):
    p = np.zeros(6)
    p[:3] = M.translation.A1
    p[3:] = matrixToRpy(M.rotation).A1
    return p

def XYZRPYToSe3(xyzrpy):
    return SE3(rpyToMatrix(xyzrpy[3:]), xyzrpy[:3])

def _computeQuantities(jiminy_model, position, velocity=None, acceleration=None,
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

    @param jiminy_model    The Jiminy model
    @param position         Joint position vector
    @param velocity         Joint velocity vector
    @param acceleration     Joint acceleration vector

    @pre model_.nq == positionIn.size()
    @pre model_.nv == velocityIn.size()
    @pre model_.nv == accelerationIn.size()
    """
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
        pnc_data = jiminy_model.pinocchio_data_th
    else:
        pnc_model = jiminy_model.pinocchio_model
        pnc_data = jiminy_model.pinocchio_data

    pnc.crba(pnc_model, pnc_data, position)
    #pnc.getJacobianComFromCrba(pnc_model, pnc_data)
    if velocity is None:
        pnc.centerOfMass(pnc_model, pnc_data, position)
    else:
        if acceleration is None:
            zero_acceleration = np.zeros((pnc_model.nv, 1))
            pnc.centerOfMass(pnc_model, pnc_data, position, velocity, zero_acceleration)
        else:
            pnc.centerOfMass(pnc_model, pnc_data, position, velocity, acceleration)
    pnc.computeJointJacobians(pnc_model, pnc_data, position)
    if velocity is not None:
        pnc.nonLinearEffects(pnc_model, pnc_data, position, velocity)
    if velocity is not None:
        pnc.kineticEnergy(pnc_model, pnc_data, position, velocity, False)
    pnc.potentialEnergy(pnc_model, pnc_data, position, False)

def getBodyIndexAndFixedness(jiminy_model, body_name, use_theoretical_model=True):
    """
    @brie Retrieve a body index in model and its fixedness from the body name.

    @details The method searchs bodyNames array and fixedBodyNames array
             for the provided bodyName.

    @param  jiminy_model    The Jiminy model
    @param	body_name       The name of the body for which index and fixedness are needed
    @return	body_id         The index of the body whose name is bodyName in model
    @return is_body_fixed   Whether or not the body is considered as a fixed-body by the model
    """
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
    else:
        pnc_model = jiminy_model.pinocchio_model

    frame_id = pnc_model.getFrameId(body_name)
    parent_frame_id = pnc_model.frames[frame_id].previousFrame
    parent_frame_type = pnc_model.frames[parent_frame_id].type
    is_body_fixed = (parent_frame_type == FrameType.FIXED_JOINT)
    if is_body_fixed:
        body_id = frame_id
    else:
        body_id = pnc_model.frames[frame_id].parent

    return body_id, is_body_fixed

def getBodyWorldTransform(jiminy_model, body_name, use_theoretical_model=True):
    """
    @brief Get the transform from world frame to body frame for a given body.

    @details It is assumed that computeQuantities has been called.

    @param  jiminy_model    The Jiminy model
    @param  body_name       The name of the body
    @return transform       Object representing the transform

    @pre body_name must be the name of a body existing in model.
    """
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
        pnc_data = jiminy_model.pinocchio_data_th
    else:
        pnc_model = jiminy_model.pinocchio_model
        pnc_data = jiminy_model.pinocchio_data

    body_id, body_is_fixed = getBodyIndexAndFixedness(jiminy_model, body_name)
    if body_is_fixed:
        transform_in_parent_frame = pnc_model.frames[body_id].placement
        last_moving_parent_id = pnc_model.frames[body_id].parent
        parent_transform_in_world = pnc_data.oMi[last_moving_parent_id]
        transform = parent_transform_in_world * transform_in_parent_frame
    else:
        transform = pnc_data.oMi[body_id]

    return transform

def getBodyWorldVelocity(jiminy_model, body_name, use_theoretical_model=True):
    """
    @brief Get the velocity wrt world in body frame for a given body.

    @details It is assumed that computeQuantities has been called.

    @return body transform in world frame.

    @param  jiminy_model        The Jiminy model
    @param  body_name           The name of the body
    @return spatial_velocity    se3 object representing the velocity

    @pre body_name must be the name of a body existing in model.
    """
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
        pnc_data = jiminy_model.pinocchio_data_th
    else:
        pnc_model = jiminy_model.pinocchio_model
        pnc_data = jiminy_model.pinocchio_data

    body_id, body_is_fixed = getBodyIndexAndFixedness(jiminy_model, body_name)
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

def getBodyWorldAcceleration(jiminy_model, body_name, use_theoretical_model=True):
    """
    @brief Get the body spatial acceleration in world frame.

    @details It is assumed that computeQuantities has been called.

    @note The moment of this tensor (i.e linear part) is \b NOT the linear
          acceleration of the center of the body frame, expressed in the world
          frame. Use getBodyWorldLinearAcceleration for that.

    @param  body_name       The name of the body
    @return acceleration    Body spatial acceleration.

    @pre body_name must be the name of a body existing in model.
    """
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
        pnc_data = jiminy_model.pinocchio_data_th
    else:
        pnc_model = jiminy_model.pinocchio_model
        pnc_data = jiminy_model.pinocchio_data

    body_id, body_is_fixed = getBodyIndexAndFixedness(jiminy_model, body_name)

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

def computeFreeflyerStateFromFixedBody(jiminy_model, fixed_body_name, position,
                                       velocity=None, acceleration=None,
                                       use_theoretical_model=True):
    """
    @brief Fill rootjoint data from articular data when a body is fixed parallel to world.

    @details The hypothesis is that 'fixed_body_name' is fixed parallel to world frame.
             So this method computes the position of freeflyer rootjoint in the fixed body frame.

    @note This function modifies internal data.

    @param model            The Jiminy model
    @param fixed_body_name  The name of the body that is considered fixed parallel to world frame
    @param[inout] position  Must contain current articular data. The rootjoint data can
                            contain any value, it will be ignored and replaced
                            The method fills in rootjoint data
    @param[inout] velocity  Same as positionInOut but for velocity

    @pre Rootjoint must be a freeflyer
    @pre positionInOut.size() == model_->nq
    """
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
    else:
        pnc_model = jiminy_model.pinocchio_model

    position[:7].fill(0)
    position[6] = 1.0
    if velocity is not None:
        velocity[:6].fill(0)
    else:
        velocity = np.zeros((pnc_model.nv, 1))
    if acceleration is not None:
        acceleration[:6].fill(0)
    else:
        acceleration = np.zeros((pnc_model.nv, 1))

    _computeQuantities(jiminy_model, position, velocity, acceleration)

    ff_M_fixed_body = getBodyWorldTransform(jiminy_model, fixed_body_name)
    w_M_ff = ff_M_fixed_body.inverse()
    base_link_translation = w_M_ff.translation
    base_link_quaternion = Quaternion(w_M_ff.rotation)
    position[:3] = base_link_translation
    position[3:7] = base_link_quaternion.coeffs()

    ff_v_fixed_body = getBodyWorldVelocity(jiminy_model, fixed_body_name)
    base_link_velocity = - ff_v_fixed_body
    velocity[:6] = base_link_velocity.vector

    ff_a_fixedBody = getBodyWorldAcceleration(jiminy_model, fixed_body_name)
    base_link_acceleration = - ff_a_fixedBody
    acceleration[:6] = base_link_acceleration.vector

    _computeQuantities(jiminy_model, position, velocity, acceleration)

def retrieve_freeflyer(trajectory_data, roll_angle=0.0, pitch_angle=0.0):
    """
    @brief   Retrieves the freeflyer positions and velocities. The reference frame is the support foot.
    """
    jiminy_model = trajectory_data['jiminy_model']
    use_theoretical_model = trajectory_data['use_theoretical_model']
    for s in trajectory_data['evolution_robot']:
        # Compute freeflyer using support foot as reference frame.
        computeFreeflyerStateFromFixedBody(jiminy_model, s.support_foot, s.q, s.v, s.a,
                                           use_theoretical_model)

        # Move freeflyer to take the foot angle into account.
        # w: world frame, st: support foot frame, ff: freeflyer frame.
        w_M_sf = XYZRPYToSe3(np.array([[roll_angle, pitch_angle, 0.0, 0.0, 0.0, 0.0]]).T)
        sf_M_ff = XYZQUATToSe3(s.q[:7]) # Px, Py, Pz, Qx, Qy, Qz, Qw
        w_M_ff = w_M_sf.act(sf_M_ff)
        s.q[:3] = w_M_ff.translation
        s.q[3:7] = Quaternion(w_M_ff.rotation).coeffs()

def compute_efforts(trajectory_data, index=(0, 0)):
    """
    @brief   Compute the efforts in the trajectory using RNEA method.

    @param   trajectory_data Sequence of States for which to compute the efforts.
    @param   index Index under which the efforts will be saved in the trajectory_data
             (usually the couple of (roll, pitch) angles of the support foot).
    """
    jiminy_model = trajectory_data['jiminy_model']
    use_theoretical_model = trajectory_data['use_theoretical_model']
    if use_theoretical_model:
        pnc_model = jiminy_model.pinocchio_model_th
        pnc_data = jiminy_model.pinocchio_data_th
    else:
        pnc_model = jiminy_model.pinocchio_model
        pnc_data = jiminy_model.pinocchio_data

    ## Compute the efforts at each time step
    root_joint_idx = pnc_model.getJointId('root_joint')
    for s in trajectory_data['evolution_robot']:
        # Apply a first run of rnea without explicit external forces
        pnc.computeJointJacobians(pnc_model, pnc_data, s.q)
        pnc.rnea(pnc_model, pnc_data, s.q, s.v, s.a)

        # Initialize vector of exterior forces to 0
        fs = pnc.StdVec_Force()
        fs.extend([pnc.Force(np.matrix([0.0, 0, 0, 0, 0, 0]).T)
                   for _ in range(len(pnc_model.names))])

        # Compute the force at the henkle level
        support_foot_idx = pnc_model.frames[pnc_model.getBodyId(s.support_foot)].parent
        fs[support_foot_idx] = pnc_data.oMi[support_foot_idx]\
            .actInv(pnc_data.oMi[root_joint_idx]).act(pnc_data.f[root_joint_idx])

        # Recompute the efforts with RNEA and the correct external forces
        s.tau[index] = pnc.rnea(pnc_model, pnc_data, s.q, s.v, s.a, fs)
        s.f_ext[index] = fs[support_foot_idx].copy()

        # Add the force to the structure
        s.f[index] = dict(list(zip(pnc_model.names, [f_ind.copy() for f_ind in pnc_data.f])))

        # Add the external force applied at the soles as if the floor was a parent of the soles
        for joint in ['LeftSole', 'RightSole']:
            ha_M_s = pnc_model.frames[pnc_model.getBodyId(joint)].placement
            if s.support_foot == joint:
                s.f[index][joint] = ha_M_s.actInv(s.f_ext[index])
            else:
                s.f[index][joint] = pnc.Force(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T)
