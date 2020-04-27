#!/usr/bin/env python

## @file jiminy_py/dynamics.py

import numpy as np

import pinocchio as pnc
from pinocchio import FrameType, Quaternion, SE3, XYZQUATToSe3
from pinocchio.rpy import rpyToMatrix, matrixToRpy


def se3ToXYZRPY(M):
    p = np.zeros((6,))
    p[:3] = M.translation
    p[3:] = matrixToRpy(M.rotation)
    return p

def XYZRPYToSe3(xyzrpy):
    return SE3(rpyToMatrix(xyzrpy[3:]), xyzrpy[:3])

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
        pnc.computeAllTerms(pnc_model, pnc_data, position, velocity)
    else:
        if update_physics:
            if velocity is not None:
                pnc.nonLinearEffects(pnc_model, pnc_data, position, velocity)
            pnc.crba(pnc_model, pnc_data, position)

        if update_jacobian:
            # if update_com:
            #     pnc.getJacobianComFromCrba(pnc_model, pnc_data)
            pnc.computeJointJacobians(pnc_model, pnc_data)

        if update_com:
            if velocity is None:
                pnc.centerOfMass(pnc_model, pnc_data, position)
            elif acceleration is None:
                pnc.centerOfMass(pnc_model, pnc_data, position, velocity)
            else:
                pnc.centerOfMass(pnc_model, pnc_data, position, velocity, acceleration)
        else:
            if velocity is None:
                pnc.forwardKinematics(pnc_model, pnc_data, position)
            elif acceleration is None:
                pnc.forwardKinematics(pnc_model, pnc_data, position, velocity)
            else:
                pnc.forwardKinematics(pnc_model, pnc_data, position, velocity, acceleration)
            pnc.framesForwardKinematics(pnc_model, pnc_data, position)

        if update_energy:
            if velocity is not None:
                pnc.kineticEnergy(pnc_model, pnc_data, position, velocity, False)
            pnc.potentialEnergy(pnc_model, pnc_data, position, False)

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
    is_body_fixed = (parent_frame_type == FrameType.FIXED_JOINT)
    if is_body_fixed:
        body_id = frame_id
    else:
        body_id = pnc_model.frames[frame_id].parent

    return body_id, is_body_fixed

def get_body_world_transform(robot, body_name, use_theoretical_model=True):
    """
    @brief Get the transform from world frame to body frame for a given body.

    @details It is assumed that computeQuantities has been called.

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

    body_id, body_is_fixed = get_body_index_and_fixedness(
        robot, body_name, use_theoretical_model)
    if body_is_fixed:
        transform_in_parent_frame = pnc_model.frames[body_id].placement
        last_moving_parent_id = pnc_model.frames[body_id].parent
        parent_transform_in_world = pnc_data.oMi[last_moving_parent_id]
        transform = parent_transform_in_world * transform_in_parent_frame
    else:
        transform = pnc_data.oMi[body_id]

    return transform

def get_body_world_velocity(robot, body_name, use_theoretical_model=True):
    """
    @brief Get the velocity wrt world in body frame for a given body.

    @details It is assumed that computeQuantities has been called.

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

    @details It is assumed that computeQuantities has been called.

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

def compute_freeflyer_state_from_fixed_body(robot, fixed_body_name, position,
                                            velocity=None, acceleration=None,
                                            use_theoretical_model=True):
    """
    @brief Fill rootjoint data from articular data when a body is fixed parallel to world.

    @details The hypothesis is that 'fixed_body_name' is fixed parallel to world frame.
             So this method computes the position of freeflyer rootjoint in the fixed body frame.

    @note This function modifies internal data.

    @param model            The jiminy robot
    @param fixed_body_name  The name of the body that is considered fixed parallel to world frame
    @param[inout] position  Must contain current articular data. The rootjoint data can
                            contain any value, it will be ignored and replaced
                            The method fills in rootjoint data
    @param[inout] velocity  Same as positionInOut but for velocity
    """
    if use_theoretical_model:
        pnc_model = robot.pinocchio_model_th
        pnc_data = robot.pinocchio_data_th
    else:
        pnc_model = robot.pinocchio_model
        pnc_data = robot.pinocchio_data

    position[:6].fill(0)
    position[6] = 1.0
    if velocity is not None:
        velocity[:6].fill(0)
    else:
        velocity = np.zeros((pnc_model.nv,))
    if acceleration is not None:
        acceleration[:6].fill(0)
    else:
        acceleration = np.zeros((pnc_model.nv,))

    pnc.forwardKinematics(pnc_model, pnc_data, position, velocity, acceleration)
    pnc.framesForwardKinematics(pnc_model, pnc_data, position)

    ff_M_fixed_body = get_body_world_transform(
        robot, fixed_body_name, use_theoretical_model)
    w_M_ff = ff_M_fixed_body.inverse()
    base_link_translation = w_M_ff.translation
    base_link_quaternion = Quaternion(w_M_ff.rotation)
    position[:3] = base_link_translation
    position[3:7] = base_link_quaternion.coeffs()

    ff_v_fixed_body = get_body_world_velocity(
        robot, fixed_body_name, use_theoretical_model)
    base_link_velocity = - ff_v_fixed_body
    velocity[:6] = base_link_velocity.vector

    ff_a_fixedBody = get_body_world_acceleration(
        robot, fixed_body_name, use_theoretical_model)
    base_link_acceleration = - ff_a_fixedBody
    acceleration[:6] = base_link_acceleration.vector
