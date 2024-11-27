# mypy: disable-error-code="attr-defined, name-defined"
"""Helpers to ease computation of kinematic and dynamic quantities.

.. warning::
    These helpers must be used with caution. They are inefficient and some may
    not even work properly due to ground profile being partially supported.
"""
# pylint: disable=invalid-name,no-member
import logging
from bisect import bisect_left
from dataclasses import dataclass, fields
from typing import List, Union, Optional, Tuple, Sequence, Callable, Literal

import numpy as np

import eigenpy
import hppfcl
import pinocchio as pin
from pinocchio.rpy import (rpyToMatrix,  # pylint: disable=import-error
                           matrixToRpy,
                           computeRpyJacobian,
                           computeRpyJacobianInverse)

from . import core as jiminy


LOGGER = logging.getLogger(__name__)


TRAJ_INTERP_TOL = 1e-12  # 0.01 * 'STEPPER_MIN_TIMESTEP'


# #####################################################################
# ######################### Generic math ##############################
# #####################################################################

def SE3ToXYZRPY(M: pin.SE3) -> np.ndarray:
    """Convert Pinocchio SE3 object to [X,Y,Z,Roll,Pitch,Yaw] vector.
    """
    return np.concatenate((M.translation, matrixToRpy(M.rotation)))


def XYZRPYToSE3(xyzrpy: np.ndarray) -> np.ndarray:
    """Convert [X,Y,Z,Roll,Pitch,Yaw] vector to Pinocchio SE3 object.
    """
    return pin.SE3(rpyToMatrix(xyzrpy[3:]), xyzrpy[:3])


def XYZRPYToXYZQuat(xyzrpy: np.ndarray) -> np.ndarray:
    """Convert [X,Y,Z,Roll,Pitch,Yaw] to [X,Y,Z,Qx,Qy,Qz,Qw].
    """
    xyz, rpy = xyzrpy[:3], xyzrpy[3:]
    return np.concatenate((xyz, pin.Quaternion(rpyToMatrix(rpy)).coeffs()))


def XYZQuatToXYZRPY(xyzquat: np.ndarray) -> np.ndarray:
    """Convert [X,Y,Z,Qx,Qy,Qz,Qw] to [X,Y,Z,Roll,Pitch,Yaw].
    """
    xyz, quat = xyzquat[:3], xyzquat[3:]
    return np.concatenate((xyz, matrixToRpy(pin.Quaternion(quat).matrix())))


def velocityXYZRPYToXYZQuat(xyzrpy: np.ndarray,
                            dxyzrpy: np.ndarray) -> np.ndarray:
    """Convert the derivative of [X,Y,Z,Roll,Pitch,Yaw] to [X,Y,Z,Qx,Qy,Qz,Qw].

    .. warning::
        Linear velocity in XYZRPY must be local-world-aligned frame, while
        returned linear velocity in XYZQuat is in local frame.
    """
    rpy = xyzrpy[3:]
    R, J_rpy = rpyToMatrix(rpy), computeRpyJacobian(rpy)
    return np.concatenate((R.T @ dxyzrpy[:3], J_rpy @ dxyzrpy[3:]))


def velocityXYZQuatToXYZRPY(xyzquat: np.ndarray,
                            v: np.ndarray) -> np.ndarray:
    """Convert the derivative of [X,Y,Z,Qx,Qy,Qz,Qw] to [X,Y,Z,Roll,Pitch,Yaw].

    .. note:
        No need to estimate yaw angle to get RPY velocity, since
        `computeRpyJacobianInverse` only depends on Roll and Pitch angles.
        However, it is not the case for the linear velocity.

    .. warning::
        Linear velocity in XYZQuat must be local frame, while returned linear
        velocity in XYZRPY is in local-world-aligned frame.
    """
    quat = pin.Quaternion(xyzquat[3:])
    rpy = matrixToRpy(quat.matrix())
    J_rpy_inv = computeRpyJacobianInverse(rpy)
    return np.concatenate((quat * v[:3], J_rpy_inv @ v[3:]))


# #####################################################################
# #################### State and Trajectory ###########################
# #####################################################################

@dataclass
class State:
    """Basic data structure storing kinematics and dynamics information at a
    given time.

    .. note::
        The user is the responsible for keeping track to which robot the state
        is associated to as this information is not stored in the state itself.
    """

    t: float
    """Time.
    """

    q: np.ndarray
    """Configuration vector.
    """

    v: Optional[np.ndarray] = None
    """Velocity vector as a 1D array.
    """

    a: Optional[np.ndarray] = None
    """Acceleration vector as a 1D array.
    """

    u: Optional[np.ndarray] = None
    """Total joint efforts as a 1D array.
    """

    command: Optional[np.ndarray] = None
    """Motor command as a 1D array.
    """

    f_external: Optional[np.ndarray] = None
    """Joint external forces as a 2D array.

    The first dimension corresponds to the N individual joints of the robot
    including 'universe', while the second gathers the 6 spatial force
    coordinates (Fx, Fy, Fz, Mx, My, Mz).
    """

    lambda_c: Optional[np.ndarray] = None
    """Lambda multipliers associated with all the constraints as a 1D array.
    """


@dataclass
class Trajectory:
    """Trajectory of a robot.

    This class is mostly a basic data structure storing a sequence of states
    along with the robot to which it is associated. On top of that, it provides
    helper methods to make it easier to manipulate these data, eg query the
    state at a given timestamp.
    """

    states: Tuple[State, ...]
    """Sequence of states of increasing time.

    .. warning::
        The time may not be strictly increasing. There may be up to two
        consecutive data point associated with the same timestep because
        quantities may vary instantaneously at acceleration-level and higher.
    """

    robot: jiminy.Robot
    """Robot associated with the trajectory.
    """

    use_theoretical_model: bool
    """Whether to use the theoretical model or the extended simulation model.
    """

    def __init__(self,
                 states: Sequence[State],
                 robot: jiminy.Robot,
                 use_theoretical_model: bool) -> None:
        """
        :param states: Trajectory data as a sequence of `State` instances of
                       increasing time.
        :param robot: Robot associated with the trajectory.
        :param use_theoretical_model: Whether the state sequence is associated
                                      with the theoretical dynamical model or
                                      extended simulation model of the robot.
        """
        # Backup user arguments
        self.states = tuple(states)
        self.robot = robot
        self.use_theoretical_model = use_theoretical_model

        # Extract time associated with all states
        self._times = tuple(state.t for state in states)
        if any(t_right - t_left < 0.0 for t_right, t_left in zip(
                self._times[1:], self._times[:-1])):
            raise ValueError(
                "Time must not be decreasing between consecutive timesteps.")

        # Define pinocchio model proxy for fast access
        if use_theoretical_model:
            self._pinocchio_model = robot.pinocchio_model_th
        else:
            self._pinocchio_model = robot.pinocchio_model

        # Compute the trajectory stride.
        # Ensure continuity of the freeflyer when time is wrapping.
        self._stride_offset_log6: Optional[np.ndarray] = None
        if self.robot.has_freeflyer and self.has_data:
            M_start = pin.XYZQUATToSE3(self.states[0].q[:7])
            M_end = pin.XYZQUATToSE3(self.states[-1].q[:7])
            self._stride_offset_log6 = pin.log6(M_end * M_start.inverse())

        # Keep track of last request to speed up nearest neighbors search
        self._t_prev = 0.0
        self._index_prev = 1

        # List of optional state fields that are provided
        # Note that looking for keys in such a small set is not worth the
        # hassle of using Python `set`, which breaks ordering and index access.
        fields_: List[str] = []
        fields_candidates = [field.name for field in fields(State)[2:]]
        for state in states:
            for field in fields_candidates:
                if getattr(state, field) is None:
                    if field in fields_:
                        raise ValueError(
                            "The state information being set must be the same "
                            "for all the timesteps of a given trajectory.")
                else:
                    fields_.append(field)
        self._fields = tuple(fields_)

    @property
    def has_data(self) -> bool:
        """Whether the trajectory has data, ie the state sequence is not empty.
        """
        return bool(self.states)

    @property
    def has_velocity(self) -> bool:
        """Whether the trajectory contains the velocity vector.
        """
        return "v" in self._fields

    @property
    def has_acceleration(self) -> bool:
        """Whether the trajectory contains the acceleration vector.
        """
        return "a" in self._fields

    @property
    def has_effort(self) -> bool:
        """Whether the trajectory contains the effort vector.
        """
        return "u" in self._fields

    @property
    def has_command(self) -> bool:
        """Whether the trajectory contains motor commands.
        """
        return "command" in self._fields

    @property
    def has_external_forces(self) -> bool:
        """Whether the trajectory contains external forces.
        """
        return "f_external" in self._fields

    @property
    def has_constraints(self) -> bool:
        """Whether the trajectory contains lambda multipliers of constraints.
        """
        return "lambda_c" in self._fields

    @property
    def optional_fields(self) -> Tuple[str, ...]:
        """Optional state information being specified for all the timesteps of
        the trajectory.
        """
        return self._fields

    @property
    def time_interval(self) -> Tuple[float, float]:
        """Time interval of the trajectory.

        It raises an exception if no data is available.
        """
        if not self.has_data:
            raise RuntimeError(
                "State sequence is empty. Time interval undefined.")
        return (self._times[0], self._times[-1])

    def get(self,
            t: float,
            mode: Literal['raise', 'wrap', 'clip'] = 'raise') -> State:
        """Query the state at a given timestamp.

        Internally, the nearest neighbor states are linearly interpolated,
        taking into account the corresponding Lie Group of all state attributes
        that are available.

        :param t: Time of the state to extract from the trajectory.
        :param mode: Fallback strategy when the query time is not in the time
                     interval 'time_interval' of the trajectory. 'raise' raises
                     an exception if the query time is out-of-bound wrt the
                     underlying state sequence of the selected trajectory.
                     'clip' forces clipping of the query time before
                     interpolation of the state sequence. 'wrap' wraps around
                     the query time wrt the time span of the trajectory. This
                     is useful to store periodic trajectories as finite state
                     sequences.
        """
        # Raise exception if state sequence is empty
        if not self.has_data:
            raise RuntimeError(
                "State sequence is empty. Impossible to interpolate data.")

        # Backup the original query time
        t_orig = t

        # Handling of the desired mode
        n_steps = 0.0
        t_start, t_end = self.time_interval
        if mode == "raise":
            if t - t_end > TRAJ_INTERP_TOL or t_start - t > TRAJ_INTERP_TOL:
                raise RuntimeError("Time is out-of-range.")
        elif mode == "wrap":
            if t_end > t_start:
                n_steps, t_rel = divmod(t - t_start, t_end - t_start)
                t = t_rel + t_start
            else:
                t = t_start
        else:
            t = max(t, t_start)  # Clipping right it is sufficient

        # Get nearest neighbors timesteps for linear interpolation.
        # Note that the left and right data points may be associated with the
        # same timestamp, corresponding respectively t- and t+. These values
        # are different for quantities that may change discontinuously such as
        # the acceleration. If the state at such a timestamp is requested, then
        # returning the left value is preferred, because it corresponds to the
        # only state that was accessible to the user, ie after call `step`.
        if t < self._t_prev:
            self._index_prev = 1
        self._index_prev = bisect_left(
            self._times, t, self._index_prev, len(self._times) - 1)
        self._t_prev = t

        # Skip interpolation if not necessary
        index_left, index_right = self._index_prev - 1, self._index_prev
        t_left, s_left = self._times[index_left], self.states[index_left]
        if t - t_left < TRAJ_INTERP_TOL:
            return s_left
        t_right, s_right = self._times[index_right], self.states[index_right]
        if t_right - t < TRAJ_INTERP_TOL:
            return s_right
        alpha = (t - t_left) / (t_right - t_left)

        # Interpolate state
        position = pin.interpolate(
            self._pinocchio_model, s_left.q, s_right.q, alpha)
        data = {"q": position}
        for field in self._fields:
            value_left = getattr(s_left, field)
            value_right = getattr(s_right, field)
            data[field] = value_left + alpha * (value_right - value_left)

        # Perform odometry if the time is wrapping
        if self._stride_offset_log6 is not None and n_steps:
            stride_offset = pin.exp6(n_steps * self._stride_offset_log6)
            ff_xyzquat = stride_offset * pin.XYZQUATToSE3(position[:7])
            position[:7] = pin.SE3ToXYZQUAT(ff_xyzquat)

        return State(t=t_orig, **data)


# #####################################################################
# ################### Kinematic and dynamics ##########################
# #####################################################################

def update_quantities(robot: jiminy.Model,
                      position: np.ndarray,
                      velocity: Optional[np.ndarray] = None,
                      acceleration: Optional[np.ndarray] = None,
                      f_external: Optional[
                        Union[List[np.ndarray], pin.StdVec_Force]] = None,
                      update_dynamics: bool = True,
                      update_centroidal: bool = True,
                      update_energy: bool = True,
                      update_jacobian: bool = False,
                      update_collisions: bool = False,
                      use_theoretical_model: bool = True) -> None:
    """Compute all quantities using position, velocity and acceleration
    configurations.

    Run multiple algorithms to compute all quantities which can be known with
    the model position, velocity and acceleration.

    This includes:
    - body and frame spatial transforms,
    - body spatial velocities,
    - body spatial drifts,
    - body spatial acceleration,
    - joint transform jacobian matrices,
    - center-of-mass position,
    - center-of-mass velocity,
    - center-of-mass drift,
    - center-of-mass acceleration,
    - center-of-mass jacobian,
    - articular inertia matrix,
    - non-linear effects (Coriolis + gravity)
    - collisions and distances

    .. note::
        Computation results are stored internally in the robot, and can be
        retrieved with associated getters.

    .. warning::
        This function modifies the internal robot data.

    :param robot: Jiminy robot.
    :param position: Configuration vector.
    :param velocity: Joint velocity vector.
    :param acceleration: Joint acceleration vector.
    :param f_external: External forces applied on each joints.
    :param update_dynamics: Whether to compute the non-linear effects and the
                            joint internal forces.
                            Optional: True by default.
    :param update_centroidal: Whether to compute the centroidal dynamics (incl.
                              CoM) of the robot.
                              Optional: False by default.
    :param update_energy: Whether to compute the energy of the robot.
                          Optional: False by default
    :param update_jacobian: Whether to compute the Jacobian matrices of the
                            joint transforms.
                            Optional: False by default.
    :param update_collisions: Whether to detect collisions and compute
                              distances between all the geometry objects.
                              Optional: False by default.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the state of the robot.
                                  Optional: True by default.
    """
    if use_theoretical_model:
        model = robot.pinocchio_model_th
        data = robot.pinocchio_data_th
    else:
        model = robot.pinocchio_model
        data = robot.pinocchio_data

    if (update_dynamics and update_centroidal and
            update_energy and update_jacobian and
            velocity is not None and acceleration is None):
        pin.computeAllTerms(model, data, position, velocity)
    else:
        if velocity is None:
            pin.forwardKinematics(model, data, position)
        elif acceleration is None:
            pin.forwardKinematics(model, data, position, velocity)
        else:
            pin.forwardKinematics(
                model, data, position, velocity, acceleration)

        if update_jacobian:
            if update_centroidal:
                pin.jacobianCenterOfMass(model, data)
            if not update_dynamics:
                pin.computeJointJacobians(model, data)

        if update_dynamics:
            if velocity is not None:
                pin.nonLinearEffects(model, data, position, velocity)
            jiminy.crba(model, data, position)

        if update_energy:
            if velocity is not None:
                jiminy.computeKineticEnergy(
                    model, data, position, velocity, update_kinematics=False)
            pin.computePotentialEnergy(model, data)

    if update_centroidal:
        pin.computeCentroidalMomentumTimeVariation(model, data)
        if acceleration is not None:
            pin.centerOfMass(model, data, pin.ACCELERATION, False)

    if (update_dynamics and velocity is not None and
            acceleration is not None and f_external is not None):
        jiminy.rnea(model, data, position, velocity, acceleration, f_external)

    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(
        model, data, robot.collision_model, robot.collision_data)

    if update_collisions:
        pin.computeCollisions(
            robot.collision_model, robot.collision_data,
            stop_at_first_collision=False)
        pin.computeDistances(robot.collision_model, robot.collision_data)
        for dist_req in robot.collision_data.distanceResults:
            if np.linalg.norm(dist_req.normal) < 1e-6:
                pin.computeDistances(
                    robot.collision_model, robot.collision_data)
                break


def get_body_world_transform(robot: jiminy.Model,
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
                                  the state of the robot.
    :param copy: Whether to return the internal buffers (which could be
                 altered) or copy them.
                 Optional: True by default. It is less efficient but safer.

    :returns: Body transform.
    """
    # Pick the right pinocchio model and data
    if use_theoretical_model:
        model = robot.pinocchio_model_th
        data = robot.pinocchio_data_th
    else:
        model = robot.pinocchio_model
        data = robot.pinocchio_data

    # Get frame index and make sure it exists
    body_id = model.getFrameId(body_name)
    assert body_id < model.nframes, f"Frame '{body_name}' does not exits."

    # Get body transform in world frame
    transform = data.oMf[body_id]
    if copy:
        transform = transform.copy()

    return transform


def get_body_world_velocity(robot: jiminy.Model,
                            body_name: str,
                            use_theoretical_model: bool = True) -> pin.SE3:
    """Get the spatial velocity in world frame.

    .. warning::
        It is assumed that `update_quantities` has been called beforehand.

    :param robot: Jiminy robot.
    :param body_name: Name of the body.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the state of the robot.
                                  Optional: True by default.

    :returns: Spatial velocity.
    """
    # Pick the right pinocchio model and data
    if use_theoretical_model:
        model = robot.pinocchio_model_th
        data = robot.pinocchio_data_th
    else:
        model = robot.pinocchio_model
        data = robot.pinocchio_data

    # Get frame index and make sure it exists
    body_id = model.getFrameId(body_name)
    assert body_id < model.nframes, f"Frame '{body_name}' does not exits."

    return pin.getFrameVelocity(model, data, body_id, pin.WORLD)


def get_body_world_acceleration(robot: jiminy.Model,
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
                                  the state of the robot.
                                  Optional: True by default.

    :returns: Spatial acceleration.
    """
    # Pick the right pinocchio model and data
    if use_theoretical_model:
        model = robot.pinocchio_model_th
        data = robot.pinocchio_data_th
    else:
        model = robot.pinocchio_model
        data = robot.pinocchio_data

    # Get frame index and make sure it exists
    body_id = model.getFrameId(body_name)
    assert body_id < model.nframes, f"Frame '{body_name}' does not exits."

    return pin.getFrameAcceleration(model, data, body_id, pin.WORLD)


def compute_transform_contact(
        robot: jiminy.Model,
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
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object

    # Proxy for convenience
    collision_model = robot.collision_model

    # Compute the transform in the world of the contact points
    contact_frames_transform = []
    for frame_index in robot.contact_frame_indices:
        transform = robot.pinocchio_data.oMf[frame_index]
        contact_frames_transform.append(transform)

    # Compute the transform of the ground at these points
    if ground_profile is not None:
        contact_ground_transform = []
        ground_pos = np.zeros(3)
        for frame_transform in contact_frames_transform:
            ground_pos[2], normal = ground_profile(
                frame_transform.translation[:2])
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
    contact_frames_order = np.argsort([
        pos[2] for pos in contact_frames_pos_rel])
    contact_frames_pos_rel = [
        contact_frames_pos_rel[i] for i in contact_frames_order]

    # Compute the contact plane normal
    if len(contact_frames_pos_rel) > 2:
        # Try to compute a valid normal using three deepest points
        contact_edge_ref = \
            contact_frames_pos_rel[0] - contact_frames_pos_rel[1]
        contact_edge_ref /= np.linalg.norm(contact_edge_ref)
        contact_edge_alt = \
            contact_frames_pos_rel[0] - contact_frames_pos_rel[2]
        contact_edge_alt /= np.linalg.norm(contact_edge_alt)
        normal_offset = np.cross(contact_edge_ref, contact_edge_alt)

        # Make sure that the normal is valid, otherwise use the default one
        if np.linalg.norm(normal_offset) < 0.6:
            normal_offset = np.array([0.0, 0.0, 1.0])

        # Make sure the normal is pointing upward
        if normal_offset[2] < 0.0:
            normal_offset *= -1.0
    else:
        # Fallback to world aligned if no reference can be computed
        normal_offset = np.array([0.0, 0.0, 1.0])

    # Compute the translation and rotation to apply the touch the ground
    rot_offset = pin.Quaternion.FromTwoVectors(
        normal_offset, np.array([0.0, 0.0, 1.0])).matrix()
    if contact_frames_pos_rel:
        contact_frame_pos = contact_frames_transform[
            contact_frames_order[0]].translation
        pos_shift = (
            rot_offset @ contact_frame_pos)[2] - contact_frame_pos[2]
        pos_offset = np.array([
            0.0, 0.0, - pos_shift - contact_frames_pos_rel[0][2]])
    else:
        pos_offset = np.zeros(3)
    transform_offset = pin.SE3(rot_offset, pos_offset)

    # Take into account the collision bodies
    # FIXME: Take into account the ground profile
    min_distance = float('inf')
    deepest_index = None
    for i, dist_req in enumerate(robot.collision_data.distanceResults):
        if np.linalg.norm(dist_req.normal) > 1e-6:
            body_index = collision_model.collisionPairs[0].first
            body_geom = robot.collision_model.geometryObjects[body_index]
            if dist_req.normal[2] > 0.0 and \
                    isinstance(body_geom.geometry, hppfcl.Box):
                ground_index = collision_model.collisionPairs[0].second
                ground_geom = collision_model.geometryObjects[ground_index]
                box_size = 2.0 * ground_geom.geometry.halfSide
                body_size = 2.0 * body_geom.geometry.halfSide
                distance = - body_size[2] - box_size[2] - dist_req.min_distance
            else:
                distance = dist_req.min_distance
            if distance < min_distance:
                min_distance = distance
                deepest_index = i
        else:
            LOGGER.warning("Collision computation failed for some reason. "
                           "Skipping this collision pair.")
    if deepest_index is not None and (
            not contact_frames_pos_rel or
            transform_offset.translation[2] < -min_distance):
        transform_offset.translation[2] = -min_distance
        if not contact_frames_pos_rel:
            geom_index = collision_model.collisionPairs[deepest_index].first
            geom = collision_model.geometryObjects[geom_index]
            if isinstance(geom.geometry, hppfcl.Box):
                dist_rslt = robot.collision_data.distanceResults[deepest_index]
                collision_position = dist_rslt.getNearestPoint1()
                transform_offset.rotation = \
                    robot.collision_data.oMg[geom_index].rotation.T
                transform_offset.translation[2] += (
                    collision_position -
                    transform_offset.rotation @ collision_position)[2]

    return transform_offset


def compute_freeflyer_state_from_fixed_body(
        robot: jiminy.Model,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        fixed_body_name: Optional[str] = None,
        ground_profile: Optional[Callable[
            [np.ndarray], Tuple[float, np.ndarray]]] = None,
        use_theoretical_model: Optional[bool] = None) -> None:
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
    :param position: Must contain current articular data. The freeflyer data
                     can contain any value, it will be ignored and replaced.
    :param velocity: See position.
    :param acceleration: See position.
    :param fixed_body_name: Name of the body frame that is considered fixed
                            parallel to world frame.
                            Optional: It will be inferred from the set of
                            contact points and collision bodies.
    :param ground_profile: Ground profile callback.
    :param use_theoretical_model:
        Whether the state corresponds to the theoretical model when updating
        and fetching the state of the robot. Must be False if `fixed_body_name`
        is not specified.
        Optional: True by default if `fixed_body_name` is specified, False
        otherwise.

    :returns: Name of the contact frame, if any.
    """
    # Early return if no freeflyer
    if not robot.has_freeflyer:
        return

    # Handling of default arguments
    if use_theoretical_model is None:
        use_theoretical_model = fixed_body_name is not None

    # Clear freeflyer position, velocity and acceleration
    position[:6].fill(0.0)
    position[6] = 1.0
    if velocity is not None:
        velocity[:6].fill(0.0)
    if acceleration is not None:
        acceleration[:6].fill(0.0)

    # Update kinematics, frame placements and collision information
    update_quantities(robot,
                      position,
                      velocity,
                      acceleration,
                      update_dynamics=False,
                      update_centroidal=False,
                      update_energy=False,
                      use_theoretical_model=use_theoretical_model)

    if fixed_body_name is None:
        if use_theoretical_model:
            raise RuntimeError(
                "Cannot infer contact transform for the theoretical model.")
        w_M_ff = compute_transform_contact(robot, ground_profile)
    else:
        ff_M_fixed_body = get_body_world_transform(
            robot, fixed_body_name, use_theoretical_model, copy=False)
        if ground_profile is not None:
            ground_translation = np.zeros(3)
            ground_translation[2], normal = ground_profile(
                ff_M_fixed_body.translation[:2])
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


def compute_efforts_from_fixed_body(
        robot: jiminy.Model,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        fixed_body_name: str,
        use_theoretical_model: bool = True) -> Tuple[np.ndarray, pin.Force]:
    """Compute the efforts using RNEA method.

    .. warning::
        This function modifies the internal robot data.

    :param robot: Jiminy robot.
    :param position: Robot configuration vector.
    :param velocity: Robot velocity vector.
    :param acceleration: Robot acceleration vector.
    :param fixed_body_name: Name of the body frame.
    :param use_theoretical_model: Whether the state corresponds to the
                                  theoretical model when updating and fetching
                                  the state of the robot.
                                  Optional: True by default.

    :returns: articular efforts and external forces.
    """
    # Pick the right pinocchio model and data
    if use_theoretical_model:
        model = robot.pinocchio_model_th
        data = robot.pinocchio_data_th
    else:
        model = robot.pinocchio_model
        data = robot.pinocchio_data

    # Apply a first run of rnea without explicit external forces
    jiminy.rnea(model, data, position, velocity, acceleration)

    # Initialize vector of exterior forces to zero
    f_ext = pin.StdVec_Force()
    f_ext.extend(len(model.names) * (pin.Force.Zero(),))

    # Compute the force at the contact frame
    pin.forwardKinematics(model, data, position)
    support_joint_index = model.frames[
        model.getBodyId(fixed_body_name)].parent
    f_ext[support_joint_index] = data.oMi[support_joint_index].actInv(
        data.oMi[1]).act(data.f[1])

    # Recompute the efforts with RNEA and the correct external forces
    u = jiminy.rnea(model, data, position, velocity, acceleration, f_ext)
    f_ext = f_ext[support_joint_index]

    return u, f_ext


def compute_inverse_dynamics(robot: jiminy.Model,
                             position: np.ndarray,
                             velocity: np.ndarray,
                             acceleration: np.ndarray,
                             use_theoretical_model: bool = False
                             ) -> np.ndarray:
    """Compute the motor torques through inverse dynamics, assuming to external
    forces except the one resulting from the analytical constraints applied on
    the model.

    .. warning::
        This function modifies the internal robot data.

    :param robot: Jiminy robot.
    :param position: Robot configuration vector.
    :param velocity: Robot velocity vector.
    :param acceleration: Robot acceleration vector.
    :param use_theoretical_model:
        Whether the position, velocity and acceleration are associated with the
        theoretical model instead of the extended one.
        Optional: False by default.

    :returns: motor torques.
    """
    if not robot.has_constraints:
        raise NotImplementedError(
            "Robot without active constraints is not supported for now.")

    # Convert theoretical position, velocity and acceleration if necessary
    if use_theoretical_model and robot.is_flexibility_enabled:
        position = robot.get_extended_position_from_theoretical(position)
        velocity = robot.get_extended_velocity_from_theoretical(velocity)
        acceleration = (
            robot.get_extended_velocity_from_theoretical(acceleration))

    # Define some proxies for convenience
    model = robot.pinocchio_model
    data = robot.pinocchio_data
    motor_velocity_indices = [
        model.joints[motor.joint_index].idx_v for motor in robot.motors]

    # Updating kinematics quantities
    pin.forwardKinematics(model, data, position, velocity, acceleration)
    pin.updateFramePlacements(model, data)

    # Compute constraint jacobian and drift
    robot.compute_constraints(position, velocity)
    J, drift = robot.get_constraints_jacobian_and_drift()

    # No need to compute the internal matrix using `crba` nor to perform the
    # cholesky decomposition since it is already done by `compute_constraints`
    # internally.
    M_inv = pin.cholesky.computeMinv(model, data)
    M_inv_mcol = M_inv[:, motor_velocity_indices]

    # Compute non-linear effects
    pin.nonLinearEffects(model, data, position, velocity)
    nle = data.nle

    # Compute constraint forces
    jiminy.computeJMinvJt(model, data, J)
    a_f = jiminy.solveJMinvJtv(data, + J @ M_inv @ nle - drift)
    B_f = jiminy.solveJMinvJtv(data, - J @ M_inv_mcol, False)

    # compute feedforward term
    a_ydd = (M_inv @ (J.T @ a_f - nle) - acceleration)[motor_velocity_indices]
    B_ydd = (M_inv_mcol + M_inv @ J.T @ B_f)[motor_velocity_indices]

    # Compute motor torques
    u = eigenpy.LDLT(B_ydd).solve(- a_ydd)

    return u
