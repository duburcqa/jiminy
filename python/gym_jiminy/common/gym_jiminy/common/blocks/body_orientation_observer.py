"""Implementation of a stateless body orientation state estimator block
compatible with gym_jiminy reinforcement learning pipeline environment design.
"""
from collections import OrderedDict
from collections.abc import Mapping
from typing import List, Dict, Sequence, Union, Optional

import numpy as np
import numba as nb
import gymnasium as gym

from jiminy_py.core import ImuSensor  # pylint: disable=no-name-in-module

from ..bases import BaseAct, BaseObs, BaseObserverBlock, InterfaceJiminyEnv
from ..wrappers.observation_layout import NestedData
from ..utils import (DataNested,
                     fill,
                     quat_to_rpy,
                     matrices_to_quat,
                     quat_multiply,
                     quat_apply,
                     remove_twist_from_quat)


@nb.jit(nopython=True, cache=True)
def update_twist(q: np.ndarray,
                 twist: np.ndarray,
                 omega: np.ndarray,
                 time_constant_inv: float,
                 dt: float) -> None:
    """Update the twist estimate of the Twist-after-Swing decomposition of
    given orientations in quaternion representation using leaky integrator.

    :param q: Current swing estimate as a quaternion. It will be updated
              in-place to add the estimated twist.
    :param twist: Current twist estimate to update in-place.
    :param omega: Current angular velocity estimate in local frame.
    :param time_constant_inv: Inverse of the time constant of the leaky
                              integrator used to update the twist estimate.
    :param dt: Time step, in seconds, between consecutive Quaternions.
    """
    # Compute the derivative of the twist angle:
    # The element-wise time-derivative of a quaternion is given by:
    #   dq = 0.5 * q * Quaternion(axis=gyro, w=0.0)        [1]
    # The twist around a given axis can be computed as follows:
    #   p = q_axis.dot(twist_axis) * twist_axis            [2]
    #   twist = pin.Quaternion(np.array((*p, q_w))).normalized()
    # The twist angle can be inferred from this expression:
    #   twist = 2 * atan2(q_axis.dot(twist_axis), q_w)     [3]
    # The derivative of twist angle can be derived:
    #  dtwist = 2 * (
    #     (dq_axis * q_w - q_axis * dq_w).dot(twist_axis)  [4]
    #  ) / (q_axis.dot(twist_axis) ** 2 + q_w ** 2)
    # One can show that if q_s is the swing part of the orientation, then:
    #   q_s_axis.dot(twist_axis) = 0
    # It yields:
    #  dtwist = 2 * dq_s_axis.dot(twist_axis) / q_s_w      [5]
    q_x, q_y, _, q_w = q
    dtwist = (- q_y * omega[0] + q_x * omega[1]) / q_w + omega[2]

    # Update twist angle using Leaky Integrator scheme to avoid long-term drift
    twist *= max(0.0, 1.0 - time_constant_inv * dt)
    twist += dtwist * dt

    # Update quaternion to add estimated twist
    p_z, p_w = np.sin(0.5 * twist), np.cos(0.5 * twist)
    q[0], q[1], q[2], q[3] = (
        p_w * q_x - p_z * q_y,
        p_z * q_x + p_w * q_y,
        p_z * q_w,
        p_w * q_w)


class BodyObserver(BaseObserverBlock[
        Dict[str, np.ndarray], np.ndarray, BaseObs, BaseAct]):
    """Compute the orientation and angular velocity in local frame of the
    parent body associated with all the IMU sensors of the robot.

    .. note::
        The twist angle is not part of the internal state, even if being
        integrated over time, because it is uniquely determined from the
        orientation estimate.
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 *,
                 nested_imu_quat_key: NestedData = (
                    "features", "mahony_filter", "quat"),
                 nested_imu_omega_key: NestedData = (
                    "features", "mahony_filter", "omega",),
                 twist_time_constant: Optional[float] = None,
                 compute_rpy: bool = True,
                 update_ratio: int = 1) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param nested_imu_quat_key: Nested key from environment observation
                                    mapping to the IMU quaternion estimates.
                                    Their ordering must be consistent with the
                                    true IMU sensors of the robot.
        :param nested_imu_omega_key: Nested key from environment observation
                                     mapping to the IMU angular velocity in
                                     local frame estimates. Their ordering must
                                     be consistent with the true IMU sensors of
                                     the robot.
        :param twist_time_constant:
            If specified, it corresponds to the time constant of the leaky
            integrator (Exponential Moving Average) used to estimate the twist
            part of twist-after-swing decomposition of the estimated
            orientation in place of the Mahony Filter. If `0.0`, then its is
            kept constant equal to zero. `None` to kept the original estimate
            provided by Mahony Filter. See `remove_twist_from_quat` and
            `update_twist` documentations for details.
            Optional: `0.0` by default.
        :param compute_rpy: Whether to compute the Yaw-Pitch-Roll Euler angles
                            representation for the 3D orientation of the IMU,
                            in addition to the quaternion representation.
                            Optional: True by default.
        :param update_ratio: Ratio between the update period of the observer
                             and the one of the subsequent observer. -1 to
                             match the simulation timestep of the environment.
                             Optional: `1` by default.
        """
        # Backup some of the user-argument(s)
        self.compute_rpy = compute_rpy

        # Keep track of how the twist must be computed
        self.twist_time_constant_inv: Optional[float]
        if twist_time_constant is None:
            self.twist_time_constant_inv = None
        else:
            if twist_time_constant > 0.0:
                self.twist_time_constant_inv = 1.0 / twist_time_constant
            else:
                self.twist_time_constant_inv = float("inf")
        self._remove_twist = self.twist_time_constant_inv is not None
        self._update_twist = (
            self.twist_time_constant_inv is not None and
            np.isfinite(self.twist_time_constant_inv))

        # Define observed / estimated IMU data proxies for fast access
        obs_imu_data_list: List[np.ndarray] = []
        for nested_imu_key in (nested_imu_quat_key, nested_imu_omega_key):
            obs_imu_data: DataNested = env.observation
            for key in nested_imu_key:
                if isinstance(key, str):
                    assert isinstance(obs_imu_data, Mapping)
                    obs_imu_data = obs_imu_data[key]
                elif isinstance(key, int):
                    assert isinstance(obs_imu_data, Sequence)
                    obs_imu_data = obs_imu_data[key]
                else:
                    assert isinstance(obs_imu_data, np.ndarray)
                    slices: List[Union[int, slice]] = []
                    for start_end in key:
                        if isinstance(start_end, int):
                            slices.append(start_end)
                        elif not start_end:
                            slices.append(slice(None,))
                        else:
                            slices.append(slice(*start_end))
                    obs_imu_data = obs_imu_data[tuple(slices)]
            assert isinstance(obs_imu_data, np.ndarray)
            obs_imu_data_list.append(obs_imu_data)
        self._obs_imu_quats, self._obs_imu_omegas = obs_imu_data_list

        # Extract the relative IMU frame placement wrt its parent body
        imu_rel_rot_mats: List[np.ndarray] = []
        for sensor in env.robot.sensors[ImuSensor.type]:
            frame = env.robot.pinocchio_model.frames[sensor.frame_index]
            imu_rel_rot_mats.append(frame.placement.rotation)
        self._imu_rel_quats = matrices_to_quat(tuple(imu_rel_rot_mats))

        # Allocate twist angle estimate around z-axis in world frame.
        num_imu_sensors = len(env.robot.sensors[ImuSensor.type])
        self._twist = np.zeros((1, num_imu_sensors))

        # Initialize the observer
        super().__init__(name, env, update_ratio)

        # Define proxies for the body orientations and angular velocities
        self._quat = self.observation["quat"]
        if self.compute_rpy:
            self._rpy = self.observation["rpy"]
        else:
            self._rpy = np.array([])
        self._omega = self.observation["omega"]

    def _initialize_observation_space(self) -> None:
        num_imu_sensors = len(self.env.robot.sensors[ImuSensor.type])
        observation_space: Dict[str, gym.Space] = OrderedDict()
        observation_space["quat"] = gym.spaces.Box(
            low=np.full((4, num_imu_sensors), -1.0 - 1e-9),
            high=np.full((4, num_imu_sensors), 1.0 + 1e-9),
            dtype=np.float64)
        if self.compute_rpy:
            high = np.array([np.pi, np.pi/2, np.pi]) + 1e-9
            observation_space["rpy"] = gym.spaces.Box(
                low=-high[:, np.newaxis].repeat(num_imu_sensors, axis=1),
                high=high[:, np.newaxis].repeat(num_imu_sensors, axis=1),
                dtype=np.float64)
        observation_space["omega"] = gym.spaces.Box(
            low=float('-inf'),
            high=float('inf'),
            shape=(3, num_imu_sensors),
            dtype=np.float64)
        self.observation_space = gym.spaces.Dict(
            **observation_space)  # type: ignore[arg-type]

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Reset the twist estimate
        fill(self._twist, 0)

        # Fix initialization of the observation to be valid quaternions
        self._quat[-1] = 1.0

    @property
    def fieldnames(self) -> Dict[str, List[List[str]]]:
        imu_sensors = self.env.robot.sensors[ImuSensor.type]
        fieldnames: Dict[str, List[List[str]]] = {}
        fieldnames["quat"] = [
            [".".join((sensor.name, f"Quat{e}")) for sensor in imu_sensors]
            for e in ("X", "Y", "Z", "W")]
        if self.compute_rpy:
            fieldnames["rpy"] = [
                [".".join((sensor.name, e)) for sensor in imu_sensors]
                for e in ("Roll", "Pitch", "Yaw")]
        fieldnames["omega"] = [
            [".".join((sensor.name, e)) for sensor in imu_sensors]
            for e in ("X", "Y", "Z")]
        return fieldnames

    def refresh_observation(self, measurement: BaseObs) -> None:
        # Compute the parent body orientations
        quat_multiply(self._obs_imu_quats,
                      self._imu_rel_quats,
                      out=self._quat,
                      is_right_conjugate=True)

        # Compute the parent body angular velocities in local frame.
        # Note that batched "quaternion apply" is faster than sequential
        # "rotation matrix apply" in practice, both when using `np.einsum` or
        # single-threaded numba jitted implementation.
        quat_apply(self._imu_rel_quats,
                   self._obs_imu_omegas,
                   out=self._omega)

        # Remove twist if requested
        if self._remove_twist:
            remove_twist_from_quat(self._quat)

        # Update twist if requested
        if self._update_twist:
            update_twist(self._quat,
                         self._twist,
                         self._omega,
                         self.twist_time_constant_inv,
                         self.observe_dt)

        # Compute the RPY representation if requested
        if self.compute_rpy:
            quat_to_rpy(self._quat, self._rpy)
