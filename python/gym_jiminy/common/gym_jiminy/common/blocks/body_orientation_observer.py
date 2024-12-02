"""Implementation of a stateless body orientation state estimator block
compatible with gym_jiminy reinforcement learning pipeline environment design.
"""
from collections import OrderedDict
from collections.abc import Mapping
from typing import List, Dict, Sequence, Union

import numpy as np
import gymnasium as gym

from jiminy_py.core import ImuSensor  # pylint: disable=no-name-in-module

from ..bases import BaseAct, BaseObs, BaseObserverBlock, InterfaceJiminyEnv
from ..wrappers.observation_layout import NestedData
from ..utils import (DataNested,
                     quat_to_rpy,
                     matrices_to_quat,
                     quat_multiply,
                     quat_apply,
                     remove_twist_from_quat)


class BodyObserver(BaseObserverBlock[
        Dict[str, np.ndarray], np.ndarray, BaseObs, BaseAct]):
    """Compute the orientation and angular velocity in local frame of the
    parent body associated with all the IMU sensors of the robot.
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 *,
                 nested_imu_quat_key: NestedData = (
                    "features", "mahony_filter", "quat"),
                 nested_imu_omega_key: NestedData = (
                    "features", "mahony_filter", "omega",),
                 ignore_twist: bool = False,
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
        :param ignore_twist: Whether to ignore the twist of the IMU quaternion
                             estimate.
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
        self.ignore_twist = ignore_twist
        self.compute_rpy = compute_rpy

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
        self.observation_space = gym.spaces.Dict(observation_space)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

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

        # Remove twist if requested
        if self.ignore_twist:
            remove_twist_from_quat(self._quat)

        # Compute the parent body angular velocities in local frame.
        # Note that batched "quaternion apply" is faster than sequential
        # "rotation matrix apply" in practice, both when using `np.einsum` or
        # single-threaded numba jitted implementation.
        quat_apply(self._imu_rel_quats,
                   self._obs_imu_omegas,
                   out=self._omega)

        # Compute the RPY representation if requested
        if self.compute_rpy:
            quat_to_rpy(self._quat, self._rpy)
