"""Implementation of Mahony filter block compatible with gym_jiminy
reinforcement learning pipeline environment design.
"""
import logging
from collections import OrderedDict
from typing import List, Sequence, Union, Dict

import numpy as np
import numba as nb
import gymnasium as gym

from jiminy_py.core import ImuSensor  # pylint: disable=no-name-in-module

from ..bases import BaseObs, BaseAct, BaseObserverBlock, InterfaceJiminyEnv
from ..utils import (fill,
                     quat_to_rpy,
                     matrices_to_quat,
                     swing_from_vector,
                     compute_tilt_from_quat,
                     remove_twist_from_quat)


EARTH_SURFACE_GRAVITY = 9.81

LOGGER = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True)
def mahony_filter(q: np.ndarray,
                  omega: np.ndarray,
                  cf: np.ndarray,
                  gyro: np.ndarray,
                  acc: np.ndarray,
                  bias_hat: np.ndarray,
                  kp: float,
                  ki: float,
                  dt: float) -> None:
    """Attitude Estimation using Mahony filter.

    .. note::
        This method is fully vectorized, which means that multiple IMU signals
        can be processed at once. The first dimension corresponds to individual
        IMU data.

    :param q: Current orientation estimate as a quaternion to update in-place.
              The twist part of its twist-after-swing decomposition may have
              been removed.
    :param omega: Pre-allocated memory that will be updated to store the
                  estimate of the angular velocity in local frame (unbiased).
    :param cf: Pre-allocated memory that will be updated to store the value of
               the complementary filter in local frame.
    :param gyro: Sample of tri-axial Gyroscope in rad/s. It corresponds to the
                 angular velocity in local frame.
    :param acc: Sample of tri-axial Accelerometer in m/s^2. It corresponds to
                classical (as opposed to spatial) acceleration of the IMU in
                local frame minus gravity.
    :param bias_hat: Current gyroscope bias estimate to update in-place.
    :param kp: Proportional gain used for gyro-accel sensor fusion.
    :param ki: Integral gain used for gyro bias estimate.
    :param dt: Time step, in seconds, between consecutive Quaternions.
    """
    # Compute expected Earth's gravity (Euler-Rodrigues Formula): R(q).T @ e_z
    v_x, v_y, v_z = compute_tilt_from_quat(q)

    # Update the estimate of the angular velocity
    omega[:] = gyro - bias_hat

    # Compute a correction term based on measured IMU acceleration (eq. 32c):
    # omega_mes = (- v_a) x v_a_hat, where x is the cross product.
    v_x_hat, v_y_hat, v_z_hat = acc / EARTH_SURFACE_GRAVITY
    omega_mes = np.stack((
        v_y_hat * v_z - v_z_hat * v_y,
        v_z_hat * v_x - v_x_hat * v_z,
        v_x_hat * v_y - v_y_hat * v_x), 0)

    # Apply Explicit Complementary Filter (eq. 32a - right hand)
    cf[:] = omega + kp * omega_mes

    # Early return if there is no IMU motion
    if (np.abs(cf) < 1e-6).all():
        return

    # Compute Axis-Angle repr. of the angular velocity: exp3(dt * cf)
    theta = np.sqrt(np.sum(cf * cf, 0))
    axis = cf / theta
    theta *= dt / 2
    (p_x, p_y, p_z), p_w = (axis * np.sin(theta)), np.cos(theta)

    # Integrate the orientation (eq. 32a - left hand): q * exp3(dt * cf)
    q_x, q_y, q_z, q_w = q
    q[0], q[1], q[2], q[3] = (
        q_x * p_w + q_w * p_x - q_z * p_y + q_y * p_z,
        q_y * p_w + q_z * p_x + q_w * p_y - q_x * p_z,
        q_z * p_w - q_y * p_x + q_x * p_y + q_w * p_z,
        q_w * p_w - q_x * p_x - q_y * p_y - q_z * p_z)

    # First order quaternion normalization to prevent compounding of errors
    q *= (3.0 - np.sum(np.square(q), 0)) / 2

    # Update Gyro bias (eq. 32b)
    bias_hat -= ki * dt * omega_mes


class MahonyFilter(BaseObserverBlock[
        Dict[str, np.ndarray], Dict[str, np.ndarray], BaseObs, BaseAct]):
    """Mahony's Nonlinear Complementary Filter on SO(3).

    This observer estimate the 3D orientation of all the IMU of the robot from
    the Gyrometer and Accelerometer measurements, i.e. the angular velocity in
    local frame and the linear acceleration minus gravity in local frame.

    .. seealso::
        Robert Mahony, Tarek Hamel, and Jean-Michel Pflimlin "Nonlinear
        Complementary Filters on the Special Orthogonal Group" IEEE
        Transactions on Automatic Control, Institute of Electrical and
        Electronics Engineers, 2008, 53 (5), pp.1203-1217:
        https://hal.archives-ouvertes.fr/hal-00488376/document

    .. note::
        The feature space of this observer is a dictionary storing quaternion
        estimates under key 'quat', optionally, their corresponding
        Yaw-Pitch-Roll Euler angles representations under key 'rpy' if
        `compute_rpy=True`, and finally, the angular velocity in local frame
        estimates under key 'omega'. Both leaves are 2D array of shape (N, M),
        where N is the number of components of the representation while M is
        the number of IMUs of the robot. Specifically, `N=4` for quaternions
        (Qx, Qy, Qz, Qw), 'N=3' for both the Euler angles (Roll-Pitch-Yaw) and
        the angular velocity. The Yaw angle of the Yaw-Pitch-Roll Euler angles
        representations is systematically included in the feature space, even
        if its value is meaningless, i.e. `ignore_twist=True`.

    .. warning::
        This filter works best for 'observe_dt' smaller or equal to 5ms. Its
        performance drops rapidly beyond this point. Having 'observe_dt' equal
        to 10ms is generally acceptable but the yaw estimate is drifting anyway
        even for fairly slow motions and without sensor noise and bias.
    """
    def __init__(self,
                 name: str,
                 env: InterfaceJiminyEnv[BaseObs, BaseAct],
                 *,
                 ignore_twist: bool = False,
                 exact_init: bool = True,
                 kp: Union[np.ndarray, Sequence[float], float] = 1.0,
                 ki: Union[np.ndarray, Sequence[float], float] = 0.1,
                 compute_rpy: bool = False,
                 update_ratio: int = 1) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param ignore_twist: Whether to ignore the twist of the IMU quaternion
                             estimate.
        :param exact_init: Whether to initialize orientation estimate using
                           accelerometer measurements or ground truth. `False`
                           is not recommended because the robot is often
                           free-falling at init, which is not realistic anyway.
                           Optional: `True` by default.
        :param kp: Proportional gain used for gyro-accel sensor fusion. Set it
                   to 0.0 to use only the gyroscope. In such a case, the
                   orientation estimate would be exact if the sensor is bias-
                   and noise-free, and the update period matches the
                   simulation integrator timestep.
                   Optional: `1.0` by default.
        :param ki: Integral gain used for gyro bias estimate.
                   Optional: `0.1` by default.
        :param compute_rpy: Whether to compute the Yaw-Pitch-Roll Euler angles
                            representation for the 3D orientation of the IMU,
                            in addition to the quaternion representation.
                            Optional: False by default.
        :param update_ratio: Ratio between the update period of the observer
                             and the one of the subsequent observer. -1 to
                             match the simulation timestep of the environment.
                             Optional: `1` by default.
        """
        # Handling of default argument(s)
        num_imu_sensors = len(env.robot.sensors[ImuSensor.type])
        if isinstance(kp, float):
            kp = (kp,) * num_imu_sensors
        if isinstance(ki, float):
            ki = (ki,) * num_imu_sensors

        # Backup some of the user arguments
        self.ignore_twist = ignore_twist
        self.exact_init = exact_init
        self.kp = np.asarray(kp)
        self.ki = np.asarray(ki)
        self.compute_rpy = compute_rpy

        # Whether the observer has been initialized.
        # This flag must be managed internally because relying on
        # `self.env.is_simulation_running` is not possible for observer blocks.
        # Unlike `compute_command`, the simulation is already running when
        # `refresh_observation` is called for the first time of an episode.
        self._is_initialized = False

        # Whether the observer has been compiled already.
        # This is necessary avoid raising a timeout exception during the first
        # simulation step. It is not reliable to only check if `mahony_filter`
        # has been compiled once, because a different environment may have been
        # involved, for which `mahony_filter` may be another signature,
        # triggering yet another compilation.
        self._is_compiled = False

        # Define gyroscope and accelerometer proxies for fast access
        self.gyro, self.acc = np.split(
            env.measurement["measurements"][ImuSensor.type], 2)

        # Gyroscope bias estimate
        self._bias = np.zeros((3, num_imu_sensors))

        # Explicit complementary filter
        self._cf = np.zeros((3, num_imu_sensors))

        # Define the state of the filter
        self._state = {"bias": self._bias}

        # Initialize the observer
        super().__init__(name, env, update_ratio)

        # Define some proxies for fast access
        self._quat = self.observation["quat"]
        if self.compute_rpy:
            self._rpy = self.observation["rpy"]
        else:
            self._rpy = np.array([])
        self._omega = self.observation["omega"]

    def _initialize_state_space(self) -> None:
        """Configure the internal state space of the observer.

        It corresponds to the current gyroscope bias estimate.
        """
        # Strictly speaking, 'q_prev' is part of the internal state of the
        # observer since it is involved in its computations. Yet, it is not an
        # internal state of the (partially observable) MDP since the previous
        # observation must be provided anyway when integrating the observable
        # dynamics by definition.
        num_imu_sensors = len(self.env.robot.sensors[ImuSensor.type])
        state_space: Dict[str, gym.Space] = OrderedDict()
        state_space["bias"] = gym.spaces.Box(
            low=np.full((3, num_imu_sensors), -np.inf),
            high=np.full((3, num_imu_sensors), np.inf),
            dtype=np.float64)
        self.state_space = gym.spaces.Dict(
            **state_space)  # type: ignore[arg-type]

    def _initialize_observation_space(self) -> None:
        """Configure the observation space of the observer.

        .. note::
            if `compute_rpy` is enabled, then the Roll, Pitch and Yaw angles
            are guaranteed to be within range [-pi,pi], [-pi/2,pi/2], [-pi,pi].

        It corresponds to the current orientation estimate for all the IMUs of
        the robot at once, with special treatment for their twist part. See
        `__init__` documentation for details.
        """
        observation_space: Dict[str, gym.Space] = OrderedDict()
        num_imu_sensors = len(self.env.robot.sensors[ImuSensor.type])
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

        # Initialize observation
        fill(self.observation, 0)

        # Fix initialization of the observation to be valid quaternions
        self._quat[-1] = 1.0

        # Make sure observe update is discrete-time
        if self.env.observe_dt <= 0.0:
            raise ValueError(
                "This block does not support time-continuous update.")

        # Reset the sensor bias estimate
        fill(self._bias, 0)

        # Warn if 'observe_dt' is too large to provide a meaningful
        if self.observe_dt > 0.01 + 1e-6:
            LOGGER.warning(
                "Beware 'observe_dt' (%s) is too large for Mahony filters to "
                "provide a meaningful estimate of the IMU orientations. It "
                "should not exceed 10ms.", self.observe_dt)

        # Call `refresh_observation` manually to make sure that all the jitted
        # method of it control flow has been compiled.
        # Note that setup must be called once again because compilation will
        # mess up with the internal state of the filter.
        if not self._is_compiled:
            self._is_initialized = False
            for _ in range(2):
                self.refresh_observation(self.env.observation)
            self._is_compiled = True
            self._setup()
            return

        # Consider that the observer is not initialized anymore
        self._is_initialized = False

    def get_state(self) -> Dict[str, np.ndarray]:
        return self._state

    @property
    def fieldnames(self) -> Dict[str, List[List[str]]]:
        imu_sensors = self.env.robot.sensors[ImuSensor.type]
        fieldnames: Dict[str, List[List[str]]] = {}
        fieldnames["quat"] = [
            [".".join((sensor.name, f"Quat{e}")) for sensor in imu_sensors]
            for e in ("X", "Y", "Z", "W")]
        fieldnames["omega"] = [
            [".".join((sensor.name, e)) for sensor in imu_sensors]
            for e in ("X", "Y", "Z")]
        if self.compute_rpy:
            fieldnames["rpy"] = [
                [".".join((sensor.name, e)) for sensor in imu_sensors]
                for e in ("Roll", "Pitch", "Yaw")]
        return fieldnames

    def refresh_observation(self, measurement: BaseObs) -> None:
        # Re-initialize the quaternion estimate if no simulation running.
        # It corresponds to the rotation transforming 'acc' in 'e_z'.
        if not self._is_initialized:
            if not self.exact_init:
                if (np.abs(self.acc) < 0.1 * EARTH_SURFACE_GRAVITY).all():
                    if self._is_compiled:
                        LOGGER.warning(
                            "The robot is free-falling. Impossible to "
                            "initialize Mahony filter for 'exact_init=False'.")
                else:
                    # Try to determine the orientation of the IMU from its
                    # measured acceleration at initialization. This approach is
                    # not very accurate because the initial acceleration is
                    # often jerky, plus the tilt is not observable at all.
                    acc = self.acc / np.linalg.norm(self.acc, axis=0)
                    swing_from_vector(acc, self._quat)

                    self._is_initialized = True
            if not self._is_initialized:
                # Get true orientation of IMU frames
                imu_rots = []
                robot = self.env.robot
                for sensor in robot.sensors[ImuSensor.type]:
                    assert isinstance(sensor, ImuSensor)
                    rot = robot.pinocchio_data.oMf[sensor.frame_index].rotation
                    imu_rots.append(rot)

                # Convert the rotation matrices of the IMUs to quaternions
                matrices_to_quat(tuple(imu_rots), self._quat)

                self._is_initialized = True

            # Compute the RPY representation if requested
            if self.compute_rpy:
                quat_to_rpy(self._quat, self._rpy)

            return

        # Run an iteration of the filter, computing the next state estimate
        mahony_filter(self._quat,
                      self._omega,
                      self._cf,
                      self.gyro,
                      self.acc,
                      self._bias,
                      self.kp,
                      self.ki,
                      self.observe_dt)

        # Remove twist if requested
        if self.ignore_twist:
            remove_twist_from_quat(self._quat)

        # Compute the RPY representation if requested
        if self.compute_rpy:
            quat_to_rpy(self._quat, self._rpy)
