from typing import Any, List, Optional

import numpy as np
import numba as nb
import gymnasium as gym

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    ImuSensor as imu)
import pinocchio as pin

from ..bases import BaseObsT, BaseActT, BaseObserverBlock, JiminyEnvInterface
from ..utils import DataNested, fill


EARTH_SURFACE_GRAVITY = 9.81


@nb.jit(nopython=True, nogil=True)
def mahony_filter(q: np.ndarray,
                  gyro: np.ndarray,
                  acc: np.ndarray,
                  bias_hat: np.ndarray,
                  dt: float,
                  k_P: float,
                  k_I: float) -> None:
    """Attitude Estimation using Mahony filter.

    .. note::
        This method is fully vectorized, which means that multiple IMU signals
        can be processed at once. The first dimension corresponds to individual
        IMU data.

    :param q: Current quaternion estimate.
    :param gyro: Sample of tri-axial Gyroscope in rad/s.
    :param acc: Sample of tri-axial Accelerometer in m/s^2.
    :param dt: Time step, in seconds, between consecutive Quaternions.
    """
    # Compute expected Earth's gravity: R(q).T @ e_z
    q_x, q_y, q_z, q_w = np.atleast_2d(q).T
    v_a = np.stack((
        2 * (q_x * q_z - q_y * q_w),
        2 * (q_y * q_z + q_w * q_x),
        1 - 2 * (q_x * q_x + q_y * q_y),
    ), axis=-1)

    # Compute the angular velocity using Explicit Complementary Filter
    v_a_hat = acc / EARTH_SURFACE_GRAVITY
    omega_mes = np.cross(v_a_hat, v_a)
    omega = gyro - bias_hat + k_P * omega_mes

    # Compute Axis-Angle repr. of the angular velocity: exp3(dt * omega)
    theta = np.sqrt(
        np.sum(omega * omega, axis=-1)).reshape((*omega.shape[:-1], 1))
    axis = omega / theta
    theta *= dt / 2
    (p_x, p_y, p_z), p_w = (axis * np.sin(theta)).T, np.cos(theta).reshape(-1)

    # Integrate the orientation: q * exp3(dt * omega)
    q[:] = np.stack((
        q_x * p_w + q_w * p_x - q_z * p_y + q_y * p_z,
        q_y * p_w + q_z * p_x + q_w * p_y - q_x * p_z,
        q_z * p_w - q_y * p_x + q_x * p_y + q_w * p_z,
        q_w * p_w - q_x * p_x - q_y * p_y - q_z * p_z,
    ), axis=-1)

    # First order quaternion normalization to prevent compounding of errors
    q *= (3 - np.sum(np.atleast_2d(
        q * q), axis=-1).reshape((*q.shape[:-1], 1))) / 2

    # Update Gyro bias
    bias_hat -= dt * k_I * omega_mes.reshape(bias_hat.shape)


class MahonyFilter(
        BaseObserverBlock[np.ndarray, np.ndarray, BaseObsT, BaseActT]):
    """Mahony's Nonlinear Complementary Filter on SO(3).

    .. seealso::
        Robert Mahony, Tarek Hamel, and Jean-Michel Pflimlin "Nonlinear
        Complementary Filters on the Special Orthogonal Group" IEEE
        Transactions on Automatic Control, Institute of Electrical and
        Electronics Engineers, 2008, 53 (5), pp.1203-1217:
        https://hal.archives-ouvertes.fr/hal-00488376/document
    """
    def __init__(self,
                 name: str,
                 env: JiminyEnvInterface[BaseObsT, BaseActT],
                 update_ratio: int = 1,
                 exact_init: bool = True,
                 mahony_kp: float = 0.0,
                 mahony_ki: float = 0.0,
                 **kwargs: Any) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
        :param exact_init: Whether to initialize orientation estimate using
                           accelerometer measurements or ground truth.
        :param mahony_kp: Proportional gain used for gyro-accel sensor fusion.
        :param mahony_ki: Integral gain used for gyro bias estimate.
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup some of the user arguments
        self.exact_init = exact_init
        self.kp = mahony_kp
        self.ki = mahony_ki

        # Allocate bias and orientation estimates
        num_imu_sensors = len(env.robot.sensors_names[imu.type])
        self._bias = np.zeros((num_imu_sensors, 3))

        # Initialize the observer
        super().__init__(name, env, update_ratio)

    def _initialize_state_space(self) -> None:
        # Strictly speaking, 'q_prev' is part of the internal state of the
        # observer since it is involved in its computations. Yet, it is not an
        # internal state of the (partially observable) MDP since the previous
        # observation must be provided anyway when integrating the observable
        # dynamics by definition.
        num_imu_sensors = len(self.env.robot.sensors_names[imu.type])
        self.state_space = gym.spaces.Box(
            low=np.full((num_imu_sensors, 3), -np.inf),
            high=np.full((num_imu_sensors, 3), np.inf),
            dtype=np.float64)

    def _initialize_observation_space(self) -> None:
        num_imu_sensors = len(self.env.robot.sensors_names[imu.type])
        self.observation_space = gym.spaces.Box(
            low=np.full((num_imu_sensors, 4), -1.0 - 1e-12),
            high=np.full((num_imu_sensors, 4), 1.0 + 1e-12),
            dtype=np.float64)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Reset the sensor bias and quaternion estimate
        fill(self._bias, 0)
        fill(self._q_prev, float('nan'))

    def get_state(self) -> DataNested:
        return self._bias

    @property
    def fieldnames(self) -> List[List[str]]:
        """Get mapping between each scalar element of the observation space of
        the observer block and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        observation space, but having lists of string as leaves. Generic
        fieldnames are used by default.
        """
        return [[f"{name}.Quat{e}" for e in ("x", "y", "z", "w")]
                for name in self.env.robot.sensors_names[imu.type]]

    def refresh_observation(self, measurement: BaseObsT) -> None:
        # Extract gyroscope and accelerometer data
        gyro, acc = np.split(self.env.sensors_data[imu.type], 2)

        # Re-initialize the quaternion estimate if no simulation running.
        # It corresponds to the rotation transforming 'acc' in 'e_z'.
        is_simulation_running = self.env.simulator.is_simulation_running
        if not is_simulation_running:
            if self.exact_init:
                for i, name in enumerate(
                        self.env.robot.sensors_names[imu.type]):
                    sensor: imu = self.env.robot.get_sensor(imu.type, name)
                    oMf = self.robot.pinocchio_model.oMf[sensor.frame_idx]
                    self._q_prev[i] = pin.Quaternion(oMf.rotation).coeffs()
            else:
                acc = acc / np.linalg.norm(acc, axis=0)
                axis = np.array((acc[1], -acc[0], 0.0))
                s = np.sqrt(2 * (1 + acc[2]))
                self._q_prev[:] = np.stack((*(axis / s), s / 2), axis=0)

        # Run an iteration of the filter, computing the next state estimate
        mahony_filter(self._q_prev,
                      gyro.T,
                      acc.T,
                      self._bias,
                      self.observe_dt,
                      self.kp,
                      self.ki)
