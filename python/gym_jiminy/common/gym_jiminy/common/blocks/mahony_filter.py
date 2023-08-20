"""Implementation of Mahony filter block compatible with gym_jiminy
reinforcement learning pipeline environment design.
"""
import logging
from typing import List, Union, Optional

import numpy as np
import numba as nb
import gymnasium as gym

from jiminy_py.core import (  # pylint: disable=no-name-in-module
    ImuSensor as imu)
import pinocchio as pin

from ..bases import BaseObsT, BaseActT, BaseObserverBlock, JiminyEnvInterface
from ..utils import fill


EARTH_SURFACE_GRAVITY = 9.81
TWIST_SWING_SINGULARITY_THR = 1e-6

LOGGER = logging.getLogger(__name__)


@nb.jit(nopython=True, nogil=True, cache=True)
def mahony_filter(q: np.ndarray,
                  omega: np.ndarray,
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
                  estimate of the angular velocity in local frame.
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
    q_x, q_y, q_z, q_w = q
    v_x = 2 * (q_x * q_z - q_y * q_w)
    v_y = 2 * (q_y * q_z + q_w * q_x)
    v_z = 1 - 2 * (q_x * q_x + q_y * q_y)

    # Compute the angular velocity using Explicit Complementary Filter:
    # omega_mes = v_a_hat x v_a, where x is the cross product.
    v_x_hat, v_y_hat, v_z_hat = acc / EARTH_SURFACE_GRAVITY
    omega_mes = np.stack((
        v_y_hat * v_z - v_z_hat * v_y,
        v_z_hat * v_x - v_x_hat * v_z,
        v_x_hat * v_y - v_y_hat * v_x), axis=0)
    omega[:] = gyro - bias_hat + kp * omega_mes

    # Early return if there is no IMU motion
    if (np.abs(omega) < 1e-6).all():
        return

    # Compute Axis-Angle repr. of the angular velocity: exp3(dt * omega)
    theta = np.sqrt(np.sum(omega * omega, axis=0))
    axis = omega / theta
    theta *= dt / 2
    (p_x, p_y, p_z), p_w = (axis * np.sin(theta)), np.cos(theta)

    # Integrate the orientation: q * exp3(dt * omega)
    q[0], q[1], q[2], q[3] = (
        q_x * p_w + q_w * p_x - q_z * p_y + q_y * p_z,
        q_y * p_w + q_z * p_x + q_w * p_y - q_x * p_z,
        q_z * p_w - q_y * p_x + q_x * p_y + q_w * p_z,
        q_w * p_w - q_x * p_x - q_y * p_y - q_z * p_z,
    )

    # First order quaternion normalization to prevent compounding of errors
    q *= (3.0 - np.sum(q * q, axis=0)) / 2

    # Update Gyro bias
    bias_hat -= dt * ki * omega_mes


@nb.jit(nopython=True, nogil=True, cache=False)
def remove_twist(q: np.ndarray) -> None:
    """Remove the twist part of the Twist-after-Swing decomposition of given
    orientations in quaternion representation.

    Any rotation R can be decomposed as:

        R = R_z * R_s

    where R_z (the twist) is a rotation around e_z and R_s (the swing) is
    the "smallest" rotation matrix such that t(R_s) = t(R).

    .. seealso::
        * See "Estimation and control of the deformations of an exoskeleton
          using inertial sensors", PhD Thesis, M. Vigne, 2021, p. 130.
        * See "Swing-twist decomposition in Clifford algebra", P. Dobrowolski,
          2015 (https://arxiv.org/abs/1506.05481)


    :param q: Array whose rows are the 4 components of quaternions (x, y, z, w)
              and columns are N independent orientations from which to remove
              the swing part. It will be updated in-place.
    """
    # Compute e_z in R(q) frame (Euler-Rodrigues Formula): R(q).T @ e_z
    q_x, q_y, q_z, q_w = q
    v_x = 2 * (q_x * q_z - q_y * q_w)
    v_y = 2 * (q_y * q_z + q_w * q_x)
    v_z = 1 - 2 * (q_x * q_x + q_y * q_y)

    # Compute the "smallest" rotation transforming vector 'v_a' in 'e_z'.
    # There is a singularity when the rotation axis of orientation estimate
    # and z-axis are nearly opposites, i.e. v_z ~= -1. One solution that
    # ensure continuity of q_w is picked arbitrarily using SVD decomposition.
    # See `Eigen::Quaternion::FromTwoVectors` implementation for details.
    if q.ndim > 1:
        is_singular = np.any(v_z < -1.0 + TWIST_SWING_SINGULARITY_THR)
    else:
        is_singular = v_z < -1.0 + TWIST_SWING_SINGULARITY_THR
    if is_singular:
        if q.ndim > 1:
            for q_i in q.T:
                remove_twist(q_i)
        else:
            _, _, v_h = np.linalg.svd(np.array((
                (v_x, v_y, v_z),
                (0.0, 0.0, 1.0))
            ), full_matrices=True)
            w_2 = (1 + max(v_z, -1)) / 2
            q[:3], q[3] = v_h[-1] * np.sqrt(1 - w_2), np.sqrt(w_2)
    else:
        s = np.sqrt(2 * (1 + v_z))
        q[0], q[1], q[2], q[3] = v_y / s, v_x / s, 0.0, s / 2

    # First order quaternion normalization to prevent compounding of errors.
    # If not done, shit may happen with removing twist again and again on the
    # same quaternion, which is typically the case when the IMU is steady, so
    # that the mahony filter updated is actually skipped internally.
    q *= (3.0 - np.sum(q * q, axis=0)) / 2


@nb.jit(nopython=True, nogil=True, cache=True)
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
    twist *= 1.0 - time_constant_inv * dt
    twist += dtwist * dt

    # Update quaternion to add estimated twist
    p_z, p_w = np.sin(0.5 * twist), np.cos(0.5 * twist)
    q[0], q[1], q[2], q[3] = (
        p_w * q_x - p_z * q_y,
        p_z * q_x + p_w * q_y,
        p_z * q_w,
        p_w * q_w,
    )


class MahonyFilter(
        BaseObserverBlock[np.ndarray, np.ndarray, BaseObsT, BaseActT]):
    """Mahony's Nonlinear Complementary Filter on SO(3).

    .. seealso::
        Robert Mahony, Tarek Hamel, and Jean-Michel Pflimlin "Nonlinear
        Complementary Filters on the Special Orthogonal Group" IEEE
        Transactions on Automatic Control, Institute of Electrical and
        Electronics Engineers, 2008, 53 (5), pp.1203-1217:
        https://hal.archives-ouvertes.fr/hal-00488376/document

    .. warning::
        This filter works best for 'observe_dt' smaller or equal to 5ms. Its
        performance drops rapidly beyond this point. Having 'observe_dt' equal
        to 10ms is generally acceptable but the yaw estimate is drifting anyway
        even for fairly slow motions and without sensor noise and bias.
    """
    def __init__(self,
                 name: str,
                 env: JiminyEnvInterface[BaseObsT, BaseActT],
                 *,
                 update_ratio: int = 1,
                 twist_time_constant: Optional[float] = None,
                 exact_init: bool = True,
                 kp: Union[np.ndarray, float] = 1.0,
                 ki: Union[np.ndarray, float] = 0.1) -> None:
        """
        :param name: Name of the block.
        :param env: Environment to connect with.
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
                             Optional: `1` by default.
        :param twist_time_constant:
            If specified, it corresponds to the time constant of the leaky
            integrator used to estimate the twist part of twist-after-swing
            decomposition of the estimated orientation in place of the Mahony
            Filter. If `0.0`, then its is kept constant equal to zero. `None`
            to kept the original estimate provided by Mahony Filter.
            See `remove_twist` and `update_twist` documentation for details.
            Optional: `None` by default.
        :param twist_mode: Method used to compute twist part of twist-after-
                           swing decomposition of the estimated orientation.
                           It can be either 'none', 'leaky', 'default'. If
                           'none', then the twist part is removed from mahony
                           estimate. If 'leaky', it is removed and replaced
                           by a leaky integrator estimate with given time
                           constant. This choice is recommended because the
                           twist is not observable by the accelerometer, so
                           its value comes from the sole integration of the
                           gyroscope, which is drifting and therefore
                           unreliable.
                           Optional: `False` by default.
        :param exact_init: Whether to initialize orientation estimate using
                           accelerometer measurements or ground truth. `False`
                           is not recommended because the robot is often
                           free-falling at init, which is not realistic anyway.
                           Optional: `True` by default.
        :param mahony_kp: Proportional gain used for gyro-accel sensor fusion.
                          Set it to 0.0 to use only the gyroscope. In such a
                          case, the orientation estimate would be exact if the
                          sensor is bias- and noise-free, and the update period
                          matches the simulation integrator timestep.
                          Optional: `1.0` by default.
        :param mahony_ki: Integral gain used for gyro bias estimate.
                          Optional: `0.1` by default.
        """
        # Handling of default argument(s)
        num_imu_sensors = len(env.robot.sensors_names[imu.type])
        if isinstance(kp, float):
            kp = np.full((num_imu_sensors,), kp)
        if isinstance(ki, float):
            ki = np.full((num_imu_sensors,), ki)

        # Backup some of the user arguments
        self.exact_init = exact_init
        self.kp = kp
        self.ki = ki

        # Keep track of how the twist must be computed
        self.twist_time_constant_inv: Optional[float]
        if twist_time_constant is not None:
            if twist_time_constant > 0.0:
                self.twist_time_constant_inv = 1.0 / twist_time_constant
            else:
                self.twist_time_constant_inv = float("inf")
        else:
            self.twist_time_constant_inv = None
        self._remove_twist = self.twist_time_constant_inv is not None
        self._update_twist = (
            self.twist_time_constant_inv is not None and
            np.isfinite(self.twist_time_constant_inv))

        # Extract gyroscope and accelerometer data for fast access
        self.gyro, self.acc = np.split(env.sensors_data[imu.type], 2)

        # Allocate gyroscope bias estimate
        self._bias = np.zeros((3, num_imu_sensors))

        # Allocate twist angle estimate around z-axis in world frame
        self._twist = np.zeros((1, num_imu_sensors))

        # Store the estimate angular velocity to avoid redundant computations
        self._omega = np.zeros((3, num_imu_sensors))

        # Initialize the observer
        super().__init__(name, env, update_ratio)

    def _initialize_state_space(self) -> None:
        """Configure the internal state space of the observer.

        It corresponds to the current gyroscope bias estimate. The twist angle
        is not part of the internal state although being integrated over time
        because it is uniquely determined from the orientation estimate.
        """
        # Strictly speaking, 'q_prev' is part of the internal state of the
        # observer since it is involved in its computations. Yet, it is not an
        # internal state of the (partially observable) MDP since the previous
        # observation must be provided anyway when integrating the observable
        # dynamics by definition.
        num_imu_sensors = len(self.env.robot.sensors_names[imu.type])
        self.state_space = gym.spaces.Box(
            low=np.full((3, num_imu_sensors), -np.inf),
            high=np.full((3, num_imu_sensors), np.inf),
            dtype=np.float64)

    def _initialize_observation_space(self) -> None:
        """Configure the observation space of the observer.

        It corresponds to the current orientation estimate for all the IMUs of
        the robot at once, with special treatment for their twist part. See
        `__init__` documentation for details.
        """
        num_imu_sensors = len(self.env.robot.sensors_names[imu.type])
        self.observation_space = gym.spaces.Box(
            low=np.full((4, num_imu_sensors), -1.0 - 1e-9),
            high=np.full((4, num_imu_sensors), 1.0 + 1e-9),
            dtype=np.float64)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Make sure observe update is discrete-time
        if self.env.observe_dt <= 0.0:
            raise ValueError(
                "This block does not support time-continuous update.")

        # Refresh gyroscope and accelerometer proxies
        self.gyro, self.acc = np.split(self.env.sensors_data[imu.type], 2)

        # Reset the sensor bias
        fill(self._bias, 0)

        # Reset the twist estimate
        fill(self._twist, 0)

        # Warn if 'observe_dt' is too large to provide a meaningful
        if self.observe_dt > 0.01 + 1e-6:
            LOGGER.warning(
                "Beware 'observe_dt' (%s) is too large for Mahony filters to "
                "provide a meaningful estimate of the IMU orientations. It "
                "should not exceed 10ms.", self.observe_dt)

    def get_state(self) -> np.ndarray:
        return self._bias

    @property
    def fieldnames(self) -> List[List[str]]:
        """Get mapping between each scalar element of the observation space of
        the observer block and the associated fieldname for logging.

        It is expected to return an object with the same structure than the
        observation space, but having lists of string as leaves. Generic
        fieldnames are used by default.
        """
        sensor_names = self.env.robot.sensors_names[imu.type]
        return [[f"{name}.Quat{e}" for name in sensor_names]
                for e in ("x", "y", "z", "w")]

    def refresh_observation(self, measurement: BaseObsT) -> None:
        # Re-initialize the quaternion estimate if no simulation running.
        # It corresponds to the rotation transforming 'acc' in 'e_z'.
        if not self.env.is_simulation_running:
            is_initialized = False
            if not self.exact_init:
                if (np.abs(self.acc) < 0.1 * EARTH_SURFACE_GRAVITY).all():
                    LOGGER.warning(
                        "The acceleration at reset is too small. Impossible "
                        "to initialize Mahony filter for 'exact_init=False'.")
                else:
                    acc = self.acc / np.linalg.norm(self.acc, axis=0)
                    axis = np.stack(
                        (acc[1], -acc[0], np.zeros(acc.shape[1:])), axis=0)
                    s = np.sqrt(2 * (1 + acc[2]))
                    self.observation[:] = *(axis / s), s / 2
                    is_initialized = True
            if not is_initialized:
                robot = self.env.robot
                for i, name in enumerate(robot.sensors_names[imu.type]):
                    sensor = robot.get_sensor(imu.type, name)
                    assert isinstance(sensor, imu)
                    rot = robot.pinocchio_data.oMf[sensor.frame_idx].rotation
                    self.observation[:, i] = pin.Quaternion(rot).coeffs()
                    if self._update_twist:
                        self._twist[i] = np.arctan2(
                            self.observation[2, i], self.observation[3, i])
            if mahony_filter.signatures:
                return

        # Run an iteration of the filter, computing the next state estimate
        mahony_filter(self.observation,
                      self._omega,
                      self.gyro,
                      self.acc,
                      self._bias,
                      self.kp,
                      self.ki,
                      self.observe_dt)

        # Remove twist if requested
        if self._remove_twist:
            remove_twist(self.observation)

        if self._update_twist:
            update_twist(self.observation,
                         self._twist,
                         self._omega,
                         self.twist_time_constant_inv,
                         self.observe_dt)
