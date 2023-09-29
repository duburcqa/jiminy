""" TODO: Write documentation.
"""
import math
from typing import Union

import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True, inline='always')
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared array elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))


@nb.jit(nopython=True, nogil=True, cache=True, inline='always')
def matrix_to_yaw(rotation_matrix: np.ndarray) -> float:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    rotation matrix.
    """
    return math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])


@nb.jit(nopython=True, nogil=True, cache=True, inline='always')
def quat_to_yaw_cos_sin(quat: np.ndarray) -> np.ndarray:
    """Compute cosine and sine of the yaw from Yaw-Pitch-Roll Euler angles
    representation of a single or a batch of quaternions.

    :param quat: 1D array or a 2D array whose first dimension corresponds to
                 the number of individual quaternions, and the second to the 4
                 coordinates [qx, qy, qz, qw].
    """
    qx, qy, qz, qw = np.atleast_2d(quat)[:, -4:].T
    cos_yaw = 2 * (qw * qw + qx * qx) - 1.0
    sin_yaw = 2 * (qw * qz + qx * qy)
    yaw_cos_sin = np.stack((cos_yaw, sin_yaw), axis=-1)
    if quat.ndim == 1:
        return yaw_cos_sin[0]
    return yaw_cos_sin


@nb.jit(nopython=True, nogil=True, cache=True, inline='always')
def quat_to_yaw(quat: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    single or a batch of quaternions.

    :param quat: 1D array or a 2D array whose first dimension corresponds to
                 the number of individual quaternions, and the second to the 4
                 coordinates [qx, qy, qz, qw].
    """
    cos_yaw, sin_yaw = quat_to_yaw_cos_sin(quat).T
    return np.arctan2(sin_yaw, cos_yaw)


@nb.jit(nopython=True, nogil=True, cache=True, inline='always')
def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of quaternions.

    :param quat: 1D array or a 2D array whose first dimension corresponds to
                 the number of individual quaternions, and the second to the 4
                 coordinates [qx, qy, qz, qw].
    """
    qx, qy, qz, qw = np.atleast_2d(quat)[:, -4:].T
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1.0 - 2 * (qx * qx + qy * qy))
    pitch = -np.pi / 2 + 2 * np.arctan2(
        np.sqrt(1 + 2 * (qw * qy - qx * qz)),
        np.sqrt(1 - 2 * (qw * qy - qx * qz)),
    )
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1.0 - 2 * (qy * qy + qz * qz))
    rpy = np.stack((roll, pitch, yaw), axis=-1)
    if quat.ndim == 1:
        return rpy[0]
    return rpy
