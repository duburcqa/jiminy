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

    :param quat: 1D array, a 2D array or nD array whose first dimensions
                 corresponds to the number of individual quaternions, and the
                 last to the 4 coordinates [qx, qy, qz, qw].
    """
    qx, qy, qz, qw = np.atleast_2d(quat)[..., -4:].T
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


@nb.jit(nopython=True, nogil=True, inline='always')
def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a
    batch of quaternions.

    :param quat: 1D array, a 2D array or nD array whose first dimensions
                 corresponds to the number of individual quaternions, and the
                 last to the 4 coordinates [qx, qy, qz, qw].
    """
    qx, qy, qz, qw = np.atleast_2d(quat)[..., -4:].T
    col_1 = np.stack(
        (
            qx * qx - qy * qy - qz * qz + qw * qw,
            2 * (qx * qy + qz * qw),
            2 * (qx * qz - qy * qw),
        ),
        -1,
    )
    col_2 = np.stack(
        (
            2 * (qx * qy - qz * qw),
            -qx * qx + qy * qy - qz * qz + qw * qw,
            2 * (qy * qz + qx * qw),
        ),
        -1,
    )
    col_3 = np.stack(
        (
            2 * (qx * qz + qy * qw),
            2 * (qy * qz - qx * qw),
            -qx * qx - qy * qy + qz * qz + qw * qw,
        ),
        -1,
    )
    matrix = np.stack((col_1, col_2, col_3), -1)
    if quat.ndim == 1:
        return matrix[0, :, :]
    return matrix


@nb.jit(nopython=True, nogil=True, inline='always')
def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of a single or a
    batch of rotation matrices.

    :param matrix: 2D array, a 3D array or a nD array whose first dimensions
                   corresponds to the number of individual rotation matrices,
                   and the two last to the 3-by-3 rotation matrices.
    """
    quat = np.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
            1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2],
        ),
        -1,
    )
    norm = np.sqrt(np.sum(np.atleast_2d(quat * quat), -1))
    quat /= norm.reshape((*quat.shape[:-1], 1))
    return quat


@nb.jit(nopython=True, nogil=True, inline='always')
def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: 1D array, a 2D array or a nD array whose first dimensions
                corresponds to the number of individual Euler angles, and the
                last to the 3 coordinates [Roll, Pitch, Yaw].
    """
    c_r, c_p, c_y = np.cos(np.atleast_2d(rpy)[..., -3:].T)
    s_r, s_p, s_y = np.sin(np.atleast_2d(rpy)[..., -3:].T)
    col_1 = np.stack((c_p * c_y, c_p * s_y, -s_p), -1)
    col_2 = np.stack(
        (-c_r * s_y + s_r * s_p * c_y, c_r * c_y + s_r * s_p * s_y, s_r * c_p),
        -1,
    )
    col_3 = np.stack(
        (s_r * s_y + c_r * s_p * c_y, -s_r * c_y + c_r * s_p * s_y, c_r * c_p),
        -1,
    )
    matrix = np.stack((col_1, col_2, col_3), -1)
    if rpy.ndim == 1:
        return matrix[0, :, :]
    return matrix


@nb.jit(nopython=True, nogil=True, inline='always')
def matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of rotation matrices.

    :param matrix: 2D array, a 3D array or a nD array whose first dimensions
                   corresponds to the number of individual rotation matrices,
                   and the two last to the 3-by-3 rotation matrices.
    """
    return quat_to_rpy(matrix_to_quat(matrix))


@nb.jit(nopython=True, nogil=True, inline='always')
def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: 1D array, a 2D array or a nD array whose first dimensions
                corresponds to the number of individual Euler angles, and the
                last to the 3 coordinates [Roll, Pitch, Yaw].
    """
    return matrix_to_quat(rpy_to_matrix(rpy))
