""" TODO: Write documentation.
"""
import math
from typing import Union

import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared array elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))


@nb.jit(nopython=True, cache=True)
def matrix_to_yaw(mat: np.ndarray) -> float:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    rotation matrix in 3D Euclidean space.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    """
    assert mat.ndim >= 2
    return np.arctan2(mat[1, 0], mat[0, 0])


@nb.jit(nopython=True, cache=True)
def quat_to_yaw_cos_sin(quat: np.ndarray) -> np.ndarray:
    """Compute cosine and sine of the yaw from Yaw-Pitch-Roll Euler angles
    representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    """
    assert quat.ndim >= 1
    (qxy, qyy), (qzz, qzw) = quat[-3] * quat[-4:-2], quat[-2] * quat[-2:]
    cos_yaw, sin_yaw = 1.0 - 2 * (qyy + qzz), 2 * (qxy + qzw)
    if quat.ndim == 1:
        return np.array((cos_yaw, sin_yaw))
    return np.stack((cos_yaw, sin_yaw))


@nb.jit(nopython=True, cache=True)
def quat_to_yaw(quat: np.ndarray) -> Union[float, np.ndarray]:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    single or a batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    """
    assert quat.ndim >= 1
    cos_yaw, sin_yaw = quat_to_yaw_cos_sin(quat)
    return np.arctan2(sin_yaw, cos_yaw)


@nb.jit(nopython=True, cache=True)
def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    """
    assert quat.ndim >= 1
    qxx, qxy, qxz, qxw = quat[-4] * quat[-4:]
    qyy, qyz, qyw = quat[-3] * quat[-3:]
    qzz, qzw = quat[-2] * quat[-2:]
    roll = np.arctan2(2 * (qxw + qyz), 1.0 - 2 * (qxx + qyy))
    pitch = -np.pi / 2 + 2 * np.arctan2(
        np.sqrt(1.0 + 2 * (qyw - qxz)),
        np.sqrt(1.0 - 2 * (qyw - qxz)),
    )
    yaw = np.arctan2(2 * (qzw + qxy), 1.0 - 2 * (qyy + qzz))
    if quat.ndim == 1:
        return np.array((roll, pitch, yaw))
    return np.stack((roll, pitch, yaw))


@nb.jit(nopython=True, cache=True)
def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a
    batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    """
    assert quat.ndim >= 1
    qxx, qxy, qxz, qxw = quat[-4] * quat[-4:]
    qyy, qyz, qyw = quat[-3] * quat[-3:]
    qzz, qzw = quat[-2] * quat[-2:]
    mat_flat = (
        1.0 - 2 * (qyy + qzz), 2 * (qxy - qzw), 2 * (qxz + qyw),
        2 * (qxy + qzw), 1.0 - 2 * (qxx + qzz), 2 * (qyz - qxw),
        2 * (qxz - qyw), 2 * (qyz + qxw), 1.0 - 2 * (qxx + qyy),
    )
    if quat.ndim == 1:
        return np.array(mat_flat).reshape((3, 3))
    return np.stack(mat_flat).reshape((3, 3, *quat.shape[1:]))


@nb.jit(nopython=True, cache=True)
def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of a single or a
    batch of rotation matrices.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    """
    assert mat.ndim >= 2
    quat_flat = (
        mat[2, 1] - mat[1, 2],
        mat[0, 2] - mat[2, 0],
        mat[1, 0] - mat[0, 1],
        1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2])
    quat = np.array(quat_flat) if mat.ndim == 2 else np.stack(quat_flat)
    quat /= np.sqrt(np.sum(quat * quat, 0))
    return quat


@nb.jit(nopython=True, cache=True)
def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    """
    (c_r, c_p, c_y), (s_r, s_p, s_y) = np.cos(rpy[-3:]), np.sin(rpy[-3:])
    mat_flat = (
        c_p * c_y, -c_r * s_y + s_r * s_p * c_y,  s_r * s_y + c_r * s_p * c_y,
        c_p * s_y,  c_r * c_y + s_r * s_p * s_y, -s_r * c_y + c_r * s_p * s_y,
        -s_p, s_r * c_p, c_r * c_p
    )
    if rpy.ndim == 1:
        return np.array(mat_flat).reshape((3, 3))
    return np.stack(mat_flat).reshape((3, 3, *rpy.shape[1:]))


@nb.jit(nopython=True, cache=True)
def matrix_to_rpy(mat: np.ndarray) -> np.ndarray:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of rotation matrices.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    """
    return quat_to_rpy(matrix_to_quat(mat))


@nb.jit(nopython=True, cache=True)
def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    """
    return matrix_to_quat(rpy_to_matrix(rpy))
