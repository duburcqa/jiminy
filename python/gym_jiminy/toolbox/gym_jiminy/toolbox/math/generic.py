""" TODO: Write documentation.
"""
import math

import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True)
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared arrray elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))


@nb.jit(nopython=True, nogil=True)
def matrix_to_yaw(rotation_matrix: np.ndarray) -> float:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation from a
    rotation matrix.
    """
    return math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])


@nb.jit(nopython=True, nogil=True)
def quat_to_yaw(quaternion: np.ndarray) -> float:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation from a
    quaternion.
    """
    return math.atan2(
        2.0 * (quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1]),
        - 1.0 + 2.0 * (quaternion[3] ** 2 + quaternion[0] ** 2))
