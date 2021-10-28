""" TODO: Write documentation.
"""
import math
from typing import Tuple

import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True)
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared arrray elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))


@nb.jit(nopython=True, nogil=True)
def matrix_to_yaw(rotation_matrix: np.ndarray) -> float:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    rotation matrix.
    """
    return math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])


@nb.jit(nopython=True, nogil=True)
def quat_to_yaw_cos_sin(quaternion: np.ndarray) -> Tuple[float, float]:
    """Compute the cosinus and sinus of the yaw from Yaw-Pitch-Roll Euler
    angles representation of a quaternion.
    """
    cos_yaw = - 1.0 + 2.0 * (quaternion[3] ** 2 + quaternion[0] ** 2)
    sin_yaw = 2.0 * (
        quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1])
    return cos_yaw, sin_yaw


@nb.jit(nopython=True, nogil=True)
def quat_to_yaw(quaternion: np.ndarray) -> float:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    quaternion.
    """
    cos_yaw, sin_yaw = quat_to_yaw_cos_sin(quaternion)
    return math.atan2(sin_yaw, cos_yaw)
