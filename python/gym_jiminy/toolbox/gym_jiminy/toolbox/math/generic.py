""" TODO: Write documentation.
"""
from typing import Union, Tuple, Optional

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
def quat_to_yaw_cos_sin(quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cosine and sine of the yaw from Yaw-Pitch-Roll Euler angles
    representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    """
    assert quat.ndim >= 1
    (q_xy, q_yy), (q_zz, q_zw) = quat[-3] * quat[-4:-2], quat[-2] * quat[-2:]
    cos_yaw, sin_yaw = 1.0 - 2 * (q_yy + q_zz), 2 * (q_xy + q_zw)
    return cos_yaw, sin_yaw


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
def quat_to_rpy(quat: np.ndarray,
                out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((3, *quat.shape[1:]))
    else:
        assert out.shape == (3, *quat.shape[1:])
        out_ = out
    roll, pitch, yaw = out_
    q_xx, q_xy, q_xz, q_xw = quat[-4] * quat[-4:]
    q_yy, q_yz, q_yw = quat[-3] * quat[-3:]
    q_zz, q_zw = quat[-2] * quat[-2:]
    roll[:] = np.arctan2(2 * (q_xw + q_yz), 1.0 - 2 * (q_xx + q_yy))
    pitch[:] = - np.pi / 2 + 2 * np.arctan2(
        np.sqrt(1.0 + 2 * (q_yw - q_xz)), np.sqrt(1.0 - 2 * (q_yw - q_xz)))
    yaw[:] = np.arctan2(2 * (q_zw + q_xy), 1.0 - 2 * (q_yy + q_zz))
    return out_


@nb.jit(nopython=True, cache=True)
def quat_to_matrix(quat: np.ndarray,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a
    batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates [qx, qy, qz, qw].
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((3, 3, *quat.shape[1:]))
    else:
        assert out.shape == (3, 3, *quat.shape[1:])
        out_ = out
    q_xx, q_xy, q_xz, q_xw = quat[-4] * quat[-4:]
    q_yy, q_yz, q_yw = quat[-3] * quat[-3:]
    q_zz, q_zw = quat[-2] * quat[-2:]
    out_[0][0] = 1.0 - 2 * (q_yy + q_zz)
    out_[0][1] = 2 * (q_xy - q_zw)
    out_[0][2] = 2 * (q_xz + q_yw)
    out_[1][0] = 2 * (q_xy + q_zw)
    out_[1][1] = 1.0 - 2 * (q_xx + q_zz)
    out_[1][2] = 2 * (q_yz - q_xw)
    out_[2][0] = 2 * (q_xz - q_yw)
    out_[2][1] = 2 * (q_yz + q_xw)
    out_[2][2] = 1.0 - 2 * (q_xx + q_yy)
    return out_


@nb.jit(nopython=True, cache=True)
def matrix_to_quat(mat: np.ndarray,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of a single or a
    batch of rotation matrices.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert mat.ndim >= 2
    if out is None:
        out_ = np.empty((4, *mat.shape[2:]))
    else:
        assert out.shape == (4, *mat.shape[2:])
        out_ = out
    q_x, q_y, q_z, q_w = out_
    q_x[:] = mat[2, 1] - mat[1, 2]
    q_y[:] = mat[0, 2] - mat[2, 0]
    q_z[:] = mat[1, 0] - mat[0, 1]
    q_w[:] = 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]
    out_ /= np.sqrt(np.sum(out_ * out_, 0))
    return out_


@nb.jit(nopython=True, cache=True)
def rpy_to_matrix(rpy: np.ndarray,
                  out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert rpy.ndim >= 1
    if out is None:
        out_ = np.empty((3, 3, *rpy.shape[1:]))
    else:
        assert out.shape == (3, 3, *rpy.shape[1:])
        out_ = out
    cos_roll, cos_pitch, cos_yaw = np.cos(rpy[-3:])
    sin_roll, sin_pitch, sin_yaw = np.sin(rpy[-3:])
    out_[0][0] = cos_pitch * cos_yaw
    out_[0][1] = - cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw
    out_[0][2] = sin_roll * sin_yaw + cos_roll * sin_pitch * cos_yaw
    out_[1][0] = cos_pitch * sin_yaw
    out_[1][1] = cos_roll * cos_yaw + sin_roll * sin_pitch * sin_yaw
    out_[1][2] = - sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw
    out_[2][0] = - sin_pitch
    out_[2][1] = sin_roll * cos_pitch
    out_[2][2] = cos_roll * cos_pitch
    return out_


@nb.jit(nopython=True, cache=True)
def matrix_to_rpy(mat: np.ndarray,
                  out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of rotation matrices.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert mat.ndim >= 2
    if out is None:
        out_ = np.empty((3, *mat.shape[2:]))
    else:
        assert out.shape == (3, *mat.shape[2:])
        out_ = out
    roll, pitch, yaw = out_
    yaw[:] = np.arctan2(mat[1, 0], mat[0, 0])
    cos_pitch = np.sqrt(mat[2, 2] ** 2 + mat[2, 1] ** 2)
    pitch[:] = np.arctan2(- mat[2, 0], np.sign(yaw) * cos_pitch)
    yaw[:] += np.pi * (yaw < 0.0)
    sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
    roll[:] = np.arctan2(
        sin_yaw * mat[0, 2] - cos_yaw * mat[1, 2],
        cos_yaw * mat[1, 1] - sin_yaw * mat[0, 1])
    return out_


@nb.jit(nopython=True, cache=True)
def rpy_to_quat(rpy: np.ndarray,
                out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert rpy.ndim >= 1
    if out is None:
        out_ = np.empty((4, *rpy.shape[1:]))
    else:
        assert out.shape == (4, *rpy.shape[1:])
        out_ = out
    q_x, q_y, q_z, q_w = out_
    roll, pitch, yaw = rpy
    cos_roll, sin_roll = np.cos(roll / 2), np.sin(roll / 2)
    cos_pitch, sin_pitch = np.cos(pitch / 2), np.sin(pitch / 2)
    cos_yaw, sin_yaw = np.cos(yaw / 2), np.sin(yaw / 2)
    q_x[:] = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    q_y[:] = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    q_z[:] = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    q_w[:] = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    return out_


@nb.jit(nopython=True, cache=True)
def quat_multiply(quat_left: np.ndarray,
                  quat_right: np.ndarray,
                  out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the composition of rotations as the product of two single or
    batch of quaternions [qx, qy, qz, qw], namely `quat_left * quat_right`

    .. warning::
        Beware the argument order is important because the composition of
        rotations is not commutative.

    .. seealso::
        See `https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation`.

    :param quat_left: Left-hand side of the quaternion product, as a
                      N-dimensional array whose first dimension gathers the 4
                      quaternion coordinates [qx, qy, qz, qw].
    :param quat_right: Right-hand side of the quaternion product, as a
                       N-dimensional array whose first dimension gathers the 4
                       quaternion coordinates [qx, qy, qz, qw].
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    assert quat_left.ndim >= 1 and quat_left.shape == quat_right.shape
    if out is None:
        out_ = np.empty_like(quat_left)
    else:
        assert out.shape == quat_left.shape
        out_ = out
    qx_out, qy_out, qz_out, qw_out = out_
    (qx_l, qy_l, qz_l, qw_l), (qx_r, qy_r, qz_r, qw_r) = quat_left, quat_right
    qx_out[:] = qw_l * qx_r + qx_l * qw_r + qy_l * qz_r - qz_l * qy_r
    qy_out[:] = qw_l * qy_r - qx_l * qz_r + qy_l * qw_r + qz_l * qx_r
    qz_out[:] = qw_l * qz_r + qx_l * qy_r - qy_l * qx_r + qz_l * qw_r
    qw_out[:] = qw_l * qw_r - qx_l * qx_r - qy_l * qy_r - qz_l * qz_r
    return out_


def quat_average(quat: np.ndarray,
                 axis: Optional[Union[Tuple[int, ...], int]] = None
                 ) -> np.ndarray:
    """Compute the average of a batch of quaternions [qx, qy, qz, qw] over some
    or all axes.

    Here, the average is defined as a quaternion minimizing the mean error
    wrt every individual quaternion. The distance metric used as error is the
    dot product of quaternions `p.dot(q)`, which is directly related to the
    angle between them `cos(angle(p.conjugate() * q) / 2)`. This metric as the
    major advantage to yield a quadratic problem, which can be solved very
    efficiently, unlike the squared angle `angle(p.conjugate() * q) ** 2`.

    :param quat: N-dimensional (N >= 2) array whose first dimension gathers the
                 4 quaternion coordinates [qx, qy, qz, qw].
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    # TODO: This function cannot be jitted because numba does not support
    # batched matrix multiplication for now.
    assert quat.ndim >= 2
    if axis is None:
        axis = tuple(range(1, quat.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    assert len(axis) > 0 and 0 not in axis
    q_perm = quat.transpose((
        *(i for i in range(1, quat.ndim) if i not in axis), 0, *axis))
    q_flat = q_perm.reshape((*q_perm.shape[:-len(axis)], -1))
    _, eigvec = np.linalg.eigh(q_flat @ np.swapaxes(q_flat, -1, -2))
    return np.moveaxis(eigvec[..., -1], -1, 0)
