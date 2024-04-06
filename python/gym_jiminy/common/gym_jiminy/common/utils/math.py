"""Mathematic utilities heavily optimized for speed.

They combine batch-processing with Just-In-Time (JIT) compiling via Numba when
possible for optimal performance. Most of them are dealing with rotations (SO3)
to perform transformations or convert from one representation to another.
"""
from typing import Union, Tuple, Optional, no_type_check

import numpy as np
import numba as nb

from .spaces import ArrayOrScalar


TWIST_SWING_SINGULARITY_THR = 1e-6


@nb.jit(nopython=True, cache=True, inline='always')
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared array elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))


@nb.jit(nopython=True, cache=True)
def matrix_to_yaw(mat: np.ndarray,
                  out: Optional[np.ndarray] = None
                  ) -> Union[float, np.ndarray]:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    rotation matrix in 3D Euclidean space.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    """
    assert mat.ndim >= 2

    # Allocate memory for the output array
    if out is None:
        out_ = np.empty(mat.shape[2:])
    else:
        assert out.shape == mat.shape[2:]
        out_ = out

    out_[:] = np.arctan2(mat[1, 0], mat[0, 0])

    return out_


@nb.jit(nopython=True, cache=True, inline='always')
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

    # Allocate memory for the output array
    if out is None:
        out_ = np.empty((3, *quat.shape[1:]))
    else:
        assert out.shape == (3, *quat.shape[1:])
        out_ = out

    # Compute some intermediary quantities
    q_xx, q_xy, q_xz, q_xw = quat[-4] * quat[-4:]
    q_yy, q_yz, q_yw = quat[-3] * quat[-3:]
    q_zz, q_zw = quat[-2] * quat[-2:]

    # First-order normalization (by copy) to avoid numerical instabilities
    q_ww = quat[-1] * quat[-1]
    norm_inv = ((3.0 - (q_xx + q_yy + q_zz + q_ww)) / 2)
    q_yw *= norm_inv
    q_xz *= norm_inv

    # Compute Roll, Pitch and Yaw separately
    # roll, pitch, yaw = out_
    out_[0] = np.arctan2(2 * (q_xw + q_yz), 1.0 - 2 * (q_xx + q_yy))
    out_[1] = - np.pi / 2 + 2 * np.arctan2(
        np.sqrt(1.0 + 2 * (q_yw - q_xz)), np.sqrt(1.0 - 2 * (q_yw - q_xz)))
    out_[2] = np.arctan2(2 * (q_zw + q_xy), 1.0 - 2 * (q_yy + q_zz))

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
    # q_x, q_y, q_z, q_w = out_
    out_[0] = mat[2, 1] - mat[1, 2]
    out_[1] = mat[0, 2] - mat[2, 0]
    out_[2] = mat[1, 0] - mat[0, 1]
    out_[3] = 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]
    out_ /= np.sqrt(np.sum(np.square(out_), 0))
    return out_


# TODO: Merge this method with `matrix_to_quat` by leverage compile-time
# implementation dispatching via `nb.generated_jit` or `nb.overload`.
@nb.jit(nopython=True, cache=True)
def matrices_to_quat(mat_list: Tuple[np.ndarray, ...],
                     out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the [qx, qy, qz, qw] Quaternion representation of multiple
    rotation matrices.

    .. seealso::
        See https://math.stackexchange.com/a/3183435.

    :param mat: Tuple of N arrays corresponding to independent 3D rotation
                matrices.
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    if out is None:
        out_ = np.empty((4, len(mat_list)))
    else:
        assert out.shape == (4, len(mat_list))
        out_ = out
    # q_x, q_y, q_z, q_w = out_
    t = np.empty((len(mat_list),))
    for i, mat in enumerate(mat_list):
        if mat[2, 2] < 0:
            if mat[0, 0] > mat[1, 1]:
                t[i] = 1 + mat[0, 0] - mat[1, 1] - mat[2, 2]
                out_[0][i] = t[i]
                out_[1][i] = mat[1, 0] + mat[0, 1]
                out_[2][i] = mat[0, 2] + mat[2, 0]
                out_[3][i] = mat[2, 1] - mat[1, 2]
            else:
                t[i] = 1 - mat[0, 0] + mat[1, 1] - mat[2, 2]
                out_[0][i] = mat[1, 0] + mat[0, 1]
                out_[1][i] = t[i]
                out_[2][i] = mat[2, 1] + mat[1, 2]
                out_[3][i] = mat[0, 2] - mat[2, 0]
        else:
            if mat[0, 0] < -mat[1, 1]:
                t[i] = 1 - mat[0, 0] - mat[1, 1] + mat[2, 2]
                out_[0][i] = mat[0, 2] + mat[2, 0]
                out_[1][i] = mat[2, 1] + mat[1, 2]
                out_[2][i] = t[i]
                out_[3][i] = mat[1, 0] - mat[0, 1]
            else:
                t[i] = 1 + mat[0, 0] + mat[1, 1] + mat[2, 2]
                out_[0][i] = mat[2, 1] - mat[1, 2]
                out_[1][i] = mat[0, 2] - mat[2, 0]
                out_[2][i] = mat[1, 0] - mat[0, 1]
                out_[3][i] = t[i]
    out_ /= 2 * np.sqrt(t)
    return out_


@nb.jit(nopython=True, cache=True)
def transforms_to_vector(
        transform_list: Tuple[Tuple[np.ndarray, np.ndarray], ...],
        out: Optional[np.ndarray] = None) -> np.ndarray:
    """Stack the translation vector [x, y, z] and the quaternion representation
    [qx, qy, qz, qw] of the orientation of multiple transform tuples.

    .. note::
        Internally, it copies the translation unaffected and convert rotation
        matrices to quaternions using `matrices_to_quat`.

    :param transform_list: Tuple of N transforms, each of which represented as
                           pairs gathering the translation as a vector and the
                           orientation as a 3D rotation matrix.
    :param out: A pre-allocated array into which the result is stored. If not
                provided, a new array is freshly-allocated, which is slower.
    """
    # Allocate memory if necessart
    if out is None:
        out_ = np.empty((7, len(transform_list)))
    else:
        out2d = out[:, np.newaxis] if out.ndim == 1 else out
        assert out2d.shape == (7, len(transform_list))
        out_ = out2d

    # Simply copy the translation
    for i, (translation, _) in enumerate(transform_list):
        out_[:3, i] = translation

    # Convert all rotation matrices to quaternions at once
    rotation_list = [rotation for _, rotation in transform_list]
    matrices_to_quat(rotation_list, out_[-4:])

    # Revel extra dimension before returning if not present initially
    if out is not None and out.ndim == 1:
        return out_[:, 0]
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

    # Direct translation of Eigen: `R.eulerAngles(2, 1, 0).reverse()`
    # roll, pitch, yaw = out_
    cos_pitch = np.sqrt(mat[2, 2] ** 2 + mat[2, 1] ** 2)
    out_[1] = np.arctan2(- mat[2, 0], cos_pitch)
    out_[2] = np.arctan2(mat[1, 0], mat[0, 0])
    sin_yaw, cos_yaw = np.sin(out_[2]), np.cos(out_[2])
    out_[0] = np.arctan2(
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
    roll, pitch, yaw = rpy
    cos_roll, sin_roll = np.cos(roll / 2), np.sin(roll / 2)
    cos_pitch, sin_pitch = np.cos(pitch / 2), np.sin(pitch / 2)
    cos_yaw, sin_yaw = np.cos(yaw / 2), np.sin(yaw / 2)
    # q_x, q_y, q_z, q_w = out_
    out_[0] = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    out_[1] = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    out_[2] = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    out_[3] = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    return out_


@nb.jit(nopython=True, cache=True)
def quat_multiply(quat_left: np.ndarray,
                  quat_right: np.ndarray,
                  out: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the composition of rotations as pair-wise product of two single
    or batches of quaternions [qx, qy, qz, qw], ie `quat_left * quat_right`.

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
    (qx_l, qy_l, qz_l, qw_l), (qx_r, qy_r, qz_r, qw_r) = quat_left, quat_right
    # qx_out, qy_out, qz_out, qw_out = out_
    out_[0] = qw_l * qx_r + qx_l * qw_r + qy_l * qz_r - qz_l * qy_r
    out_[1] = qw_l * qy_r - qx_l * qz_r + qy_l * qw_r + qz_l * qx_r
    out_[2] = qw_l * qz_r + qx_l * qy_r - qy_l * qx_r + qz_l * qw_r
    out_[3] = qw_l * qw_r - qx_l * qx_r - qy_l * qy_r - qz_l * qz_r
    return out_


@nb.jit(nopython=True, cache=True)
def compute_tilt_from_quat(q: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute e_z in R(q) frame (Euler-Rodrigues Formula): R(q).T @ e_z.

    :param q: Array whose rows are the 4 components of quaternions (x, y, z, w)
              and columns are N independent orientations.

    :returns: Tuple of arrays corresponding to the 3 individual components
              (a_x, a_y, a_z) of N independent tilt axes.
    """
    q_x, q_y, q_z, q_w = q
    v_x = 2 * (q_x * q_z - q_y * q_w)
    v_y = 2 * (q_y * q_z + q_w * q_x)
    v_z = 1 - 2 * (q_x * q_x + q_y * q_y)
    return (v_x, v_y, v_z)


# FIXME: Enabling cache causes segfault on Apple Silicon
@no_type_check
@nb.jit(nopython=True, cache=False)
def swing_from_vector(
        v_a: Tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar],
        q: np.ndarray) -> None:
    """Compute the "smallest" rotation transforming vector 'v_a' in 'e_z'.

    :param v_a: Tuple of arrays corresponding to the 3 individual components
                (a_x, a_y, a_z) of N independent tilt axes.
    :param q: Array where the result will be stored. The rows are the 4
              components of quaternions (x, y, z, w) and columns are the N
              independent orientations.
    """
    # Extract individual tilt components
    v_x, v_y, v_z = v_a

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
            for i, q_i in enumerate(q.T):
                swing_from_vector((v_x[i], v_y[i], v_z[i]), q_i)
        else:
            _, _, v_h = np.linalg.svd(np.array((
                (v_x, v_y, v_z),
                (0.0, 0.0, 1.0))
            ), full_matrices=True)
            w_2 = (1 + max(v_z, -1)) / 2
            q[:3], q[3] = v_h[-1] * np.sqrt(1 - w_2), np.sqrt(w_2)
    else:
        s = np.sqrt(2 * (1 + v_z))
        q[0], q[1], q[2], q[3] = v_y / s, - v_x / s, 0.0, s / 2

    # First order quaternion normalization to prevent compounding of errors.
    # If not done, shit may happen with removing twist again and again on the
    # same quaternion, which is typically the case when the IMU is steady, so
    # that the mahony filter update is actually skipped internally.
    q *= (3.0 - np.sum(np.square(q), 0)) / 2


# FIXME: Enabling cache causes segfault on Apple Silicon
@nb.jit(nopython=True, cache=False)
def remove_twist_from_quat(q: np.ndarray) -> None:
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
    v_a = compute_tilt_from_quat(q)

    # Compute the "smallest" rotation transforming vector 'v_a' in 'e_z'
    swing_from_vector(v_a, q)


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
    # batched matrix multiplication for now. See official issue for details:
    # https://github.com/numba/numba/issues/3804
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
