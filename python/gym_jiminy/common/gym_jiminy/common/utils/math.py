"""Mathematic utilities heavily optimized for speed.

They combine batch-processing with Just-In-Time (JIT) compiling via Numba when
possible for optimal performance. Most of them are dealing with rotations (SO3)
to perform transformations or convert from one representation to another.
"""
from typing import Union, Tuple, Optional, Literal, no_type_check, overload

import numpy as np
import numba as nb

from .spaces import ArrayOrScalar


TWIST_SWING_SINGULAR_THR = 1e-5


@overload
def matrix_to_yaw(mat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def matrix_to_yaw(mat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def matrix_to_yaw(mat: np.ndarray,
                  out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
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
    out1d = np.atleast_1d(out_)

    out1d[:] = np.arctan2(mat[1, 0], mat[0, 0])

    if out is None:
        return out_
    return None


@nb.jit(nopython=True, cache=True, inline='always')
def quat_to_yaw_cos_sin(quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cosine and sine of the yaw from Yaw-Pitch-Roll Euler angles
    representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    """
    assert quat.ndim >= 1
    (q_xy, q_yy), (q_zz, q_zw) = quat[-3] * quat[-4:-2], quat[-2] * quat[-2:]
    cos_yaw, sin_yaw = 1.0 - 2 * (q_yy + q_zz), 2 * (q_xy + q_zw)
    return cos_yaw, sin_yaw


@overload
def quat_to_yaw(quat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def quat_to_yaw(quat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def quat_to_yaw(quat: np.ndarray,
                out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the yaw from Yaw-Pitch-Roll Euler angles representation of a
    single or a batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert quat.ndim >= 1

    # Allocate memory for the output array
    if out is None:
        out_ = np.empty(quat.shape[1:])
    else:
        assert out.shape == quat.shape[1:]
        out_ = out
    out1d = np.atleast_1d(out_)

    cos_yaw, sin_yaw = quat_to_yaw_cos_sin(quat)
    out1d[:] = np.arctan2(sin_yaw, cos_yaw)

    if out is None:
        return out_
    return None


@overload
def quat_to_rpy(quat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def quat_to_rpy(quat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def quat_to_rpy(quat: np.ndarray,
                out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of quaternions.

    The Roll, Pitch and Yaw angles are guaranteed to be within range [-pi,pi],
    [-pi/2,pi/2], [-pi,pi], respectively.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    # Allocate memory for the output array
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((3, *quat.shape[1:]))
    else:
        assert out.shape == (3, *quat.shape[1:])
        out_ = out

    # Compute some intermediary quantities
    q_xx, q_xy, q_xz, q_xw = quat[-4] * quat[-4:]
    q_yy, q_yz, q_yw = quat[-3] * quat[-3:]
    q_zz, q_zw = quat[-2] * quat[-2:]
    q_ww = quat[-1] * quat[-1]

    # First-order normalization (by copy) to avoid numerical instabilities
    norm_2_inv = ((3.0 - (q_xx + q_yy + q_zz + q_ww)) / 2)
    q_yw *= norm_2_inv
    q_xz *= norm_2_inv

    # Compute Roll, Pitch and Yaw separately
    # roll, pitch, yaw = out_
    out_[0] = np.arctan2(2 * (q_xw + q_yz), 1.0 - 2 * (q_xx + q_yy))
    out_[1] = - np.pi / 2 + 2 * np.arctan2(
        np.sqrt(1.0 + 2 * (q_yw - q_xz)), np.sqrt(1.0 - 2 * (q_yw - q_xz)))
    out_[2] = np.arctan2(2 * (q_zw + q_xy), 1.0 - 2 * (q_yy + q_zz))

    if out is None:
        return out_
    return None


@overload
def quat_to_matrix(quat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def quat_to_matrix(quat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def quat_to_matrix(quat: np.ndarray,
                   out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the Rotation Matrix representation of a single or a
    batch of quaternions.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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

    if out is None:
        return out_
    return None


@overload
def matrix_to_quat(mat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def matrix_to_quat(mat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def matrix_to_quat(mat: np.ndarray,
                   out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the (qx, qy, qz, qw) Quaternion representation of a single or a
    batch of rotation matrices.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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

    if out is None:
        return out_
    return None


@overload
def matrices_to_quat(mat_list: Tuple[np.ndarray, ...],
                     out: np.ndarray) -> None:
    ...


@overload
def matrices_to_quat(mat_list: Tuple[np.ndarray, ...],
                     out: Literal[None] = ...) -> np.ndarray:
    ...


# TODO: Merge this method with `matrix_to_quat` by leverage compile-time
# implementation dispatching via `nb.generated_jit` or `nb.overload`.
@nb.jit(nopython=True, cache=True)
def matrices_to_quat(mat_list: Tuple[np.ndarray, ...],
                     out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the (qx, qy, qz, qw) Quaternion representation of multiple
    rotation matrices.

    .. seealso::
        See https://math.stackexchange.com/a/3183435.

    :param mat: Tuple of N arrays corresponding to independent 3D rotation
                matrices.
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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

    if out is None:
        return out_
    return None


@overload
def transforms_to_xyzquat(
        transform_list: Tuple[Tuple[np.ndarray, np.ndarray], ...],
        out: np.ndarray) -> None:
    ...


@overload
def transforms_to_xyzquat(
        transform_list: Tuple[Tuple[np.ndarray, np.ndarray], ...],
        out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def transforms_to_xyzquat(
        transform_list: Tuple[Tuple[np.ndarray, np.ndarray], ...],
        out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Stack the translation vector (x, y, z) and the quaternion representation
    (qx, qy, qz, qw) of the orientation of multiple transform tuples.

    .. note::
        Internally, it copies the translation unaffected and convert rotation
        matrices to quaternions using `matrices_to_quat`.

    :param transform_list: Tuple of N transforms, each of which represented as
                           pairs gathering the translation as a vector and the
                           orientation as a 3D rotation matrix.
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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
    matrices_to_quat(rotation_list, out_[-4:])  # type: ignore[call-overload]

    # Ravel extra dimension before returning if not present initially
    if out is not None:
        return out_[:, 0] if out.ndim == 1 else out_
    return None


@overload
def rpy_to_matrix(rpy: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def rpy_to_matrix(rpy: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def rpy_to_matrix(rpy: np.ndarray,
                  out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the Rotation Matrix representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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

    if out is None:
        return out_
    return None


@overload
def matrix_to_rpy(mat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def matrix_to_rpy(mat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def matrix_to_rpy(mat: np.ndarray,
                  out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a
    batch of rotation matrices.

    :param mat: N-dimensional array whose first and second dimensions gathers
                the 3-by-3 rotation matrix elements.
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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

    if out is None:
        return out_
    return None


@overload
def rpy_to_quat(rpy: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def rpy_to_quat(rpy: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def rpy_to_quat(rpy: np.ndarray,
                out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the (qx, qy, qz, qw) Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
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

    if out is None:
        return out_
    return None


@overload
def quat_multiply(quat_left: np.ndarray,
                  quat_right: np.ndarray,
                  out: np.ndarray,
                  is_left_conjugate: bool = False,
                  is_right_conjugate: bool = False) -> np.ndarray:
    ...


@overload
def quat_multiply(quat_left: np.ndarray,
                  quat_right: np.ndarray,
                  out: Literal[None] = ...,
                  is_left_conjugate: bool = False,
                  is_right_conjugate: bool = False) -> None:
    ...


@nb.jit(nopython=True, cache=True)
def quat_multiply(quat_left: np.ndarray,
                  quat_right: np.ndarray,
                  out: Optional[np.ndarray] = None,
                  is_left_conjugate: bool = False,
                  is_right_conjugate: bool = False) -> Optional[np.ndarray]:
    """Compute the composition of rotations as pair-wise product of two single
    or batches of quaternions (qx, qy, qz, qw), ie `quat_left * quat_right`.

    .. warning::
        Beware the argument order is important because the composition of
        rotations is not commutative.

    .. seealso::
        See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for
        mathematical details.

    :param quat_left: Left-hand side of the quaternion product, as a
                      N-dimensional array whose first dimension gathers the 4
                      quaternion coordinates (qx, qy, qz, qw).
    :param quat_right: Right-hand side of the quaternion product, as a
                       N-dimensional array whose first dimension gathers the 4
                       quaternion coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    :param is_left_conjugate: Whether to conjugate the left-hand side
                              quaternion before computing the product.
                              Optional: False by default.
    :param is_right_conjugate: Whether to conjugate the right-hand side
                               quaternion before computing the product.
                               Optional: False by default.
    """
    assert quat_left.ndim >= 1
    out_shape = np.broadcast_shapes(quat_left.shape, quat_right.shape)
    if out is None:
        out_ = np.empty(out_shape)
    else:
        assert out.shape == out_shape
        out_ = out

    (qx_l, qy_l, qz_l, qw_l), (qx_r, qy_r, qz_r, qw_r) = quat_left, quat_right
    s_l = -1 if is_left_conjugate else 1
    s_r = -1 if is_right_conjugate else 1

    # Note that we assign all components at once to allow multiply in-place
    # qx_out, qy_out, qz_out, qw_out = out_
    out_[0], out_[1], out_[2], out_[3] = (
        s_l * qw_l * qx_r + qx_l * s_r * qw_r + qy_l * qz_r - qz_l * qy_r,
        s_l * qw_l * qy_r - qx_l * qz_r + qy_l * s_r * qw_r + qz_l * qx_r,
        s_l * qw_l * qz_r + qx_l * qy_r - qy_l * qx_r + qz_l * s_r * qw_r,
        s_l * qw_l * s_r * qw_r - qx_l * qx_r - qy_l * qy_r - qz_l * qz_r,
    )

    if out is None:
        return out_
    return None


@overload
def quat_apply(quat: np.ndarray,
               vec: np.ndarray,
               out: np.ndarray,
               is_conjugate: bool = False) -> np.ndarray:
    ...


@overload
def quat_apply(quat: np.ndarray,
               vec: np.ndarray,
               out: Literal[None] = ...,
               is_conjugate: bool = False) -> None:
    ...


@nb.jit(nopython=True, cache=True)
def quat_apply(quat: np.ndarray,
               vec: np.ndarray,
               out: Optional[np.ndarray] = None,
               is_conjugate: bool = False) -> Optional[np.ndarray]:
    """Apply rotations to position vectors as pair-wise transform of a single
    or batch of position vectors (x, y, z) by a single or batch of quaternions
    (qx, qy, qz, qw), ie `quat * (vec, 0) * quat.conjugate()`.

    .. seealso::
        See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for
        mathematical details.

    .. warning::
        Applying rotation to position vectors using quaternions is much slower
        than using rotation matrices. In case where the same rotation must be
        applied to a batch of position vectors, it is faster to first convert
        the quaternion to a rotation matrix and use batched matrix product.
        However, if a different rotation must be applied to each position
        vector, then it is faster to apply batched quaternion transformation
        directly, because the cost of converting all quaternions to rotation
        matrices exceeds its benefit overall. This holds true even for a single
        pair quaternion-position.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    :param vec: N-dimensional array whose first dimension gathers the 3
                position coordinates (x, y, z).
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    :param is_conjugate: Whether to conjugate the quaternion before applying
                         the rotation.
                         Optional: False by default.
    """
    assert quat.ndim >= 1 and vec.ndim >= 1
    if out is None:
        out_ = np.empty(vec.shape)
    else:
        assert out.shape == vec.shape
        out_ = out

    q_xx, q_xy, q_xz, q_xw = quat[-4] * quat[-4:]
    q_yy, q_yz, q_yw = quat[-3] * quat[-3:]
    q_zz, q_zw = quat[-2] * quat[-2:]
    q_ww = quat[-1] * quat[-1]
    x, y, z = vec
    s = -1 if is_conjugate else 1

    # Note that we assign all components at once to allow rotation in-place
    # px, py, pz = out_
    (out_[0], out_[1], out_[2]) = (
        x * (q_xx + q_ww - q_yy - q_zz) +
        y * (2 * q_xy - 2 * s * q_zw) +
        z * (2 * q_xz + 2 * s * q_yw),
        x * (2 * s * q_zw + 2 * q_xy) +
        y * (q_ww - q_xx + q_yy - q_zz) +
        z * (- 2 * s * q_xw + 2 * q_yz),
        x * (- 2 * s * q_yw + 2 * q_xz) +
        y * (2 * s * q_xw + 2 * q_yz) +
        z * (q_ww - q_xx - q_yy + q_zz))

    if out is None:
        return out_
    return None


@overload
def log3(quat: np.ndarray, out: np.ndarray, theta: np.ndarray) -> None:
    ...


@overload
def log3(quat: np.ndarray,
         out: Literal[None] = ...,
         theta: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def log3(quat: np.ndarray,
         out: Optional[np.ndarray] = None,
         theta: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the angle-axis representation theta * (ax, ay, az) of a single
    or a batch of quaternions (qx, qy, qz, qz).

    As a reminder, any element of the Lie Group of rotation group SO(3) can be
    mapped to an element of its Lie Algebra so(3) ⊂ R3 at identity, which
    identifies to its tangent space, through the pseudo-inverse of the
    exponential map. See `pinocchio.log3` documentation for technical details.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    :param theta: Pre-allocated array into which to store the angle. This is
                  useful to avoid redundant computations in some cases.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((3, *quat.shape[1:]))
    else:
        assert out.shape == (3, *quat.shape[1:])
        out_ = out
    if theta is None:
        theta_ = np.empty(quat.shape[1:])
    else:
        assert theta.shape == quat.shape[1:]
        theta_ = theta
    theta1d = np.atleast_1d(theta_)

    # Split real (qx, qy, qz) and imaginary (qw,) quaternion parts
    q_vec, q_w = quat[:3], quat[3]

    # Compute the angle-axis representation of the relative rotation.
    # Note that one must deal with undefined behavior asymptotically.
    # FIXME: Ideally, a taylor expansion should be used to handle theta ~ 0,
    # but it is tricky to implement without having to compute both branches
    # systematically. In practice, float64 computations are precise enough not
    # to have to worry too much about it.
    eps = np.finfo(np.float64).tiny
    theta_sin_2 = np.sqrt(np.sum(np.square(q_vec), 0))
    theta1d[:] = 2 * np.arctan2(theta_sin_2, np.abs(q_w))
    inv_sinc = theta_ / np.maximum(theta_sin_2, eps)
    out_[:] = inv_sinc * q_vec * np.sign(q_w)

    if out is None:
        return out_
    return None


quat_to_angle_axis = log3


@overload
def exp3(angle_axis: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def exp3(angle_axis: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def exp3(angle_axis: np.ndarray,
         out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the quaternion representation (qx, qy, qz, qz) of a single
    or a batch of angle-axis vectors theta * (ax, ay, az).

    As a reminder, it also corresponds to the inverse exponential map from the
    rotation Lie Group SO3 to its Lie Algebra so3.

    :param angle_axis: N-dimensional array whose first dimension gathers the 3
                       angle-axis components theta * (ax, ay, az).
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert angle_axis.ndim >= 1
    if out is None:
        out_ = np.empty((4, *angle_axis.shape[1:]))
    else:
        assert out.shape == (4, *angle_axis.shape[1:])
        out_ = out

    # Compute unit axis and positive angle separately
    # Note that one must deal with undefined behavior asymptotically.
    # FIXME: Taylor expansion should be used to handle theta ~ 0.
    eps = np.finfo(np.float64).tiny
    theta = np.sqrt(np.sum(np.square(angle_axis), 0))
    axis = angle_axis / np.maximum(theta, eps)

    # Compute the quaternion representation
    out_[:3] = np.sin(0.5 * theta) * axis
    out_[3] = np.cos(0.5 * theta)

    if out is None:
        return out_
    return None


angle_axis_to_quat = exp3


@overload
def log6(xyzquat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def log6(xyzquat: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def log6(xyzquat: np.ndarray,
         out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Apply SE3 to se3 inverse exponential map on a single or a batch of
    transform vectors (x, y, z, qx, qy, qz, qw) defining the pose (position
    plus orientation) of a frame in 3D space.

    As a reminder, the resulting vector is homogeneous to a spatial velocity
    vector, aka. a motion vector.

    :param xyzquat: N-dimensional array whose first dimension gathers the 7
                    position and quaternion coordinates (x, y, z),
                    (qx, qy, qz, qw) respectively.
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert xyzquat.ndim >= 1
    if out is None:
        out_ = np.empty((6, *xyzquat.shape[1:]))
    else:
        assert out.shape == (6, *xyzquat.shape[1:])
        out_ = out

    # Split linear and angular parts for input and output representations
    v_lin, v_ang = out_[:3], out_[3:]
    pos, quat = xyzquat[:3], xyzquat[3:]
    qvec, qw = quat[:-1], quat[-1]

    # Compute the angular part
    theta = np.empty(xyzquat.shape[1:])
    log3(quat, v_ang, theta)

    # Compute the linear part.
    # FIXME: Taylor expansion should be used to handle theta ~ 0.
    eps = np.finfo(np.float64).tiny ** (1 / 2)
    theta_cos_2 = np.abs(qw)
    theta_sin_2 = np.maximum(np.sqrt(np.sum(np.square(qvec), 0)), eps)
    theta_cot_2 = theta_cos_2 / theta_sin_2
    np.maximum(theta, eps, theta)
    beta = 1.0 / np.square(theta) - 0.5 * theta_cot_2 / theta
    wxv_x = v_ang[1] * pos[2] - v_ang[2] * pos[1]
    wxv_y = v_ang[2] * pos[0] - v_ang[0] * pos[2]
    wxv_z = v_ang[0] * pos[1] - v_ang[1] * pos[0]
    w2xv_x = v_ang[1] * wxv_z - v_ang[2] * wxv_y
    w2xv_y = v_ang[2] * wxv_x - v_ang[0] * wxv_z
    w2xv_z = v_ang[0] * wxv_y - v_ang[1] * wxv_x
    v_lin[0] = pos[0] - 0.5 * wxv_x + beta * w2xv_x
    v_lin[1] = pos[1] - 0.5 * wxv_y + beta * w2xv_y
    v_lin[2] = pos[2] - 0.5 * wxv_z + beta * w2xv_z

    if out is None:
        return out_
    return None


@overload
def exp6(v_spatial: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def exp6(v_spatial: np.ndarray, out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def exp6(v_spatial: np.ndarray,
         out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Apply se3 to SE3 exponential map on a single or a batch of spatial
    velocity vectors (vx, vy, vz, wx, wy, wz), also called motion vectors,
    using quaternions (qx, qy, qz, qw) to represent the rotation.

    :param v_spatial: N-dimensional array whose first dimension gathers the 6
                      linear and angular velocity components (vx, vy, vz),
                      (wx, wy, wz) respectively.
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert v_spatial.ndim >= 1
    if out is None:
        out_ = np.empty((7, *v_spatial.shape[1:]))
    else:
        assert out.shape == (7, *v_spatial.shape[1:])
        out_ = out

    # Split linear and angular velocity for convenience
    v_lin, v_ang = v_spatial[:3], v_spatial[3:]

    # Compute the linear part.
    # FIXME: Taylor expansion should be used to handle theta ~ 0.
    eps = np.finfo(np.float64).tiny ** (2 / 3)
    theta_sq = np.maximum(np.sum(np.square(v_ang), 0), eps)
    theta = np.sqrt(theta_sq)
    theta_cos, theta_sin = np.cos(theta), np.sin(theta)
    alpha_wxv = (1.0 - theta_cos) / theta_sq
    alpha_w2 = (theta - theta_sin) / theta_sq / theta
    wxv_x = v_ang[1] * v_lin[2] - v_ang[2] * v_lin[1]
    wxv_y = v_ang[2] * v_lin[0] - v_ang[0] * v_lin[2]
    wxv_z = v_ang[0] * v_lin[1] - v_ang[1] * v_lin[0]
    w2xv_x = v_ang[1] * wxv_z - v_ang[2] * wxv_y
    w2xv_y = v_ang[2] * wxv_x - v_ang[0] * wxv_z
    w2xv_z = v_ang[0] * wxv_y - v_ang[1] * wxv_x
    out_[0] = v_lin[0] + alpha_wxv * wxv_x + alpha_w2 * w2xv_x
    out_[1] = v_lin[1] + alpha_wxv * wxv_y + alpha_w2 * w2xv_y
    out_[2] = v_lin[2] + alpha_wxv * wxv_z + alpha_w2 * w2xv_z

    # Compute the angular part
    exp3(v_ang, out_[-4:])

    if out is None:
        return out_
    return None


@overload
def quat_difference(quat_left: np.ndarray,
                    quat_right: np.ndarray,
                    out: np.ndarray) -> None:
    ...


@overload
def quat_difference(quat_left: np.ndarray,
                    quat_right: np.ndarray,
                    out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def quat_difference(quat_left: np.ndarray,
                    quat_right: np.ndarray,
                    out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Compute the pair-wise SO3 difference between two batches of quaternions
    (qx, qy, qz, qz). For each pairs, it returns a angular velocity vector
    (wx, wy, wz) in tangent space of SO3 Lie Group.

    First, it computes the residual rotation for all pairs, ie
    `quat_diff = quat_left.conjugate() * quat_right`. Then, it computes the
    angle-axis representation of the residual rotations, ie `log3(quat_diff)`.
    See `pinocchio.liegroups.SO3.difference` documentation for reference.

    .. note::
        Calling this method is faster than `pinocchio.liegroups.SO3.difference`
        if at least 2 pairs of quaternions with pre-allocated output. This is
        not surprising since vectorization does not have any effect in this
        case. This expected speed up is about x5 and x15 for 10 and 100 pairs
        respectively with pre-allocated output.

    :param quat_left: Left-hand side of SO3 difference, as a N-dimensional
                      array whose first dimension gathers the 4 quaternion
                      coordinates (qx, qy, qz, qw).
    :param quat_right: Right-hand side of SO3 difference, as a N-dimensional
                       array whose first dimension gathers the 4 quaternion
                       coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    # Compute the quaternion representation of the residual rotation
    quat_diff = quat_multiply(quat_left, quat_right, is_left_conjugate=True)

    # Compute the angle-axis representation of the residual rotation
    return log3(quat_diff, out)  # type: ignore[call-overload]


@nb.jit(nopython=True, cache=True)
def xyzquat_difference(xyzquat_left: np.ndarray,
                       xyzquat_right: np.ndarray,
                       out: Optional[np.ndarray] = None
                       ) -> Optional[np.ndarray]:
    """Compute the pair-wise SE3 difference between two batches of transform
    vectors (x, y, z, qx, qy, qz, qz). For each pairs, it returns a spatial
    velocity vector (vx, vy, vz, wx, wy, wz), also called motion vector, in
    tangent space of SE3 Lie Group.

    First, it computes the residual transform in local frame for all pairs.
    Then, it applies the inverse exponential map `log6` of it. See
    `pinocchio.liegroups.SE3.difference` documentation for reference.

    :param xyzquat_left: Left-hand side of SE3 difference, as a N-dimensional
                         array whose first dimension gathers the 7 position and
                         quaternion coordinates (x, y, z), (qx, qy, qz, qw).
    :param xyzquat_right: Right-hand side of SO3 difference, as a N-dimensional
                         array whose first dimension gathers the 7 position and
                         quaternion coordinates (x, y, z), (qx, qy, qz, qw).
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    # Compute the xyzquat representation of the residual pose
    xyzquat_diff = np.empty(xyzquat_left.shape)
    xyz_diff, quat_diff = xyzquat_diff[:3], xyzquat_diff[-4:]
    xyz_left, quat_left = xyzquat_left[:3], xyzquat_left[-4:]
    xyz_right, quat_right = xyzquat_right[:3], xyzquat_right[-4:]
    xyz_diff[:] = xyz_right - xyz_left
    quat_apply(quat_left, xyz_diff, xyz_diff, is_conjugate=True)
    quat_multiply(quat_left, quat_right, quat_diff, is_left_conjugate=True)

    # Apply inverse exponential map to cast the residual pose in tangent space
    return log6(xyzquat_diff, out)


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
    # pylint: disable=possibly-used-before-assignment

    # Extract individual tilt components
    v_x, v_y, v_z = v_a

    # There is a singularity when the rotation axis of orientation estimate
    # and z-axis are nearly opposites, i.e. v_z ~= -1. One solution that
    # ensure continuity of q_w is picked arbitrarily using SVD decomposition.
    # See `Eigen::Quaternion::FromTwoVectors` implementation for details.
    if q.ndim > 1:
        is_singular = np.any(v_z < -1.0 + TWIST_SWING_SINGULAR_THR)
    else:
        is_singular = v_z < -1.0 + TWIST_SWING_SINGULAR_THR
    if is_singular:
        if q.ndim > 1:
            for i, q_i in enumerate(q.T):
                swing_from_vector((v_x[i], v_y[i], v_z[i]), q_i)
        else:
            eps_thr = np.sqrt(TWIST_SWING_SINGULAR_THR)
            eps_x = -TWIST_SWING_SINGULAR_THR < v_x < TWIST_SWING_SINGULAR_THR
            eps_y = -TWIST_SWING_SINGULAR_THR < v_y < TWIST_SWING_SINGULAR_THR
            if eps_x and not eps_y:
                ratio = v_x / v_y
                esp_ratio = - eps_thr < ratio < eps_thr
            elif eps_y and not eps_x:
                ratio = v_y / v_x
                esp_ratio = - eps_thr < ratio < eps_thr
            w_2 = (1.0 + max(v_z, -1.0)) / 2.0
            if eps_x and eps_y:
                # Both q_x and q_y would do fine. Picking q_y arbitrarily.
                q[0] = 0.0
                q[1] = np.sqrt(1.0 - w_2)
            elif esp_ratio and eps_x:
                q[0] = - np.sqrt(1.0 - w_2) * (1 - 0.5 * ratio ** 2)
                q[1] = + np.sqrt(1.0 - w_2) * (ratio - 0.5 * ratio ** 3)
            elif esp_ratio and eps_y:
                q[0] = - np.sqrt(1.0 - w_2) * (ratio - 0.5 * ratio ** 3)
                q[1] = + np.sqrt(1.0 - w_2) * (1 - 0.5 * ratio ** 2)
            else:
                q[0] = - np.sqrt((1.0 - w_2) / (1 + (v_x / v_y) ** 2))
                q[1] = + np.sqrt((1.0 - w_2) / (1 + (v_y / v_x) ** 2))
            q[2] = 0.0
            q[3] = np.sqrt(w_2)
            # _, _, v_h = np.linalg.svd(np.array((
            #     (v_x, v_y, v_z),
            #     (0.0, 0.0, 1.0))
            # ), full_matrices=True)
            # q[:3], q[3] = v_h[-1] * np.sqrt(1.0 - w_2), np.sqrt(w_2)
    else:
        s = np.sqrt(2.0 * (1.0 + v_z))
        q[0], q[1], q[2], q[3] = v_y / s, - v_x / s, 0.0, s / 2

    # First order quaternion normalization to prevent compounding of errors.
    # If not done, shit may happen with removing twist again and again on the
    # same quaternion, which is typically the case when the IMU is steady, so
    # that the mahony filter update is actually skipped internally.
    q *= (3.0 - np.sum(np.square(q), 0)) / 2


@overload
def remove_yaw_from_quat(quat: np.ndarray, out: np.ndarray) -> None:
    ...


@overload
def remove_yaw_from_quat(quat: np.ndarray,
                         out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True)
def remove_yaw_from_quat(quat: np.ndarray,
                         out: Optional[np.ndarray] = None
                         ) -> Optional[np.ndarray]:
    """Remove the rotation around z-axis of a single or batch of quaternions.

    .. note::
        Note that this decomposition is rarely used in practice, mainly because
        of singularity issues related to the Roll-Pitch-Yaw decomposition. It
        is usually preferable to remove the twist part of the Twist-after-Swing
        decomposition. Note that in both cases, the Roll and Pitch angles from
        their corresponding Yaw-Pitch-Roll Euler angles representation matches
        exactly. See `remove_twist_from_quat` documentation for details.

    :param quat: N-dimensional array whose first dimension gathers the 4
                 quaternion coordinates (qx, qy, qz, qw).
    :param out: Pre-allocated array into which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    # Allocate memory for the output array
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty(quat.shape)
    else:
        assert out.shape == quat.shape
        out_ = out

    # Compute some intermediary quantities
    q_xx, (q_xz, q_xw) = quat[-4] * quat[-4], quat[-4] * quat[-2:]
    q_yy, q_yz, q_yw = quat[-3] * quat[-3:]

    # Compute some intermediary quantities
    cos_roll = 1.0 - 2 * (q_xx + q_yy)
    sin_roll = 2 * (q_xw + q_yz)
    cos_roll /= np.sqrt(cos_roll ** 2 + sin_roll ** 2)
    cos_roll_2 = np.sqrt(0.5 * (1.0 + cos_roll))
    sin_roll_2 = np.sign(sin_roll) * np.sqrt(0.5 * (1.0 - cos_roll))
    sin_pitch = 2 * (q_yw - q_xz)
    cos_pitch = np.sqrt(1.0 - sin_pitch ** 2)
    cos_pitch_2 = np.sqrt(0.5 * (1.0 + cos_pitch))
    sin_pitch_2 = np.sign(sin_pitch) * np.sqrt(0.5 * (1.0 - cos_pitch))

    # q_x, q_y, q_z, q_w = out_
    out_[0] = + sin_roll_2 * cos_pitch_2
    out_[1] = + cos_roll_2 * sin_pitch_2
    out_[2] = - sin_roll_2 * sin_pitch_2
    out_[3] = + cos_roll_2 * cos_pitch_2

    if out is None:
        return out_
    return None


# FIXME: Enabling cache causes segfault on Apple Silicon
@nb.jit(nopython=True, cache=False)
def remove_twist_from_quat(quat: np.ndarray,
                           out: Optional[np.ndarray] = None) -> None:
    """Remove the twist part of the Twist-after-Swing decomposition of given
    orientations in quaternion representation.

    Any rotation R can be decomposed as:

        R = R_z * R_s

    where R_z (the twist) is a rotation around e_z and R_s (the swing) is
    the "smallest" rotation matrix (in terms of angle of its corresponding
    Axis-Angle representation) such that s(R_s) = s(R). Note that although the
    swing is not free of rotation around z-axis, the latter only depends on the
    rotation around e_x, e_y, which is the main motivation for using this
    decomposition. One must use `remove_yaw_from_quat` to completely cancel you
    the rotation around z-axis.

    .. seealso::
        * See "Estimation and control of the deformations of an exoskeleton
          using inertial sensors", PhD Thesis, M. Vigne, 2021, p. 130.
        * See "Swing-twist decomposition in Clifford algebra", P. Dobrowolski,
          2015 (https://arxiv.org/abs/1506.05481)

    :param q: Array whose rows are the 4 components of quaternions (x, y, z, w)
              and columns are N independent orientations from which to remove
              the swing part.
    :param out: Pre-allocated array into which to store the result. `None` to
                update the input quaternion in-place.
                Optional: `None` by default.
    """
    # Update in-place in no out has been specified
    if out is None:
        out_ = quat
    else:
        assert out.shape == quat.shape
        out_ = out

    # Compute e_z in R(q) frame (Euler-Rodrigues Formula): R(q).T @ e_z
    v_a = compute_tilt_from_quat(quat)

    # Compute the "smallest" rotation transforming vector 'v_a' in 'e_z'
    swing_from_vector(v_a, out_)


def quat_average(quat: np.ndarray,
                 axes: Optional[Union[Tuple[int, ...], int]] = None
                 ) -> np.ndarray:
    """Compute the average of a batch of quaternions (qx, qy, qz, qw) over some
    or all axes.

    Here, the average is defined as a quaternion minimizing the mean error
    wrt every individual quaternion. The distance metric used as error is the
    dot product of quaternions `p.dot(q)`, which is directly related to the
    angle between them `cos(angle(p.conjugate() * q) / 2)`. This metric as the
    major advantage to yield a quadratic problem, which can be solved very
    efficiently, unlike the squared angle `angle(p.conjugate() * q) ** 2`.

    :param quat: N-dimensional (N >= 2) array whose first dimension gathers the
                 4 quaternion coordinates (qx, qy, qz, qw).
    :param axes: Batch dimensions to preserve without computing the average.
    """
    # TODO: This function cannot be jitted because numba does not support
    # batched matrix multiplication for now. See official issue for details:
    # https://github.com/numba/numba/issues/3804
    assert quat.ndim >= 2
    if axes is None:
        axes = tuple(range(1, quat.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    assert len(axes) > 0 and 0 not in axes

    q_perm = quat.transpose((
        *[i for i in range(1, quat.ndim) if i not in axes], 0, *axes))
    q_flat = q_perm.reshape((*q_perm.shape[:-len(axes)], -1))
    _, eigvec = np.linalg.eigh(q_flat @ np.swapaxes(q_flat, -1, -2))
    return np.moveaxis(eigvec[..., -1], -1, 0)


@overload
def quat_interpolate_middle(quat1: np.ndarray,
                            quat2: np.ndarray,
                            out: np.ndarray) -> None:
    ...


@overload
def quat_interpolate_middle(quat1: np.ndarray,
                            quat2: np.ndarray,
                            out: Literal[None] = ...) -> np.ndarray:
    ...


@nb.jit(nopython=True, cache=True, fastmath=True)
def quat_interpolate_middle(quat1: np.ndarray,
                            quat2: np.ndarray,
                            out: Optional[np.ndarray] = None
                            ) -> Optional[np.ndarray]:
    """Compute the midpoint interpolation between two batches of quaternions
    (qx, qy, qz, qw).

    The midpoint interpolation of two quaternion is defined as the integration
    of half the difference between them, starting from the first one, ie
    `q_mid = integrate(q1, 0.5 * difference(q1, d2))`, which is a special case
    of the `slerp` method (spherical linear interpolation) for `alpha=0.5`.

    For the midpoint in particular, one can show that the middle quaternion is
    simply normalized sum of the previous and next quaternions.

    :param quat1: First batch of quaternions as a N-dimensional array whose
                  first dimension gathers the 4 quaternion coordinates.
    :param quat2: Second batch of quaternions as a N-dimensional array.
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert quat1.ndim >= 1 and quat1.shape == quat2.shape
    if out is None:
        out_ = np.empty((4, *quat1.shape[1:]))
    else:
        assert out.shape == (4, *quat1.shape[1:])
        out_ = out

    dot = np.sum(quat1 * quat2, axis=0)
    dot_ = dot if quat1.ndim == 1 else np.expand_dims(dot, axis=0)
    out_[:] = (quat1 + np.sign(dot_) * quat2) / np.sqrt(2 * (1 + np.abs(dot_)))

    if out is None:
        return out_
    return None
