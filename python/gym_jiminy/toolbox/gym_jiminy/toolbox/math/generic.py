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
