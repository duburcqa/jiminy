""" TODO: Write documentation.
"""
from typing import Optional

import numba as nb
import numpy as np
from numpy.lib.stride_tricks import as_strided


FACTORIAL_TABLE = (1, 1, 2, 6, 24, 120, 720)


@nb.jit(nopython=True, nogil=True)
def _toeplitz(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Numba-compatible implementation of `scipy.linalg.toeplitz` method.

    .. note:
        It does not handle any special case for efficiency, so the input types
        is more respective than originally.

    .. warning:
        It returns a strided matrix instead of contiguous copy for efficiency.

    :param col: First column of the matrix.
    :param row: First row of the matrix.

    see::
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """
    vals = np.concatenate((col[::-1], row[1:]))
    stride = vals.strides[0]  # pylint: disable=E1136
    return as_strided(vals[len(col)-1:],
                      shape=(len(col), len(row)),
                      strides=(-stride, stride))


@nb.jit(nopython=True, nogil=True)
def _integrate_zoh_impl(state_prev: np.ndarray,
                        dt: float,
                        state_min: np.ndarray,
                        state_max: np.ndarray) -> np.ndarray:
    """ TODO: Write documentation.
    """
    # Compute integration matrix
    order = len(state_prev)
    integ_coeffs = np.array([
        pow(dt, k) / FACTORIAL_TABLE[k] for k in range(order)])
    integ_matrix = _toeplitz(integ_coeffs, np.zeros(order)).T
    integ_zero = integ_matrix[:, :-1].copy() @ state_prev[:-1]
    integ_drift = np.expand_dims(integ_matrix[:, -1], axis=-1)

    # Propagate derivative bounds to compute highest derivative bounds
    deriv = state_prev[-1]
    deriv_min_stack = (state_min - integ_zero) / integ_drift
    deriv_max_stack = (state_max - integ_zero) / integ_drift
    deriv_min = np.full_like(deriv, fill_value=-np.inf)
    deriv_max = np.full_like(deriv, fill_value=np.inf)
    for deriv_min_i, deriv_max_i in zip(deriv_min_stack, deriv_max_stack):
        deriv_min_i_valid = np.logical_and(
            deriv_min < deriv_min_i, deriv_min_i < deriv_max)
        deriv_min[deriv_min_i_valid] = deriv_min_i[deriv_min_i_valid]
        deriv_max_i_valid = np.logical_and(
            deriv_min < deriv_max_i, deriv_max_i < deriv_max)
        deriv_max[deriv_max_i_valid] = deriv_max_i[deriv_max_i_valid]

    # Clip highest derivative to ensure every derivative are withing bounds
    # it possible, lowest orders in priority otherwise.
    deriv = np.minimum(np.maximum(deriv, deriv_min), deriv_max)

    # Integrate, taking into account clipped highest derivative
    return integ_zero + integ_drift * deriv


def integrate_zoh(state_prev: np.ndarray,
                  dt: float,
                  state_min: Optional[np.ndarray] = None,
                  state_max: Optional[np.ndarray] = None) -> np.ndarray:
    """N-order integration scheme assuming Zero-Order-Hold for highest
    derivative, taking state bounds into account to make sure integrated state
    is wthin bounds.

    :param state_prev: Previous state update, ordered from lowest to highest
                       derivative order, which means:
                       s[i](t) = s[i](t-1) + integ_{t-1}^{t}(s[i+1](t))
    :param state_min: Lower bounds of the state.
                      Optional: -Inf by default.
    :param state_max: Upper bounds of the state.
                      Optional: Inf by default.
    :param dt: Integration delta of time since previous state update.
    """
    # Make sure dt is not zero, otherwise return early
    if abs(dt) < 1e-9:
        return state_prev.copy()

    # Handling of default arguments
    if state_min is None:
        state_min = np.full_like(state_prev, fill_value=-np.inf)
    if state_max is None:
        state_max = np.full_like(state_prev, fill_value=-np.inf)

    # Call internal implementation
    return _integrate_zoh_impl(state_prev, dt, state_min, state_max)
