""" TODO: Write documentation.
"""
from typing import Optional, Union, Dict, Sequence

import numba as nb
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import UnivariateSpline


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


def smoothing_filter(
        time_in: np.ndarray,
        val_in: np.ndarray,
        time_out: Optional[np.ndarray] = None,
        relabel: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, Sequence[float]]]] = None
        ) -> np.ndarray:
    """Smoothing filter with relabeling and resampling features.
    It supports evenly sampled multidimensional input signal. Relabeling can be
    used to infer the value of samples at time steps before and after the
    explicitly provided samples.
    .. note::
        As a reminder, relabeling is a generalization of periodicity.
    :param time_in: Time steps of the input signal.
    :param val_in: Sampled values of the input signal.
                   (2D numpy array: row = sample, column = time)
    :param time_out: Time steps of the output signal.
    :param relabel: Relabeling matrix (identity for periodic signals).
                    Optional: Disable if omitted.
    :param params:
        .. raw:: html
            Parameters of the filter. Dictionary with keys:
        - **'mixing_ratio_1':** Relative time at the begining of the signal
          during the output signal corresponds to a linear mixing over time of
          the filtered and original signal (only used if relabel is omitted).
        - **'mixing_ratio_2':** Relative time at the end of the signal during
          the output signal corresponds to a linear mixing over time of the
          filtered and original signal (only used if relabel is omitted).
        - **'smoothness'[0]:** Smoothing factor to filter the begining of the
          signal (only used if relabel is omitted).
        - **'smoothness'[1]:** Smoothing factor to filter the end of the signal
          (only used if relabel is omitted).
        - **'smoothness'[2]:** Smoothing factor to filter the middle part of
          the signal.
    :returns: Filtered signal (2D numpy array: row = sample, column = time).
    """
    if time_out is None:
        time_out = time_in
    if params is None:
        params = {}
        params['mixing_ratio_1'] = 0.12
        params['mixing_ratio_2'] = 0.04
        params['smoothness'] = [0.0] * 3
        params['smoothness'][0] = 5e-3
        params['smoothness'][1] = 5e-3
        params['smoothness'][2] = 3e-3

    if relabel is None:
        def t_rescaled(t: float, start: float) -> float:
            return (t - start) / (time_in[-1] - time_in[0])

        mix_fit = [None, None, None]
        mix_fit[0] = lambda t: 0.5 * (np.sin(
            1 / params['mixing_ratio_1'] *
            t_rescaled(t, time_in[0]) * np.pi - np.pi/2) + 1)
        mix_fit[1] = lambda t: 0.5 * (np.sin(
            1 / params['mixing_ratio_2'] *
            t_rescaled(t, params['mixing_ratio_2']) * np.pi + np.pi/2) + 1)
        mix_fit[2] = lambda t: 1.0

        val_fit = []
        for v in val_in:  # Loop over the rows
            val_fit_j = []
            for smoothness in params['smoothness']:
                val_fit_j.append(UnivariateSpline(time_in, v, s=smoothness))
            val_fit.append(val_fit_j)

        time_out_mixing = [None] * 3
        time_out_mixing_ind = [None] * 3
        time_out_mixing_ind[0] = \
            time_out < time_out[-1] * params['mixing_ratio_1']
        time_out_mixing[0] = time_out[time_out_mixing_ind[0]]
        time_out_mixing_ind[1] = \
            time_out > time_out[-1] * (1 - params['mixing_ratio_2'])
        time_out_mixing[1] = time_out[time_out_mixing_ind[1]]
        time_out_mixing_ind[2] = \
            (not time_out_mixing_ind[0]) & (not time_out_mixing_ind[1])
        time_out_mixing[2] = time_out[time_out_mixing_ind[2]]

        val_out = []
        for val in val_fit:
            val_out_j = []
            for t, v, fit in zip(time_out_mixing, val, mix_fit):
                val_out_j.append((1 - fit(t)) * v(t) + fit(t) * val[-1](t))
            val_out.append(val_out_j)
        val_out = np.array(val_out)
    else:
        _time = np.concatenate((time_in[:-1] - time_in[-1],
                                time_in, time_in[1:] + time_in[-1]))
        _val_in = np.concatenate([relabel.dot(val_in[:, :-1]),
                                  val_in,
                                  relabel.dot(val_in[:, 1:])], axis=1)

        val_out = []
        for v in _val_in:
            val_out.append(UnivariateSpline(
                _time, v, s=params['smoothness'][-1])(time_out))
        val_out = np.concatenate(val_out, axis=0)

    return val_out
