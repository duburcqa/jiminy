from math import factorial
from typing import Optional, Dict, Union, Sequence

import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import UnivariateSpline


def integrate_zoh(state_prev: np.ndarray,
                  dt: float) -> np.ndarray:
    """N-order integration scheme assuming Zero-Order-Hold for highest
    derivative.

    :param state_prev: Previous state update, ordered from lowest to highest
                       derivative order, which means:
                       s[i](t) = s[i](t-1) + integ_{t-1}^{t}(s[i+1](t))
    :param dt: Integration delta of time since previous state update.
    """
    integ_coeffs = [pow(dt, k) / factorial(k) for k in range(len(state_prev))]
    integ_matrix = toeplitz(integ_coeffs, np.zeros(len(state_prev))).T
    return integ_matrix @ state_prev


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
