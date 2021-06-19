from typing import Optional, Dict, Union, Sequence

import numba as nb
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import UnivariateSpline
from scipy.spatial.qhull import _Qhull


FACTORIAL_TABLE = (1, 1, 2, 6, 24, 120, 720)


class ConvexHull:
    def __init__(self, points: np.ndarray) -> None:
        """Compute the convex hull defined by a set of points.

        :param points: N-D points whose to computed the associated convex hull,
                       as a 2D array whose first dimension corresponds to the
                       number of points, and the second to the N-D coordinates.
        """
        assert len(points) > 0, "The length of 'points' must be at least 1."

        # Backup user argument(s)
        self._points = points

        # Create convex full if possible
        if len(self._points) > 2:
            self._hull = _Qhull(points=self._points,
                                options=b"",
                                mode_option=b"i",
                                required_options=b"Qt",
                                furthest_site=False,
                                incremental=False,
                                interior_point=None)
        else:
            self._hull = None

        # Buffer to cache center computation
        self._center = None

    @property
    def center(self) -> np.ndarray:
        """Get the center of the convex hull.

        .. note::
            Degenerated convex hulls corresponding to len(points) == 1 or 2 are
            handled separately.

        :returns: 1D float vector with N-D coordinates of the center.
        """
        if self._center is None:
            if len(self._points) > 3:
                vertices = self._points[self._hull.get_extremes_2d()]
            else:
                vertices = self._points
            self._center = np.mean(vertices, axis=0)
        return self._center

    def get_distance(self, queries: np.ndarray) -> np.ndarray:
        """Compute the signed distance of query points from the convex hull.

        Positive distance corresponds to a query point lying outside the convex
        hull.

        .. note::
            Degenerated convex hulls corresponding to len(points) == 1 or 2 are
            handled separately. The distance from a point and a segment is used
            respectevely.

        :param queries: N-D query points for which to compute distance from the
                        convex hull, as a 2D array.

        :returns: 1D float vector of the same length than `queries`.
        """
        if len(self._points) > 2:
            equations = self._hull.get_simplex_facet_array()[2].T
            return np.max(queries @ equations[:-1] + equations[-1], axis=1)
        elif len(self._points) == 2:
            vec = self._points[1] - self._points[0]
            ratio = (queries - self._points[0]) @ vec / squared_norm_2(vec)
            proj = self._points[0] + np.outer(np.clip(ratio, 0.0, 1.0), vec)
            return np.linalg.norm(queries - proj, 2, axis=1)
        else:
            return np.linalg.norm(queries - self._points, 2, axis=1)


@nb.jit(nopython=True, nogil=True)
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared arrray elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))


@nb.jit(nopython=True, nogil=True)
def _toeplitz(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Numba-compatible implementation of `scipy.linalg.toeplitz` method.

    .. note:
        It does not handle any special case for efficiency, so the input types
        is more respective than originally.

    .. warning:
        It returns a strided matrix instead of contiguous copy for efficiency.

    :param c: First column of the matrix.
    :param r: First row of the matrix.

    see::
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """
    vals = np.concatenate((c[::-1], r[1:]))
    n = vals.strides[0]
    return as_strided(
        vals[len(c)-1:], shape=(len(c), len(r)), strides=(-n, n))


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


__all__ = [
    "ConvexHull",
    "squared_norm_2",
    "integrate_zoh",
    "smoothing_filter"
]
