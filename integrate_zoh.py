import numpy as np
import matplotlib.pyplot as plt

import numba as nb

from gym_jiminy.common.blocks.proportional_derivative_controller import (
    toeplitz, INV_FACTORIAL_TABLE)

#@nb.jit(nopython=True, cache=True, inline='always', fastmath=True)
def integrate_zoh(state_prev: np.ndarray,
                  state_min: np.ndarray,
                  state_max: np.ndarray,
                  dt: float) -> np.ndarray:
    """N-th order exact integration scheme assuming Zero-Order-Hold for the
    highest-order derivative, taking state bounds into account.

    .. warning::
        This method tries its best to keep the state within bounds, but it is
        not always possible if the order is strictly larger than 1. Indeed, the
        bounds of different derivative order may be conflicting. In such a
        case, it gives priority to lower orders.

    :param state_prev: Previous state update, ordered from lowest to highest
                       derivative order, which means:
                       s[i](t) = s[i](t-1) + integ_{t-1}^{t}(s[i+1](t))
    :param state_min: Lower bounds of the state.
    :param state_max: Upper bounds of the state.
    :param dt: Integration delta of time since previous state update.
    """
    # Make sure that dt is not negative
    assert dt >= 0.0, "The integration timestep 'dt' must be positive."

    # Early return if the timestep is too small
    if abs(dt) < 1e-9:
        return state_prev.copy()

    # Propagate derivative bounds to compute highest-order derivative bounds
    state_min_stack = [state_min.copy() for _ in range(100, 0, -1)]
    state_max_stack = [state_max.copy() for _ in range(100, 0, -1)]
    dim, size = state_prev.shape
    deriv_min, deriv_max = np.full((size,), -np.inf), np.full((size,), np.inf)
    for j in range(dim):
        for i in range(100, 0, -1):
            # Compute i-step ahead integration problem
            integ_coeffs = np.array([
                pow(i * dt, k) * INV_FACTORIAL_TABLE[k]
                for k in range(dim - j)])
            integ_zero = integ_coeffs[:-1].dot(state_prev[j:-1])
            integ_drift = integ_coeffs[-1]

            # Propagate derivative bounds at a given order only
            deriv_min_j = (state_min_stack[i-1][j] - integ_zero) / integ_drift
            deriv_max_j = (state_max_stack[i-1][j] - integ_zero) / integ_drift
            for k in range(size):
                if deriv_min[k] < deriv_min_j[k] <= deriv_max[k]:
                    deriv_min[k] = deriv_min_j[k]
                if deriv_min[k] <= deriv_max_j[k] < deriv_max[k]:
                    deriv_max[k] = deriv_max_j[k]
                if deriv_max_j[k] <= state_prev[-1, k]:
                    print(f"{i}: state_max_tmp")
                    breakpoint()
                    state_max_stack[i-1][(j + 1):, k] = 0.0
                if state_prev[-1, k] <= deriv_min_j[k]:
                    print(f"{i}: state_min_tmp")
                    breakpoint()
                    state_min_stack[i-1][(j + 1):, k] = 0.0

    # Clip highest-order derivative to ensure every derivative are within
    # bounds if possible, lowest orders in priority otherwise.
    deriv = np.minimum(np.maximum(state_prev[-1], deriv_min), deriv_max)

    # Compute 1-step ahead integration matrix
    integ_coeffs = np.array([
        pow(dt, k) * INV_FACTORIAL_TABLE[k] for k in range(dim)])
    integ_matrix = toeplitz(integ_coeffs, np.zeros(dim)).T
    integ_zero = integ_matrix[:, :-1].copy() @ state_prev[:-1]
    integ_drift = integ_matrix[:, -1:]

    # Integrate, taking into account clipped highest derivative
    return integ_zero + integ_drift * deriv


ORDER = 4

state_prev = np.zeros((ORDER, 1))
state_min = np.full((ORDER, 1), np.nan)
state_max = np.full((ORDER, 1), np.nan)
dt = 0.001

t = 0.0
times, states = [t], [state_prev.copy()]
states_min, states_max = [state_min.copy()], [state_max.copy()]
while t < 2.5:
    #print(t)
    state_prev[-1] = 1.0
    state_min[:, 0] = (-1.0, -np.inf, *((-np.inf,) * (ORDER - 2)))
    state_max[:, 0] = (+1.0, +np.inf, *((+np.inf,) * (ORDER - 2)))
    state_prev[:] = integrate_zoh(state_prev, state_min, state_max, dt)
    t += dt
    times.append(t)
    states.append(state_prev.copy())
    states_min.append(state_min.copy())
    states_max.append(state_max.copy())

fig, axes = plt.subplots(len(state_prev), 1, sharex=True)
for ax, data, data_min, data_max in zip(axes, *(
        np.concatenate(data, axis=-1)
        for data in (states, states_min, states_max))):
    ax.plot(times, data, '.-', color="tab:blue")
    ax.plot(times, data_min, '--', color="tab:red")
    ax.plot(times, data_max, '--', color="tab:red")
    ax.set_xlim(left=1.7)
plt.show()
