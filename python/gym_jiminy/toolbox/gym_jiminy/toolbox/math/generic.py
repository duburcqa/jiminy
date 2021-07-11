""" TODO: Write documentation.
"""
import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True)
def squared_norm_2(array: np.ndarray) -> float:
    """Fast implementation of the sum of squared arrray elements, optimized for
    small to medium size 1D arrays.
    """
    return np.sum(np.square(array))
