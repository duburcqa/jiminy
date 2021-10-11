""" TODO: Write documentation.
"""
import math
import ctypes
from typing import Callable, Union, Tuple

import numpy as np
import numba as nb
from numpy.lib.stride_tricks import as_strided
from numba.extending import get_cython_function_address

from .core import HeightMapFunctor, heightMapType_t


murmurhash3_32_addr = get_cython_function_address(
    "sklearn.utils.murmurhash", "murmurhash3_int_u32")
murmurhash3_32_functype = ctypes.CFUNCTYPE(
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint)
murmurhash3_32_ptr = murmurhash3_32_functype(murmurhash3_32_addr)


@nb.generated_jit(nopython=True, nogil=True)
def murmurhash3_32(key: int, seed: np.uint32) -> Union[int, float]:
    """Compute the 32 bit murmurhash3 encoding of key at seed.

    :param key: integer to hash.
    :param seed: Unsigned integer seed for the hashing algorithm.
                 Optional: 0 by default.

    :returns: Unsigned integer corresponding to the hash of the key.
    """
    if isinstance(key, nb.types.Array):
        def _murmurhash3_32_impl(key, seed):
            encoding = 0
            for i, val in enumerate(key):
                encoding = murmurhash3_32_ptr(encoding + val, seed + i)
            return encoding
    elif isinstance(key, nb.types.Integer):
        def _murmurhash3_32_impl(key, seed):
            return murmurhash3_32_ptr(key, seed)
    else:
        raise RuntimeError("'key' must have type `int` for `np.ndarray`.")

    return _murmurhash3_32_impl


@nb.jit(nopython=True, nogil=True)
def _random_height(key: int,
                   proba_inv: int,
                   seed: np.uint32) -> Union[int, float]:
    """ TODO: Write documentation.
    """
    encoding = murmurhash3_32(key, seed)
    if encoding % proba_inv == 0:
        encoding /= (2 ** 32 - 1)
    else:
        encoding = 0
    return encoding


@nb.jit(nopython=True, nogil=True)
def _tile_2d_interp_1d(p_idx: np.ndarray,
                       p_rel: np.ndarray,
                       dim: int,
                       tile_size: np.ndarray,
                       tile_proba_inv: float,
                       tile_height_max: float,
                       tile_interp_threshold: np.ndarray,
                       seed: np.uint32) -> Tuple[float, float]:
    """ TODO: Write documentation.
    """
    z = tile_height_max * _random_height(p_idx, tile_proba_inv, seed)
    if p_rel[dim] < tile_interp_threshold[dim]:
        p_idx = p_idx.copy()
        p_idx[dim] -= 1
        z_m = tile_height_max * _random_height(p_idx, tile_proba_inv, seed)
        ratio = (1.0 - p_rel[dim] / tile_interp_threshold[dim]) / 2.0
        height = z + (z_m - z) * ratio
        dheight = (z - z_m) / (
            2.0 * tile_size[dim] * tile_interp_threshold[dim])
    elif 1.0 - p_rel[dim] < tile_interp_threshold[dim]:
        p_idx = p_idx.copy()
        p_idx[dim] += 1
        z_p = tile_height_max * _random_height(p_idx, tile_proba_inv, seed)
        ratio = (1.0 + (p_rel[dim] - 1.0) / tile_interp_threshold[dim]) / 2.0
        height = z + (z_p - z) * ratio
        dheight = (z_p - z) / (
            2.0 * tile_size[dim] * tile_interp_threshold[dim])
    else:
        height = z
        dheight = 0.0

    return height, dheight


def get_random_tile_ground(tile_size: np.ndarray,
                           tile_proba_inv: float,
                           tile_height_max: float,
                           tile_interp_delta: np.ndarray,
                           seed: np.uint32) -> Callable[
                               [float, float], Tuple[float, np.ndarray]]:
    """ TODO: Write documentation.
    """
    # Make sure the arguments are valid
    assert (np.all(0.01 <= tile_interp_delta) and np.all(
        tile_interp_delta <= tile_size / 2.0)), (
            "'tile_interp_delta' must be in range [0.01, 'tile_size'/2.0].")

    # Compute some proxies
    tile_interp_threshold = tile_interp_delta / tile_size

    @nb.jit(nopython=True, nogil=True)
    def _random_tile_ground_impl(x: float,
                                 y: float,
                                 height: np.ndarray,
                                 normal: np.ndarray) -> None:
        nonlocal tile_size, tile_proba_inv, tile_height_max, \
            tile_interp_threshold, seed

        # Compute the tile index and relative coordinate
        p = np.array([x, y]) + tile_size / 2
        p_idx = (p // tile_size).astype(np.int32)
        p_rel = p / tile_size - p_idx

        # Interpolate height based on nearby tiles if necessary
        is_edge = np.logical_or(p_rel < tile_interp_threshold,
                                1.0 - p_rel < tile_interp_threshold)
        if is_edge[0] and not is_edge[1]:
            height[()], dheight_x = _tile_2d_interp_1d(
                p_idx, p_rel, 0, tile_size, tile_proba_inv, tile_height_max,
                tile_interp_threshold, seed)
            dheight_y = 0.0
        elif is_edge[1] and not is_edge[0]:
            height[()], dheight_y = _tile_2d_interp_1d(
                p_idx, p_rel, 1, tile_size, tile_proba_inv, tile_height_max,
                tile_interp_threshold, seed)
            dheight_x = 0.0
        elif is_edge[1] and is_edge[0]:
            height_0, dheight_x = _tile_2d_interp_1d(
                p_idx, p_rel, 0, tile_size, tile_proba_inv, tile_height_max,
                tile_interp_threshold, seed)
            if p_rel[1] < tile_interp_threshold[1]:
                p_idx[1] -= 1
                height_m, dheight_x_m = _tile_2d_interp_1d(
                    p_idx, p_rel, 0, tile_size, tile_proba_inv,
                    tile_height_max, tile_interp_threshold, seed)
                ratio = (1.0 - p_rel[1] / tile_interp_threshold[1]) / 2.0
                height[()] = height_0 + (height_m - height_0) * ratio
                dheight_x = dheight_x + (dheight_x_m - dheight_x) * ratio
                dheight_y = (height_0 - height_m) / (
                    2.0 * tile_size[1] * tile_interp_threshold[1])
            else:
                p_idx[1] += 1
                height_p, dheight_x_p = _tile_2d_interp_1d(
                    p_idx, p_rel, 0, tile_size, tile_proba_inv,
                    tile_height_max, tile_interp_threshold, seed)
                ratio = (
                    1.0 + (p_rel[1] - 1.0) / tile_interp_threshold[1]) / 2.0
                height[()] = height_0 + (height_p - height_0) * ratio
                dheight_x = dheight_x + (dheight_x_p - dheight_x) * ratio
                dheight_y = (height_p - height_0) / (
                    2.0 * tile_size[1] * tile_interp_threshold[1])
        else:
            height[()] = tile_height_max * _random_height(
                p_idx, tile_proba_inv, seed)
            dheight_x, dheight_y = 0.0, 0.0

        # Compute the resulting normal to the surface
        # normal = (-d.height/d.x, -d.height/d.y, 1.0)
        normal[:] = -dheight_x, -dheight_y, 1.0
        normal /= math.sqrt(dheight_x ** 2 + dheight_y ** 2 + 1.0)

    return HeightMapFunctor(_random_tile_ground_impl, heightMapType_t.GENERIC)


@nb.jit(nopython=True, nogil=True)
def _discretize_heightmap_impl(heightmap: nb.core.dispatcher.Dispatcher,
                               grid_size: float,
                               grid_unit: float) -> np.ndarray:
    """ TODO: Write documentation.
    """
    # Allocate empty discrete grid
    grid_dim = int(np.ceil(grid_size / grid_unit)) + 1
    height_grid = np.empty((grid_dim, grid_dim, 6))

    # Set X and Y coordinates
    values = np.arange(grid_dim) * grid_unit - (grid_dim - 1) * grid_unit / 2
    height_grid[..., 0] = as_strided(
        values, shape=(grid_dim, grid_dim), strides=(0, values.itemsize))
    height_grid[..., 1] = height_grid[..., 0].T

    # Fill discrete grid
    height = np.array(0.0)
    for i in range(grid_dim):
        for j in range(grid_dim):
            x, y = height_grid[i, j][:2]
            normal = height_grid[i, j][3:]
            heightmap(x, y, height, normal)
            height_grid[i, j][2] = height[()]

    return height_grid


def discretize_heightmap(heightmap: HeightMapFunctor,
                         grid_size: float,
                         grid_unit: float) -> np.ndarray:
    """ TODO: Write documentation.
    """
    # Try to unwrap python method for fast access if possible.
    heightmap_py = heightmap.py_function
    if heightmap_py is not None and isinstance(
            heightmap_py, nb.core.dispatcher.Dispatcher):
        return _discretize_heightmap_impl(heightmap_py, grid_size, grid_unit)

    # Allocate empty discrete grid
    grid_dim = int(np.ceil(grid_size / grid_unit)) + 1
    values = np.arange(grid_dim) * grid_unit - (grid_dim - 1) * grid_unit / 2
    height_grid = np.empty((grid_dim, grid_dim, 6))
    height_grid[..., 0], height_grid[..., 1] = np.meshgrid(
        values, values, copy=False)

    # Fill discrete grid
    for i in range(grid_dim):
        for j in range(grid_dim):
            height_grid[i, j][2], height_grid[i, j][3:] = \
                heightmap(height_grid[i, j][:3])

    return height_grid
