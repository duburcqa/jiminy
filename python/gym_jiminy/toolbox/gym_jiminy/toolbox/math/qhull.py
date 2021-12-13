""" TODO: Write documentation.
"""
from typing import Optional

import numpy as np
import numba as nb
from numba.np.extensions import cross2d
from scipy.spatial.qhull import _Qhull, QhullError

from .generic import squared_norm_2


@nb.jit("float64[:](float64[:, :])", nopython=True, nogil=True)
def _amin_last_axis(array: np.ndarray) -> np.ndarray:
    """ TODO: Write documentation.
    """
    res = np.empty(array.shape[0])
    for i in range(array.shape[0]):
        res[i] = np.min(array[i])
    return res


@nb.jit("boolean[:](boolean[:, :])", nopython=True, nogil=True)
def _all_last_axis(array: np.ndarray) -> np.ndarray:
    """ TODO: Write documentation.
    """
    res = np.empty(array.shape[0], dtype=np.bool_)
    for i in range(array.shape[0]):
        res[i] = np.all(array[i])
    return res


@nb.jit(nopython=True, nogil=True)
def compute_distance_convex_to_point(points: np.ndarray,
                                     vertex_indices: np.ndarray,
                                     queries: np.ndarray) -> np.ndarray:
    """ TODO: Write documentation.
    """
    # Compute the equations of the edges
    points_1 = points[np.roll(vertex_indices, 1)].T
    points_0 = points[vertex_indices].T
    vectors = points_1 - points_0
    normals = np.stack((-vectors[1], vectors[0]), axis=0)
    normals /= np.sqrt(np.sum(np.square(normals), axis=0))
    offsets = - np.sum(normals * points_0, axis=0)
    equations = np.concatenate((normals, np.expand_dims(offsets, 0)))

    # Determine for each query point if it lies inside or outside
    queries = np.ascontiguousarray(queries)
    sign_dist = 1.0 - 2.0 * _all_last_axis(
        queries @ equations[:-1] + equations[-1] < 0.0)

    # Compute the distance from the convex hull, as the min distance
    # from every segment of the convex hull.
    ratios = np.sum(
        (np.expand_dims(queries, -1) - points_0) * vectors, axis=1
        ) / np.sum(np.square(vectors), axis=0)
    ratios = np.minimum(np.maximum(ratios, 0.0), 1.0)
    projs = np.expand_dims(ratios, 1) * vectors + points_0
    dist = np.sqrt(_amin_last_axis(np.sum(np.square(
        np.expand_dims(queries, -1) - projs), axis=1)))

    # Compute the resulting signed distance (negative if inside)
    signed_dist = sign_dist * dist

    return signed_dist


@nb.jit(nopython=True, nogil=True)
def compute_distance_convex_to_ray(
        points: np.ndarray,
        vertex_indices: np.ndarray,
        query_vector: np.ndarray,
        query_origin: np.ndarray) -> float:
    """ TODO: Write documentation.
    """
    # Compute the direction vectors of the edges
    points_1 = points[np.roll(vertex_indices, 1)]
    points_0 = points[vertex_indices]
    vectors = points_1 - points_0

    # Compute the distance from the convex hull, as the only edge intersecting
    # with the oriented line.
    ratios = cross2d(query_origin - points_0, query_vector) / \
        cross2d(vectors, query_vector)

    if np.sum(np.logical_and(0.0 <= ratios, ratios < 1.0)) != 2:
        raise ValueError("Query point origin not lying inside convex hull.")

    for j, ratio in enumerate(ratios):
        if 0.0 <= ratio < 1.0:
            proj = ratio * vectors[j] + points_0[j] - query_origin
            if proj.dot(query_vector) > 0.0:
                return np.linalg.norm(proj)

    return 0.0  # This case cannot happens because for the explicit check


class ConvexHull:
    """ TODO: Write documentation.
    """
    def __init__(self, points: np.ndarray) -> None:
        """Compute the convex hull defined by a set of points.

        :param points: N-D points whose to computed the associated convex hull,
                       as a 2D array whose first dimension corresponds to the
                       number of points, and the second to the N-D coordinates.
        """
        assert len(points) > 0, "The length of 'points' must be at least 1."

        # Backup user argument(s)
        self._points = np.ascontiguousarray(points)

        # Create convex full if possible
        if len(self._points) > 2:
            try:
                self._hull = _Qhull(points=self._points,
                                    options=b"",
                                    mode_option=b"i",
                                    required_options=b"Qt",
                                    furthest_site=False,
                                    incremental=False,
                                    interior_point=None)
            except QhullError as e:
                raise ValueError(
                    f"Impossible to compute convex hull ({self._points})."
                    ) from e
            self._vertex_indices = self._hull.get_extremes_2d()
        else:
            self._hull = None

        # Buffer to cache center computation
        self._center: Optional[np.ndarray] = None

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
                vertices = self._points[self._vertex_indices]
            else:
                vertices = self._points
            self._center = np.mean(vertices, axis=0)
        return self._center

    def get_distance_to_point(self, queries: np.ndarray) -> np.ndarray:
        """Compute the signed distance of query points from the convex hull.

        Positive distance corresponds to a query point lying outside the convex
        hull.

        .. note::
            Degenerated convex hulls corresponding to len(points) == 1 or 2 are
            handled separately. The distance from a point and a segment is used
            respectevely.

        ..warning::
            This method only supports 2D space for the non-degenerated case.

        :param queries: N-D query points for which to compute distance from the
                        convex hull, as a 2D array.

        :returns: 1D float vector of the same length than `queries`.
        """
        if len(self._points) > 2:
            # Compute the signed distance between query points and convex hull
            if self._points.shape[1] != 2:
                raise NotImplementedError
            return compute_distance_convex_to_point(
                self._points, self._vertex_indices, queries)

        if len(self._points) == 2:
            # Compute the distance between query points and segment
            vec = self._points[1] - self._points[0]
            ratio = (queries - self._points[0]) @ vec / squared_norm_2(vec)
            proj = self._points[0] + np.outer(np.clip(ratio, 0.0, 1.0), vec)
            return np.linalg.norm(queries - proj, 2, axis=1)

        # Compute the distance between query points and point
        return np.linalg.norm(queries - self._points, 2, axis=1)

    def get_distance_to_ray(self,
                            query_vector: np.ndarray,
                            query_origin: np.ndarray) -> np.ndarray:
        """Compute the distance of single ray from the convex hull.

        .. warning::
            It is assumed that the query origins are lying inside the convex
            hull.

        ..warning::
            This method only supports 2D space.

        .. warning::
            Degenerated convex hulls corresponding to len(points) == 1 or 2 are
            not supported.

        :param query_vector: Direction of the ray.
        :param query_origin: Origin of the ray.
        """
        if len(self._points) < 3:
            raise NotImplementedError
        if self._points.shape[1] != 2:
            raise NotImplementedError
        return compute_distance_convex_to_ray(
            self._points, self._vertex_indices, query_vector, query_origin)
