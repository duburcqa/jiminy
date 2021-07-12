""" TODO: Write documentation.
"""
import numpy as np
from scipy.spatial.qhull import _Qhull

from .generic import squared_norm_2


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
        if len(self._points) == 2:
            vec = self._points[1] - self._points[0]
            ratio = (queries - self._points[0]) @ vec / squared_norm_2(vec)
            proj = self._points[0] + np.outer(np.clip(ratio, 0.0, 1.0), vec)
            return np.linalg.norm(queries - proj, 2, axis=1)
        return np.linalg.norm(queries - self._points, 2, axis=1)
