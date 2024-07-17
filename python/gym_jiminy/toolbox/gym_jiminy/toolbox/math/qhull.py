"""This module proposes a basic yet reasonably efficient representation of the
convex hull of a set of points in 2D Euclidean space.

This comes with helper methods to compute various types of distances, including
for degenerated cases.
"""
import math
from functools import cached_property
from typing import Tuple, List

import numpy as np
import numba as nb
from numba.np.extensions import cross2d

from scipy.optimize import linprog

try:
    from matplotlib.pyplot import Figure
except ImportError:
    Figure = type(None)  # type: ignore[misc,assignment,unused-ignore]

from jiminy_py.viewer import interactive_mode


@nb.jit(nopython=True, cache=True, inline='always')
def _amin_last_axis(array: np.ndarray) -> np.ndarray:
    """Compute the minimum of a 2-dimensional array along the second axis.

    :param array: Input array.
    """
    res = np.empty(array.shape[0])
    for i in range(array.shape[0]):
        res[i] = np.min(array[i])
    return res


@nb.jit(nopython=True, cache=True, inline='always')
def _all_last_axis(array: np.ndarray) -> np.ndarray:
    """Test whether all array elements along the second axis of a 2-dimensional
    array evaluate to True.

    :param array: Input array.
    """
    res = np.empty(array.shape[0], dtype=np.bool_)
    for i in range(array.shape[0]):
        res[i] = np.all(array[i])
    return res


@nb.jit(nopython=True, cache=True, fastmath=True)
def compute_convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Determine the vertices of the convex hull defined by a set of points in
    2D Euclidean space.

    As a reminder, the convex hull of a set of points is defined as the
    smallest convex polygon that can enclose all the points.

    .. seealso::
        Internally, it leverages using Andrew's monotone chain algorithm, which
        as almost optimal complexity O(n*log(n)) but is only applicable in 2D
        space. For an overview of all the existing algorithms to this day, see:
        https://en.wikipedia.org/wiki/Convex_hull_algorithms

    :param points: Set of 2D points from which to compute the convex hull,
                   as a 2D array whose first dimension corresponds to the
                   number of points, and the second gathers the 2 position
                   coordinates (X, Y).
    """
    # Sorting points lexicographically (by x and then by y).
    indices = np.argsort(points[:, 0], kind='mergesort')
    indices = indices[np.argsort(points[indices, 1], kind='mergesort')]

    # If there is less than 3 points, the hull comprises all the points
    npoints = len(points)
    if npoints <= 3:
        return indices

    # Combining the lower and upper hulls to get the full convex hull.
    # The first point of each hull is omitted because it is the same as the
    # last point of the other hull.
    vertex_indices: List[int] = []

    # Build the upper hull.
    # The upper hull is similar to the lower hull but runs along the top of the
    # set of points. It is constructed by starting with the right-most point
    # and moving left.
    upper: List[np.ndarray] = []
    for i in indices[::-1]:
        point = points[i]
        while len(upper) > 1:
            # Check if the point is inside or outside of the hull at the time
            # being by checking the sign of the 2D cross product.
            if (upper[-1][0] - upper[-2][0]) * (point[1] - upper[-2][1]) > (
                    point[0] - upper[-2][0]) * (upper[-1][1] - upper[-2][1]):
                break
            upper.pop()
            vertex_indices.pop()
        if upper:
            vertex_indices.append(i)
        upper.append(point)

    # Build the lower hull.
    # The lower hull is the part of the convex hull that runs along the bottom
    # of the set of points when they are sorted by their x-coordinates (from
    # left to right). It is constructed by starting with the left-most point,
    # points are added to the lower hull. If adding a new point creates a
    # "right turn" (or non-left turn) with the last two points in the lower
    # hull, the second-to-last point is removed.
    lower: List[np.ndarray] = []
    for i in indices:
        point = points[i]
        while len(lower) > 1:
            if (lower[-1][0] - lower[-2][0]) * (point[1] - lower[-2][1]) > (
                    lower[-1][1] - lower[-2][1]) * (point[0] - lower[-2][0]):
                break
            lower.pop()
            vertex_indices.pop()
        if lower:
            vertex_indices.append(i)
        lower.append(point)

    return np.array(vertex_indices)


@nb.jit(nopython=True, cache=True, inline='always')
def compute_vectors_from_convex(vertices: np.ndarray) -> np.ndarray:
    """Compute the un-normalized oriented direction vector of the edges.

    A point is inside the convex hull if it lies on the left side of all the
    edges.

    :param vertices: Vertices of the convex hull with counter-clockwise
                     ordering, as a 2D array whose first dimension corresponds
                     to individual vertices while the second dimensions gathers
                     the 2 position coordinates (X, Y).

    :returns: Direction of all the edges with the same ordering of the provided
    vertices, as a 2D array whose first dimension corresponds to individual
    edges while the second gathers the 2 components of the direction.
    """
    vectors = np.empty((2, len(vertices))).T
    vectors[0] = vertices[-1] - vertices[0]
    vectors[1:] = vertices[:-1] - vertices[1:]
    return vectors


@nb.jit(nopython=True, cache=True, inline='always')
def compute_equations_from_convex(vertices: np.ndarray,
                                  vectors: np.ndarray) -> np.ndarray:
    """Compute the (normalized) equations of the edges for a convex hull in 2D
    Euclidean space.

    The equation of a edge is fully specified by its normal vector 'a' and a
    scalar floating-point offset 'c'. A given point 'x' is on the line
    defined by the edge of `np.dot(a, x) + d = 0.0`, inside if negative,
    outside otherwise.

    :param vertices: Vertices of the convex hull with counter-clockwise
                     ordering, as a 2D array whose first dimension corresponds
                     to individual vertices while the second dimensions gathers
                     the 2 position coordinates (X, Y).
    :param vectors: Direction of all the edges with the same ordering of the
                    provided vertices, as a 2D array whose first dimension
                    corresponds to individual edges while the second gathers
                    the 2 components of the direction.

    :returns: Equations of all the edges with the same ordering of the provided
    vertices, as a 2D array whose first dimension corresponds to individual
    edges while the second gathers the 2 components of the normal (ax, ay) plus
    the offset d. The normal vector is normalized.
    """
    equations = np.empty((3, len(vertices)))
    normals, offsets = equations[:2], equations[-1]
    normals[0] = - vectors[:, 1]
    normals[1] = + vectors[:, 0]
    normals /= np.sqrt(np.sum(np.square(normals), axis=0))
    offsets[:] = - np.sum(normals * vertices.T, axis=0)
    return equations.T


@nb.jit(nopython=True, cache=True, inline='always')
def compute_distance_convex_to_point(vertices: np.ndarray,
                                     vectors: np.ndarray,
                                     queries: np.ndarray) -> np.ndarray:
    """Compute the signed distance of query points from a convex hull in 2D
    Euclidean space.

    Positive distance corresponds to a query point lying outside the convex
    hull.

    .. warning:
        The convex hull must be non-degenerated, ie having at least 3 points.

    :param vertices: Vertices of the convex hull with counter-clockwise
                     ordering, as a 2D array whose first dimension corresponds
                     to individual vertices while the second dimensions gathers
                     the 2 position coordinates (X, Y).
    :param vectors: Direction of all the edges with the same ordering of the
                    provided vertices, as a 2D array whose first dimension
                    corresponds to individual edges while the second gathers
                    the 2 components of the direction.
    :param queries: Set of 2D points for which to compute the distance from the
                    convex hull, as a 2D array whose first dimension
                    corresponds to the individual query points while the second
                    dimensions gathers the 2 position coordinates (X, Y).
    """
    # Determine for each query point if it lies inside or outside
    queries_rel = np.expand_dims(queries, -1) - vertices.T
    sign_dist = 1.0 - 2.0 * _all_last_axis(
        queries_rel[:, 0] * vectors[:, 1] > queries_rel[:, 1] * vectors[:, 0])

    # Compute the distance from the convex hull, as the min distance
    # from every segment of the convex hull.
    ratios = np.expand_dims(np.sum(
        queries_rel * vectors.T, axis=1
        ), 1) / np.sum(np.square(vectors), axis=1)
    ratios = np.minimum(np.maximum(ratios, 0.0), 1.0)
    projs = ratios * vectors.T + vertices.T
    dist = np.sqrt(_amin_last_axis(np.sum(np.square(
        np.expand_dims(queries, -1) - projs), axis=1)))

    # Resulting the signed distance (negative if inside)
    return sign_dist * dist


@nb.jit(nopython=True, cache=True, inline='always')
def compute_distance_convex_to_ray(
        vertices: np.ndarray,
        vectors: np.ndarray,
        query_dir: np.ndarray,
        query_origin: np.ndarray) -> float:
    """Compute ray-casting signed distance (aka. time-of-flight) from a convex
    hull to a oriented ray originating at a given position and pointing in a
    specific direction, in 2D Euclidean space.

    The distance is negative if the origin of the ray lies inside the convex
    hull. The distance is 'inf' is there is no intersection between the
    oriented ray and the convex hull. It never happens if the origin lays
    inside the convex hull, which means that the distance is negative, but
    there is no guarantee otherwise.

    .. warning:
        The convex hull must be non-degenerated, ie having at least 3 points.

    :param vertices: Vertices of the convex hull with counter-clockwise
                     ordering, as a 2D array whose first dimension corresponds
                     to individual vertices while the second dimensions gathers
                     the 2 position coordinates (X, Y).
    :param vectors: Direction of all the edges with the same ordering of the
                    provided vertices, as a 2D array whose first dimension
                    corresponds to individual edges while the second gathers
                    the 2 components of the direction.
    :param query_dir: Direction in which the ray is casting, as a 1D array.
                      It does not have to be normalized.
    :param query_origin: Position from which the ray is casting, as a 1D array.
    """
    # Compute the distance from the convex hull.
    # The distance only edge intersecting with the oriented line.
    # The follow ratio corresponds to the relative position of the intersection
    # point from each edge, 0.0 and 1.0 correspond to start vertex 'point_0'
    # and end vertex 'point_1' respectively. This ratio is unbounded. Values
    # outside range [0.0, 1.0] means that there is no intersection with the
    # corresponding edge.
    ratios = (cross2d(query_origin - vertices, query_dir) /
              cross2d(vectors, query_dir))

    # Compute the minimum casting distance and count how many edges are crossed
    collide_num = 0
    casting_dist = math.inf
    for j, ratio in enumerate(ratios):
        if 0.0 <= ratio < 1.0:
            collision = ratio * vectors[j] + vertices[j]
            oriented_ray = collision - query_origin
            if oriented_ray.dot(query_dir) > 0.0:
                casting_dist = min(  # type: ignore[assignment]
                    np.linalg.norm(oriented_ray), casting_dist)
                collide_num += 1

    # If the ray is intersecting with two edges and the sign of the oriented
    # casting ray from origin to collision point is positive for only one of
    # them, then the origin is laying inside the convex hull, which means that
    # the sign of the distance should be negative. On the contrary, if both
    # are positive, then it is outside and the distance is the minimum between
    # them. In all other cases, the distance is undefine, returning 'inf'.
    if collide_num == 1:
        casting_dist *= -1
    return casting_dist


@nb.jit(nopython=True, cache=True, inline='always')
def compute_distance_convex_to_convex(vertices_1: np.ndarray,
                                      vectors_1: np.ndarray,
                                      vertices_2: np.ndarray,
                                      vectors_2: np.ndarray) -> float:
    """Compute the distance between two convex hulls in 2D Euclidean space.

    .. warning:
        Both convex hull must be non-degenerated, ie having at least 3 points
        each.

    :param vertices_1: Vertices of the first convex hull with counter-clockwise
                       ordering, as a 2D array whose first dimension
                       corresponds to individual vertices while the second
                       dimensions gathers the 2 position coordinates (X, Y).
    :param vectors_1: Direction of all the edges of the first convex hull with
                      the same ordering of the provided vertices, as a 2D array
                      whose first dimension corresponds to individual edges and
                      the second gathers the 2 components of the direction.
    :param vertices_2: Vertices of the second convex hull with counter-
                       clockwise ordering, as a 2D array. See `vertices_1` for
                       details.
    :param vectors_2: Direction of all the edges of the second convex hull with
                      the same ordering of the provided vertices, as a 2D
                      array. See `vertices_2` for details.
    """
    distance_1 = np.min(
        compute_distance_convex_to_point(vertices_1, vectors_1, vertices_2))
    distance_2 = np.min(
        compute_distance_convex_to_point(vertices_2, vectors_2, vertices_1))
    return min(distance_1, distance_2)


def compute_convex_chebyshev_center(
        equations: np.ndarray) -> Tuple[np.ndarray, float]:
    r"""Compute the Chebyshev center of a convex polyhedron in N-dimensional
    Euclidean space.

    The Chebyshev center is the point that is furthest inside a convex
    polyhedron. Alternatively, it is the center of the largest hypersphere of
    inscribed in the polyhedron. This can easily be computed using linear
    programming. Considering halfspaces of the form :math:`Ax + b \leq 0`,
    solving the linear program:

    .. math::
        max \: y
        s.t. Ax + y ||A_i|| \leq -b

    With :math:`A_i` being the rows of A, i.e. the normals to each plane. The
    equations outputted by Qhull are always normalized. For reference, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html

    .. warning:
        The convex hull must be non-degenerated, ie having at least 3 points.

    :param equations: Equations of the edges as a 2D array whose first dimension
                      corresponds to individual edges while the second gathers
                      the 2 components of the normal plus the offset.

    :return: Pair (center, radius) where 'center' is a 1D array, and 'radius'
    is a positive scalar floating point value.
    """  # noqa: E501  # pylint: disable=line-too-long
    # Compute the centroid of the polyhedron as the initial guess
    num_dims = equations.shape[1]
    A = np.concatenate((
        equations[:, :-1], np.ones((len(equations), 1))), axis=1)
    b = - equations[:, -1:]
    c = np.array([*((0.0,) * (num_dims - 1)), -1.0])
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    return res.x[:-1], res.x[-1]


class ConvexHull2D:
    """Class representing the convex hulls of a set of points in 2D Euclidean
    space.
    """

    def __init__(self, points: np.ndarray) -> None:
        """Compute the convex hull defined by a set of points in 2D Euclidean
        space.

        :param points: Set of 2D points from which to compute the convex hull,
                       as a 2D array whose first dimension corresponds to the
                       number of points, and the second gathers the 2 position
                       coordinates (X, Y).
        """
        assert len(points) > 0, "The length of 'points' must be at least 1."

        # Backup user argument(s)
        self.points = points
        self.npoints = len(self.points)

        # Compute the vertices of the convex hull
        self.indices = compute_convex_hull_vertices(self.points)

        # Extract vertices of the convex hull with counter-clockwise ordering
        self.vertices = self.points[self.indices]

    @cached_property
    def vectors(self) -> np.ndarray:
        """Un-normalized oriented direction vector of the edges.
        """
        return compute_vectors_from_convex(self.vertices)

    @cached_property
    def equations(self) -> np.ndarray:
        """Normalized equations of the edges.
        """
        return compute_equations_from_convex(-self.vertices, self.vectors)

    @cached_property
    def center(self) -> np.ndarray:
        """Barycenter.

        .. note::
            The barycenter must be distinguished from the Chebyshev center,
            which is defined as the center of the largest circle inscribed in
            the polyhedron. Computing the latter involves solving a Linear
            Program, which is known to have a unique solution that can always
            be found in finite time. However, it is several order of magnitude
            to compute than the barycenter. For details about the Chebyshev
            center, see `compute_convex_chebyshev_center`.
        """
        return np.mean(self.vertices, axis=0)

    def get_distance_to_point(self, points: np.ndarray) -> np.ndarray:
        """Compute the signed distance of a single or a batch of query points
        from the convex hull.

        Positive distance corresponds to a query point lying outside the convex
        hull.

        .. note::
            Degenerated convex hulls are handled separately. The distance from
            a point and a segment is used respectively.

        :param points: Set of 2D points for which to compute the distance from
                       the convex hull, as a 2D array whose first dimension
                       corresponds to the individual query points while the
                       second dimensions gathers the 2 position coordinates.
                       Note that the first dimension can be omitted if there is
                       a single query point.
        """
        # Make sure that the input is at least 2D
        if points.ndim < 2:
            points = np.atleast_2d(points)

        # Compute the signed distance between query points and convex hull
        if self.npoints > 2:
            return compute_distance_convex_to_point(
                self.vertices, self.vectors, points)

        # Compute the distance between query points and segment
        if self.npoints == 2:
            vec = self.vertices[1] - self.vertices[0]
            ratio = (points - self.vertices[0]) @ vec / np.dot(vec, vec)
            proj = self.vertices[0] + np.outer(np.clip(ratio, 0.0, 1.0), vec)
            return np.linalg.norm(points - proj, axis=1)

        # Compute the distance between query points and point
        return np.linalg.norm(points - self.vertices, axis=1)

    def get_distance_to_ray(self,
                            vector: np.ndarray,
                            origin: np.ndarray) -> np.ndarray:
        """Compute the signed distance of single oriented ray from this convex
        hull.

        The distance is negative if the origin of the ray lies inside the
        convex hull. The distance is 'inf' is there is no intersection between
        the oriented ray and the convex hull.

        .. warning::
            Degenerated convex hulls are not supported.

        :param vector: Direction in which the ray is casting, as a 1D array.
                       This vector does not have to be normalized.
        :param origin: Position from which the ray is casting, as a 1D array.
        """
        if self.npoints < 3:
            raise NotImplementedError
        return compute_distance_convex_to_ray(
            self.vertices, self.vectors, vector, origin)

    def get_distance_to_convex(self, other: "ConvexHull2D") -> float:
        """Compute the distance between two convex hulls in 2D Euclidean space.

        :param other: Convex hull from which to compute the distance wrt the
                      one characterized by this instance.
        """
        return compute_distance_convex_to_convex(
            self.vertices, self.vectors, other.vertices, other.vectors)

    def plot(self) -> Figure:
        """Display the original points along their convex hull.
        """
        # Make sure matplotlib is available
        try:
            # pylint: disable=import-outside-toplevel
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("Matplotlib library not available. Please "
                              "install it before calling this method.") from e

        # Create new figure and return it to the user
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.plot(*self.points.T, 'o')
        plt.plot(*np.stack((
            self.vertices.T,
            np.roll(self.vertices, 1, axis=0).T), axis=-1), 'k-')
        plt.show(block=False)

        # Show figure, without blocking for interactive python sessions only
        if interactive_mode() < 2:
            fig.show()

        return fig
