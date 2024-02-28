# pylint: disable=missing-module-docstring

from .qhull import ConvexHull, compute_distance_convex_to_point
from .generic import squared_norm_2, quat_average

__all__ = [
    "ConvexHull",
    "compute_distance_convex_to_point",
    "squared_norm_2",
    "quat_average"
]

try:
    from .spline import Spline
    __all__ += ['Spline']
except ImportError:
    pass
