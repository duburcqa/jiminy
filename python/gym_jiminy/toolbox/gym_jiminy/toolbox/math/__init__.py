# pylint: disable=missing-module-docstring

from .qhull import ConvexHull, compute_distance_convex_to_point
from .generic import (squared_norm_2,
                      matrix_to_yaw,
                      quat_to_yaw_cos_sin,
                      quat_to_yaw)
from .signal import integrate_zoh


__all__ = [
    "ConvexHull",
    "compute_distance_convex_to_point",
    "squared_norm_2",
    "matrix_to_yaw",
    "quat_to_yaw_cos_sin",
    "quat_to_yaw",
    "integrate_zoh"
]

try:
    from .spline import Spline
    __all__ += ['Spline']
except ImportError:
    pass
