# pylint: disable=missing-module-docstring

from .qhull import ConvexHull, compute_distance_convex_to_point
from .generic import (squared_norm_2,
                      matrix_to_quat,
                      matrix_to_rpy,
                      matrix_to_yaw,
                      quat_to_matrix,
                      quat_to_rpy,
                      quat_to_yaw,
                      quat_to_yaw_cos_sin,
                      quat_multiply,
                      quat_average,
                      rpy_to_matrix,
                      rpy_to_quat)

__all__ = [
    "ConvexHull",
    "compute_distance_convex_to_point",
    "squared_norm_2",
    "matrix_to_quat",
    "matrix_to_rpy",
    "matrix_to_yaw",
    "quat_to_matrix",
    "quat_to_rpy",
    "quat_to_yaw",
    "quat_to_yaw_cos_sin",
    "quat_multiply",
    "quat_average",
    "rpy_to_matrix",
    "rpy_to_quat"
]

try:
    from .spline import Spline
    __all__ += ['Spline']
except ImportError:
    pass
