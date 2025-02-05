# pylint: disable=missing-module-docstring

from .math import (mean,
                   exp3,
                   exp6,
                   log3,
                   log6,
                   matrix_to_quat,
                   matrices_to_quat,
                   matrix_to_rpy,
                   matrix_to_yaw,
                   angle_axis_to_quat,
                   quat_to_angle_axis,
                   quat_to_matrix,
                   quat_to_rpy,
                   quat_to_yaw,
                   quat_to_yaw_cos_sin,
                   quat_apply,
                   quat_multiply,
                   quat_difference,
                   xyzquat_difference,
                   quat_average,
                   quat_interpolate_middle,
                   rpy_to_matrix,
                   rpy_to_quat,
                   transforms_to_xyzquat,
                   compute_tilt_from_quat,
                   swing_from_vector,
                   remove_yaw_from_quat,
                   remove_twist_from_quat)
from .spaces import (DataNested,
                     FieldNested,
                     ArrayOrScalar,
                     get_robot_state_space,
                     get_robot_measurements_space,
                     get_bounds,
                     zeros,
                     fill,
                     copyto,
                     copy,
                     clip,
                     contains,
                     build_reduce,
                     build_map,
                     build_copyto,
                     build_clip,
                     build_contains,
                     build_normalize,
                     build_flatten)
from .misc import (is_breakpoint,
                   is_nan,
                   get_fieldnames,
                   register_variables,
                   sample)
from .pipeline import (save_trajectory_to_hdf5,
                       load_trajectory_from_hdf5,
                       build_pipeline,
                       load_pipeline)


__all__ = [
    'mean',
    'exp3',
    'exp6',
    'log3',
    'log6',
    'matrix_to_quat',
    'matrices_to_quat',
    'matrix_to_rpy',
    'matrix_to_yaw',
    'angle_axis_to_quat',
    'quat_to_angle_axis',
    'quat_to_matrix',
    'quat_to_rpy',
    'quat_to_yaw',
    'quat_to_yaw_cos_sin',
    'quat_apply',
    'quat_multiply',
    'quat_difference',
    'xyzquat_difference',
    'quat_average',
    'quat_interpolate_middle',
    'rpy_to_matrix',
    'rpy_to_quat',
    'transforms_to_xyzquat',
    'compute_tilt_from_quat',
    'swing_from_vector',
    'remove_yaw_from_quat',
    'remove_twist_from_quat',
    'DataNested',
    'FieldNested',
    'ArrayOrScalar',
    'get_robot_state_space',
    'get_robot_measurements_space',
    'get_bounds',
    'sample',
    'zeros',
    'fill',
    'copyto',
    'copy',
    'clip',
    'contains',
    'build_reduce',
    'build_map',
    'build_copyto',
    'build_clip',
    'build_contains',
    'build_normalize',
    'build_flatten',
    'is_breakpoint',
    'is_nan',
    'get_fieldnames',
    'register_variables',
    'save_trajectory_to_hdf5',
    'load_trajectory_from_hdf5',
    'build_pipeline',
    'load_pipeline'
]
