import sys
import warnings
from multiprocessing.context import _force_start_method

from .core import *  # noqa: F403


if sys.platform == 'darwin' and sys.version_info < (3, 8):
    # Backport bpo-33725 fix to python < 3.8
    warnings.warn(
        "'fork' context is not properly supported on Mac OSX but is used by "
        "default on Python < 3.8. Forcing using 'spawn' context globally.")
    _force_start_method('spawn')

__all__ = [
    'get_cmake_module_path',
    'get_include',
    'get_libraries',
    '__version__',
    '__raw_version__'
]
