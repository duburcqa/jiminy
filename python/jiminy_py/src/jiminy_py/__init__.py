import sys as _sys
import warnings as _warnings
from multiprocessing.context import _force_start_method

from .core import *  # noqa: F403
from . import robot
from . import dynamics
from . import log
from . import simulator


if _sys.platform == 'darwin' and _sys.version_info < (3, 8):
    # Backport bpo-33725 fix to python < 3.8
    _warnings.warn(
        "'fork' context is not properly supported on Mac OSX but is used by "
        "default on Python < 3.8. Forcing using 'spawn' context globally.")
    _force_start_method('spawn')

__all__ = [
    'get_cmake_module_path',
    'get_include',
    'get_libraries',
    '__version__',
    '__raw_version__',
    'robot',
    'dynamics',
    'log',
    'simulator'
]

try:
    from . import plot
    __all__ += ['plot']
except ImportError:
    pass
