import sys as _sys
from .core import (core,
                   get_include,
                   get_libraries,
                   __version__,
                   __raw_version__)


_sys.modules[".".join((__name__, "core"))] = core

__all__ = [
    'core',
    'get_include',
    'get_libraries',
    '__version__',
    '__raw_version__',
]
