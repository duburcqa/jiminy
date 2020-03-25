###############################################################################
## @brief             Entry point for jiminy_pywrap python module.
###############################################################################

import os as _os # private import
import sys as _sys

# Try to preload pinocchio if possible since parts of the code rely on it
is_pinocchio_available = False

if (_sys.version_info > (3, 0)):
    from contextlib import redirect_stderr as _redirect_stderr
    with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
        try:
            import pinocchio as _pnc
            is_pinocchio_available = True
        except ImportError:
            pass
        from .libjiminy_pywrap import *
        from .libjiminy_pywrap import __version__, __raw_version__
else:
    with open(_os.devnull, 'w') as stderr:
        old_target = _sys.stderr
        _sys.stderr = stderr

        try:
            import pinocchio as _pnc
            is_pinocchio_available = True
        except ImportError:
            pass
        from .libjiminy_pywrap import *

        _sys.stderr = old_target

if is_pinocchio_available:
    from .. import _pinocchio_init as _patch
