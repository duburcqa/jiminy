###############################################################################
## @brief             Entry point for jiminy_pywrap python module.
###############################################################################

import os as _os # private import
import sys as _sys

if (_sys.version_info > (3, 0)):
    from contextlib import redirect_stderr as _redirect_stderr
    with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
        import pinocchio as _pnc
        _sys.path.append(_os.path.dirname(_pnc.__file__)) # Required to be able to find libpinocchio_pywrap.so
        import libpinocchio_pywrap as _pin # Preload the dynamic library Python binding
        from .libjiminy_pywrap import *
else:
    with open(_os.devnull, 'w') as stderr:
        old_target = _sys.stderr
        _sys.stderr = stderr

        import pinocchio as _pnc
        _sys.path.append(_os.path.dirname(_pnc.__file__))
        import libpinocchio_pywrap as _pin
        from .libjiminy_pywrap import *

        _sys.stderr = old_target