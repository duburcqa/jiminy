###############################################################################
## @brief             Entry point for jiminy_pywrap python module.
###############################################################################

import os as _os # private import
import sys as _sys

if (_sys.version_info > (3, 0)):
    from contextlib import redirect_stderr as _redirect_stderr
    with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
        from .jiminy_pywrap import *
        from .jiminy_pywrap import __version__, __raw_version__
else:
    with open(_os.devnull, 'w') as stderr:
        old_target = _sys.stderr
        _sys.stderr = stderr
        from .jiminy_pywrap import *
        _sys.stderr = old_target
