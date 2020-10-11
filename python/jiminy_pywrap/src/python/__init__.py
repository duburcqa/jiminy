###############################################################################
## @brief             Entry point for jiminy_pywrap python module.
###############################################################################

import os as _os  # private import

from contextlib import redirect_stderr as _redirect_stderr
with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
    from .jiminy_pywrap import *
    from .jiminy_pywrap import __version__, __raw_version__
