import os as _os
import sys as _sys
import importlib as _importlib
from contextlib import redirect_stderr as _redirect_stderr

with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
    # Import Pinocchio and co (use the embedded version only if necessary)
    if _importlib.util.find_spec("eigenpy") is not None:
        import eigenpy as _eigenpy
    else:
        from . import eigenpy as _eigenpy
        _sys.modules["eigenpy"] = _eigenpy

    if _importlib.util.find_spec("hppfcl") is not None:
        import hppfcl as _hppfcl
    else:
        from . import hppfcl as _hppfcl
        _sys.modules["hppfcl"] = _hppfcl

    if _importlib.util.find_spec("pinocchio") is not None:
        import pinocchio as _pinocchio
    else:
        from . import pinocchio as _pinocchio
        _sys.modules["pinocchio"] = _pinocchio

    # Preload rpy module to avoid boost python warnings at import
    from pinocchio.pinocchio_pywrap import rpy as _rpy
    _sys.modules["pinocchio.pinocchio_pywrap.rpy"] = _rpy

    # Mock 'StdVec_StdString' converter, if deleted (aka Windows)
    if not hasattr(_pinocchio, "StdVec_StdString"):
        _pinocchio.StdVec_StdString = list
        _pinocchio.pinocchio_pywrap.StdVec_StdString = list

    # Import core submodule once every dependency has been preloaded
    from . import core
    from .core import __version__, __raw_version__

# Patch Pinocchio to fix support of numpy.ndarray as Eigen conversion.
from . import _pinocchio_init as _patch  # noqa


__all__ = [
    'core',
    '__version__',
    '__raw_version__'
]
