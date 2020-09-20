import sys as _sys
import importlib as _importlib

# Import Pinocchio and co (use the embedded version only if necessary),
# then patch Pinocchio to fix support of numpy.ndarray as Eigen conversion.
if _importlib.util.find_spec("eigenpy") is not None:
    import eigenpy
else:
    from . import eigenpy
    _sys.modules["eigenpy"] = eigenpy

if _importlib.util.find_spec("hppfcl") is not None:
    import hppfcl
else:
    from . import hppfcl
    _sys.modules["hppfcl"] = hppfcl

if _importlib.util.find_spec("pinocchio") is not None:
    import pinocchio
else:
    from . import pinocchio
    _sys.modules["pinocchio"] = pinocchio
    from pkg_resources import parse_version as _version
    if _version(pinocchio.printVersion()) >= _version("2.3.0"):
        from .pinocchio.pinocchio_pywrap import rpy as _rpy
        _sys.modules["pinocchio.pinocchio_pywrap.rpy"] = _rpy
    pinocchio.pinocchio_pywrap.StdVec_StdString = list
from . import _pinocchio_init as _patch

# Import core submodule once every dependency has been preloaded
from . import core
from .core import __version__, __raw_version__
