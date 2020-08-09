import sys as _sys

# Define helper to check if a module is available for compatibility with Python 2.7
def is_package_available(pkg_name):
    if (_sys.version_info > (3, 0)):
        return __import__('importlib').util.find_spec(pkg_name) is not None
    else:
        try:
            __import__('imp').find_module(pkg_name)
        except ImportError:
            return False
        return True

# Import Eigenpy and Pinocchio (use the embedded version only if necessary),
# then patch Pinocchio to fix support of numpy.ndarray as Eigen conversion.
if (is_package_available("eigenpy")):
    import eigenpy
else:
    from . import eigenpy
    _sys.modules["eigenpy"] = eigenpy
if (is_package_available("pinocchio")):
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

# Import core submodule
from . import core
from .core import __version__, __raw_version__
