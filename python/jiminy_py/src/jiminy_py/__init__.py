import os as _os
import sys as _sys
import inspect as _inspect
import importlib as _importlib
from pkg_resources import parse_version as _version
from contextlib import redirect_stderr as _redirect_stderr


# Fix Dll seach path on windows for Python >= 3.8
if _sys.platform.startswith('win') and _sys.version_info >= (3, 8):
    for path in _os.environ['PATH'].split(_os.pathsep):
        if _os.path.exists(path):
            _os.add_dll_directory(path)

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

# Register pinocchio_pywrap to avoid importing bindings twise, which messes
# up with boost python converters. In addition, submodules search path needs
# to be fixed for releases older than 2.5.6.
submodules = _inspect.getmembers(
    _pinocchio.pinocchio_pywrap, _inspect.ismodule)
for module_name, module_obj in submodules:
    module_path = ".".join(('pinocchio', 'pinocchio_pywrap', module_name))
    _sys.modules[module_path] = module_obj
    if _version(_pinocchio.printVersion()) <= _version("2.5.6"):
        module_path = ".".join(('pinocchio', module_name))
        _sys.modules[module_path] = module_obj

# Import core submodule once every dependencies have been preloaded
with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
    from . import core
    from .core import __version__, __raw_version__

# Patch Pinocchio to avoid loading ground geometry in viewer, and
# force numpy.ndarray type for from/to Python matrix converters.
from . import _pinocchio_init as _patch  # noqa


__all__ = [
    'core',
    '__version__',
    '__raw_version__'
]
