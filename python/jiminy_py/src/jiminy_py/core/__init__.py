import os as _os
import re as _re
import sys as _sys
import ctypes as _ctypes
import inspect as _inspect
import logging as _logging
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
from contextlib import redirect_stderr as _redirect_stderr
from sysconfig import get_config_var as _get_config_var


# Special dlopen flags are used when loading Boost Python shared library
# available in search path on the system if any. This is necessary to make sure
# the same boost python runtime is shared between every modules, even if linked
# versions are different. It is necessary to share the same boost python
# registers, required for inter-operability between modules.
pyver_suffix = "".join(map(str, _sys.version_info[:2]))
if _sys.platform.startswith('win'):
    lib_prefix = ""
    lib_suffix = ".dll"
elif _sys.platform == 'darwin':
    lib_prefix = "lib"
    lib_suffix = ".dylib"
else:
    lib_prefix = "lib"
    lib_suffix = _get_config_var('SHLIB_SUFFIX')
is_boost_shared = False
for boost_python_lib in (
        f"{lib_prefix}boost_python{pyver_suffix}{lib_suffix}",
        f"{lib_prefix}boost_python3-py{pyver_suffix}{lib_suffix}"):
    try:
        _ctypes.CDLL(boost_python_lib, _ctypes.RTLD_GLOBAL)
        is_boost_shared = True
        break
    except OSError:
        pass

# Check if all the boost python dependencies are already available on the
# system. The system dependencies will be used instead of the one embedded with
# jiminy if and only if all of them are available.
is_dependency_available = any(
    _find_spec(module_name) is not None
    for module_name in ("eigenpy", "hppfcl", "pinocchio"))
if not is_boost_shared and is_dependency_available:
    _logging.warning(
        "Boost::Python not found on the system. Impossible to import "
        "system-wide jiminy dependencies.")


# Since Python >= 3.8, PATH and the current working directory are no longer
# used for DLL resolution on Windows OS. One is expected to explicitly call
# `os.add_dll_directory` instead.
if _sys.platform.startswith('win') and _sys.version_info >= (3, 8):
    _os.add_dll_directory(_os.path.join(_os.path.dirname(__file__), "lib"))
    for path in _os.environ['PATH'].split(_os.pathsep):
        if _os.path.exists(path):
            _os.add_dll_directory(path)

# Import eigenpy first since jiminy depends on it
if is_boost_shared and _find_spec("eigenpy") is not None:
    # Module already available on the system
    _import_module("eigenpy")
else:
    # Importing the embedded version as fallback
    _sys.modules["eigenpy"] = _import_module(".".join((__name__, "eigenpy")))

# Import core submodule.
# For some reason, the serialization registration of pinocchio for hpp-fcl
# `exposeFCL` is conflicting with the one implemented by jiminy. It is
# necessary to import jiminy first to make it work.
with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
    from .core import *  # noqa: F403
    from .core import __version__, __raw_version__

# Import other dependencies to hide boost python converter errors
with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
    for module_name in ("hppfcl", "pinocchio"):
        if is_boost_shared and _find_spec(module_name) is not None:
            _import_module(module_name)
        else:
            _module = _import_module(".".join((__name__, module_name)))
            _sys.modules[module_name] = _module

# Register pinocchio_pywrap and submodules to avoid importing bindings twice,
# which messes up with boost python converters.
submodules = _inspect.getmembers(
    _sys.modules["pinocchio"].pinocchio_pywrap, _inspect.ismodule)
for module_name, module_obj in submodules:
    module_real_path = ".".join(('pinocchio', 'pinocchio_pywrap', module_name))
    _sys.modules[module_real_path] = module_obj
    module_sym_path = ".".join(('pinocchio', module_name))
    _sys.modules[module_sym_path] = module_obj

# Update core submodule to appear as member of current module
__all__ = []
for name in dir(core):
    attrib = getattr(core, name)
    if not name.startswith("_") and isinstance(attrib, type):
        __all__.append(name)
        attrib.__module__ = __name__


# Define helpers to build extension modules
def get_cmake_module_path():
    return _os.path.join(_os.path.dirname(__file__), "cmake")


def get_include():
    return _os.path.join(_os.path.dirname(__file__), "include")


def get_libraries():
    ver_short = '.'.join(__version__.split('.')[:2])
    lib_dir = _os.path.join(_os.path.dirname(__file__), "lib")
    libraries_fullpath = []
    for library_filename in _os.listdir(lib_dir):
        if _re.search(f'\\.(dll|dylib|so.{ver_short})$', library_filename):
            libraries_fullpath.append(_os.path.join(lib_dir, library_filename))
    return ";".join(libraries_fullpath)


__all__ += [
    'get_cmake_module_path',
    'get_include',
    'get_libraries',
    '__version__',
    '__raw_version__'
]
