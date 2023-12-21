# pylint: disable=missing-module-docstring,invalid-name,import-error,
# pylint: disable=undefined-variable,wrong-import-position
import os as _os
import re as _re
import sys as _sys
import ctypes as _ctypes
import inspect as _inspect
import logging as _logging
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
from sysconfig import get_config_var as _get_config_var


# Ordered list of modules on which jiminy depends
_JIMINY_REQUIRED_MODULES = ("eigenpy", "hppfcl", "pinocchio")

# Check if all the jiminy dependencies are already available on the system.
# The system dependencies will be used instead of the one bundled with
# jiminy if and only if all of them are available and boost python is shared.
_are_all_dependencies_available = any(
    _find_spec(_module_name) is not None
    for _module_name in _JIMINY_REQUIRED_MODULES)

# Load Boost Python shared library if necessary.
# This is a pre-requisite for using non-shipped dependencies while ensuring
# inter-operability between modules.
_is_boost_shared = False
if "JIMINY_FORCE_STANDALONE" not in _os.environ:
    # Try to load Boost Python shared library if available.
    # FIXME: This mechanism is fundamentally unsafe and partially broken.
    # It does work at all on OSX and causes segfault at when when mixing
    # modules linking against Boost>=1.78 and <1.78.
    if _sys.platform != 'darwin' and _are_all_dependencies_available:
        # Determine candidate names
        _pyver_suffix = "".join(map(str, _sys.version_info[:2]))
        if _sys.platform.startswith('win'):
            _lib_prefix = ""
            _lib_suffix = ".dll"
        else:
            _lib_prefix = "lib"
            _lib_suffix = _get_config_var('SHLIB_SUFFIX')

        # Try loading the shared library.
        # Note that a special dlopen flag must be set to make sure the same
        # boost python runtime is shared between every modules, even if linked
        # versions are supposed to be different. It is necessary to share the
        # same boost python registers, required for inter-operability.
        for _boost_python_lib in (
                f"{_lib_prefix}boost_python{_pyver_suffix}{_lib_suffix}",
                f"{_lib_prefix}boost_python3-py{_pyver_suffix}{_lib_suffix}"):
            try:
                _ctypes.CDLL(_boost_python_lib, _ctypes.RTLD_GLOBAL)
                _is_boost_shared = True
                break
            except OSError:
                pass

    # Force importing bundled dependencies in Boost::Python was not found
    if not _is_boost_shared and _are_all_dependencies_available:
        _logging.warning(
            "Boost.Python cannot be shared between modules. Importing bundled "
            "jiminy dependencies instead of system-wide install as fallback.")

# The env variable PATH and the current working directory are ignored by
# default for DLL resolution on Windows OS.
if _sys.platform.startswith('win'):
    _os.add_dll_directory(  # type: ignore[attr-defined]
        _os.path.join(_os.path.dirname(__file__), "lib"))
    for path in _os.environ['PATH'].split(_os.pathsep):
        if _os.path.exists(path):
            _os.add_dll_directory(path)  # type: ignore[attr-defined]

# Import all dependencies in the right order
for _module_name in _JIMINY_REQUIRED_MODULES:
    if _is_boost_shared and _are_all_dependencies_available:
        _import_module(_module_name)
    else:
        _module = _import_module(".".join((__name__, _module_name)))
        _sys.modules[_module_name] = _module

# Register pinocchio_pywrap and submodules to avoid importing bindings twice,
# which messes up with boost python converters.
_submodules = _inspect.getmembers(
    _sys.modules["pinocchio"].pinocchio_pywrap, _inspect.ismodule)
for _module_name, _module_obj in _submodules:
    _module_real_path = ".".join((
        'pinocchio', 'pinocchio_pywrap', _module_name))
    _sys.modules[_module_real_path] = _module_obj
    _module_sym_path = ".".join(('pinocchio', _module_name))
    _sys.modules[_module_sym_path] = _module_obj

# Import core submodule
from .core import *  # noqa: F403
from .core import __version__, __raw_version__

# Update core submodule to appear as member of current module
__all__ = []
for name in dir(core):  # type: ignore[name-defined]
    attrib = getattr(core, name)  # type: ignore[name-defined]
    if not name.startswith("_"):
        __all__.append(name)
    try:
        attrib.__module__ = __name__
    except AttributeError:
        pass


# Define helpers to build extension modules
def get_cmake_module_path() -> str:
    """ TODO: Write documentation.
    """
    return _os.path.join(_os.path.dirname(__file__), "cmake")


def get_include() -> str:
    """ TODO: Write documentation.
    """
    return _os.path.join(_os.path.dirname(__file__), "include")


def get_libraries() -> str:
    """ TODO: Write documentation.
    """
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
