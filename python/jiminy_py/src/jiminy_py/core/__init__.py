import os as _os
import re as _re
import sys as _sys
import ctypes as _ctypes
import inspect as _inspect
import importlib as _importlib
from contextlib import redirect_stderr as _redirect_stderr
from distutils.sysconfig import get_config_var as _get_config_var


# Special dlopen flags are used when loading Boost Python shared library to
# make sure the same boost python runtime is shared between every modules,
# even if linked versions are different. It is necessary to share the same
# boost python registers, required for inter-operability between modules. Note
# that since Python3.8, PATH and the current working directory are no longer
# used for DLL resolution on Windows OS. One is expected to explicitly call
# `os.add_dll_directory` instead.
try:
    pyver_suffix = "".join(map(str, _sys.version_info[:2]))
    if _sys.platform.startswith('win'):
        lib_prefix = ""
        lib_suffix = ".dll"
    else:
        lib_prefix = "lib"
        lib_suffix = _get_config_var('SHLIB_SUFFIX')
    boost_python_lib = f"{lib_prefix}boost_python{pyver_suffix}{lib_suffix}"
    _ctypes.CDLL(boost_python_lib, _ctypes.RTLD_GLOBAL)
except OSError:
    pass

# Fix Dll seach path on windows for Python >= 3.8
if _sys.platform.startswith('win') and _sys.version_info >= (3, 8):
    _os.add_dll_directory(_os.path.join(_os.path.dirname(__file__), "lib"))
    for path in _os.environ['PATH'].split(_os.pathsep):
        if _os.path.exists(path):
            _os.add_dll_directory(path)

# Import dependencies, using embedded versions only if necessary
for module_name in ["eigenpy", "hppfcl", "pinocchio"]:
    if _importlib.util.find_spec(module_name) is not None:
        _importlib.import_module(module_name)
    else:
        _module = _importlib.import_module(".".join((__name__, module_name)))
        _sys.modules[module_name] = _module

# Register pinocchio_pywrap to avoid importing bindings twise, which messes up
# with boost python converters. In addition, submodules search path needs to be
# fixed for releases older than 2.5.6.
submodules = _inspect.getmembers(
    _sys.modules["pinocchio"].pinocchio_pywrap, _inspect.ismodule)
for module_name, module_obj in submodules:
    module_real_path = ".".join(('pinocchio', 'pinocchio_pywrap', module_name))
    _sys.modules[module_real_path] = module_obj
    module_sym_path = ".".join(('pinocchio', module_name))
    _sys.modules[module_sym_path] = module_obj

# Import core submodule once every dependencies have been preloaded.
# Note that embedded dependencies must be imported after core if provided. This
# is necessary to make sure converters that could be missing in already
# available versions are properly initialized. Indeed, because of PEP
# specifications, jiminy_py is compile on `manylinux2014` image, which does not
# support the new C++11 ABI strings. Therefore, if the dependencies have been
# compiled with it, it would result in segmentation faults.
with open(_os.devnull, 'w') as stderr, _redirect_stderr(stderr):
    from .core import *  # noqa: F403
    from .core import __version__, __raw_version__
    for module_name in ["eigenpy", "hppfcl", "pinocchio"]:
        module_path = ".".join((__name__, module_name))
        if _importlib.util.find_spec(module_path) is not None:
            _importlib.import_module(module_path)

# Update core submodule to appear as member of current module
__all__ = []
for name in dir(core):
    attrib = getattr(core, name)
    if not name.startswith("_") and isinstance(attrib, type):
        __all__.append(name)
        attrib.__module__ = __name__

# Patch Pinocchio to avoid loading ground geometry in viewer, and force
# `np.ndarray` type for from/to Python matrix converters.
from . import _pinocchio_init  # noqa


# Define helpers to build extension modules
def get_cmake_module_path():
    return _os.path.join(_os.path.dirname(__file__), "cmake")


def get_include():
    return _os.path.join(_os.path.dirname(__file__), "include")


def get_libraries():
    lib_dir = _os.path.join(_os.path.dirname(__file__), "lib")
    libraries_fullpath = []
    for library_filename in _os.listdir(lib_dir):
        if _re.search(r'\.(dll|so[0-9\.]*)$', library_filename):
            libraries_fullpath.append(_os.path.join(lib_dir, library_filename))
    return ";".join(libraries_fullpath)


__all__ += [
    'get_cmake_module_path',
    'get_include',
    'get_libraries',
    '__version__',
    '__raw_version__'
]
