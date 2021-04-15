import os
import sys
import ctypes
import inspect
import importlib
from contextlib import redirect_stderr
from distutils.sysconfig import get_config_var


# Special dlopen flags are used when loading Boost Python shared library to
# make sure the same boost python runtime is shared between every modules,
# even if linked versions are different. It is necessary to share the same
# boost python registers, required for inter-operability between modules. Note
# that since Python3.8, PATH and the current working directory are no longer
# used for DLL resolution on Windows OS. One is expected to explicitly call
# `os.add_dll_directory` instead.
try:
    pyver_suffix = "".join(map(str, sys.version_info[:2]))
    if sys.platform.startswith('win'):
        lib_prefix = ""
        lib_suffix = ".dll"
    else:
        lib_prefix = "lib"
        lib_suffix = get_config_var('SHLIB_SUFFIX')
    boost_python_lib = f"{lib_prefix}boost_python{pyver_suffix}{lib_suffix}"
    ctypes.CDLL(boost_python_lib, ctypes.RTLD_GLOBAL)
except OSError:
    pass

# Fix Dll seach path on windows for Python >= 3.8
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    for path in os.environ['PATH'].split(os.pathsep):
        if os.path.exists(path):
            os.add_dll_directory(path)

# Import dependencies, using embedded versions only if necessary
for module_name in ["eigenpy", "hppfcl", "pinocchio"]:
    if importlib.util.find_spec(module_name) is not None:
        importlib.import_module(module_name)
    else:
        _module = importlib.import_module(".".join((__name__, module_name)))
        sys.modules[module_name] = _module

# Register pinocchio_pywrap to avoid importing bindings twise, which messes up
# with boost python converters. In addition, submodules search path needs to be
# fixed for releases older than 2.5.6.
submodules = inspect.getmembers(
    sys.modules["pinocchio"].pinocchio_pywrap, inspect.ismodule)
for module_name, module_obj in submodules:
    module_real_path = ".".join(('pinocchio', 'pinocchio_pywrap', module_name))
    sys.modules[module_real_path] = module_obj
    module_sym_path = ".".join(('pinocchio', module_name))
    sys.modules[module_sym_path] = module_obj

# Import core submodule once every dependencies have been preloaded.
# Note that embedded dependencies must be imported after core if provided. This
# is necessary to make sure converters that could be missing in already
# available versions are properly initialized. Indeed, because of PEP
# specifications, jiminy_py is compile on `manylinux2014` image, which does not
# support the new C++11 ABI strings. Therefore, if the dependencies have been
# compiled with it, it would result in segmentation faults.
with open(os.devnull, 'w') as stderr, redirect_stderr(stderr):
    from . import core
    from .core import __version__, __raw_version__
    for module_name in ["eigenpy", "hppfcl", "pinocchio"]:
        module_path = ".".join((__name__, module_name))
        if importlib.util.find_spec(module_path) is not None:
            importlib.import_module(module_path)

# Patch Pinocchio to avoid loading ground geometry in viewer, and force
# `np.ndarray` type for from/to Python matrix converters.
from . import _pinocchio_init  # noqa

# Define include and lib path
def get_include():
    return os.path.join(os.path.dirname(__file__), "include")

def get_library():
    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    return os.path.join(lib_dir, os.listdir(lib_dir)[0])
