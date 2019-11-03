# Find the Python NumPy package
# NUMPY_INCLUDE_DIRS
# NUMPY_FOUND
# will be set by this script

cmake_minimum_required(VERSION 2.6)

if (PYTHON_EXECUTABLE)
    # Find out the include path
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
        OUTPUT_VARIABLE __numpy_path)
    # And the version
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "from __future__ import print_function\ntry: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
        OUTPUT_VARIABLE __numpy_version)
else(PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Python executable not found.")
endif(PYTHON_EXECUTABLE)
        
find_path(NUMPY_INCLUDE_DIRS numpy/arrayobject.h
    HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

if(NUMPY_INCLUDE_DIRS)
    set(NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(NUMPY_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS NUMPY_INCLUDE_DIRS
    VERSION_VAR __numpy_version)