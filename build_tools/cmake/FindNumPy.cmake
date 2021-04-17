# Find the Python NumPy package
# NumPy_INCLUDE_DIRS
# NumPy_FOUND
# will be set by this script

cmake_minimum_required(VERSION 3.10)

if(Python_EXECUTABLE)
    # Find out the include path
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "try: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
        OUTPUT_VARIABLE __numpy_path)
    # And the version
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "try: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
        OUTPUT_VARIABLE __numpy_version)
else()
    message(FATAL_ERROR "Python executable not found.")
endif()

unset(NumPy_INCLUDE_DIRS)
unset(NumPy_INCLUDE_DIRS CACHE)
find_path(NumPy_INCLUDE_DIRS numpy/arrayobject.h
    HINTS "${__numpy_path}" "${Python_INCLUDE_DIRS}" NO_DEFAULT_PATH)

if(NumPy_INCLUDE_DIRS)
    set(NumPy_FOUND 1 CACHE INTERNAL "Python numpy found")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS NumPy_INCLUDE_DIRS
    VERSION_VAR __numpy_version)
