

# Get the required information to build Python bindings

# Get Python executable
if(DEFINED PYTHON_EXECUTABLE)
    get_filename_component(_PYTHON_PATH "${PYTHON_EXECUTABLE}" DIRECTORY)
    get_filename_component(_PYTHON_NAME "${PYTHON_EXECUTABLE}" NAME)
    find_program(Python_EXECUTABLE "${_PYTHON_NAME}" PATHS "${_PYTHON_PATH}" NO_DEFAULT_PATH)
else()
    if(CMAKE_VERSION VERSION_LESS "3.12.4")
        find_program(Python_EXECUTABLE "python${PYTHON_REQUIRED_VERSION}")
    else()
        if(CMAKE_VERSION VERSION_LESS "3.14")
            set(Python_COMPONENTS_FIND Interpreter)
        else()
            set(Python_COMPONENTS_FIND Interpreter NumPy)
        endif()
        if(PYTHON_REQUIRED_VERSION)
            find_package(Python ${PYTHON_REQUIRED_VERSION} EXACT COMPONENTS ${Python_COMPONENTS_FIND})
        else()
            find_package(Python COMPONENTS ${Python_COMPONENTS_FIND})
        endif()
    endif()
endif()
if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "Python executable not found, CMake will exit.")
endif()

# Get Python version
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                        "import sys ; print(';'.join(map(str, sys.version_info[:3])), end='')"
                OUTPUT_VARIABLE _VERSION)
list(GET _VERSION 0 Python_VERSION_MAJOR)
list(GET _VERSION 1 Python_VERSION_MINOR)
list(GET _VERSION 2 Python_VERSION_PATCH)
set(Python_VERSION "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
if(Python_VERSION_MAJOR EQUAL 3)
    message(STATUS "Found PythonInterp: ${Python_EXECUTABLE} (found version \"${Python_VERSION}\")")
else()
    message(FATAL_ERROR "Python3 is required if BUILD_PYTHON_INTERFACE=ON.")
endif()

# Get Python system and user site-packages
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                        "import sysconfig; print(sysconfig.get_path('purelib'), end='')"
                OUTPUT_VARIABLE Python_SYS_SITELIB)
message(STATUS "Python system site-packages: ${Python_SYS_SITELIB}")
execute_process(COMMAND "${Python_EXECUTABLE}" -m site --user-site
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE Python_USER_SITELIB)
message(STATUS "Python user site-package: ${Python_USER_SITELIB}")

# Check write permissions on Python system site-package to
# determine whether to use user site as fallback.
# It also sets the installation flags
if(WIN32)
    set(HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB FALSE)
else()
    execute_process(COMMAND bash -c
                    "if test -w ${Python_SYS_SITELIB} ; then echo 0; else echo 1; fi"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB)
endif()

set(PYTHON_INSTALL_FLAGS " --prefer-binary ")
if(${HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB})
    set(PYTHON_INSTALL_FLAGS "${PYTHON_INSTALL_FLAGS} --user ")
    set(Python_SITELIB "${Python_USER_SITELIB}")
    message(STATUS "No right on Python system site-packages: ${Python_SYS_SITELIB}.\n"
                   "--   Installing on user site as fallback.")
else()
    set(Python_SITELIB "${Python_SYS_SITELIB}")
endif()

# Get PYTHON_EXT_SUFFIX
set(PYTHON_EXT_SUFFIX "")
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                        "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'), end='')"
                OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)
if("${PYTHON_EXT_SUFFIX}" STREQUAL "")
    if(WIN32)
        set(PYTHON_EXT_SUFFIX ".pyd")
    else()
        set(PYTHON_EXT_SUFFIX ".so")
    endif()
endif()

# Include Python headers
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                        "import distutils.sysconfig as sysconfig; print(sysconfig.get_python_inc(), end='')"
                OUTPUT_VARIABLE Python_INCLUDE_DIRS)

# Add Python library directory to search path on Windows
if(WIN32)
    get_filename_component(PYTHON_ROOT ${Python_SYS_SITELIB} DIRECTORY)
    get_filename_component(PYTHON_ROOT ${PYTHON_ROOT} DIRECTORY)
    link_directories("${PYTHON_ROOT}/libs/")
    message(STATUS "Found PythonLibraryDirs: ${PYTHON_ROOT}/libs/")
endif()

# Define Python_NumPy_INCLUDE_DIRS if necessary
if (NOT Python_NumPy_INCLUDE_DIRS)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "import numpy; print(numpy.get_include(), end='')\n"
        OUTPUT_VARIABLE __numpy_path)
    find_path(Python_NumPy_INCLUDE_DIRS numpy/arrayobject.h
        HINTS "${__numpy_path}" "${Python_INCLUDE_DIRS}" NO_DEFAULT_PATH)
endif()

# Define BOOST_PYTHON_LIB
find_package(Boost QUIET REQUIRED)
if(${Boost_MINOR_VERSION} GREATER_EQUAL 67)
    # Make sure the shared library is found rather than the static one
    set(BOOST_USE_STATIC_LIBS_OLD ${Boost_USE_STATIC_LIBS})
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_LIB_PREFIX "")
    unset(Boost_LIBRARIES)
    find_package(Boost REQUIRED COMPONENTS
                "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}"
                "numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
    set(BOOST_PYTHON_LIB "${Boost_LIBRARIES}")
    unset(Boost_LIBRARIES)
    if(WIN32)
        set(Boost_LIB_PREFIX "lib")
    endif()
    set(Boost_USE_STATIC_LIBS ${BOOST_USE_STATIC_LIBS_OLD})
else()
    set(BOOST_PYTHON_LIB "boost_numpy3;boost_python3")
endif()
message(STATUS "Boost Python Libs: ${BOOST_PYTHON_LIB}")

# Define Python install helpers
function(deployPythonPackage)
    # The input arguments are [TARGET_NAME...]
    foreach(TARGET_NAME IN LISTS ARGN)
        install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${PYTHON_INSTALL_FLAGS} --upgrade .
                                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME})")
    endforeach()
endfunction()

function(deployPythonPackageDevelop)
    # The input arguments are [TARGET_NAME...]
    foreach(TARGET_NAME IN LISTS ARGN)
        install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${PYTHON_INSTALL_FLAGS} -e .
                                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${TARGET_NAME})")
    endforeach()
endfunction()
