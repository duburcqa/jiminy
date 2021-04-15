

# Get the required information to build Python bindings

# Get Python executable and version
if(NOT DEFINED PYTHON_EXECUTABLE)
    if(CMAKE_VERSION VERSION_LESS "3.12.4")
        if(PYTHON_REQUIRED_VERSION)
            message(FATAL_ERROR "Impossible to handle PYTHON_REQUIRED_VERSION for cmake older than 3.12.4, Cmake will exit.")
        endif()
        find_program(PYTHON_EXECUTABLE python)
        if(NOT PYTHON_EXECUTABLE)
            message(FATAL_ERROR "No Python executable found, CMake will exit.")
        endif()
    else()
        if(PYTHON_REQUIRED_VERSION)
            find_package(Python ${PYTHON_REQUIRED_VERSION} EXACT REQUIRED COMPONENTS Interpreter)
        else()
            find_package(Python REQUIRED COMPONENTS Interpreter)
        endif()
        set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
    endif()
else()
    find_program(PYTHON_EXECUTABLE python PATHS "${PYTHON_EXECUTABLE}" NO_DEFAULT_PATH)
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "User-specified Python executable not valid, CMake will exit.")
    endif()
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                        "import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE _VERSION)
list(GET _VERSION 0 PYTHON_VERSION_MAJOR)
list(GET _VERSION 1 PYTHON_VERSION_MINOR)
list(GET _VERSION 2 PYTHON_VERSION_PATCH)
set(PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
if(PYTHON_VERSION_MAJOR EQUAL 3)
    message(STATUS "Found PythonInterp: ${PYTHON_EXECUTABLE} (found version \"${PYTHON_VERSION_STRING}\")")
else()
    message(FATAL_ERROR "Python3 is required if BUILD_PYTHON_INTERFACE=ON.")
endif()

# Get Python system and user site-packages
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                        "import sysconfig; print(sysconfig.get_paths()['purelib'])"
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PYTHON_SYS_SITELIB)
message(STATUS "Python system site-packages: ${PYTHON_SYS_SITELIB}")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m site --user-site
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PYTHON_USER_SITELIB)
message(STATUS "Python user site-package: ${PYTHON_USER_SITELIB}")

# Check write permissions on Python system site-package to
# determine whether to use user site as fallback.
# It also sets the installation flags
if(WIN32)
    set(HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB FALSE)
else()
    execute_process(COMMAND bash -c
                            "if test -w ${PYTHON_SYS_SITELIB} ; then echo 0; else echo 1; fi"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB)
endif()

if(${HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB})
    set(PYTHON_INSTALL_FLAGS " --user ")
    set(PYTHON_SITELIB "${PYTHON_USER_SITELIB}")
    message(STATUS "No right on Python system site-packages: ${PYTHON_SYS_SITELIB}.\n"
                   "--   Installing on user site as fallback.")
else()
    set(PYTHON_SITELIB "${PYTHON_SYS_SITELIB}")
endif()

# Get PYTHON_EXT_SUFFIX
set(PYTHON_EXT_SUFFIX "")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                        "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)
if("${PYTHON_EXT_SUFFIX}" STREQUAL "")
    if(WIN32)
        set(PYTHON_EXT_SUFFIX ".pyd")
    else()
        set(PYTHON_EXT_SUFFIX ".so")
    endif()
ENDIF()

# Include Python headers
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                        "import distutils.sysconfig as sysconfig; print(sysconfig.get_python_inc())"
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)

# Add Python library directory to search path on Windows
if(WIN32)
    get_filename_component(PYTHON_ROOT ${PYTHON_SYS_SITELIB} DIRECTORY)
    get_filename_component(PYTHON_ROOT ${PYTHON_ROOT} DIRECTORY)
    link_directories("${PYTHON_ROOT}/libs/")
    message(STATUS "Found PythonLibraryDirs: ${PYTHON_ROOT}/libs/")
endif()

# Define NUMPY_INCLUDE_DIRS
find_package(NumPy REQUIRED)

# Define BOOST_PYTHON_LIB
find_package(Boost QUIET REQUIRED)
if(${Boost_MINOR_VERSION} GREATER_EQUAL 67)
    # Make sure the shared library is found rather than the static one
    set(BOOST_USE_STATIC_LIBS_OLD ${Boost_USE_STATIC_LIBS})
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_LIB_PREFIX "")
    unset(Boost_LIBRARIES)
    find_package(Boost REQUIRED COMPONENTS
                "python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}"
                "numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
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
        install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PYTHON_INSTALL_FLAGS} --upgrade .
                                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME})")
    endforeach()
endfunction()

function(deployPythonPackageDevelop)
    # The input arguments are [TARGET_NAME...]
    foreach(TARGET_NAME IN LISTS ARGN)
        install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PYTHON_INSTALL_FLAGS} -e .
                                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${TARGET_NAME})")
    endforeach()
endfunction()
