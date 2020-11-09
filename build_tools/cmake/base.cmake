# Minimum version required
cmake_minimum_required (VERSION 3.10)

# Check if network is available for compiling gtest
if(NOT WIN32)
    unset(BUILD_OFFLINE)
    unset(BUILD_OFFLINE CACHE)
    execute_process(COMMAND bash -c
                            "if ping -q -c 1 -W 1 8.8.8.8 ; then echo 0; else echo 1; fi"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE BUILD_OFFLINE)
else()
    set(BUILD_OFFLINE 0)
endif()
if(${BUILD_OFFLINE})
    message(STATUS "No internet connection. Not building external projects.")
endif()

# Set default warning flags
set(WARN_FULL "-Wall -Wextra -Weffc++ -pedantic -pedantic-errors \
               -Wcast-align -Wcast-qual -Wfloat-equal -Wformat=2 \
               -Wformat-nonliteral -Wformat-security -Wformat-y2k \
               -Wimport -Winit-self -Winvalid-pch -Wlong-long \
               -Wmissing-field-initializers -Wmissing-format-attribute \
               -Wmissing-noreturn -Wpacked -Wpointer-arith \
               -Wredundant-decls -Wshadow -Wstack-protector \
               -Wstrict-aliasing=2 -Wswitch-default -Wswitch-enum \
               -Wunreachable-code -Wunused"
)

# Shared libraries need PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define GNU standard installation directories
include(GNUInstallDirs)

# Custom cmake module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

# Build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Define search strategy for Boost package
# TODO: Remove hard-coded path
option(Boost_NO_SYSTEM_PATHS "Do not search for boost on system." ON)
if(Boost_NO_SYSTEM_PATHS AND (NOT DEFINED BOOST_ROOT))
    set(BOOST_ROOT "/opt/install/pc/")
endif()

# Determine if the old legacy old for Ubuntu 18 must be used.
# It will not use "find_package" but instead plain old "link_directories"
# and "include_directories" directives.
# Thus it requires the dependencies to be installed from robotpkg apt repository.
# TODO: Remove legacy mode after dropping support of Ubuntu 18 and moving to
# Eigen >= 3.3.7, Boost >= 1.71, and Pinocchio >=2.4.0.
find_package(Boost QUIET)
string(REPLACE "_" "." BOOST_VERSION "${Boost_LIB_VERSION}")
if("${BOOST_VERSION}" VERSION_LESS "1.71.0")
    set(LEGACY_MODE ON)
endif()
if(LEGACY_MODE)
    if(WIN32)
        message(FATAL_ERROR "Boost >= 1.71.0 required.")
    else()
        message(STATUS "Old boost version detected. Fallback to Ubuntu 18 legacy mode. Make sure depedencies have been installed using apt-get.")
    endif()
endif()

# Add Fallback search paths for headers and libraries
if(LEGACY_MODE)
    link_directories("/opt/openrobots/lib/")
    link_directories("/opt/install/pc/lib/")
    include_directories(SYSTEM "/opt/openrobots/include/")
    include_directories(SYSTEM "/opt/install/pc/include/")
    include_directories(SYSTEM "/opt/install/pc/include/eigen3/")
endif()
list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots/")

# Get the required information to build Python bindings
if(BUILD_PYTHON_INTERFACE)
    # Get Python executable and version
    if(NOT DEFINED PYTHON_EXECUTABLE)
        if(${CMAKE_VERSION} VERSION_LESS "3.12.4")
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
    endif()

    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                            "import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _VERSION)
    string(REPLACE ";" "." PYTHON_VERSION_STRING "${_VERSION}")
    list(GET _VERSION 0 PYTHON_VERSION_MAJOR)
    list(GET _VERSION 1 PYTHON_VERSION_MINOR)
    list(GET _VERSION 2 PYTHON_VERSION_PATCH)
    set(PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    if(${PYTHON_VERSION_MAJOR} EQUAL 3)
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
    if(NOT WIN32)
        execute_process(COMMAND bash -c
                                "if test -w ${PYTHON_SYS_SITELIB} ; then echo 0; else echo 1; fi"
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        OUTPUT_VARIABLE HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB)
    else()
        set(HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB FALSE)
    endif()

    set(PYTHON_INSTALL_FLAGS "--upgrade ")
    if(${HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB})
        set(PYTHON_INSTALL_FLAGS "${PYTHON_INSTALL_FLAGS} --user ")
        set(PYTHON_SITELIB "${PYTHON_USER_SITELIB}")
        message(STATUS "No right on Python system site-packages: ${PYTHON_SYS_SITELIB}. Installing on user site as fallback.")
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
        if(NOT WIN32)
            set(PYTHON_EXT_SUFFIX ".so")
        else()
            set(PYTHON_EXT_SUFFIX ".pyd")
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
        find_package(Boost REQUIRED COMPONENTS
                    "python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}"
                    "numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
        set(BOOST_PYTHON_LIB "${Boost_LIBRARIES}")
        unset(Boost_LIBRARIES)
        unset(Boost_LIBRARIES CACHE)
    else()
        set(BOOST_PYTHON_LIB "boost_numpy3;boost_python3")
    endif()
    message(STATUS "Boost Python Libs: ${BOOST_PYTHON_LIB}")

    # Define Python install helpers
    function(deployPythonPackage)
        # The input arguments are [TARGET_NAME...]
        foreach(TARGET_NAME IN LISTS ARGN)
            install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PYTHON_INSTALL_FLAGS} .
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
endif()

# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)

# Add a helper to link target libraries as system dependencies to avoid generating warnings
function(target_link_libraries_system target)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
    target_link_libraries(${target} ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)
