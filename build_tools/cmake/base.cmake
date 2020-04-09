# Minimum version required
cmake_minimum_required (VERSION 3.10)

# Check if network is available for compiling gtest
option(BUILD_TESTING "Build the gtest testing tree." ON)
if(BUILD_TESTING)
    if(NOT WIN32)
        unset(BUILD_OFFLINE)
        unset(BUILD_OFFLINE CACHE)
        execute_process(COMMAND bash -c
                                "if ping -q -c 1 -W 1 8.8.8.8 ; then echo 0; else echo 1; fi"
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        OUTPUT_VARIABLE BUILD_OFFLINE)
    else(NOT WIN32)
        set(BUILD_OFFLINE 0)
    endif(NOT WIN32)
    if(${BUILD_OFFLINE})
        message("-- No internet connection. Not building external projects.")
    endif()
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
               -Wunreachable-code -Wunused -Wunused-parameter"
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

# Determine if Python bindings must be generated
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_EXAMPLES "Build the C++ examples" ON)

# Add missing include & lib directory(ies)
# TODO: Remove hard-coded paths after support of find_package for Eigen and Pinocchio,
# namely after migration to Eigen 3.3.7 / Boost 1.71, and Pinocchio 2.4.X
if(NOT WIN32)
    link_directories(SYSTEM /opt/openrobots/lib)
    link_directories(SYSTEM /opt/install/pc/lib)
    include_directories(SYSTEM /opt/openrobots/include/)
    include_directories(SYSTEM /opt/install/pc/include/)
    include_directories(SYSTEM /opt/install/pc/include/eigen3/)
else(NOT WIN32)
    link_directories(SYSTEM "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    include_directories(SYSTEM "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
endif(NOT WIN32)

# Define search strategy for Boost package and find it
# TODO: Remove hard-coded path
option(Boost_NO_SYSTEM_PATHS "Do not search for boost on system." ON)
if(Boost_NO_SYSTEM_PATHS AND (NOT DEFINED BOOST_ROOT))
    set(BOOST_ROOT "/opt/install/pc/")
endif()
find_package(Boost REQUIRED)

if(BUILD_PYTHON_INTERFACE)
    # Get Python executable and version
    unset(PYTHON_EXECUTABLE)
    unset(PYTHON_EXECUTABLE CACHE)
    if(NOT WIN32)
        find_program(PYTHON_EXECUTABLE python)
    else(NOT WIN32)
        execute_process(COMMAND python -c
                                "import sys; sys.stdout.write(sys.executable)"
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        OUTPUT_VARIABLE PYTHON_EXECUTABLE)
    endif(NOT WIN32)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                            "import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _VERSION)
    string(REPLACE ";" "." PYTHON_VERSION_STRING "${_VERSION}")
    list(GET _VERSION 0 PYTHON_VERSION_MAJOR)
    list(GET _VERSION 1 PYTHON_VERSION_MINOR)
    list(GET _VERSION 2 PYTHON_VERSION_PATCH)
    set(PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    message("-- Found PythonInterp: ${PYTHON_EXECUTABLE} (found version \"${PYTHON_VERSION_STRING}\")")

    ## Get Python system and user site-packages
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                            "import sysconfig; print(sysconfig.get_paths()['purelib'])"
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE PYTHON_SYS_SITELIB)
    message("-- Python system site-packages: ${PYTHON_SYS_SITELIB}")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m site --user-site
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PYTHON_USER_SITELIB)
    message("-- Python user site-package: ${PYTHON_USER_SITELIB}")
    
    # Check write permissions on Python system site-package to 
    # determine whether to use user site as fallback.
    # It also sets the installation flags
    if(NOT WIN32)
        execute_process(COMMAND bash -c
                                "if test -w ${PYTHON_SYS_SITELIB} ; then echo 0; else echo 1; fi"
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        OUTPUT_VARIABLE HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB)
    else(NOT WIN32)
        set(HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB FALSE)
    endif(NOT WIN32)
                    
    set(PYTHON_INSTALL_FLAGS "--upgrade ")
    if(${HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB})
        set(PYTHON_INSTALL_FLAGS "${PYTHON_INSTALL_FLAGS} --user ")
        set(PYTHON_SITELIB "${PYTHON_USER_SITELIB}")
        message("-- No right on Python system site-packages: ${PYTHON_SYS_SITELIB}. Installing on user site as fallback.")
    else(${HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB})
        set(PYTHON_SITELIB "${PYTHON_SYS_SITELIB}")
    endif(${HAS_NO_WRITE_PERMISSION_ON_PYTHON_SYS_SITELIB})

    # Get PYTHON_EXT_SUFFIX
    set(PYTHON_EXT_SUFFIX "")
    if(PYTHON_VERSION_MAJOR EQUAL 3)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                                "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)
    endif(PYTHON_VERSION_MAJOR EQUAL 3)
    if("${PYTHON_EXT_SUFFIX}" STREQUAL "")
        if(NOT WIN32)
            SET(PYTHON_EXT_SUFFIX ".so")
        else(WIN32)
            SET(PYTHON_EXT_SUFFIX ".pyd")
        endif(WIN32)
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
        link_directories(SYSTEM "${PYTHON_ROOT}/libs/")
        message("-- Found PythonLibraryDirs: ${PYTHON_ROOT}/libs/")
    endif()

    # Define NUMPY_INCLUDE_DIRS
    find_package(NumPy REQUIRED)

    # Define Python install helpers
    function(deployPythonPackage TARGET_NAME)
        install(CODE "execute_process(COMMAND pip install ${PYTHON_INSTALL_FLAGS} .
                                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pypi/${TARGET_NAME})")
    endfunction()

    function(deployPythonPackageDevelop TARGET_NAME)
        install (CODE "EXECUTE_PROCESS (COMMAND pip install ${PYTHON_INSTALL_FLAGS} -e .
                                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${TARGET_NAME})")
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
