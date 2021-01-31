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
               -Wunreachable-code -Wunused -Wundef -Wlogical-op \
               -Wdisabled-optimization -Wmissing-braces -Wtrigraphs \
               -Wparentheses -Wwrite-strings -Werror=return-type \
               -Wsequence-point -Wdeprecated")  # -Wconversion

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
