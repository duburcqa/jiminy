# Minimum version required
cmake_minimum_required (VERSION 3.10)

# Check if network is available before compiling external projects
if(WIN32)
    find_program(HAS_PING "ping")
endif()
if(HAS_PING)
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
    message(WARNING "No internet connection. Not building external projects.")
endif()

# Set default warning flags
if(WIN32)
    # It would be great to have the same quality standard for Windows but
    # system inlcude of dependencies is not working properly so far.
    # - Disabling warning 4820 generated by std time.h, which is included by std chrono
    set(WARN_FULL "/W2 /wd4068 /wd4715 /wd4820 /wd4244 /wd4005 /wd4834 /WX")
else()
    set(WARN_FULL "-Wall -Wextra -Weffc++ -pedantic -pedantic-errors \
                   -Wcast-align -Wcast-qual -Wfloat-equal -Wformat=2 \
                   -Wformat-nonliteral -Wformat-security -Wformat-y2k \
                   -Wimport -Winit-self -Winvalid-pch -Wlong-long \
                   -Wmissing-field-initializers -Wmissing-noreturn \
                   -Wmissing-format-attribute -Wctor-dtor-privacy \
                   -Wpointer-arith -Wold-style-cast -Wpacked \
                   -Woverloaded-virtual -Wredundant-decls -Wshadow \
                   -Wstack-protector -Wstrict-aliasing=2 \
                   -Wswitch-default -Wswitch-enum -Wunreachable-code \
                   -Wunused -Wundef -Wdisabled-optimization \
                   -Wmissing-braces -Wtrigraphs -Wparentheses \
                   -Wwrite-strings -Wsequence-point -Wdeprecated \
                   -Wconversion -Wdelete-non-virtual-dtor \
                   -Wno-sign-conversion -Wno-non-virtual-dtor \
                   -Wno-delete-abstract-non-virtual-dtor \
                   -Wno-delete-non-abstract-non-virtual-dtor \
                   -Wno-unknown-pragmas -Wno-undefined-var-template \
                   -Werror=return-type")
endif()

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
option(Boost_NO_WARN_NEW_VERSIONS "Do not warn about unknown boost version." ON)
option(Boost_NO_SYSTEM_PATHS "Do not search for boost on system." ON)
if(Boost_NO_SYSTEM_PATHS AND (NOT DEFINED BOOST_ROOT))
    set(BOOST_ROOT "/opt/install/pc/")
endif()

# Add Fallback search paths for headers and libraries
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
