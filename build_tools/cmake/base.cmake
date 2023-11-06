# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Set find_package strategy to look for both upper-case and case-preserved variables
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.27.0)
    cmake_policy(SET CMP0144 NEW)
endif()

# MSVC runtime library flags are defined by 'CMAKE_MSVC_RUNTIME_LIBRARY'
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.15.7)
    cmake_policy(SET CMP0091 NEW)
endif()

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
if(MSVC)
    # It would be great to have the same quality standard for Windows but
    # system include of dependencies is not working properly so far.
    set(WARN_FULL "/W2 /wd4068 /wd4715 /wd4820 /wd4244 /wd4005 /wd4834 /wd5105 /wd4251 /WX")
else()
    set(WARN_FULL "-Wall -Wextra -pedantic \
                   -pedantic-errors -Wcast-align -Wcast-qual \
                   -Wfloat-equal -Wformat=2 -Wformat-nonliteral \
                   -Wformat-security -Wformat-y2k -Wimport \
                   -Winit-self -Winvalid-pch -Wlong-long \
                   -Wmissing-field-initializers -Wmissing-noreturn \
                   -Wmissing-format-attribute -Wctor-dtor-privacy \
                   -Wpointer-arith -Wold-style-cast -Wpacked \
                   -Wshadow -Woverloaded-virtual -Wredundant-decls \
                   -Wstack-protector -Wstrict-aliasing=2 \
                   -Wswitch-default -Wswitch-enum -Wunreachable-code \
                   -Wunused -Wundef -Wdisabled-optimization \
                   -Wmissing-braces -Wtrigraphs -Wparentheses \
                   -Wwrite-strings -Wsequence-point -Wdeprecated \
                   -Wconversion -Wdelete-non-virtual-dtor \
                   -Wno-sign-conversion -Wno-non-virtual-dtor \
                   -Wno-unknown-pragmas -Wno-unknown-warning-option \
                   -Wno-unknown-warning -Wno-undefined-var-template \
                   -Wno-long-long -Wno-error=maybe-uninitialized \
                   -Wno-error=uninitialized -Wno-error=deprecated \
                   -Wno-error=array-bounds")
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

# Add a helper to link target libraries as system dependencies to avoid generating warnings
function(target_link_libraries_system target)
    set(libs ${ARGN})
    foreach(lib ${libs})
        get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
        target_link_libraries(${target} ${lib})
    endforeach(lib)
endfunction(target_link_libraries_system)
