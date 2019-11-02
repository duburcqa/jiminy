# Minimum version required
cmake_minimum_required (VERSION 3.10)

# Set various flags
set(WARN_FULL "-Wall -Wextra -Weffc++ -pedantic -pedantic-errors \
               -Wcast-align -Wcast-qual -Wfloat-equal -Wformat=2 \
               -Wformat-nonliteral -Wformat-security -Wformat-y2k \
               -Wimport -Winit-self -Winvalid-pch -Wlong-long \
               -Wmissing-field-initializers -Wmissing-format-attribute \
               -Wmissing-include-dirs -Wmissing-noreturn -Wpacked \
               -Wpointer-arith -Wredundant-decls -Wshadow -Wstack-protector \
               -Wstrict-aliasing=2 -Wswitch-default -Wswitch-enum \
               -Wunreachable-code -Wunused -Wunused-parameter"
)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wfatal-errors ${WARN_FULL}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${WARN_FULL} -Wno-unused-parameter")

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

# Set Python version
unset(PYTHON_EXECUTABLE)
unset(PYTHON_EXECUTABLE CACHE)
find_program(PYTHON_EXECUTABLE python)
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                        "import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
                OUTPUT_VARIABLE _VERSION)
string(REPLACE ";" "." PYTHON_VERSION_STRING "${_VERSION}")
list(GET _VERSION 0 PYTHON_VERSION_MAJOR)
list(GET _VERSION 1 PYTHON_VERSION_MINOR)
set(PYTHON_VERSION ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
get_filename_component(PYTHON_ROOT ${PYTHON_EXECUTABLE} DIRECTORY)
get_filename_component(PYTHON_ROOT ${PYTHON_ROOT} DIRECTORY)
set(PYTHON_SITELIB ${PYTHON_ROOT}/lib/python${PYTHON_VERSION}/site-packages)

# Add Python dependencies
set(PYTHON_INCLUDE_DIRS "")
if(EXISTS ${PYTHON_ROOT}/include/python${PYTHON_VERSION})
    list(APPEND PYTHON_INCLUDE_DIRS ${PYTHON_ROOT}/include/python${PYTHON_VERSION})
else(EXISTS ${PYTHON_ROOT}/include/python${PYTHON_VERSION})
    list(APPEND PYTHON_INCLUDE_DIRS ${PYTHON_ROOT}/include/python${PYTHON_VERSION}m)
endif(EXISTS ${PYTHON_ROOT}/include/python${PYTHON_VERSION})

# Find Numpy and add it as a dependency
find_package(NumPy REQUIRED)

if(${PYTHON_VERSION_MAJOR} EQUAL 3)
    set(BOOST_PYTHON_LIB "boost_numpy3")
    list(APPEND BOOST_PYTHON_LIB "boost_python3")
else(${PYTHON_VERSION_MAJOR} EQUAL 3)
    set(BOOST_PYTHON_LIB "boost_numpy")
    list(APPEND BOOST_PYTHON_LIB "boost_python")
endif(${PYTHON_VERSION_MAJOR} EQUAL 3)

# Add missing include & lib directory(ies)
link_directories(SYSTEM /opt/openrobots/lib)
link_directories(SYSTEM /opt/install/pc/lib)
include_directories(SYSTEM /opt/install/pc/include/)
include_directories(SYSTEM /opt/install/pc/include/eigen3/)

# Python install helper
function(deploy_python_gym_package TARGET_NAME)
    install (CODE "EXECUTE_PROCESS (COMMAND pip install -e .
                                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME})")
endfunction()

# Add utils to define package version
include(CMakePackageConfigHelpers)