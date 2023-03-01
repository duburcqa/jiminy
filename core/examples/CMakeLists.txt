# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project name
project(${LIBRARY_NAME}_examples VERSION ${BUILD_VERSION})

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FULL}")

# Sub-projects
add_subdirectory("double_pendulum")
