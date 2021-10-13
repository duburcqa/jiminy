# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_examples VERSION ${BUILD_VERSION})

# Sub-projects
add_subdirectory("cpp/double_pendulum")
