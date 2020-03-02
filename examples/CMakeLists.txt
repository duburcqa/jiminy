# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_examples VERSION ${BUILD_VERSION})

# Compile and install python bindings
add_subdirectory(double_pendulum)