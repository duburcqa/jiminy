# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_python VERSION ${BUILD_VERSION})

# Compile and install python bindings
add_subdirectory(${LIBRARY_NAME}_pywrap)
