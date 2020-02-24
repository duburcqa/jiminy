# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(${LIBRARY_NAME}_python VERSION ${BUILD_VERSION})

# Compile and install python bindings
add_subdirectory(${LIBRARY_NAME}_pywrap)

# Copy Python bindings as Python native 'core' submodule
file(GLOB_RECURSE ${PROJECT_NAME}_NATIVE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_NAME}_pywrap/src/python/*.py")
install(FILES ${${PROJECT_NAME}_NATIVE_SOURCES}
        DESTINATION "${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core"
)
install(CODE "file(COPY \"${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_pywrap/\"
                   DESTINATION \"${CMAKE_BINARY_DIR}/pypi/${LIBRARY_NAME}_py/src/${LIBRARY_NAME}_py/core/\"
                   FILES_MATCHING PATTERN \"lib${LIBRARY_NAME}_pywrap.so*\")"
)

# Build Python wheels and archives
buildPythonWheel(python/${LIBRARY_NAME}_py)
buildPythonWheel(gym_${LIBRARY_NAME})

# Install Python package(s)
deployPythonPackage(${LIBRARY_NAME}_py)

# Install Python package(s) in develop mode
deployPythonPackageDevelop(gym_${LIBRARY_NAME})