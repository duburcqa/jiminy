# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Set the build version
set(BUILD_VERSION 0.9.0)

# Project and library name
project(jiminy VERSION ${BUILD_VERSION})
set(LIBRARY_NAME ${PROJECT_NAME})

# Set build environment and standard dependencies
include(${CMAKE_SOURCE_DIR}/build/cmake/base.cmake)
include(${CMAKE_SOURCE_DIR}/build/cmake/doxygen.cmake)
include(${CMAKE_SOURCE_DIR}/build/cmake/boostPythonDocstring.cmake)

# Sub-projects
add_subdirectory(core)
add_subdirectory(python)
add_subdirectory(scenarios)

# Build documentation from Doxygen.
build_DOC ()

# Install native python packages
deploy_python_gym_package(gym_${LIBRARY_NAME})

# Install the source cmake files
file(GLOB_RECURSE SOURCE_CMAKE_FILES "${CMAKE_SOURCE_DIR}/build/cmake/Find*.cmake")
install(FILES ${SOURCE_CMAKE_FILES}
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)
install(FILES ${CMAKE_SOURCE_DIR}/build/cmake/base.cmake
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
        RENAME ${LIBRARY_NAME}_base.cmake
)