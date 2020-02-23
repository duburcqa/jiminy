# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Set the build version
set(BUILD_VERSION 1.0.2)

# Project and library name
project(jiminy VERSION ${BUILD_VERSION})
set(LIBRARY_NAME ${PROJECT_NAME})

# Set build environment and standard dependencies
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/base.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/doxygen.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/boostPythonDocstring.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/exportCmakeConfigFiles.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/buildPythonWheel.cmake)

# Sub-projects
add_subdirectory(core)
add_subdirectory(python)
add_subdirectory(scenarios)
add_subdirectory(unit)

# Build documentation from Doxygen.
build_DOC()

# Install the source cmake files
file(GLOB_RECURSE SOURCE_CMAKE_FILES "${CMAKE_SOURCE_DIR}/build_tools/cmake/Find*.cmake")
install(FILES ${SOURCE_CMAKE_FILES}
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)
install(FILES ${CMAKE_SOURCE_DIR}/build_tools/cmake/base.cmake
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
        RENAME ${LIBRARY_NAME}_base.cmake
)
install(FILES ${CMAKE_SOURCE_DIR}/build_tools/cmake/boostPythonDocstring.cmake
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake"
)
