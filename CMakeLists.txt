# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Set the build version
set(BUILD_VERSION 1.7.9)

# Set compatibility
if(CMAKE_VERSION VERSION_GREATER "3.11.0")
    set(COMPATIBILITY_VERSION SameMinorVersion)
else()
    set(COMPATIBILITY_VERSION ExactVersion)
endif()

# Extract major, minor and patch version
string(REPLACE "." ";" _VERSION "${BUILD_VERSION}")
list(GET _VERSION 0 BUILD_VERSION_MAJOR)
list(GET _VERSION 1 BUILD_VERSION_MINOR)
list(GET _VERSION 2 BUILD_VERSION_PATCH)

# Add definition of Jiminy version for C++ headers
add_definitions("-DJIMINY_VERSION=\"${BUILD_VERSION}\"")

# Enable C++ language
enable_language(CXX)

# Project and library name
project(jiminy VERSION ${BUILD_VERSION})
set(LIBRARY_NAME ${PROJECT_NAME})
set(PYTHON_LIBRARY_NAME "core")

# Set build environment and standard dependencies
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/base.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/docs.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/boostPythonDocstring.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/exportCmakeConfigFiles.cmake)
include(${CMAKE_SOURCE_DIR}/build_tools/cmake/buildPythonWheel.cmake)

# Set the compilation flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /permissive- -DNOMINMAX")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG /O2 /Ob3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -Wfatal-errors -Werror")
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Sub-projects
add_subdirectory(soup)
add_subdirectory(core)

option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
if(BUILD_PYTHON_INTERFACE)
    include(${CMAKE_SOURCE_DIR}/build_tools/cmake/setupPython.cmake)
    add_subdirectory(python)
endif()

option(BUILD_EXAMPLES "Build the C++ examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(BUILD_TESTING "Build the C++ unit tests." ON)
if(BUILD_TESTING)
    include(CTest)
    include(GoogleTest)
    add_subdirectory(unit)
endif()

# Create 'docs' optional cmake component to generate documentation.
create_component_docs()

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
