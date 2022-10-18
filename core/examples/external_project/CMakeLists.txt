# RPATH is not supported by Windows, thus jiminy DLL directory must be
# added to the search path manually, or copied alonside this executable.

# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Enable C++ language
enable_language(CXX)

# Set the compilation flags
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /O2")
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Define GNU standard installation directories
include(GNUInstallDirs)

# Project name
project(pip_double_pendulum VERSION 0.0)

# Find libraries and headers
# jiminy headers and library is provided with `pip install jiminy_py`
find_package(jiminy REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

# Make executables
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/double_pendulum.cc")

# Set include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME} jiminy::core Boost::filesystem)
if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
