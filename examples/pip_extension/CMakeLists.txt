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
# jiminy headers and library is provided with `pip install jiminy_py`,  but
# pinocchio is not despite being mantadory to compile any extension module.
# Yet, only pinocchio headers are required so it is straighforward to install.
find_package(jiminy REQUIRED)
# On windows, pinocchio is not able to find boost components for some reasons
if(WIN32)
    find_package(Boost REQUIRED COMPONENTS chrono system filesystem serialization date_time thread atomic)
endif()
find_package(pinocchio 2.5.2 REQUIRED NO_MODULE)
find_package(Eigen3 3.3.0 REQUIRED NO_MODULE)

# Make executables
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/double_pendulum.cc")

# Set include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME} jiminy::core pinocchio::pinocchio Eigen3::Eigen)
if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

# Note that RPATH is not supported by Windows, thus jiminy DLL directory must
# be added to the search path manually, so copied alonside with executable...

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
