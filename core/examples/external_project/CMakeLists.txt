# RPATH is not supported by Windows, thus jiminy DLL directory must be
# added to the search path manually, or copied alongside this executable.

# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Set the compilation flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} /Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG /O2 /Ob3")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG:FULL /INCREMENTAL:NO /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Define GNU standard installation directories
include(GNUInstallDirs)

# Project name
project(pip_double_pendulum VERSION 0.0)

# Enable C++ language
enable_language(CXX)

# Find libraries and headers
# jiminy headers and library is provided with `pip install jiminy_py`
find_package(jiminy REQUIRED)

# Make executables
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/double_pendulum.cc")

# Set include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME} jiminy::core)

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
