# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project name
project(${LIBRARY_NAME}_double_pendulum VERSION ${BUILD_VERSION})

# Make executables
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/double_pendulum.cc")

# Set include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
)

# Link with other libraries
target_link_libraries(${PROJECT_NAME} ${LIBRARY_NAME}_core)

# Configure MSVC multithreading library to link against
set_property(TARGET ${PROJECT_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
