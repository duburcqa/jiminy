# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(unit VERSION ${BUILD_VERSION})

# Find pthread if available
find_package(Threads)

# Create empty file to declare executable without any source files.
file(WRITE ${CMAKE_BINARY_DIR}/null.cc "")

# Create the unit test executable
add_executable(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/null.cc)

# Define the list of unit test files
set(UNIT_TEST_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/EngineSanityCheck.cc"
)

# Add the unit test files and data folder to the executable
target_sources(${PROJECT_NAME} PRIVATE ${UNIT_TEST_FILES})
set(UNIT_CURRENT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
if(EXISTS ${UNIT_CURRENT_DATA_DIR})
	file (COPY ${UNIT_CURRENT_DATA_DIR} DESTINATION "${CMAKE_BINARY_DIR}/unit/data/")
endif()

# Include core library for unit tests.
target_link_libraries(${PROJECT_NAME} ${LIBRARY_NAME}_core)

# Configure gtest dependency
add_dependencies(${PROJECT_NAME} gtest_external)
EXTERNALPROJECT_GET_PROPERTY(gtest_external SOURCE_DIR)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
     ${SOURCE_DIR}/googletest/include
     ${SOURCE_DIR}/googlemock/include
)
target_link_libraries(${PROJECT_NAME} gtest gtest_main gmock gmock_main "${CMAKE_THREAD_LIBS_INIT}")
set_property(TARGET ${PROJECT_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
include(CTest)
enable_testing()
