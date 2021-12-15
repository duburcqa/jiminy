# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(unit VERSION ${BUILD_VERSION})

# Find pthread if available
find_package(Threads)

# Define the list of unit test files
set(UNIT_TEST_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/EngineSanityCheck.cc"
)

# Create the unit test executable
add_executable(${PROJECT_NAME} ${UNIT_TEST_FILES})

# Add tests with CTest
gtest_discover_tests(${PROJECT_NAME})

# Add definition of unit test data folder
target_compile_definitions(${PROJECT_NAME} PUBLIC
    "-DUNIT_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/data/\""
)

# Link with Jiminy core library
target_link_libraries(${PROJECT_NAME} ${LIBRARY_NAME}_core)

# Configure gtest dependency
add_dependencies(${PROJECT_NAME} gtest_external)
externalproject_get_property(gtest_external SOURCE_DIR)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
     $<BUILD_INTERFACE:${SOURCE_DIR}/googletest/include>
     $<BUILD_INTERFACE:${SOURCE_DIR}/googlemock/include>
)
target_link_libraries(${PROJECT_NAME} gtest::gtest gtest::gtest_main gtest::gmock gtest::gmock_main)
target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
set_property(TARGET ${PROJECT_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)
