# Setup cpp unit tests.

# Download and install gtest.
include(ExternalProject)
EXTERNALPROJECT_ADD (gtest_EXTERNAL
     GIT_REPOSITORY    https://github.com/google/googletest.git
     GIT_TAG           v1.10.x
     CMAKE_ARGS
     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
     INSTALL_COMMAND ""
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
     )

# Create empty file to declare executable without any source files.
file (WRITE ${CMAKE_BINARY_DIR}/null.cc "")
set (CPP_UNIT unit)
add_executable (${CPP_UNIT} ${CMAKE_BINARY_DIR}/null.cc)

file (GLOB_RECURSE UNIT_SOURCES "*.cc")
target_sources (${CPP_UNIT} PRIVATE ${UNIT_SOURCES})
set (UNIT_CURRENT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
if (EXISTS ${UNIT_CURRENT_DATA_DIR})
	file (COPY ${UNIT_CURRENT_DATA_DIR} DESTINATION "${CMAKE_BINARY_DIR}/unit/data/")
endif ()

# Include core library for unit tests.
target_link_libraries(${CPP_UNIT} jiminy_core)

# Finalize setup of cpp unit tests (gtest).
add_dependencies (${CPP_UNIT} gtest_EXTERNAL)
EXTERNALPROJECT_GET_PROPERTY (gtest_EXTERNAL SOURCE_DIR)
EXTERNALPROJECT_GET_PROPERTY (gtest_EXTERNAL BINARY_DIR)
target_include_directories(${CPP_UNIT} SYSTEM PUBLIC ${SOURCE_DIR}/googletest/include ${SOURCE_DIR})
target_link_libraries(${CPP_UNIT} "-L${BINARY_DIR}/lib -lgtest -lgtest_main -lgmock -lpthread")
include(CTest)
enable_testing()
