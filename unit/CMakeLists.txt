# Setup cpp unit tests.

# Create empty file to declare executable without any source files.
file(WRITE ${CMAKE_BINARY_DIR}/null.cc "")
set(CPP_UNIT unit)
add_executable(${CPP_UNIT} ${CMAKE_BINARY_DIR}/null.cc)

file(GLOB_RECURSE UNIT_SOURCES "*.cc")
target_sources(${CPP_UNIT} PRIVATE ${UNIT_SOURCES})
set(UNIT_CURRENT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
if(EXISTS ${UNIT_CURRENT_DATA_DIR})
	file (COPY ${UNIT_CURRENT_DATA_DIR} DESTINATION "${CMAKE_BINARY_DIR}/unit/data/")
endif()

# Include core library for unit tests.
target_link_libraries(${CPP_UNIT} ${LIBRARY_NAME}_core)

# Finalize setup of cpp unit tests (gtest).
add_dependencies(${CPP_UNIT} gtest)
EXTERNALPROJECT_GET_PROPERTY(gtest SOURCE_DIR)
EXTERNALPROJECT_GET_PROPERTY(gtest BINARY_DIR)
target_include_directories(${CPP_UNIT} SYSTEM PUBLIC
     ${SOURCE_DIR}/googletest/include
     ${SOURCE_DIR}
)
target_link_libraries(${CPP_UNIT} "-L${BINARY_DIR}/lib -lgtest -lgtest_main -lgmock -lpthread")
include(CTest)
enable_testing()
