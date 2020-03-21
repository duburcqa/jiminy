# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(gtest_external)

# Download and build gtest.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/google/googletest.git
     GIT_TAG           release-1.10.0
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}
     CMAKE_ARGS
     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
     -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
     INSTALL_COMMAND ""
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

EXTERNALPROJECT_GET_PROPERTY(${PROJECT_NAME} BINARY_DIR)

# Import the generated libraries as targets
add_library(gtest STATIC IMPORTED GLOBAL)
add_library(gtest_main STATIC IMPORTED GLOBAL)
add_library(gmock STATIC IMPORTED GLOBAL)
add_library(gmock_main STATIC IMPORTED GLOBAL)

if(NOT WIN32)
     set_target_properties(gtest PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/libgtest.a
     )
     set_target_properties(gtest_main PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/libgtest_main.a
     )
     set_target_properties(gmock PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/libgmock.a
     )
     set_target_properties(gmock_main PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/libgmock_main.a
     )
else(NOT WIN32)
     set_target_properties(gtest PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/Release/gtest.lib
     )
     set_target_properties(gtest_main PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/Release/gtest_main.lib
     )
     set_target_properties(gmock PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/Release/gmock.lib
     )
     set_target_properties(gmock_main PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/lib/Release/gmock_main.lib
     )
endif(NOT WIN32)