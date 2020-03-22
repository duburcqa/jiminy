# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(gtest_external)

# Set the path of the generated libraries
set(${PROJECT_NAME}_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-prefix/src/${PROJECT_NAME}-build")

if(NOT WIN32)
     set(${PROJECT_NAME}_LIB_DIR "${${PROJECT_NAME}_BUILD_DIR}/lib")
     set(gtest_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/libgtest.a")
     set(gtest_main_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/libgtest_main.a")
     set(gmock_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/libgmock.a")
     set(gmock_main_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/libgmock_main.a")
else(NOT WIN32)
     set(${PROJECT_NAME}_LIB_DIR "${${PROJECT_NAME}_BUILD_DIR}/lib/${CMAKE_BUILD_TYPE}")
     set(gtest_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/gtest.lib")
     set(gtest_main_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/gtest_main.lib")
     set(gmock_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/gmock.lib")
     set(gmock_main_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/gmock_main.lib")
endif(NOT WIN32)

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

     INSTALL_COMMAND "" # Disable install of googletest on the system

     BUILD_BYPRODUCTS "${gtest_lib_PATH}"
     BUILD_BYPRODUCTS "${gtest_main_lib_PATH}"
     BUILD_BYPRODUCTS "${gmock_lib_PATH}"
     BUILD_BYPRODUCTS "${gmock_main_lib_PATH}"

     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Import the generated libraries as targets
add_library(gtest STATIC IMPORTED GLOBAL)
set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${gtest_lib_PATH})
add_library(gtest_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest_main PROPERTIES IMPORTED_LOCATION ${gtest_main_lib_PATH})
add_library(gmock STATIC IMPORTED GLOBAL)
set_target_properties(gmock PROPERTIES IMPORTED_LOCATION ${gmock_lib_PATH})
add_library(gmock_main STATIC IMPORTED GLOBAL)
set_target_properties(gmock_main PROPERTIES IMPORTED_LOCATION ${gmock_main_lib_PATH})
