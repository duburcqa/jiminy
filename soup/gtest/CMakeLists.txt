# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(gtest_external)

# Set the path of the generated libraries
set(${PROJECT_NAME}_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-prefix/src/${PROJECT_NAME}-build/lib")

if(NOT WIN32)
     set(gtest_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgtestd.a")
     set(gtest_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgtest.a")
     set(gtest_lib_NINJA BUILD_BYPRODUCTS "${gtest_lib_PATH_${CMAKE_BUILD_TYPE}}")
     set(gtest_main_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgtest_maind.a")
     set(gtest_main_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgtest_main.a")
     set(gtest_main_lib_NINJA BUILD_BYPRODUCTS "${gtest_main_lib_PATH_${CMAKE_BUILD_TYPE}}")
     set(gmock_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgmockd.a")
     set(gmock_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgmock.a")
     set(gmock_lib_NINJA BUILD_BYPRODUCTS "${gmock_lib_PATH_${CMAKE_BUILD_TYPE}}")
     set(gmock_main_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgmock_maind.a")
     set(gmock_main_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgmock_main.a")
     set(gmock_main_lib_NINJA BUILD_BYPRODUCTS "${gmock_main_lib_PATH_${CMAKE_BUILD_TYPE}}")
else(NOT WIN32)
     set(gtest_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gtest.lib")
     set(gtest_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gtest.lib")
     set(gtest_main_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gtest_main.lib")
     set(gtest_main_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gtest_main.lib")
     set(gmock_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gmock.lib")
     set(gmock_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gmock.lib")
     set(gmock_main_lib_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gmock_main.lib")
     set(gmock_main_lib_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gmock_main.lib")
endif(NOT WIN32)

# Download and build gtest.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/google/googletest.git
     GIT_TAG           release-1.10.0
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
     -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
     ${EXTERNALPROJECT_BUILD_TYPE_CMD}

     INSTALL_COMMAND "" # Disable install of googletest on the system

     ${gtest_lib_NINJA}
     ${gtest_main_lib_NINJA}
     ${gmock_lib_NINJA}
     ${gmock_main_lib_NINJA}

     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Import the generated libraries as targets
add_library(gtest STATIC IMPORTED GLOBAL)
set_target_properties(gtest PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gtest_lib_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gtest_lib_PATH_Release}
)
add_library(gtest_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest_main PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gtest_main_lib_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gtest_main_lib_PATH_Release}
)
add_library(gmock STATIC IMPORTED GLOBAL)
set_target_properties(gmock PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gmock_lib_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gmock_lib_PATH_Release}
)
add_library(gmock_main STATIC IMPORTED GLOBAL)
set_target_properties(gmock_main PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gmock_main_lib_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gmock_main_lib_PATH_Release}
)
