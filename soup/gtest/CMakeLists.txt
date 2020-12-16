# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(gtest_external)

# Download and build gtest.
externalproject_add(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/google/googletest.git
     GIT_TAG           release-1.10.0
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_CXX_FLAGS:STRING=${CMAKE_COMPILE_FLAGS_EXTERNAL}
          -Dgtest_force_shared_crt=ON  # This option is required on Github Windows CI for some reasons..
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}
          -Wno-dev  # Silent Cmake warnings about deprecated support of Cmake < 2.8.12

     ${gtest_NINJA}
     ${gtest_main_NINJA}
     ${gmock_NINJA}
     ${gmock_main_NINJA}

     INSTALL_COMMAND ""  # Disable install of googletest on the system
     UPDATE_COMMAND ""  # Avoid reinstalling systematically everything
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Get the path of the generated libraries
externalproject_get_property(${PROJECT_NAME} BINARY_DIR)
set(${PROJECT_NAME}_LIB_DIR "${BINARY_DIR}/lib")
if(NOT WIN32)
     set(gtest_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgtestd.a")
     set(gtest_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgtest.a")
     set(gtest_NINJA BUILD_BYPRODUCTS "${gtest_PATH_${CMAKE_BUILD_TYPE}}")
     set(gtest_main_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgtest_maind.a")
     set(gtest_main_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgtest_main.a")
     set(gtest_main_NINJA BUILD_BYPRODUCTS "${gtest_main_PATH_${CMAKE_BUILD_TYPE}}")
     set(gmock_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgmockd.a")
     set(gmock_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgmock.a")
     set(gmock_NINJA BUILD_BYPRODUCTS "${gmock_PATH_${CMAKE_BUILD_TYPE}}")
     set(gmock_main_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/libgmock_maind.a")
     set(gmock_main_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/libgmock_main.a")
     set(gmock_main_NINJA BUILD_BYPRODUCTS "${gmock_main_PATH_${CMAKE_BUILD_TYPE}}")
else()
     set(gtest_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gtest.lib")
     set(gtest_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gtest.lib")
     set(gtest_main_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gtest_main.lib")
     set(gtest_main_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gtest_main.lib")
     set(gmock_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gmock.lib")
     set(gmock_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gmock.lib")
     set(gmock_main_PATH_Debug "${${PROJECT_NAME}_LIB_DIR}/Debug/gmock_main.lib")
     set(gmock_main_PATH_Release "${${PROJECT_NAME}_LIB_DIR}/Release/gmock_main.lib")
endif()

# Import the generated libraries as targets
add_library(gtest::gtest STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gtest PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gtest_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gtest_PATH_Release}
)
add_library(gtest::gtest_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gtest_main PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gtest_main_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gtest_main_PATH_Release}
)
add_library(gtest::gmock STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gmock PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gmock_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gmock_PATH_Release}
)
add_library(gtest::gmock_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gmock_main PROPERTIES
     IMPORTED_LOCATION_DEBUG ${gmock_main_PATH_Debug}
     IMPORTED_LOCATION_RELEASE ${gmock_main_PATH_Release}
)
