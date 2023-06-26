# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project and library name
project(gtest_external)

# Get the paths of the generated libraries
if(WIN32)
     set(gtest_PATH "<BINARY_DIR>/lib/Release/gtest.lib")
     set(gtest_main_PATH "<BINARY_DIR>/lib/Release/gtest_main.lib")
     set(gmock_PATH "<BINARY_DIR>/lib/Release/gmock.lib")
     set(gmock_main_PATH "<BINARY_DIR>/lib/Release/gmock_main.lib")
else()
     set(gtest_PATH "<BINARY_DIR>/lib/libgtest.a")
     set(gtest_NINJA BUILD_BYPRODUCTS "${gtest_PATH}")
     set(gtest_main_PATH "<BINARY_DIR>/lib/libgtest_main.a")
     set(gtest_main_NINJA BUILD_BYPRODUCTS "${gtest_main_PATH}")
     set(gmock_PATH "<BINARY_DIR>/lib/libgmock.a")
     set(gmock_NINJA BUILD_BYPRODUCTS "${gmock_PATH}")
     set(gmock_main_PATH "<BINARY_DIR>/lib/libgmock_main.a")
     set(gmock_main_NINJA BUILD_BYPRODUCTS "${gmock_main_PATH}")
endif()

# Download and build gtest.
externalproject_add(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/google/googletest.git
     GIT_TAG           v1.13.0
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
          -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG_EXTERNAL}
          -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
          -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
          -Dgtest_force_shared_crt=ON  # This option is required on Github Windows CI for some reasons...
          -Wno-dev  # Silent Cmake warnings about deprecated support of Cmake < 2.8.12
          ${EXTERNALPROJECT_OSX_CONFIG}
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}

     ${gtest_NINJA}
     ${gtest_main_NINJA}
     ${gmock_NINJA}
     ${gmock_main_NINJA}

     INSTALL_COMMAND ""  # Disable install of googletest on the system
     UPDATE_COMMAND ""  # Avoid reinstalling systematically everything
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Replace generator expression by actual build directory in the paths of the generated libraries
externalproject_get_property(${PROJECT_NAME} BINARY_DIR)
string(REPLACE "<BINARY_DIR>" "${BINARY_DIR}" gtest_PATH "${gtest_PATH}")
string(REPLACE "<BINARY_DIR>" "${BINARY_DIR}" gtest_main_PATH "${gtest_main_PATH}")
string(REPLACE "<BINARY_DIR>" "${BINARY_DIR}" gmock_PATH "${gmock_PATH}")
string(REPLACE "<BINARY_DIR>" "${BINARY_DIR}" gmock_main_PATH "${gmock_main_PATH}")

# Import the generated libraries as targets
add_library(gtest::gtest STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gtest PROPERTIES
     IMPORTED_LOCATION ${gtest_PATH}
)
add_library(gtest::gtest_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gtest_main PROPERTIES
     IMPORTED_LOCATION ${gtest_main_PATH}
)
add_library(gtest::gmock STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gmock PROPERTIES
     IMPORTED_LOCATION ${gmock_PATH}
)
add_library(gtest::gmock_main STATIC IMPORTED GLOBAL)
set_target_properties(gtest::gmock_main PROPERTIES
     IMPORTED_LOCATION ${gmock_main_PATH}
)
