# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(gtest)

# Download and install gtest.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/google/googletest.git
     GIT_TAG           v1.10.x
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false
     CMAKE_ARGS
     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
     -DCMAKE_CXX_FLAGS="-w"
     INSTALL_COMMAND ""
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)