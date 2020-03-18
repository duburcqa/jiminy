# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(gtest)

# Download and install gtest.
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