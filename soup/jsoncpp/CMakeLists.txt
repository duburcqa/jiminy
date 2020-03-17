# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(jsoncpp)

# Generate patch argument
EXTERNALPROJECT_APPLY_PATCH(${CMAKE_CURRENT_SOURCE_DIR}/patch PATCH_ARGS)

# Download and install jsoncpp.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/open-source-parsers/jsoncpp
     GIT_TAG           1.9.2
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false
     CMAKE_ARGS
     -DJSONCPP_WITH_TESTS=OFF
     -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF
     -DJSONCPP_WITH_PKGCONFIG_SUPPORT=OFF
     -DJSONCPP_WITH_CMAKE_PACKAGE=ON
     -DBUILD_SHARED_LIBS=OFF
     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
     -DCMAKE_CXX_FLAGS="-w" # Disable all warnings
     ${PATCH_ARGS}
     INSTALL_COMMAND ""
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)
