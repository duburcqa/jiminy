# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(jsoncpp_external)

# Generate patch argument
EXTERNALPROJECT_APPLY_PATCH(${CMAKE_CURRENT_SOURCE_DIR}/patch PATCH_ARGS)

# Download, build and install jsoncpp.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/open-source-parsers/jsoncpp.git
     GIT_TAG           1.9.2
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}
     CMAKE_ARGS
     -DJSONCPP_WITH_TESTS=OFF
     -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF
     -DJSONCPP_WITH_PKGCONFIG_SUPPORT=OFF
     -DJSONCPP_WITH_CMAKE_PACKAGE=ON
     -DBUILD_SHARED_LIBS=OFF
     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
     -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
     -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
     ${PATCH_ARGS}
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

EXTERNALPROJECT_GET_PROPERTY(${PROJECT_NAME} BINARY_DIR)

# Import the generated library as a target
add_library(jsoncpp_lib STATIC IMPORTED GLOBAL)

if(NOT WIN32)
     set_target_properties(jsoncpp_lib PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/src/lib_json/libjsoncpp.a
     )
else(NOT WIN32)
     set_target_properties(jsoncpp_lib PROPERTIES
          IMPORTED_LOCATION ${BINARY_DIR}/src/lib_json/Release/jsoncpp.lib
     )
endif(NOT WIN32)
