# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(jsoncpp_external)

# Set the path of the generated library
set(${PROJECT_NAME}_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-prefix/src/${PROJECT_NAME}-build")

if(NOT WIN32)
     set(${PROJECT_NAME}_LIB_DIR "${${PROJECT_NAME}_BUILD_DIR}/src/lib_json")
     set(jsoncpp_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/libjsoncpp.a")
else(NOT WIN32)
     set(${PROJECT_NAME}_LIB_DIR "${${PROJECT_NAME}_BUILD_DIR}/src/lib_json/${CMAKE_BUILD_TYPE}")
     set(jsoncpp_lib_PATH "${${PROJECT_NAME}_LIB_DIR}/jsoncpp.lib")
endif(NOT WIN32)

# Set the library path

# Generate patch argument
EXTERNALPROJECT_APPLY_PATCH(${CMAKE_CURRENT_SOURCE_DIR}/patch PATCH_ARGS)

# Download, build and install jsoncpp.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/open-source-parsers/jsoncpp.git
     GIT_TAG           1.9.2
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     ${PATCH_ARGS}

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

     BUILD_BYPRODUCTS "${jsoncpp_lib_PATH}"

     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

EXTERNALPROJECT_GET_PROPERTY(${PROJECT_NAME} BINARY_DIR)

# Import the generated library as a target
add_library(jsoncpp_lib STATIC IMPORTED GLOBAL)
set_target_properties(jsoncpp_lib PROPERTIES IMPORTED_LOCATION ${jsoncpp_lib_PATH})
