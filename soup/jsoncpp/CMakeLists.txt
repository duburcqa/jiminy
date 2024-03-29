# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project and library name
project(jsoncpp_external)

# Get the path of the generated libraries
include(GNUInstallDirs)
if(WIN32)
     set(jsoncpp_PATH "${CMAKE_INSTALL_PREFIX}/lib/jsoncpp.lib")
else()
     set(jsoncpp_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libjsoncpp.a")
endif()
if(CMAKE_GENERATOR STREQUAL "Ninja")
     set(jsoncpp_NINJA BUILD_BYPRODUCTS "${jsoncpp_PATH}")
endif()

# Download, build and install jsoncpp.
externalproject_add(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/open-source-parsers/jsoncpp.git
     GIT_TAG           1.9.5
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DJSONCPP_WITH_TESTS=OFF
          -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF
          -DJSONCPP_WITH_EXAMPLE=OFF
          -DJSONCPP_WITH_PKGCONFIG_SUPPORT=OFF
          -DJSONCPP_WITH_CMAKE_PACKAGE=ON
          -DBUILD_SHARED_LIBS=OFF
          -DBUILD_OBJECT_LIBS=OFF
          -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
          -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
          -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
          -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG_EXTERNAL}
          -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
          -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
          -DCMAKE_SHARED_LINKER_FLAGS_DEBUG:STRING=${CMAKE_SHARED_LINKER_FLAGS_DEBUG}
          -DCMAKE_SHARED_LINKER_FLAGS_RELEASE:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELEASE}
          -DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}
          ${EXTERNALPROJECT_OSX_CONFIG}
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}

     ${jsoncpp_NINJA}

     INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
     UPDATE_COMMAND ""  # Avoid reinstalling systematically everything
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Import the generated library as a target
add_library(jsoncpp::jsoncpp STATIC IMPORTED GLOBAL)
set_target_properties(jsoncpp::jsoncpp PROPERTIES
     IMPORTED_LOCATION ${jsoncpp_PATH}
)
