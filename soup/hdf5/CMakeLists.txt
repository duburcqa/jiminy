# Minimum version required
cmake_minimum_required(VERSION 3.10)

# Project and library name
project(hdf5_external)

# Download, build and install zlib.
# TGZ overwrites some global cmake flags for no reason, and official GIT repo does
# not have fPIC enabled by default, making the whole build process fail. So zlib is
# installed manually and set add dependency of hdf5
EXTERNALPROJECT_ADD(zlib_external
     GIT_REPOSITORY    https://github.com/madler/zlib.git
     GIT_TAG           v1.2.11
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DSKIP_INSTALL_FILES=ON  # Disable install of manual and pkgconfig files
          -DSKIP_INSTALL_LIBRARIES=ON  # Do not install libraries automatically. It will be handled manually to avoid installing shared libs
          -DBUILD_SHARED_LIBS=OFF
          -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
          -DCMAKE_C_FLAGS:STRING=${CMAKE_COMPILE_FLAGS_EXTERNAL}
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}
     INSTALL_DIR ${CMAKE_INSTALL_PREFIX}

     ${zlib_NINJA}

     UPDATE_COMMAND ""  # For some reason, it is required to avoid rebuild systematically everything...
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)
if(NOT WIN32)
     set(zlib_BUILD_LIB_PATH "<BINARY_DIR>/libz.a")
     set(zlib_PATH "${CMAKE_INSTALL_PREFIX}/lib/libz.a")
else()
     set(zlib_BUILD_LIB_PATH "<BINARY_DIR>/Release/zlibstatic.lib")
     set(zlib_PATH "${CMAKE_INSTALL_PREFIX}/lib/zlibstatic.lib")
endif()
ExternalProject_Add_Step(
     zlib_external zlib_install_static_only
     COMMENT "Manually installing only static library"
     DEPENDEES install
     COMMAND ${CMAKE_COMMAND} -E copy ${zlib_BUILD_LIB_PATH} ${zlib_PATH}
)
 
# Download, build and install hdf5.
EXTERNALPROJECT_ADD(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/HDFGroup/hdf5.git
     GIT_TAG           hdf5_1_12-initial  # 'hdf5-1_10_6' and 'hdf5_1_12-initial' are the latest versions not requiring CMake >= 3.12.
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DHDF5_ENABLE_COVERAGE=OFF
          -DHDF5_ENABLE_USING_MEMCHECKER=OFF
          -DHDF5_MEMORY_ALLOC_SANITY_CHECK=OFF
          -DHDF5_ENABLE_PREADWRITE=ON
          -DHDF5_ENABLE_DEPRECATED_SYMBOLS=OFF
          #-DHDF5_MINGW_STATIC_GCC_LIBS=OFF
          -DHDF5_BUILD_GENERATORS=OFF
          -DHDF5_ENABLE_TRACE=OFF
          #-DHDF5_ENABLE_INSTRUMENT=OFF
          -DHDF5_ENABLE_EMBEDDED_LIBINFO=ON
          -DHDF5_ENABLE_HDFS=OFF
          -DHDF5_ENABLE_PARALLEL=OFF
          -DHDF5_ENABLE_THREADSAFE=OFF
          #-DHDF5_ENABLE_MAP_API=OFF
          #-DHDF5_BUILD_UTILS=OFF
          -DHDF5_BUILD_TOOLS=OFF
          -DHDF5_BUILD_EXAMPLES=OFF
          -DHDF5_BUILD_HL_LIB=OFF
          -DHDF5_BUILD_FORTRAN=OFF
          -DHDF5_BUILD_CPP_LIB=ON
          -DHDF5_BUILD_JAVA=OFF
          -DHDF5_USE_PREGEN=OFF
          -DHDF5_BATCH_H5DETECT=OFF
          -DHDF5_PACKAGE_EXTLIBS=OFF
          -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING=NO
          -DHDF5_ENABLE_Z_LIB_SUPPORT=ON
          -DHDF5_ENABLE_SZIP_SUPPORT=OFF
          -DZLIB_PACKAGE_NAME:STRING=NOT_FOUND  # Hack to avoid finding some zlib config files already installed and not skipping find_package(ZLIB)
          -DZLIB_ROOT:PATH=${CMAKE_INSTALL_PREFIX}
          -DZLIB_LIBRARY:PATH=${zlib_PATH}  # Using ZLIB_ROOT only is finding the shared lib instead of the static one
          -DHDF5_DISABLE_COMPILER_WARNINGS=ON
          -DHDF5_ENABLE_SYMBOLS=OFF
          -DHDF5_ENABLE_OPTIMIZATION=ON
          -DBUILD_TESTING=OFF
          #-DBUILD_STATIC_LIBS=ON
          -DBUILD_SHARED_LIBS=OFF
          -DBUILD_STATIC_EXECS=OFF
          -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
          -DCMAKE_CXX_FLAGS:STRING=${CMAKE_COMPILE_FLAGS_EXTERNAL}
          -DCMAKE_C_FLAGS:STRING=${CMAKE_COMPILE_FLAGS_EXTERNAL}
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}
     INSTALL_DIR ${CMAKE_INSTALL_PREFIX}

     ${hdf5_NINJA}
     ${hdf5_cpp_NINJA}

     DEPENDS zlib_external

     UPDATE_COMMAND ""  # For some reason, it is required to avoid rebuild systematically everything...
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Get the path of the generated libraries
if(NOT WIN32)
     set(zlib_NINJA BUILD_BYPRODUCTS "${zlib_PATH_${CMAKE_BUILD_TYPE}}")
     set(hdf5_PATH "${CMAKE_INSTALL_PREFIX}/lib/libhdf5.a")
     set(hdf5_NINJA BUILD_BYPRODUCTS "${hdf5_PATH_${CMAKE_BUILD_TYPE}}")
     set(hdf5_cpp_PATH "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_cpp.a")
     set(hdf5_cpp_NINJA BUILD_BYPRODUCTS "${hdf5_cpp_PATH_${CMAKE_BUILD_TYPE}}")
else()
     set(hdf5_PATH "${CMAKE_INSTALL_PREFIX}/lib/libhdf5.lib")
     set(hdf5_cpp_PATH "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_cpp.lib")
endif()

# Import the generated libraries as targets
add_library(hdf5::zlib STATIC IMPORTED GLOBAL)
set_target_properties(hdf5::zlib PROPERTIES
     IMPORTED_LOCATION ${zlib_PATH}
)
add_library(hdf5::hdf5 STATIC IMPORTED GLOBAL)
set_target_properties(hdf5::hdf5 PROPERTIES
     IMPORTED_LOCATION ${hdf5_PATH}
)
add_library(hdf5::hdf5_cpp STATIC IMPORTED GLOBAL)
set_target_properties(hdf5::hdf5_cpp PROPERTIES
     IMPORTED_LOCATION ${hdf5_cpp_PATH}
)
