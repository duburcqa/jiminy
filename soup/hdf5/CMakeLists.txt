# Minimum version required
cmake_minimum_required(VERSION 3.12.4)

# Project and library name
project(hdf5_external)

# Define path of the generated zlib library
if(WIN32)
     set(zlib_BUILD_LIB_PATH "<BINARY_DIR>/$<CONFIG>/zlibstatic$<$<CONFIG:Debug>:d>.lib")
     set(zlib_INSTALL_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib/$<CONFIG>/zlibstatic$<$<CONFIG:Debug>:d>.lib")
     set(zlib_PATH_Release "${CMAKE_INSTALL_PREFIX}/lib/Release/zlibstatic.lib")
     set(zlib_PATH_Debug "${CMAKE_INSTALL_PREFIX}/lib/Debug/zlibstaticd.lib")
else()
     set(zlib_BUILD_LIB_PATH "<BINARY_DIR>/libz$<$<CONFIG:Debug>:d>.a")
     set(zlib_INSTALL_LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib/libz$<$<CONFIG:Debug>:d>.a")
     set(zlib_PATH_Release "${CMAKE_INSTALL_PREFIX}/lib/libz.a")
     set(zlib_PATH_Debug "${CMAKE_INSTALL_PREFIX}/lib/libzd.a")
endif()
if(CMAKE_GENERATOR STREQUAL "Ninja")
     set(zlib_NINJA BUILD_BYPRODUCTS "${zlib_PATH_${CMAKE_BUILD_TYPE}}")
endif()

# Download, build and install zlib.
# TGZ overwrites some global cmake flags for no reason, and official GIT repo does
# not have fPIC enabled by default, making the whole build process fail. So zlib is
# installed manually and set add dependency of hdf5
externalproject_add(zlib_external
     GIT_REPOSITORY    https://github.com/madler/zlib.git
     GIT_TAG           v1.3
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DSKIP_INSTALL_FILES=ON  # Disable install of manual and pkgconfig files
          -DSKIP_INSTALL_LIBRARIES=ON  # Install manually to skip shared libs
          -DCMAKE_DEBUG_POSTFIX=d  # Force adding "d" suffix for config debug
          -DBUILD_SHARED_LIBS=OFF
          -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
          -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
          -DCMAKE_C_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
          -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG_EXTERNAL}
          -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
          -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
          -DCMAKE_SHARED_LINKER_FLAGS_DEBUG:STRING=${CMAKE_SHARED_LINKER_FLAGS_DEBUG}
          -DCMAKE_SHARED_LINKER_FLAGS_RELEASE:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELEASE}
          -DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}
          -Wno-dev  # Silent Cmake warnings about deprecated support of Cmake < 2.8.12
          ${EXTERNALPROJECT_OSX_CONFIG}
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}

     ${zlib_NINJA}

     INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
     UPDATE_COMMAND ""  # For some reason, it is required to avoid rebuild systematically everything...
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

ExternalProject_Add_Step(
     zlib_external zlib_install_static_only
     COMMENT "Manually installing static zlib library"
     DEPENDEES install
     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${zlib_BUILD_LIB_PATH} ${zlib_INSTALL_LIB_PATH}
)

# Define paths of the generated hdf5 libraries
if(WIN32)
     set(libhdf5_PATH_Release "${CMAKE_INSTALL_PREFIX}/lib/libhdf5.lib")
     set(libhdf5_PATH_Debug "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_D.lib")
     set(libhdf5_cpp_PATH_Release "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_cpp.lib")
     set(libhdf5_cpp_PATH_Debug "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_cpp_D.lib")
else()
     set(libhdf5_PATH_Release "${CMAKE_INSTALL_PREFIX}/lib/libhdf5.a")
     set(libhdf5_PATH_Debug "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_debug.a")
     set(libhdf5_cpp_PATH_Release "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_cpp.a")
     set(libhdf5_cpp_PATH_Debug "${CMAKE_INSTALL_PREFIX}/lib/libhdf5_cpp_debug.a")
endif()
if(CMAKE_GENERATOR STREQUAL "Ninja")
     set(libhdf5_NINJA BUILD_BYPRODUCTS "${libhdf5_PATH_${CMAKE_BUILD_TYPE}}")
     set(libhdf5_cpp_NINJA BUILD_BYPRODUCTS "${libhdf5_cpp_PATH_${CMAKE_BUILD_TYPE}}")
endif()

# Download, build and install hdf5.
# macOS Universal Binaries is supported since 2021/02/07, github-311.
externalproject_add(${PROJECT_NAME}
     GIT_REPOSITORY    https://github.com/HDFGroup/hdf5.git
     GIT_TAG           hdf5-1_14_3
     GIT_SHALLOW       TRUE
     GIT_CONFIG        advice.detachedHead=false;${GIT_CREDENTIAL_EXTERNAL}

     CMAKE_ARGS
          -DHDF5_ENABLE_DEV_WARNINGS=OFF
          -DHDF5_ENABLE_ALL_WARNINGS=OFF
          -DHDF5_DISABLE_COMPILER_WARNINGS=ON
          -DHDF5_ENABLE_COVERAGE=OFF
          -DHDF5_ENABLE_USING_MEMCHECKER=OFF
          -DHDF5_ENABLE_PREADWRITE=ON
          -DHDF5_ENABLE_DEPRECATED_SYMBOLS=ON
          -DHDF5_MINGW_STATIC_GCC_LIBS=OFF
          -DHDF5_BUILD_GENERATORS=OFF
          -DHDF5_ENABLE_TRACE=OFF
          -DHDF5_ENABLE_INSTRUMENT=OFF
          -DHDF5_ENABLE_EMBEDDED_LIBINFO=ON
          -DHDF5_ENABLE_HDFS=OFF
          -DHDF5_ENABLE_PARALLEL=OFF
          -DHDF5_ENABLE_THREADSAFE=OFF
          -DHDF5_ENABLE_MAP_API=OFF
          -DHDF5_BUILD_UTILS=OFF
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
          -DHDF5_ENABLE_SYMBOLS=OFF
          -DHDF5_ENABLE_OPTIMIZATION=ON
          -DBUILD_TESTING=OFF
          -DBUILD_STATIC_LIBS=ON
          -DBUILD_SHARED_LIBS=OFF
          -DBUILD_STATIC_EXECS=OFF
          -DZLIB_PACKAGE_NAME:STRING=NOT_FOUND  # Hack to avoid finding some zlib config files already installed and not skipping find_package(ZLIB)
          -DZLIB_ROOT:PATH=${CMAKE_INSTALL_PREFIX}
          -DZLIB_LIBRARY:PATH=${zlib_INSTALL_LIB_PATH}  # Using ZLIB_ROOT only is finding the shared lib instead of the static one
          -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
          -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}
          -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
          -DCMAKE_C_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
          -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG_EXTERNAL}
          -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
          -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
          -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS_EXTERNAL}
          -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG_EXTERNAL}
          -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
          -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
          -DCMAKE_SHARED_LINKER_FLAGS_DEBUG:STRING=${CMAKE_SHARED_LINKER_FLAGS_DEBUG}
          -DCMAKE_SHARED_LINKER_FLAGS_RELEASE:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELEASE}
          -DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}
          ${EXTERNALPROJECT_OSX_CONFIG}
          ${EXTERNALPROJECT_BUILD_TYPE_CMD}

     ${libhdf5_NINJA}
     ${libhdf5_cpp_NINJA}

     DEPENDS zlib_external

     INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
     UPDATE_COMMAND ""  # For some reason, it is required to avoid rebuild systematically everything...
     UPDATE_DISCONNECTED ${BUILD_OFFLINE}
)

# Import the generated libraries as targets
add_library(hdf5::zlib STATIC IMPORTED GLOBAL)
set_target_properties(hdf5::zlib PROPERTIES
     IMPORTED_CONFIGURATIONS "Debug;Release"
     IMPORTED_LOCATION ${zlib_PATH_Release}
     IMPORTED_LOCATION_DEBUG ${zlib_PATH_Debug}
)
add_library(hdf5::hdf5 STATIC IMPORTED GLOBAL)
set_target_properties(hdf5::hdf5 PROPERTIES
     IMPORTED_CONFIGURATIONS "Debug;Release"
     IMPORTED_LOCATION ${libhdf5_PATH_Release}
     IMPORTED_LOCATION_DEBUG ${libhdf5_PATH_Debug}
)
add_library(hdf5::hdf5_cpp STATIC IMPORTED GLOBAL)
set_target_properties(hdf5::hdf5_cpp PROPERTIES
     IMPORTED_CONFIGURATIONS "Debug;Release"
     IMPORTED_LOCATION ${libhdf5_cpp_PATH_Release}
     IMPORTED_LOCATION_DEBUG ${libhdf5_cpp_PATH_Debug}
)
