#!/bin/bash
set -eo

################################## Configure the environment ###########################################

### Set the build type to "Release" if undefined
if [ -z ${BUILD_TYPE} ]; then
  BUILD_TYPE="Release"
  echo "BUILD_TYPE is unset. Defaulting to '${BUILD_TYPE}'."
fi

### Set the macos sdk version min if undefined
if [ -z ${MACOSX_DEPLOYMENT_TARGET} ]; then
  MACOSX_DEPLOYMENT_TARGET="10.9"
  echo "MACOSX_DEPLOYMENT_TARGET is unset. Defaulting to '${MACOSX_DEPLOYMENT_TARGET}'."
fi

### Set the build architecture if undefined
if [ -z ${OSX_ARCHITECTURES} ]; then
  OSX_ARCHITECTURES="x86_64"
  echo "OSX_ARCHITECTURES is unset. Defaulting to '${OSX_ARCHITECTURES}'."
fi

### Set common CMAKE_C/CXX_FLAGS
CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC"
if [ "${BUILD_TYPE}" == "Release" ]; then
  CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -O3 -DNDEBUG"
else
  CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -O0 -g"
fi

### Get the fullpath of Jiminy project
ScriptDir="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
RootDir="$(dirname $ScriptDir)"

### Set the fullpath of the install directory, then creates it
InstallDir="$RootDir/install"
mkdir -p "$InstallDir"

### Eigenpy and Pinocchio are using the deprecated FindPythonInterp
#   cmake helper to detect Python executable, which is not working
#   properly when several executables exist.
if [ -z ${PYTHON_EXECUTABLE} ]; then
  PYTHON_EXECUTABLE=$(python -c "import sys; sys.stdout.write(sys.executable)")
  echo "PYTHON_EXECUTABLE is unset. Defaulting to '${PYTHON_EXECUTABLE}'."
fi

### Remove the preinstalled boost library from search path
unset Boost_ROOT

################################## Checkout the dependencies ###########################################

#   Patches can be generated using `git diff --submodule=diff` command.

### Checkout boost and its submodules
#   Note that boost python must be patched to fix error handling at import (boost < 1.76),
#   and fix support of PyPy (boost < 1.75).
#   Boost >= 1.75 is required to compile ouf-of-the-box on MacOS for intel and Apple Silicon.
if [ ! -d "$RootDir/boost" ]; then
  git clone https://github.com/boostorg/boost.git "$RootDir/boost"
fi
cd "$RootDir/boost"
git reset --hard
git checkout --force "boost-1.76.0"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8

### Checkout eigen3
if [ ! -d "$RootDir/eigen3" ]; then
  git clone https://gitlab.com/libeigen/eigen.git "$RootDir/eigen3"
fi
cd "$RootDir/eigen3"
git reset --hard
git checkout --force "3.3.9"

### Checkout eigenpy and its submodules
if [ ! -d "$RootDir/eigenpy" ]; then
  git clone https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
fi
cd "$RootDir/eigenpy"
git reset --hard
git checkout --force "v2.6.4"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/eigenpy.patch"

### Checkout tinyxml (robotology fork for cmake compatibility)
if [ ! -d "$RootDir/tinyxml" ]; then
  git clone https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"
fi
cd "$RootDir/tinyxml"
git reset --hard
git checkout --force "master"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/tinyxml.patch"

### Checkout console_bridge, then apply some patches (generated using `git diff --submodule=diff`)
if [ ! -d "$RootDir/console_bridge" ]; then
  git clone https://github.com/ros/console_bridge.git "$RootDir/console_bridge"
fi
cd "$RootDir/console_bridge"
git reset --hard
git checkout --force "0.4.4"

### Checkout urdfdom_headers
if [ ! -d "$RootDir/urdfdom_headers" ]; then
  git clone https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"
fi
cd "$RootDir/urdfdom_headers"
git reset --hard
git checkout --force "1.0.5"

### Checkout urdfdom, then apply some patches (generated using `git diff --submodule=diff`)
if [ ! -d "$RootDir/urdfdom" ]; then
  git clone https://github.com/ros/urdfdom.git "$RootDir/urdfdom"
fi
cd "$RootDir/urdfdom"
git checkout --force "1.0.4"
git reset --hard
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/urdfdom.patch"

### Checkout assimp
if [ ! -d "$RootDir/assimp" ]; then
  git clone https://github.com/assimp/assimp.git "$RootDir/assimp"
fi
cd "$RootDir/assimp"
git reset --hard
git checkout --force "v5.0.1"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/assimp.patch"

### Checkout hpp-fcl
if [ ! -d "$RootDir/hpp-fcl" ]; then
  git clone https://github.com/humanoid-path-planner/hpp-fcl.git "$RootDir/hpp-fcl"
  git config --global url."https://".insteadOf git://
fi
cd "$RootDir/hpp-fcl"
git reset --hard
git checkout --force "v1.7.4"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/hppfcl.patch"
cd "$RootDir/hpp-fcl/third-parties/qhull"
git checkout --force "v8.0.2"

### Checkout pinocchio and its submodules
if [ ! -d "$RootDir/pinocchio" ]; then
  git clone https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
  git config --global url."https://".insteadOf git://
fi
cd "$RootDir/pinocchio"
git reset --hard
git checkout --force "v2.5.6"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/pinocchio.patch"

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR

### Build and install the build tool b2 (build-ception !)
cd "$RootDir/boost"
./bootstrap.sh --prefix="$InstallDir" --with-python="${PYTHON_EXECUTABLE}"

### File "project-config.jam" create by bootstrap must be edited manually
#   to specify Python included dir manually, since it is not detected
#   successfully in some cases.
PYTHON_VERSION="$(${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('py_version_short'))")"
PYTHON_INCLUDE_DIRS="$(${PYTHON_EXECUTABLE} -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_python_inc())")"
PYTHON_CONFIG_JAM="using python : ${PYTHON_VERSION} : ${PYTHON_EXECUTABLE} : ${PYTHON_INCLUDE_DIRS} ;"
sed -i.old "/using python/c\\
${PYTHON_CONFIG_JAM}
" project-config.jam

### Build and install and install boost
#   (Replace -d0 option by -d1 and remove -q option to check compilation errors)
BuildTypeB2="$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
CMAKE_CXX_FLAGS_B2="-std=c++17"
if [ "${OSTYPE//[0-9.]/}" == "darwin" ]; then
  CMAKE_CXX_FLAGS_B2="${CMAKE_CXX_FLAGS_B2} -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
fi
if grep -q ";" <<< "${OSX_ARCHITECTURES}" ; then
    CMAKE_CXX_FLAGS_B2="${CMAKE_CXX_FLAGS_B2} $(echo "-arch ${OSX_ARCHITECTURES}" | sed "s/;/ -arch /g")"
fi

mkdir -p "$RootDir/boost/build"
./b2 --prefix="$InstallDir" --build-dir="$RootDir/boost/build" \
     --with-chrono --with-timer --with-date_time --with-system --with-test \
     --with-filesystem --with-atomic --with-serialization --with-thread \
     --build-type=minimal --layout=system --lto=off \
     architecture=${B2_ARCHITECTURE_TYPE} address-model=64 \
     threading=single link=static runtime-link=static debug-symbols=off \
     cxxflags="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_B2}" \
     linkflags="${CMAKE_CXX_FLAGS_B2}" \
     variant="$BuildTypeB2" install -q -d0 -j2
./b2 --prefix="$InstallDir" --build-dir="$RootDir/boost/build" \
     --with-python \
     --build-type=minimal --layout=system --lto=off \
     architecture=${B2_ARCHITECTURE_TYPE} address-model=64 \
     threading=single link=shared runtime-link=shared debug-symbols=off \
     cxxflags="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_B2}" \
     linkflags="${CMAKE_CXX_FLAGS_B2}" \
     variant="$BuildTypeB2" install -q -d0 -j2

#################################### Build and install eigen3 ##########################################

mkdir -p "$RootDir/eigen3/build"
cd "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=ON \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################### Build and install eigenpy ##########################################

mkdir -p "$RootDir/eigenpy/build"
cd "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-strict-aliasing" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################## Build and install tinyxml ###########################################

mkdir -p "$RootDir/tinyxml/build"
cd "$RootDir/tinyxml/build"
cmake "$RootDir/tinyxml" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DTIXML_USE_STL" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################## Build and install console_bridge ########################################

mkdir -p "$RootDir/console_bridge/build"
cd "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################### Build and install urdfdom_headers ######################################

mkdir -p "$RootDir/urdfdom_headers/build"
cd "$RootDir/urdfdom_headers/build"
cmake "$RootDir/urdfdom_headers" -Wno-dev -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################## Build and install urdfdom ###########################################

mkdir -p "$RootDir/urdfdom/build"
cd "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DBUILD_TESTING=OFF \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

###################################### Build and install assimp ########################################

mkdir -p "$RootDir/assimp/build"
cd "$RootDir/assimp/build"
cmake "$RootDir/assimp" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=OFF \
      -DASSIMP_BUILD_SAMPLES=OFF -DBUILD_DOCS=OFF -DCMAKE_C_FLAGS="${CMAKE_CXX_FLAGS}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-strict-overflow -Wno-tautological-compare" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################# Build and install qhull and hpp-fcl ######################################

mkdir -p "$RootDir/hpp-fcl/third-parties/qhull/build"
cd "$RootDir/hpp-fcl/third-parties/qhull/build"
cmake "$RootDir/hpp-fcl/third-parties/qhull" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_STATIC_LIBS=ON \
      -DCMAKE_C_FLAGS="${CMAKE_CXX_FLAGS}" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-conversion" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

mkdir -p "$RootDir/hpp-fcl/build"
cd "$RootDir/hpp-fcl/build"
cmake "$RootDir/hpp-fcl" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBUILD_PYTHON_INTERFACE=ON -DHPP_FCL_HAS_QHULL=ON \
      -DINSTALL_DOCUMENTATION=OFF -DENABLE_PYTHON_DOXYGEN_AUTODOC=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-ignored-qualifiers" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################# Build and install Pinocchio ##########################################

### Build and install pinocchio, finally !
mkdir -p "$RootDir/pinocchio/build"
cd "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -Wno-dev -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON \
      -DBUILD_WITH_AUTODIFF_SUPPORT=OFF -DBUILD_WITH_CASADI_SUPPORT=OFF -DBUILD_WITH_CODEGEN_SUPPORT=OFF \
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DBOOST_BIND_GLOBAL_PLACEHOLDERS -Wno-unused-local-typedefs $(
      ) -Wno-uninitialized" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2
