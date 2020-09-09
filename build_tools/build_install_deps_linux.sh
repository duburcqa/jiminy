#!/bin/bash
set -eo

################################## Configure the environment ###########################################

### Set the build type to "Release" if undefined
if [ -z ${BUILD_TYPE} ]; then
  BUILD_TYPE="Release"
  echo "BUILD_TYPE is unset. Defaulting to 'Release'."
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

### Checkout boost and its submodules.
#   Boost numeric odeint < 1.71 does not support eigen3 > 3.2,
#   and eigen < 3.3 build fails on windows because of a cmake error
#   Note that Boost 1.72 is not yet officially supported by Cmake 3.16, which is the "default" version used on Windows 10.
git clone -b "boost-1.71.0" https://github.com/boostorg/boost.git "$RootDir/boost"
cd "$RootDir/boost"
git submodule --quiet update --init --recursive --jobs 8

### Checkout eigen3
git clone -b "3.3.7" https://github.com/eigenteam/eigen-git-mirror.git "$RootDir/eigen3"

### Checkout eigenpy and its submodules
git clone -b "v2.5.0" https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
cd "$RootDir/eigenpy"
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_linux/eigenpy.patch"

### Checkout tinyxml (robotology fork for cmake compatibility)
git clone -b "master" https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"
cd "$RootDir/tinyxml"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_linux/tinyxml.patch"

### Checkout console_bridge
git clone -b "0.4.4" https://github.com/ros/console_bridge.git "$RootDir/console_bridge"

### Checkout urdfdom_headers
git clone -b "1.0.5" https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"

### Checkout urdfdom
git clone -b "1.0.4" https://github.com/ros/urdfdom.git "$RootDir/urdfdom"
cd "$RootDir/urdfdom"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_linux/urdfdom.patch"

### Checkout assimp
git clone -b "v5.0.1" https://github.com/assimp/assimp.git "$RootDir/assimp"
cd "$RootDir/assimp"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_linux/assimp.patch"

### Checkout hpp-fcl
git clone -b "v1.5.3" https://github.com/humanoid-path-planner/hpp-fcl.git "$RootDir/hpp-fcl"
cd "$RootDir/hpp-fcl"
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_linux/hppfcl.patch"

### Checkout pinocchio and its submodules
git clone -b "v2.5.0" https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
cd "$RootDir/pinocchio"
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_linux/pinocchio.patch"

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR

### Build and install the build tool b2 (build-ception !)
cd "$RootDir/boost"
./bootstrap.sh --prefix="$InstallDir" --with-python="${PYTHON_EXECUTABLE}"

### File "project-config.jam" create by bootstrap must be edited manually
#   to specify Python included dir manually, since it is not detected
#   sucessfully in some cases.
PYTHON_VERSION="$(${PYTHON_EXECUTABLE} -c "import sys; print (\"%d.%d\" % (sys.version_info[0], sys.version_info[1]))")"
PYTHON_ROOT="$(${PYTHON_EXECUTABLE} -c "import sys; print(sys.prefix)")"
PYTHON_INCLUDE_DIRS="$(${PYTHON_EXECUTABLE} -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_python_inc())")"
PYTHON_CONFIG_JAM="using python : ${PYTHON_VERSION} : ${PYTHON_ROOT} : ${PYTHON_INCLUDE_DIRS} ;"
sed -i "/using python/c ${PYTHON_CONFIG_JAM}" ./project-config.jam

### Build and install and install boost
#   (Replace -d0 option by -d1 and remove -q option to check compilation errors)
BuildTypeB2="$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
mkdir -p "$RootDir/boost/build"
./b2 --prefix="$InstallDir" --build-dir="$RootDir/boost/build" \
     --with-chrono --with-timer --with-date_time --with-headers --with-math \
     --with-stacktrace --with-system --with-filesystem --with-atomic \
     --with-thread --with-serialization --with-test --with-python \
     --build-type=minimal architecture=x86 address-model=64 threading=multi \
     --layout=system link=shared runtime-link=shared \
     toolset=gcc cxxflags="-std=c++11 -fPIC" variant="$BuildTypeB2" install -q -d0 -j2

#################################### Build and install eigen3 ##########################################

mkdir -p "$RootDir/eigen3/build"
cd "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=ON \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################### Build and install eigenpy ##########################################

mkdir -p "$RootDir/eigenpy/build"
cd "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" -DBoost_USE_STATIC_LIBS=OFF \
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################## Build and install tinyxml ###########################################

mkdir -p "$RootDir/tinyxml/build"
cd "$RootDir/tinyxml/build"
cmake "$RootDir/tinyxml" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC -DTIXML_USE_STL" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################## Build and install console_bridge ########################################

mkdir -p "$RootDir/console_bridge/build"
cd "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################## Build and install urdfdom_headers ######################################

mkdir -p "$RootDir/urdfdom_headers/build"
cd "$RootDir/urdfdom_headers/build"
cmake "$RootDir/urdfdom_headers" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################# Build and install urdfdom ###########################################

mkdir -p "$RootDir/urdfdom/build"
cd "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DBUILD_TESTING=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################## Build and install assimp ######################################

mkdir -p "$RootDir/assimp/build"
cd "$RootDir/assimp/build"
cmake "$RootDir/assimp" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=OFF \
      -DASSIMP_BUILD_SAMPLES=OFF -DBUILD_DOCS=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC -Wno-strict-overflow" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################# Build and install hpp-fcl ###########################################

mkdir -p "$RootDir/hpp-fcl/build"
cd "$RootDir/hpp-fcl/build"
cmake "$RootDir/hpp-fcl" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" -DBoost_USE_STATIC_LIBS=OFF \
      -DBUILD_PYTHON_INTERFACE=ON -DHPP_FCL_HAS_QHULL=OFF -DINSTALL_DOCUMENTATION=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC -Wno-unused-parameter" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################ Build and install Pinocchio ##########################################

### Build and install pinocchio, finally !
mkdir -p "$RootDir/pinocchio/build"
cd "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" -DBoost_USE_STATIC_LIBS=OFF \
      -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF \
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC -Wno-unused-local-typedefs -Wno-uninitialized" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2
