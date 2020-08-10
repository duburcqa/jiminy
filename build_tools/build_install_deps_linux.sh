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
git clone -b "boost-1.72.0" https://github.com/boostorg/boost.git "$RootDir/boost"
cd "$RootDir/boost"
git submodule --quiet update --init --recursive --jobs 8

### Checkout eigen3
git clone -b "3.3.7" https://github.com/eigenteam/eigen-git-mirror.git "$RootDir/eigen3"

### Checkout eigenpy and its submodules
git clone -b "v2.4.3" https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
cd "$RootDir/eigenpy"
git submodule --quiet update --init --recursive --jobs 8

### Checkout tinyxml (robotology fork for cmake compatibility)
git clone -b "master" https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"

### Checkout console_bridge
git clone -b "0.4.4" https://github.com/ros/console_bridge.git "$RootDir/console_bridge"

### Checkout urdfdom_headers
git clone -b "1.0.5" https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"

### Checkout urdfdom
git clone -b "1.0.4" https://github.com/ros/urdfdom.git "$RootDir/urdfdom"

### Checkout pinocchio and its submodules
git clone -b "v2.4.7" https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
cd "$RootDir/pinocchio"
git submodule --quiet update --init --recursive --jobs 8

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

### Build and install and install boost (Replace -d0 option by -d1 to check compilation errors)
BuildTypeB2="$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
mkdir -p "$RootDir/boost/build"
./b2 --prefix="$InstallDir" --build-dir="$RootDir/boost/build" \
     --with-date_time --with-filesystem --with-headers --with-math --with-python \
     --with-serialization --with-stacktrace --with-system --with-test --with-thread \
     --with-python --build-type=minimal architecture=x86 address-model=64 \
     threading=multi --layout=system link=shared runtime-link=shared \
     toolset=gcc cxxflags="-std=c++11 -fPIC" variant="$BuildTypeB2" install -q -d0 -j2

#################################### Build and install eigen3 ##########################################

mkdir -p "$RootDir/eigen3/build"
cd "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=ON \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################### Build and install eigenpy ##########################################

### Must patch line 92 of cmake/python.cmake for centOS 6 and above to avoid looking for PYTHON_LIBRARY,
#   since it is both irrelevant and failing in some cases. Indeed, it is the case for
#   the official 'manylinux2010' docker image, for which such library is not available.
if ( rpm -q centos-release >/dev/null 2>&1 ) ; then
    sed -i '92s/.*/'"\
    "'FIND_PACKAGE("Python${_PYTHON_VERSION_MAJOR}" COMPONENTS Interpreter) \n '"\
    "'execute_process(COMMAND "${Python${_PYTHON_VERSION_MAJOR}_EXECUTABLE}" -c '"\
    "'                        "import distutils.sysconfig as sysconfig; print(sysconfig.get_python_inc())" '"\
    "'                OUTPUT_STRIP_TRAILING_WHITESPACE '"\
    "'                OUTPUT_VARIABLE Python${_PYTHON_VERSION_MAJOR}_INCLUDE_DIRS) /' \
    "$RootDir/eigenpy/cmake/python.cmake"
fi

### Remove line 73 of boost.cmake to disable library type enforced SHARED
sed -i '73s/.*/ /' "$RootDir/eigenpy/cmake/boost.cmake"

### Must patch /CMakefile.txt to disable library type enforced SHARED
sed -i 's/SHARED //g' "$RootDir/eigenpy/CMakeLists.txt"

mkdir -p "$RootDir/eigenpy/build"
cd "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DBoost_USE_STATIC_LIBS=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################## Build and install tinyxml ###########################################

### Patch line 19 of CMakeLists.txt to set the right default library directory depending to the linux distro.
sed -i '19s/.*/'\
'include(GNUInstallDirs) \n '\
'set(TINYXML_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Installation directory for libraries") /' \
"$RootDir/tinyxml/CMakeLists.txt"

mkdir -p "$RootDir/tinyxml/build"
cd "$RootDir/tinyxml/build"
cmake "$RootDir/tinyxml" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################## Build and install console_bridge ########################################

mkdir -p "$RootDir/console_bridge/build"
cd "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

############################## Build and install urdfdom_headers ######################################

mkdir -p "$RootDir/urdfdom_headers/build"
cd "$RootDir/urdfdom_headers/build"
cmake "$RootDir/urdfdom_headers" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################# Build and install urdfdom ###########################################

### Must patch line 71 of CMakeLists.txt to add TinyXML dependency to cmake configuration files generator
sed -i '71s/.*/ set(PKG_DEPENDS urdfdom_headers console_bridge TinyXML) /' \
"$RootDir/urdfdom/CMakeLists.txt"

### Must patch line 81 of CMakeLists.txt to add TinyXML dependency to pkgconfig files generator
sed -i '81s/.*/ set(PKG_URDF_LIBS "-lurdfdom_sensor -lurdfdom_model_state -lurdfdom_model -lurdfdom_world -ltinyxml") /' \
"$RootDir/urdfdom/CMakeLists.txt"

### Must patch /urdf_parser/CMakeLists.txt to disable library type enforced SHARED
sed -i 's/SHARED //g' "$RootDir/urdfdom/urdf_parser/CMakeLists.txt"

mkdir -p "$RootDir/urdfdom/build"
cd "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DBUILD_TESTING=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC -DURDFDOM_STATIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

################################ Build and install Pinocchio ##########################################

### Must patch line 92 of cmake/python.cmake for centOS 6 and above.
if ( rpm -q centos-release >/dev/null 2>&1 ) ; then
    sed -i '92s/.*/'"\
    "'FIND_PACKAGE("Python${_PYTHON_VERSION_MAJOR}" COMPONENTS Interpreter) \n '"\
    "'execute_process(COMMAND "${Python${_PYTHON_VERSION_MAJOR}_EXECUTABLE}" -c '"\
    "'                        "import distutils.sysconfig as sysconfig; print(sysconfig.get_python_inc())" '"\
    "'                OUTPUT_STRIP_TRAILING_WHITESPACE '"\
    "'                OUTPUT_VARIABLE Python${_PYTHON_VERSION_MAJOR}_INCLUDE_DIRS) /' \
    "$RootDir/pinocchio/cmake/python.cmake"
fi

### Remove line 73 of boost.cmake to disable library type enforced SHARED
sed -i '73s/.*/ /' "$RootDir/pinocchio/cmake/boost.cmake"

### Must patch /src/CMakefile.txt to disable library type enforced SHARED
sed -i 's/SHARED //g' "$RootDir/pinocchio/src/CMakeLists.txt"

### Remove every std::vector bindings of native types, since it makes absolutely no sense to bind such ambiguous types
find "$RootDir/pinocchio" -type f -name "*.hpp" -exec ex -sc "g/StdVectorPythonVisitor</d" -cx {} ';'

### Build and install pinocchio, finally !
mkdir -p "$RootDir/pinocchio/build"
cd "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DPYTHON_STANDARD_LAYOUT=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBUILD_WITH_COLLISION_SUPPORT=OFF -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF \
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON -DBoost_USE_STATIC_LIBS=OFF \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC -DURDFDOM_STATIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2
