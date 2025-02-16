#!/bin/bash
set -eo

################################## Configure the environment ###########################################

### Set the build type to "Release" if undefined
if [ -z ${BUILD_TYPE} ]; then
  BUILD_TYPE="Release"
  echo "BUILD_TYPE is unset. Defaulting to '${BUILD_TYPE}'."
fi

### Set the macos sdk version min if undefined
if [ "$(uname -s)" = "Darwin" ]; then
  if [ -z ${OSX_DEPLOYMENT_TARGET} ]; then
    OSX_DEPLOYMENT_TARGET="10.15"
    echo "OSX_DEPLOYMENT_TARGET is unset. Defaulting to '${OSX_DEPLOYMENT_TARGET}'."
  fi

  ### Set the build architecture if undefined
  if [ -z ${OSX_ARCHITECTURES} ]; then
    OSX_ARCHITECTURES="x86_64;arm64"
    echo "OSX_ARCHITECTURES is unset. Defaulting to '${OSX_ARCHITECTURES}'."
  fi
fi

### Set the compiler(s) if undefined
if [ -z ${C_COMPILER} ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
      C_COMPILER="$(which clang)"
  else
      C_COMPILER="$(which gcc)"
  fi
  echo "C_COMPILER is unset. Defaulting to '${C_COMPILER}'."
fi
if [ -z ${CXX_COMPILER} ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
      CXX_COMPILER="$(which clang++)"
  else
      CXX_COMPILER="$(which g++)"
  fi
  echo "CXX_COMPILER is unset. Defaulting to '${CXX_COMPILER}'."
fi

### Set common CMAKE_C/CXX_FLAGS
CXX_FLAGS="${CXX_FLAGS} -Wno-unknown-warning-option -Wno-enum-constexpr-conversion"
if [ "${BUILD_TYPE}" == "Release" ]; then
  CXX_FLAGS="${CXX_FLAGS} -DNDEBUG -O3"
elif [ "${BUILD_TYPE}" == "Debug" ]; then
  CXX_FLAGS="${CXX_FLAGS} -DBOOST_PYTHON_DEBUG -O0 -g"
elif [ "${BUILD_TYPE}" == "RelWithDebInfo" ]; then
  CXX_FLAGS="${CXX_FLAGS} -DNDEBUG -O2 -g"
fi
echo "CXX_FLAGS: ${CXX_FLAGS}"

### Get the fullpath of Jiminy project
ScriptDir="$(cd "$(dirname -- $0)" >/dev/null 2>&1 && pwd)"
RootDir="$(dirname -- ${ScriptDir})"

### Set the fullpath of the install directory, then creates it
InstallDir="${RootDir}/install"
mkdir -p "${InstallDir}"

### Eigenpy and Pinocchio are using the deprecated FindPythonInterp
#   cmake helper to detect Python executable, which is not working
#   properly when several executables exist.
if [ -z ${PYTHON_EXECUTABLE} ]; then
  PYTHON_EXECUTABLE=$(python3 -c "import sys; sys.stdout.write(sys.executable)")
  echo "PYTHON_EXECUTABLE is unset. Defaulting to '${PYTHON_EXECUTABLE}'."
fi

### Configure site-packages pythonic "symlink" pointing to install directory
PYTHON_USER_SITELIB="$("${PYTHON_EXECUTABLE}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if [ ! -w "$PYTHON_USER_SITELIB" ]; then
    PYTHON_USER_SITELIB="$("${PYTHON_EXECUTABLE}" -m site --user-site)"
fi
PYTHON_VERSION="$(${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('py_version_short'))")"
mkdir -p "${PYTHON_USER_SITELIB}"
echo "${InstallDir}/lib/python${PYTHON_VERSION}/site-packages" > "${PYTHON_USER_SITELIB}/install_site.pth"

### Add install library to path. This is necessary to generate stubs.
LD_LIBRARY_PATH="${InstallDir}/lib:${InstallDir}/lib64:/usr/local/lib"
DYLD_FALLBACK_LIBRARY_PATH="$LD_LIBRARY_PATH"

### Remove the preinstalled boost library from search path
unset Boost_ROOT

################################## Checkout the dependencies ###########################################

#   Patches can be generated using `git diff --submodule=diff` command.

### Checkout boost and its submodules
#   - Boost.Python == 1.75 fixes support of PyPy
#   - Boost.Python == 1.76 fixes error handling at import
#   - Boost >= 1.75 is required to compile ouf-of-the-box on MacOS for intel and Apple Silicon
#   - Boost < 1.77 causes compilation failure with gcc-12 if not patched
#   - Boost >= 1.77 affects the memory layout to improve alignment, breaking retro-compatibility
#   - Boost.Python == 1.87 fixes memory alignment issues
if [ ! -d "${RootDir}/boost" ]; then
  git clone --depth 1 https://github.com/boostorg/boost.git "${RootDir}/boost"
fi
rm -rf "${RootDir}/boost/build"
mkdir -p "${RootDir}/boost/build"
cd "${RootDir}/boost"
git reset --hard
git fetch origin "boost-1.87.0" && git checkout --force FETCH_HEAD || true
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8

### Checkout eigen3
if [ ! -d "${RootDir}/eigen3" ]; then
  git clone --depth 1 https://gitlab.com/libeigen/eigen.git "${RootDir}/eigen3"
fi
rm -rf "${RootDir}/eigen3/build"
mkdir -p "${RootDir}/eigen3/build"
cd "${RootDir}/eigen3"
git reset --hard
git fetch origin "3.4.0" && git checkout --force FETCH_HEAD || true

### Checkout eigenpy and its submodules
if [ ! -d "${RootDir}/eigenpy" ]; then
  git clone --depth 1 https://github.com/stack-of-tasks/eigenpy.git "${RootDir}/eigenpy"
fi
rm -rf "${RootDir}/eigenpy/build"
mkdir -p "${RootDir}/eigenpy/build"
cd "${RootDir}/eigenpy"
git reset --hard
git fetch origin "v3.5.0" && git checkout --force FETCH_HEAD || true
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
git apply --reject --whitespace=fix "${RootDir}/build_tools/patch_deps_unix/eigenpy.patch"

### Checkout tinyxml2
if [ ! -d "${RootDir}/tinyxml2" ]; then
  git clone --depth 1 https://github.com/leethomason/tinyxml2 "${RootDir}/tinyxml2"
fi
rm -rf "${RootDir}/tinyxml2/build"
mkdir -p "${RootDir}/tinyxml2/build"
cd "${RootDir}/tinyxml2"
git reset --hard
git fetch origin "master" && git checkout --force FETCH_HEAD || true

### Checkout console_bridge, then apply some patches (generated using `git diff --submodule=diff`)
if [ ! -d "${RootDir}/console_bridge" ]; then
  git clone --depth 1 https://github.com/ros/console_bridge.git "${RootDir}/console_bridge"
fi
rm -rf "${RootDir}/console_bridge/build"
mkdir -p "${RootDir}/console_bridge/build"
cd "${RootDir}/console_bridge"
git reset --hard
git fetch origin "1.0.2" && git checkout --force FETCH_HEAD || true

### Checkout urdfdom_headers
if [ ! -d "${RootDir}/urdfdom_headers" ]; then
  git clone --depth 1 https://github.com/ros/urdfdom_headers.git "${RootDir}/urdfdom_headers"
fi
rm -rf "${RootDir}/urdfdom_headers/build"
mkdir -p "${RootDir}/urdfdom_headers/build"
cd "${RootDir}/urdfdom_headers"
git reset --hard
git fetch origin "1.1.2" && git checkout --force FETCH_HEAD || true

### Checkout urdfdom, then apply some patches (generated using `git diff --submodule=diff`)
if [ ! -d "${RootDir}/urdfdom" ]; then
  git clone --depth 1 https://github.com/ros/urdfdom.git "${RootDir}/urdfdom"
fi
rm -rf "${RootDir}/urdfdom/build"
mkdir -p "${RootDir}/urdfdom/build"
cd "${RootDir}/urdfdom"
git reset --hard
git fetch origin "4.0.1" && git checkout --force FETCH_HEAD || true
git apply --reject --whitespace=fix "${RootDir}/build_tools/patch_deps_unix/urdfdom.patch"

### Checkout CppAD
if [ ! -d "${RootDir}/cppad" ]; then
  git clone --depth 1 https://github.com/coin-or/CppAD.git "${RootDir}/cppad"
fi
rm -rf "${RootDir}/cppad/build"
mkdir -p "${RootDir}/cppad/build"
cd "${RootDir}/cppad"
git reset --hard
git fetch origin "20240000.7" && git checkout --force FETCH_HEAD || true

### Checkout CppADCodeGen
if [ ! -d "${RootDir}/cppadcodegen" ]; then
  git clone --depth 1 https://github.com/joaoleal/CppADCodeGen.git "${RootDir}/cppadcodegen"
fi
rm -rf "${RootDir}/cppadcodegen/build"
mkdir -p "${RootDir}/cppadcodegen/build"
cd "${RootDir}/cppadcodegen"
git reset --hard
git fetch origin "v2.5.0" && git checkout --force FETCH_HEAD || true

### Checkout assimp
if [ ! -d "${RootDir}/assimp" ]; then
  git clone --depth 1 https://github.com/assimp/assimp.git "${RootDir}/assimp"
fi
rm -rf "${RootDir}/assimp/build"
mkdir -p "${RootDir}/assimp/build"
cd "${RootDir}/assimp"
git reset --hard
git fetch origin "v5.4.3" && git checkout --force FETCH_HEAD || true

### Checkout hpp-fcl
if [ ! -d "${RootDir}/hpp-fcl" ]; then
  git clone --depth 1 https://github.com/humanoid-path-planner/hpp-fcl.git "${RootDir}/hpp-fcl"
fi
rm -rf "${RootDir}/hpp-fcl/build"
mkdir -p "${RootDir}/hpp-fcl/build"
cd "${RootDir}/hpp-fcl"
git reset --hard
git fetch origin "v2.4.4" && git checkout --force FETCH_HEAD || true
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
git apply --reject --whitespace=fix "${RootDir}/build_tools/patch_deps_unix/hppfcl.patch"
cd "${RootDir}/hpp-fcl/third-parties/qhull"
rm -rf "${RootDir}/hpp-fcl/third-parties/qhull/build"
git fetch origin "v8.0.2" && git checkout --force FETCH_HEAD || true

### Checkout pinocchio and its submodules
if [ ! -d "${RootDir}/pinocchio" ]; then
  git clone --depth 1 https://github.com/stack-of-tasks/pinocchio.git "${RootDir}/pinocchio"
fi
rm -rf "${RootDir}/pinocchio/build"
mkdir -p "${RootDir}/pinocchio/build"
cd "${RootDir}/pinocchio"
git reset --hard
git fetch origin "v2.7.0" && git checkout --force FETCH_HEAD || true
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
git apply --reject --whitespace=fix "${RootDir}/build_tools/patch_deps_unix/pinocchio.patch"

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR

# Must add toolset to search path for boost build system to find it
PATH="${PATH}:$(dirname -- ${C_COMPILER})"

### Build and install the build tool b2 (build-ception !)
cd "${RootDir}/boost"
./bootstrap.sh --with-toolset="$(basename -- ${C_COMPILER})" \
               --prefix="${InstallDir}" \
               --with-python="${PYTHON_EXECUTABLE}"

### File "project-config.jam" create by bootstrap must be edited manually
#   to specify Python include dir manually, since it is not detected
#   successfully in some cases.
PYTHON_INCLUDE_DIRS="$(${PYTHON_EXECUTABLE} -c "import sysconfig as sysconfig; print(sysconfig.get_path('include'))")"
PYTHON_CONFIG_JAM="using python : ${PYTHON_VERSION} : ${PYTHON_EXECUTABLE} : ${PYTHON_INCLUDE_DIRS} ;"
sed -i.old "/using python/c\\
${PYTHON_CONFIG_JAM}
" project-config.jam

### Build and install and install boost
#   (Replace -d0 option by -d1 and remove -q option to check compilation errors)
if [[ ${OSX_ARCHITECTURES} == *";"* ]] ; then
  ArchB2="arm+x86"
fi
if [ "${BUILD_TYPE}" == "Release" ]; then
  BuildTypeB2="release"
elif [ "${BUILD_TYPE}" == "Debug" ]; then
  BuildTypeB2="debug"
  DebugOptionsB2="debug-symbols=on python-debugging=on"
# elif [ "${BUILD_TYPE}" == "RelWithDebInfo" ]; then
#   BuildTypeB2="profile"
else
  echo "Build type '${BUILD_TYPE}' not supported." >&2
  exit 1
fi
CXX_FLAGS_B2="-fPIC -std=c++17"
if [ "${OSTYPE//[0-9.]/}" == "darwin" ]; then
  CXX_FLAGS_B2="${CXX_FLAGS_B2} -mmacosx-version-min=${OSX_DEPLOYMENT_TARGET}"
fi
if grep -q ";" <<< "${OSX_ARCHITECTURES}" ; then
  CXX_FLAGS_B2="${CXX_FLAGS_B2} $(echo "-arch ${OSX_ARCHITECTURES}" | sed "s/;/ -arch /g")"
fi

# Clean all targets first
./b2 --prefix="$InstallDir" --build-dir="$RootDir/boost/build" \
     --debug-configuration \
     --clean-all \
     architecture="$ArchB2" address-model=64 \
     variant="$BuildTypeB2"

# Compiling everything with static linkage except Boost::Python
mkdir -p "${RootDir}/boost/build"
./b2 --prefix="${InstallDir}" --build-dir="${RootDir}/boost/build" \
     --with-chrono --with-timer --with-date_time --with-system --with-test \
     --with-filesystem --with-atomic --with-serialization --with-thread \
     --build-type=minimal --layout=system --lto=on \
     architecture="$ArchB2" address-model=64 $DebugOptionsB2 \
     threading=single link=static runtime-link=static \
     cxxflags="${CXX_FLAGS} ${CXX_FLAGS_B2}" \
     linkflags="${LINKER_FLAGS} ${CXX_FLAGS_B2}" \
     toolset="$(basename -- ${C_COMPILER})" \
     variant="$BuildTypeB2" install -q -d0 -j4

# Disable Boost::Python assert systematically, even in debug, to avoid
# raising an exception for trying to re-register an existing converter.
./b2 --prefix="${InstallDir}" --build-dir="${RootDir}/boost/build" \
     --with-python \
     --build-type=minimal --layout=system --lto=on \
     architecture="$ArchB2" address-model=64 $DebugOptionsB2 \
     threading=single link=shared runtime-link=shared \
     cxxflags="${CXX_FLAGS} ${CXX_FLAGS_B2} -DNDEBUG" \
     linkflags="${LINKER_FLAGS} ${CXX_FLAGS_B2} " \
     toolset="$(basename -- ${C_COMPILER})" \
     variant="$BuildTypeB2" install -q -d0 -j4

#################################### Build and install eigen3 ##########################################

cd "${RootDir}/eigen3/build"
cmake "${RootDir}/eigen3" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=ON
make install -j4

################################### Build and install eigenpy ##########################################

cd "${RootDir}/eigenpy/build"
cmake "${RootDir}/eigenpy" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" -DCMAKE_PREFIX_PATH="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${LINKER_FLAGS}" \
      -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -DPYTHON_STANDARD_LAYOUT=ON \
      -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="${InstallDir}" -DBoost_INCLUDE_DIR="${InstallDir}/include" \
      -DGENERATE_PYTHON_STUBS=OFF -DBUILD_TESTING=OFF -DBUILD_TESTING_SCIPY=OFF -DINSTALL_DOCUMENTATION=OFF \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS} $(
      ) -Wno-strict-aliasing -Wno-maybe-uninitialized" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

################################## Build and install tinyxml ###########################################

cd "${RootDir}/tinyxml2/build"
cmake "${RootDir}/tinyxml2" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_EXE_LINKER_FLAGS="${LINKER_FLAGS}" \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

############################## Build and install console_bridge ########################################

cd "${RootDir}/console_bridge/build"
cmake "${RootDir}/console_bridge" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

############################### Build and install urdfdom_headers ######################################

cd "${RootDir}/urdfdom_headers/build"
cmake "${RootDir}/urdfdom_headers" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

################################## Build and install urdfdom ###########################################

cd "${RootDir}/urdfdom/build"
cmake "${RootDir}/urdfdom" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" -DCMAKE_PREFIX_PATH="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_EXE_LINKER_FLAGS="${LINKER_FLAGS}" \
      -DBUILD_TESTING=OFF \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

################################### Build and install CppAD ##########################################

cd "${RootDir}/cppad/build"
cmake "${RootDir}/cppad" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

################################### Build and install CppADCodeGen ##########################################

cd "${RootDir}/cppadcodegen/build"
cmake "${RootDir}/cppadcodegen" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DGOOGLETEST_GIT=ON \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

###################################### Build and install assimp ########################################

# C flag 'HAVE_HIDDEN' must be specified to hide internal symbols of zlib that may not be exposed at
# runtime causing undefined symbol error when loading hpp-fcl shared library.
cd "${RootDir}/assimp/build"
cmake "${RootDir}/assimp" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
      -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=OFF \
      -DASSIMP_BUILD_SAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_TESTING=OFF \
      -DCMAKE_C_FLAGS="${CXX_FLAGS} -DHAVE_HIDDEN" -DCMAKE_CXX_FLAGS_RELEASE_INIT="" \
      -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -Wno-strict-overflow -Wno-tautological-compare -Wno-array-compare $(
      ) -Wno-alloc-size-larger-than -Wno-unknown-warning-option -Wno-unknown-warning -Wno-error=array-bounds" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

############################# Build and install qhull and hpp-fcl ######################################

mkdir -p "${RootDir}/hpp-fcl/third-parties/qhull/build"
cd "${RootDir}/hpp-fcl/third-parties/qhull/build"
cmake "${RootDir}/hpp-fcl/third-parties/qhull" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_EXE_LINKER_FLAGS="${LINKER_FLAGS}" \
      -DCMAKE_C_FLAGS="${CXX_FLAGS}" -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -Wno-conversion" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

cd "${RootDir}/hpp-fcl/build"
cmake "${RootDir}/hpp-fcl" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" -DCMAKE_PREFIX_PATH="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${LINKER_FLAGS}" \
      -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -DPYTHON_STANDARD_LAYOUT=ON \
      -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="${InstallDir}" -DBoost_INCLUDE_DIR="${InstallDir}/include" \
      -DHPP_FCL_HAS_QHULL=ON -DBUILD_PYTHON_INTERFACE=ON -DGENERATE_PYTHON_STUBS=OFF \
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DENABLE_PYTHON_DOXYGEN_AUTODOC=OFF \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS} $(
      ) -Wno-unused-parameter -Wno-class-memaccess -Wno-sign-compare-Wno-conversion -Wno-ignored-qualifiers $(
      ) -Wno-uninitialized -Wno-maybe-uninitialized -Wno-deprecated-copy $(
      ) -Wno-unknown-warning" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

################################# Build and install Pinocchio ##########################################

### Build and install pinocchio
cd "${RootDir}/pinocchio/build"
cmake "${RootDir}/pinocchio" -Wno-dev -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="${C_COMPILER}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_INSTALL_PREFIX="${InstallDir}" -DCMAKE_PREFIX_PATH="${InstallDir}" \
      -DCMAKE_OSX_ARCHITECTURES="${OSX_ARCHITECTURES}" -DCMAKE_OSX_DEPLOYMENT_TARGET="${OSX_DEPLOYMENT_TARGET}" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${LINKER_FLAGS}" \
      -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -DPYTHON_STANDARD_LAYOUT=ON \
      -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBOOST_ROOT="${InstallDir}" -DBoost_INCLUDE_DIR="${InstallDir}/include" \
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON \
      -DBUILD_WITH_AUTODIFF_SUPPORT=ON -DBUILD_WITH_CODEGEN_SUPPORT=ON -DBUILD_WITH_CASADI_SUPPORT=OFF \
      -DBUILD_WITH_OPENMP_SUPPORT=OFF -DGENERATE_PYTHON_STUBS=OFF -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF \
      -DCMAKE_CXX_FLAGS_RELEASE_INIT="" -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -DBOOST_BIND_GLOBAL_PLACEHOLDERS $(
      ) -Wno-uninitialized -Wno-type-limits -Wno-unused-local-typedefs -Wno-extra $(
      ) -Wno-unknown-warning" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
make install -j4

# Copy cmake configuration files for cppad and cppadcodegen
cp -r ${RootDir}/pinocchio/cmake/find-external/**/*cppad* ${RootDir}/build_tools/cmake
