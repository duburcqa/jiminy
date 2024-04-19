################################## Configure the environment ###########################################

### Enable debug print mode and disable stop-on-error because it appears that some commands return 1 even if successfull
$ErrorActionPreference = "Continue"
Set-PSDebug -Trace 1

### Eigenpy and Pinocchio are using the deprecated FindPythonInterp cmake helper to detect Python executable,
#   which is not working properly when several executables exist.
if (-not (Test-Path env:PYTHON_EXECUTABLE)) {
  $PYTHON_EXECUTABLE = ( python -c "import sys; sys.stdout.write(sys.executable)" )
  Write-Output "PYTHON_EXECUTABLE is unset. Defaulting to '${PYTHON_EXECUTABLE}'."
} else {
  ${PYTHON_EXECUTABLE} = "${env:PYTHON_EXECUTABLE}"
}
$PYTHON_VERSION = ( & "${PYTHON_EXECUTABLE}" -c "import sys ; print(''.join(map(str, sys.version_info[:2])), end='')" )
Write-Output "PYTHON_VERSION: ${PYTHON_VERSION}"

### Set the build type to "Release" if undefined
if (-not (Test-Path env:BUILD_TYPE)) {
  ${BUILD_TYPE} = "Release"
  Write-Output "BUILD_TYPE is unset. Defaulting to '${BUILD_TYPE}'."
} else {
  ${BUILD_TYPE} = "${env:BUILD_TYPE}"
}

### Set the CPU architecture
$ARCHITECTURE = ( & "${PYTHON_EXECUTABLE}" -c "import platform ; print(platform.machine(), end='')" )
if (("x86_64", "AMD64").contains(${ARCHITECTURE})) {
  $ARCHITECTURE = "x64"
} elseif (${ARCHITECTURE} -eq "ARM64") {
  $ARCHITECTURE = "ARM64"
} else {
  Write-Error "Architecture '${ARCHITECTURE}' not supported." -ErrorAction:Stop
}
Write-Output "Architecture: ${ARCHITECTURE}"

### Set the default generate if undefined.
#   The appropriate toolset will be selected automatically by cmake.
if (-not (Test-Path env:GENERATOR)) {
  ${GENERATOR} = "Visual Studio 17 2022"
  Write-Output "GENERATOR is unset. Defaulting to '${GENERATOR}'."
} else {
  ${GENERATOR} = "${env:GENERATOR}"
}

### Set common CMAKE_C/CXX_FLAGS
${CMAKE_CXX_FLAGS} = "${env:CMAKE_CXX_FLAGS} /MP2 /EHsc /bigobj /Gy /Zc:inline /Zc:preprocessor /Zc:__cplusplus /permissive- /DWIN32 /D_USE_MATH_DEFINES /DNOMINMAX"
if (${BUILD_TYPE} -eq "Debug") {
  ${CMAKE_CXX_FLAGS} = "${CMAKE_CXX_FLAGS} /Zi /Od"
} else {
  ${CMAKE_CXX_FLAGS} = "${CMAKE_CXX_FLAGS} /DNDEBUG /O2 /Ob3"
}
Write-Output "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}"

### Set common CMAKE_SHARED_LINKER_FLAGS
#   - Note that, on Windows OS, linking against Python debug binaries requires
#     recompiling all Python C-modules including `numpy`, otherwise they will
#     fail at import. To get around this issue, CXX flag '/DBOOST_PYTHON_DEBUG'
#     must not be set, with cmake flag `-DBoost_USE_DEBUG_PYTHON=OFF`. See:
#     https://beta.boost.org/doc/libs/1_65_1/libs/python/doc/html/building/python_debugging_builds.html
#     https://stackoverflow.com/questions/56798472/force-linking-against-python-release-library-in-debug-mode-in-windows-visual-stu
#   - To avoid additional side-effects, it is recommanded to install Python
#     WITHOUT debugging symbols, otherwise cmake would pick the wrong one,
#     unlesss `set(Python_FIND_ABI "OFF" "ANY" "ANY")` has been specified.
${CMAKE_SHARED_LINKER_FLAGS} = "${env:CMAKE_SHARED_LINKER_FLAGS}"
if (${BUILD_TYPE} -eq "Debug") {
  ${CMAKE_SHARED_LINKER_FLAGS} = "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG:FULL /INCREMENTAL:NO /OPT:REF /OPT:ICF"
}
Write-Output "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}"

### Get the fullpath of Jiminy project
$RootDir = (Split-Path -Parent "$PSScriptRoot")
$RootDir = "$RootDir" -replace '\\', '/' # Force cmake compliant path delimiter

### Set the fullpath of the install directory, then creates it
$InstallDir = "$RootDir/install"
if (-not (Test-Path -PathType Container "$InstallDir")) {
  New-Item -ItemType "directory" -Force -Path "$InstallDir"
}

### Remove the preinstalled boost library from search path
if (Test-Path env:Boost_ROOT) {
  Remove-Item env:Boost_ROOT
}

################################## Checkout the dependencies ###########################################

### Checkout boost and its submodules
#   - Boost >= 1.78 is required to support MSVC 2022
#   - Boost >= 1.78 defines `BOOST_CORE_USE_GENERIC_CMATH` that prevents wrong substitution for
#     `boost::math::copysign` with MSVC
if (-not (Test-Path -PathType Container "$RootDir/boost")) {
  git clone --depth=1 https://github.com/boostorg/boost.git "$RootDir/boost"
}
if (Test-Path -PathType Container "$RootDir/boost/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/boost/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/boost/build"
Push-Location -Path "$RootDir/boost"
git reset --hard
git fetch origin "boost-1.78.0"
git checkout --force FETCH_HEAD
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
cd "$RootDir/boost/libs/python"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_unix/boost-python.patch"
Pop-Location

### Checkout eigen3
#   A specific commit from Aug 25, 2021 (post 3.4.0) fixes CXX STANDARD detection with MSVC
if (-not (Test-Path -PathType Container "$RootDir/eigen3")) {
  git clone --depth=1 https://gitlab.com/libeigen/eigen.git "$RootDir/eigen3"
}
if (Test-Path -PathType Container "$RootDir/eigen3/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/eigen3/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/eigen3/build"
Push-Location -Path "$RootDir/eigen3"
git fetch origin eeacbd26c8838869a491ee89ab5cf0fe7dac016f
git checkout --force FETCH_HEAD
Pop-Location

### Checkout eigenpy and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
#   eigenpy >= 2.6.8 is required to support Boost >= 1.77
if (-not (Test-Path -PathType Container "$RootDir/eigenpy")) {
  git clone --depth=1 https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
}
if (Test-Path -PathType Container "$RootDir/eigenpy/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/eigenpy/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/eigenpy/build"
Push-Location -Path "$RootDir/eigenpy"
git reset --hard
git fetch origin "v3.4.0"
git checkout --force FETCH_HEAD
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/eigenpy.patch"
Pop-Location

### Checkout tinyxml (robotology fork for cmake compatibility)
if (-not (Test-Path -PathType Container "$RootDir/tinyxml")) {
  git clone --depth=1 https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"
}
if (Test-Path -PathType Container "$RootDir/tinyxml/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/tinyxml/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/tinyxml/build"
Push-Location -Path "$RootDir/tinyxml"
git reset --hard
git fetch origin "master"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout console_bridge
if (-not (Test-Path -PathType Container "$RootDir/console_bridge")) {
  git clone --depth=1 https://github.com/ros/console_bridge.git "$RootDir/console_bridge"
}
if (Test-Path -PathType Container "$RootDir/console_bridge/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/console_bridge/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/console_bridge/build"
Push-Location -Path "$RootDir/console_bridge"
git reset --hard
git fetch origin "1.0.2"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout urdfdom_headers
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers")) {
  git clone --depth=1 https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"
}
if (Test-Path -PathType Container "$RootDir/urdfdom_headers/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/urdfdom_headers/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom_headers/build"
Push-Location -Path "$RootDir/urdfdom_headers"
git reset --hard
git fetch origin "1.0.5"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout urdfdom, then apply some patches
if (-not (Test-Path -PathType Container "$RootDir/urdfdom")) {
  git clone --depth=1 https://github.com/ros/urdfdom.git "$RootDir/urdfdom"
}
if (Test-Path -PathType Container "$RootDir/urdfdom/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/urdfdom/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom/build"
Push-Location -Path "$RootDir/urdfdom"
git reset --hard
git fetch origin "3.0.0"
git checkout --force FETCH_HEAD
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/urdfdom.patch"
Pop-Location

### Checkout CppAD
if (-not (Test-Path -PathType Container "$RootDir/cppad")) {
  git clone --depth=1 https://github.com/coin-or/CppAD.git "$RootDir/cppad"
}
if (Test-Path -PathType Container "$RootDir/cppad/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/cppad/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/cppad/build"
Push-Location -Path "$RootDir/cppad"
git reset --hard
git fetch origin "20240000.2"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout CppADCodeGen
if (-not (Test-Path -PathType Container "$RootDir/cppadcodegen")) {
  git clone --depth=1 https://github.com/joaoleal/CppADCodeGen.git "$RootDir/cppadcodegen"
}
if (Test-Path -PathType Container "$RootDir/cppadcodegen/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/cppadcodegen/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/cppadcodegen/build"
Push-Location -Path "$RootDir/cppadcodegen"
git reset --hard
git fetch origin "v2.4.3"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout assimp, then apply some patches
if (-not (Test-Path -PathType Container "$RootDir/assimp")) {
  git clone --depth=1 https://github.com/assimp/assimp.git "$RootDir/assimp"
}
if (Test-Path -PathType Container "$RootDir/assimp/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/assimp/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/assimp/build"
Push-Location -Path "$RootDir/assimp"
git reset --hard
git fetch origin "v5.2.5"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout hpp-fcl, then apply some patches
if (-not (Test-Path -PathType Container "$RootDir/hpp-fcl")) {
  git clone --depth=1 https://github.com/humanoid-path-planner/hpp-fcl.git "$RootDir/hpp-fcl"
}
if (Test-Path -PathType Container "$RootDir/hpp-fcl/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/hpp-fcl/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/hpp-fcl/build"
Push-Location -Path "$RootDir/hpp-fcl"
git reset --hard
git fetch origin "v2.4.4"
git checkout --force FETCH_HEAD
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/hppfcl.patch"
Pop-Location
if (Test-Path -PathType Container "$RootDir/hpp-fcl/third-parties/qhull/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/hpp-fcl/third-parties/qhull/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/hpp-fcl/third-parties/qhull/build"
Push-Location -Path "$RootDir/hpp-fcl/third-parties/qhull"
git fetch origin "v8.0.2"
git checkout --force FETCH_HEAD
Pop-Location

### Checkout pinocchio and its submodules, then apply some patches
if (-not (Test-Path -PathType Container "$RootDir/pinocchio")) {
  git clone --depth=1 https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
}
if (Test-Path -PathType Container "$RootDir/pinocchio/build") {
  Remove-Item -Recurse -Force -Path "$RootDir/pinocchio/build"
}
New-Item -ItemType "directory" -Force -Path "$RootDir/pinocchio/build"
Push-Location -Path "$RootDir/pinocchio"
git reset --hard
git fetch origin "v2.7.0"
git checkout --force FETCH_HEAD
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --depth 1 --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/pinocchio.patch"
Pop-Location

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR

### Build and install the build tool b2 (build-ception !)
Push-Location -Path "$RootDir/boost"
./bootstrap.bat --prefix="$InstallDir" --with-python="${PYTHON_EXECUTABLE}"

### Build and install and install boost
#   (Replace -d0 option by -d1 and remove -q option to check compilation errors)
#   Note that on Windows, the shared (C++) runtime library is used even for static
#   libraries. Indeed, "Using static runtime with shared libraries is impossible on
#   Linux, and dangerous on Windows" (see boost/Jamroot#handle-static-runtime).
#   See also https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features
#   [Because a DLL built by linking to a static CRT will have its own CRT state ...].
#   Anyway, dynamic linkage is not a big deal in practice because the (universal)
#   C++ runtime  library Windows (aka (U)CRT) ships as part of Windows 10.
#   Note that static linkage is still possible on windows but Jamroot must be edited
#   to remove line "<conditional>@handle-static-runtime".
if (${BUILD_TYPE} -eq "Release") {
  $BuildTypeB2 = "release"
} elseif (${BUILD_TYPE} -eq "Debug") {
  $BuildTypeB2 = "debug"
  $DebugOptionsB2 = @{
    "debug-symbols" = "on"
    # "python-debugging" = "on"
  }
# } elseif (${BUILD_TYPE} -eq "RelWithDebInfo") {
#   $BuildTypeB2 = "profile"
} else {
  Write-Error "Build type '${BUILD_TYPE}' not supported." -ErrorAction:Stop
}

if (${ARCHITECTURE} -eq "x64") {
  $ArchiB2 = "x86"
} elseif (${ARCHITECTURE} -eq "ARM64") {
  $ArchiB2 = "arm"
}

# Compiling everything with static linkage except Boost::Python
./b2.exe --prefix="$InstallDir" --build-dir="$RootDir/boost/build" `
         --with-chrono --with-timer --with-date_time --with-system --with-test `
         --with-filesystem --with-atomic --with-serialization --with-thread `
         --build-type=minimal --layout=system --lto=off `
         architecture=${ArchiB2} address-model=64 @DebugOptionsB2 `
         threading=single link=static runtime-link=shared `
         cxxflags="-std=c++11 ${CMAKE_CXX_FLAGS}" `
         linkflags="${CMAKE_SHARED_LINKER_FLAGS}" `
         variant="$BuildTypeB2" install -q -d0 -j2

./b2.exe --prefix="$InstallDir" --build-dir="$RootDir/boost/build" `
         --with-python `
         --build-type=minimal --layout=system --lto=off `
         architecture=${ArchiB2} address-model=64 @DebugOptionsB2 `
         threading=single link=shared runtime-link=shared `
         cxxflags="-std=c++11 ${CMAKE_CXX_FLAGS}" `
         linkflags="${CMAKE_SHARED_LINKER_FLAGS}" `
         variant="$BuildTypeB2" install -q -d0 -j2
Pop-Location

#################################### Build and install eigen3 ##########################################

Push-Location -Path "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=OFF
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

################################### Build and install eigenpy ##########################################

Push-Location -Path "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE -DGENERATE_PYTHON_STUBS=OFF `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} /wd4005 -DBOOST_ALL_NO_LIB $(
)     -DBOOST_CORE_USE_GENERIC_CMATH -DEIGENPY_STATIC"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

################################## Build and install tinyxml ###########################################

Push-Location -Path "$RootDir/tinyxml/build"
cmake "$RootDir/tinyxml" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DTIXML_USE_STL"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

############################## Build and install console_bridge ########################################

Push-Location -Path "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

############################### Build and install urdfdom_headers ######################################

Push-Location -Path "$RootDir/urdfdom_headers/build"
cmake "$RootDir/urdfdom_headers" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

################################## Build and install urdfdom ###########################################

Push-Location -Path "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF `
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DURDFDOM_STATIC"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

################################### Build and install CppAD ##########################################

Push-Location -Path "$RootDir/cppad/build"
cmake "$RootDir/cppad" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

################################### Build and install CppADCodeGen ##########################################

Push-Location -Path "$RootDir/cppadcodegen/build"
cmake "$RootDir/cppadcodegen" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DGOOGLETEST_GIT=ON -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

###################################### Build and install assimp ########################################

Push-Location -Path "$RootDir/assimp/build"
cmake "$RootDir/assimp" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_ZLIB=OFF -DASSIMP_BUILD_TESTS=OFF `
      -DASSIMP_BUILD_SAMPLES=OFF -DBUILD_DOCS=OFF -DASSIMP_INSTALL_PDB=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_C_FLAGS="${CMAKE_CXX_FLAGS}" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} /wd4005 /wd5105"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

############################# Build and install qhull and hpp-fcl ######################################

### Build qhull
#   Note that 'CMAKE_MSVC_RUNTIME_LIBRARY' is not working with qhull. So, it must be patched instead to
#   add the desired flag at the end of CMAKE_CXX_FLAGS ("/MT", "/MD"...). It will take precedence over
#   any existing flag if any.
Push-Location -Path "$RootDir/hpp-fcl/third-parties/qhull/build"
cmake "$RootDir/hpp-fcl/third-parties/qhull" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON `
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_C_FLAGS="${CMAKE_CXX_FLAGS}" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

### Build hpp-fcl
Push-Location -Path "$RootDir/hpp-fcl/build"
cmake "$RootDir/hpp-fcl" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_PYTHON_INTERFACE=ON -DHPP_FCL_HAS_QHULL=ON -DGENERATE_PYTHON_STUBS=OFF -DBUILD_TESTING=OFF `
      -DINSTALL_DOCUMENTATION=OFF -DENABLE_PYTHON_DOXYGEN_AUTODOC=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} /wd4068 /wd4267 /wd4005 /wd4081 -DBOOST_ALL_NO_LIB $(
)     -DBOOST_CORE_USE_GENERIC_CMATH -DEIGENPY_STATIC -DHPP_FCL_STATIC"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

################################ Build and install Pinocchio ##########################################

### Build and install pinocchio, finally !
Push-Location -Path "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -Wno-dev -G "${GENERATOR}" -DCMAKE_GENERATOR_PLATFORM="${ARCHITECTURE}" `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" `
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE -DGENERATE_PYTHON_STUBS=OFF `
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON `
      -DBUILD_WITH_AUTODIFF_SUPPORT=OFF -DBUILD_WITH_CASADI_SUPPORT=OFF -DBUILD_WITH_CODEGEN_SUPPORT=OFF `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" `
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} /wd4068 /wd4081 /wd4715 /wd4834 /wd4005 /wd5104 /wd5105 $(
)     -DBOOST_ALL_NO_LIB -DBOOST_CORE_USE_GENERIC_CMATH -DEIGENPY_STATIC -DURDFDOM_STATIC -DHPP_FCL_STATIC $(
)     -DPINOCCHIO_STATIC"
cmake --build . --target INSTALL --config "${BUILD_TYPE}" --parallel 2
Pop-Location

### Copy cmake configuration files for cppad and cppadcodegen
Copy-Item -Path "$RootDir/pinocchio/cmake/find-external/**/*cppad*" -Destination "$RootDir/build_tools/cmake"
