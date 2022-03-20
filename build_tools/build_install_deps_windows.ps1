################################## Configure the environment ###########################################

### Enable debug print mode and disable stop-on-error because it appears that some commands return 1 even if successfull
$ErrorActionPreference = "Continue"
Set-PSDebug -Trace 1

### Set the build type to "Release" if undefined
if (-not (Test-Path env:BUILD_TYPE)) {
  $env:BUILD_TYPE = "Release"
  Write-Output "BUILD_TYPE is unset. Defaulting to '${BUILD_TYPE}'."
}

### Set common CMAKE_C/CXX_FLAGS
$env:CMAKE_CXX_FLAGS = "$env:CMAKE_CXX_FLAGS /EHsc /bigobj /Zc:__cplusplus /permissive- -D_USE_MATH_DEFINES -DNOMINMAX"
if (${env:BUILD_TYPE} -eq "Release") {
  $env:CMAKE_CXX_FLAGS = "$env:CMAKE_CXX_FLAGS /O2 /Ob3 -DNDEBUG"
} else {
  $env:CMAKE_CXX_FLAGS = "$env:CMAKE_CXX_FLAGS /Od -g"
}

### Get the fullpath of Jiminy project
$RootDir = (Split-Path -Parent "$PSScriptRoot")
$RootDir = "$RootDir" -replace '\\', '/' # Force cmake compliant path delimiter

### Set the fullpath of the install directory, then creates it
$InstallDir = "$RootDir/install"
if (-not (Test-Path -PathType Container "$InstallDir")) {
  New-Item -ItemType "directory" -Force -Path "$InstallDir"
}

### Eigenpy and Pinocchio are using the deprecated FindPythonInterp cmake helper to detect Python executable,
#   which is not working properly when several executables exist.
$PYTHON_EXECUTABLE = ( python -c "import sys; sys.stdout.write(sys.executable)" )

### Remove the preinstalled boost library from search path
if (Test-Path env:Boost_ROOT) {
  Remove-Item env:Boost_ROOT
}

################################## Checkout the dependencies ###########################################

### Checkout boost and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
#   Note that Boost 1.72 is not yet officially supported by Cmake 3.16, which is the "default" version used
#   on Windows 10.
if (-not (Test-Path -PathType Container "$RootDir/boost")) {
  git clone https://github.com/boostorg/boost.git "$RootDir/boost"
}
Set-Location -Path "$RootDir/boost"
git reset --hard
git checkout --force "boost-1.71.0"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
Set-Location -Path "libs/python"
git checkout --force "boost-1.76.0"

### Checkout eigen3
if (-not (Test-Path -PathType Container "$RootDir/eigen3")) {
  git clone https://gitlab.com/libeigen/eigen.git "$RootDir/eigen3"
}
Set-Location -Path "$RootDir/eigen3"
git checkout --force "3.3.9"

### Checkout eigenpy and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/eigenpy")) {
  git clone https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
}
Set-Location -Path "$RootDir/eigenpy"
git reset --hard
git checkout --force "v2.6.4"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
dos2unix "$RootDir/build_tools/patch_deps_windows/eigenpy.patch"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/eigenpy.patch"

### Checkout tinyxml (robotology fork for cmake compatibility)
if (-not (Test-Path -PathType Container "$RootDir/tinyxml")) {
  git clone https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"
}
Set-Location -Path "$RootDir/tinyxml"
git reset --hard
git checkout --force "master"

### Checkout console_bridge, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/console_bridge")) {
  git clone https://github.com/ros/console_bridge.git "$RootDir/console_bridge"
}
Set-Location -Path "$RootDir/console_bridge"
git reset --hard
git checkout --force "0.4.4"

### Checkout urdfdom_headers
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers")) {
  git clone https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"
}
Set-Location -Path "$RootDir/urdfdom_headers"
git reset --hard
git checkout --force "1.0.5"

### Checkout urdfdom, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/urdfdom")) {
  git clone https://github.com/ros/urdfdom.git "$RootDir/urdfdom"
}
Set-Location -Path "$RootDir/urdfdom"
git reset --hard
git checkout --force "1.0.4"
dos2unix "$RootDir/build_tools/patch_deps_windows/urdfdom.patch"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/urdfdom.patch"

### Checkout assimp
if (-not (Test-Path -PathType Container "$RootDir/assimp")) {
  git clone https://github.com/assimp/assimp.git "$RootDir/assimp"
}
Set-Location -Path "$RootDir/assimp"
git reset --hard
git checkout --force "v5.0.1"
dos2unix "$RootDir/build_tools/patch_deps_windows/assimp.patch"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/assimp.patch"

### Checkout hpp-fcl
if (-not (Test-Path -PathType Container "$RootDir/hpp-fcl")) {
  git clone https://github.com/humanoid-path-planner/hpp-fcl.git "$RootDir/hpp-fcl"
  git config --global url."https://".insteadOf git://
}
Set-Location -Path "$RootDir/hpp-fcl"
git reset --hard
git checkout --force "v1.7.4"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
dos2unix "$RootDir/build_tools/patch_deps_windows/hppfcl.patch"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/hppfcl.patch"
Set-Location -Path "$RootDir/hpp-fcl/third-parties/qhull"
git checkout --force "v8.0.2"

### Checkout pinocchio and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/pinocchio")) {
  git clone https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
  git config --global url."https://".insteadOf git://
}
Set-Location -Path "$RootDir/pinocchio"
git reset --hard
git checkout --force "v2.5.6"
git submodule --quiet foreach --recursive git reset --quiet --hard
git submodule --quiet update --init --recursive --jobs 8
dos2unix "$RootDir/build_tools/patch_deps_windows/pinocchio.patch"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/pinocchio.patch"

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR

### Build and install the build tool b2 (build-ception !)
Set-Location -Path "$RootDir/boost"
./bootstrap.bat --prefix="$InstallDir"

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
$BuildTypeB2 = ${env:BUILD_TYPE}.ToLower()
if (-not (Test-Path -PathType Container "$RootDir/boost/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/boost/build"
}
./b2.exe --prefix="$InstallDir" --build-dir="$RootDir/boost/build" `
         --with-chrono --with-timer --with-date_time --with-system --with-test `
         --with-filesystem --with-atomic --with-serialization --with-thread `
         --build-type=minimal architecture=x86 address-model=64 threading=single `
         --layout=system --lto=off link=static runtime-link=shared debug-symbols=off `
         toolset=msvc-14.2 cxxflags="-std=c++17 $env:CMAKE_CXX_FLAGS" `
         variant="$BuildTypeB2" install -q -d0 -j2
./b2.exe --prefix="$InstallDir" --build-dir="$RootDir/boost/build" `
         --with-python `
         --build-type=minimal architecture=x86 address-model=64 threading=single `
         --layout=system --lto=off link=shared runtime-link=shared debug-symbols=off `
         toolset=msvc-14.2 cxxflags="-std=c++17 $env:CMAKE_CXX_FLAGS" `
         variant="$BuildTypeB2" install -q -d0 -j2

#################################### Build and install eigen3 ##########################################

if (-not (Test-Path -PathType Container "$RootDir/eigen3/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigen3/build"
}
Set-Location -Path "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

################################### Build and install eigenpy ##########################################

### Build eigenpy
if (-not (Test-Path -PathType Container "$RootDir/eigenpy/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigenpy/build"
}
Set-Location -Path "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS -DBOOST_ALL_NO_LIB -DEIGENPY_STATIC"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

################################## Build and install tinyxml ###########################################

if (-not (Test-Path -PathType Container "$RootDir/tinyxml/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/tinyxml/build"
}
Set-Location -Path "$RootDir/tinyxml/build"
cmake "$RootDir/tinyxml" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS -DTIXML_USE_STL"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

############################## Build and install console_bridge ########################################

###
if (-not (Test-Path -PathType Container "$RootDir/console_bridge/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/console_bridge/build"
}
Set-Location -Path "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

############################### Build and install urdfdom_headers ######################################

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom_headers/build"
}
Set-Location -Path "$RootDir/urdfdom_headers/build"
cmake "$RootDir/urdfdom_headers" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

################################## Build and install urdfdom ###########################################

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom/build"
}
Set-Location -Path "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS -DURDFDOM_STATIC"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

###################################### Build and install assimp ########################################

###
if (-not (Test-Path -PathType Container "$RootDir/assimp/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/assimp/build"
}
Set-Location -Path "$RootDir/assimp/build"
cmake "$RootDir/assimp" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=OFF `
      -DASSIMP_BUILD_SAMPLES=OFF -DBUILD_DOCS=OFF -DASSIMP_INSTALL_PDB=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS /wd4005" -DCMAKE_C_FLAGS="$env:CMAKE_CXX_FLAGS"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

############################# Build and install qhull and hpp-fcl ######################################

### Build qhull
#   Note that 'CMAKE_MSVC_RUNTIME_LIBRARY' is not working with qhull. So, it must be patched instead to
#   add the desired flag at the end of CMAKE_CXX_FLAGS ("/MT", "/MD"...). It will take precedence over
#   any existing flag if any.
Set-Location -Path "$RootDir/hpp-fcl/third-parties/qhull/build"
cmake "$RootDir/hpp-fcl/third-parties/qhull" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON `
      -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS" -DCMAKE_C_FLAGS="$env:CMAKE_CXX_FLAGS"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

### Build hpp-fcl
if (-not (Test-Path -PathType Container "$RootDir/hpp-fcl/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/hpp-fcl/build"
}
Set-Location -Path "$RootDir/hpp-fcl/build"
cmake "$RootDir/hpp-fcl" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_PYTHON_INTERFACE=ON -DHPP_FCL_HAS_QHULL=ON `
      -DINSTALL_DOCUMENTATION=OFF -DENABLE_PYTHON_DOXYGEN_AUTODOC=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS /wd4068 /wd4267 /wd4005 $(
)     -DBOOST_ALL_NO_LIB -DEIGENPY_STATIC -DHPP_FCL_STATIC"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2

################################ Build and install Pinocchio ##########################################

### Build and install pinocchio, finally !
if (-not (Test-Path -PathType Container "$RootDir/pinocchio/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/pinocchio/build"
}
Set-Location -Path "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -Wno-dev -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" `
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON `
      -DBUILD_WITH_AUTODIFF_SUPPORT=OFF -DBUILD_WITH_CASADI_SUPPORT=OFF -DBUILD_WITH_CODEGEN_SUPPORT=OFF `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="$env:CMAKE_CXX_FLAGS /wd4068 /wd4715 /wd4834 /wd4005 $(
)     -DBOOST_ALL_NO_LIB -DEIGENPY_STATIC -DURDFDOM_STATIC -DHPP_FCL_STATIC -DPINOCCHIO_STATIC"
cmake --build . --target INSTALL --config "${env:BUILD_TYPE}" --parallel 2
