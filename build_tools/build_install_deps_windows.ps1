################################## Configure the environment ###########################################

### Enable debug print mode and disable stop-on-error because it appears that some commands return 1 even if successfull
$ErrorActionPreference = "Continue"
Set-PSDebug -Trace 1

### Set the build type to "Release" if undefined
if (-not (Test-Path Env:BUILD_TYPE)) {
  $Env:BUILD_TYPE = "Release"
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
if (Test-Path Env:/Boost_ROOT) {
  Remove-Item Env:/Boost_ROOT
}

################################## Checkout the dependencies ###########################################

### Checkout boost and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
#   Note that Boost 1.72 is not yet officially supported by Cmake 3.16, which is the "default" version used on Windows 10.
if (-not (Test-Path -PathType Container "$RootDir/boost")) {
  git clone https://github.com/boostorg/boost.git "$RootDir/boost"
}
Set-Location -Path "$RootDir/boost"
git checkout --force "boost-1.71.0"
git submodule foreach --recursive git reset --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/boost.patch"

### Checkout eigen3
if (-not (Test-Path -PathType Container "$RootDir/eigen3")) {
  git clone https://github.com/eigenteam/eigen-git-mirror.git "$RootDir/eigen3"
}
Set-Location -Path "$RootDir/eigen3"
git checkout --force "3.3.7"

### Checkout eigenpy and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/eigenpy")) {
  git clone https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
}
Set-Location -Path "$RootDir/eigenpy"
git checkout --force "v2.5.0"
git submodule foreach --recursive git reset --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/eigenpy.patch"

### Checkout tinyxml (robotology fork for cmake compatibility)
if (-not (Test-Path -PathType Container "$RootDir/tinyxml")) {
  git clone https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"
}
Set-Location -Path "$RootDir/tinyxml"
git checkout --force "master"

### Checkout console_bridge, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/console_bridge")) {
  git clone https://github.com/ros/console_bridge.git "$RootDir/console_bridge"
}
Set-Location -Path "$RootDir/console_bridge"
git checkout --force "0.4.4"

### Checkout urdfdom_headers
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers")) {
  git clone https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"
}
Set-Location -Path "$RootDir/urdfdom_headers"
git checkout --force "1.0.5"

### Checkout urdfdom, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/urdfdom")) {
  git clone https://github.com/ros/urdfdom.git "$RootDir/urdfdom"
}
Set-Location -Path "$RootDir/urdfdom"
git checkout --force "1.0.4"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/urdfdom.patch"

### Checkout assimp
if (-not (Test-Path -PathType Container "$RootDir/assimp")) {
  git clone https://github.com/assimp/assimp.git "$RootDir/assimp"
}
Set-Location -Path "$RootDir/assimp"
git checkout --force "v5.0.1"
dos2unix "$RootDir/build_tools/patch_deps_windows/assimp.patch"  # Fix encoding, just in case
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/assimp.patch"

### Checkout hpp-fcl
if (-not (Test-Path -PathType Container "$RootDir/hpp-fcl")) {
  git clone https://github.com/humanoid-path-planner/hpp-fcl.git "$RootDir/hpp-fcl"
}
Set-Location -Path "$RootDir/hpp-fcl"
git checkout --force "v1.5.4"
git submodule foreach --recursive git reset --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/hppfcl.patch"
Set-Location -Path "$RootDir/hpp-fcl/third-parties/qhull"
git checkout --force "v8.0.2"

### Checkout pinocchio and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
if (-not (Test-Path -PathType Container "$RootDir/pinocchio")) {
  git clone https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
}
Set-Location -Path "$RootDir/pinocchio"
git checkout --force "v2.5.0"
git submodule foreach --recursive git reset --hard
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/pinocchio.patch"

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR

### Build and install the build tool b2 (build-ception !)
Set-Location -Path "$RootDir/boost"
./bootstrap.bat --prefix="$InstallDir"

### Build and install and install boost
#   (Replace -d0 option by -d1 and remove -q option to check compilation errors)
$BuildTypeB2 = ${Env:BUILD_TYPE}.ToLower()
if (-not (Test-Path -PathType Container "$RootDir/boost/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/boost/build"
}
./b2.exe --prefix="$InstallDir" --build-dir="$RootDir/boost/build" `
         --with-chrono --with-timer --with-date_time --with-headers --with-math `
         --with-stacktrace --with-system --with-filesystem --with-atomic `
         --with-thread --with-serialization --with-test --with-python `
         --build-type=minimal architecture=x86 address-model=64 threading=multi `
         --layout=system link=shared runtime-link=shared `
         toolset=msvc-14.2 cxxflags="-std=c++14" variant="$BuildTypeB2" install -q -d0 -j2

#################################### Build and install eigen3 ##########################################

if (-not (Test-Path -PathType Container "$RootDir/eigen3/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigen3/build"
}
Set-Location -Path "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=OFF `
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

################################### Build and install eigenpy ##########################################

### Build eigenpy
if (-not (Test-Path -PathType Container "$RootDir/eigenpy/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigenpy/build"
}
Set-Location -Path "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus $(
)     -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC -DEIGENPY_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$InstallDir/lib/site-packages/eigenpy"

################################## Build and install tinyxml ###########################################

if (-not (Test-Path -PathType Container "$RootDir/tinyxml/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/tinyxml/build"
}
Set-Location -Path "$RootDir/tinyxml/build"
cmake "$RootDir/tinyxml" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus $(
)     -DTIXML_USE_STL"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################## Build and install console_bridge ########################################

###
if (-not (Test-Path -PathType Container "$RootDir/console_bridge/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/console_bridge/build"
}
Set-Location -Path "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################### Build and install urdfdom_headers ######################################

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom_headers/build"
}
Set-Location -Path "$RootDir/urdfdom_headers/build"
cmake "$RootDir/urdfdom_headers" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

################################## Build and install urdfdom ###########################################

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom/build"
}
Set-Location -Path "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus $(
)     -D_USE_MATH_DEFINES -DURDFDOM_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

###################################### Build and install assimp ########################################

###
if (-not (Test-Path -PathType Container "$RootDir/assimp/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/assimp/build"
}
Set-Location -Path "$RootDir/assimp/build"
cmake "$RootDir/assimp" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=OFF `
      -DASSIMP_BUILD_SAMPLES=OFF -DBUILD_DOCS=OFF -DASSIMP_INSTALL_PDB=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DNDEBUG /O2 /Zc:__cplusplus $(
)     -D_USE_MATH_DEFINES"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################# Build and install qhull and hpp-fcl ######################################

### Build qhull
if (-not (Test-Path -PathType Container "$RootDir/hpp-fcl/third-parties/qhull/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/hpp-fcl/third-parties/qhull/build"
}
Set-Location -Path "$RootDir/hpp-fcl/third-parties/qhull/build"
cmake "$RootDir/hpp-fcl/third-parties/qhull" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_CXX_FLAGS="/EHsc /bigobj /Zc:__cplusplus" -DCMAKE_C_FLAGS="/EHsc /bigobj"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Build hpp-fcl
if (-not (Test-Path -PathType Container "$RootDir/hpp-fcl/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/hpp-fcl/build"
}
Set-Location -Path "$RootDir/hpp-fcl/build"
cmake "$RootDir/hpp-fcl" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_PYTHON_INTERFACE=ON -DHPP_FCL_HAS_QHULL=ON `
      -DINSTALL_DOCUMENTATION=OFF -DENABLE_PYTHON_DOXYGEN_AUTODOC=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj /wd4068 /wd4267 /permissive- /Zc:__cplusplus $(
)     -D_USE_MATH_DEFINES -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC $(
)     -DEIGENPY_STATIC -DHPP_FCL_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/lib/boost_filesystem*.dll" `
          -Destination "$InstallDir/lib/site-packages/hppfcl"
Copy-Item -Path "$InstallDir/lib/boost_timer*.dll" `
          -Destination "$InstallDir/lib/site-packages/hppfcl"
Copy-Item -Path "$InstallDir/lib/boost_chrono*.dll" `
          -Destination "$InstallDir/lib/site-packages/hppfcl"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$InstallDir/lib/site-packages/hppfcl"

################################ Build and install Pinocchio ##########################################

### Build and install pinocchio, finally !
if (-not (Test-Path -PathType Container "$RootDir/pinocchio/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/pinocchio/build"
}
Set-Location -Path "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_WITH_COLLISION_SUPPORT=ON -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj /wd4068 /wd4715 /wd4834 /permissive- /Zc:__cplusplus $(
)     -D_USE_MATH_DEFINES -DNOMINMAX -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC $(
)     -DEIGENPY_STATIC -DURDFDOM_STATIC -DHPP_FCL_STATIC -DPINOCCHIO_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/lib/boost_filesystem*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_serialization*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
