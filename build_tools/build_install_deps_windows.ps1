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

### Add the generated pkgconfig file to the search path
$Env:PKG_CONFIG_PATH = "$InstallDir/lib/pkgconfig;$InstallDir/share/pkgconfig"

################################## Checkout the dependencies ###########################################

### Checkout boost and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
#   Boost numeric odeint < 1.71 does not support eigen3 > 3.2,
#   and eigen < 3.3 build fails on windows because of a cmake error
#   Note that Boost 1.72 is not yet officially supported by Cmake 3.16, which is the "default" version used on Windows 10.
git clone -b "boost-1.71.0" https://github.com/boostorg/boost.git "$RootDir/boost"
Set-Location -Path "$RootDir/boost"
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/boost.patch"

### Checkout eigen3
git clone -b "3.3.7" https://github.com/eigenteam/eigen-git-mirror.git "$RootDir/eigen3"

### Checkout eigenpy and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
git clone -b "v2.4.3" https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
Set-Location -Path "$RootDir/eigenpy"
git submodule --quiet update --init --recursive --jobs 8
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/eigenpy.patch"

### Checkout tinyxml (robotology fork for cmake compatibility)
git clone -b "master" https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"

### Checkout console_bridge, then apply some patches (generated using `git diff --submodule=diff`)
git clone -b "0.4.4" https://github.com/ros/console_bridge.git "$RootDir/console_bridge"
Set-Location -Path "$RootDir/console_bridge"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/console_bridge.patch"

### Checkout urdfdom_headers
git clone -b "1.0.5" https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"

### Checkout urdfdom, then apply some patches (generated using `git diff --submodule=diff`)
git clone -b "1.0.4" https://github.com/ros/urdfdom.git "$RootDir/urdfdom"
Set-Location -Path "$RootDir/urdfdom"
git apply --reject --whitespace=fix "$RootDir/build_tools/patch_deps_windows/urdfdom.patch"

### Checkout pinocchio and its submodules, then apply some patches (generated using `git diff --submodule=diff`)
git clone -b "v2.4.7" https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
Set-Location -Path "$RootDir/pinocchio"
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
         --with-date_time --with-filesystem --with-headers --with-math --with-python `
         --with-serialization --with-stacktrace --with-system --with-test --with-thread `
         --with-python --build-type=minimal architecture=x86 address-model=64 `
         threading=multi --layout=system link=shared runtime-link=shared `
         toolset=msvc-14.2 variant="$BuildTypeB2" install -q -d0 -j2

#################################### Build and install eigen3 ##########################################

if (-not (Test-Path -PathType Container "$RootDir/eigen3/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigen3/build"
}
Set-Location -Path "$RootDir/eigen3/build"
cmake "$RootDir/eigen3" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=ON `
      -DCMAKE_CXX_FLAGS="/bigobj"
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
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj $(
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
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################## Build and install console_bridge ########################################

###
if (-not (Test-Path -PathType Container "$RootDir/console_bridge/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/console_bridge/build"
}
Set-Location -Path "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################## Build and install urdfdom_headers ######################################

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom_headers/build"
}
Set-Location -Path "$RootDir/urdfdom_headers/build"
cmake -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj" "$RootDir/urdfdom_headers"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

################################# Build and install urdfdom ###########################################

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom/build"
}
Set-Location -Path "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=14 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -D_USE_MATH_DEFINES -DURDFDOM_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

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
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_WITH_COLLISION_SUPPORT=OFF -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF `
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj /wd4068 /wd4715 $(
)     -D_USE_MATH_DEFINES -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC -DPINOCCHIO_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/lib/boost_filesystem*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_serialization*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
