################################## Configure the environment ###########################################

### Enable stop-on-error and debug print mode
$ErrorActionPreference = "Stop"
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

### Checkout boost and its submodules.
#   Boost numeric odeint < 1.71 does not support eigen3 > 3.2,
#   and eigen < 3.3 build fails on windows because of a cmake error
git clone -b "boost-1.72.0" https://github.com/boostorg/boost.git "$RootDir/boost"
Set-Location -Path "$RootDir/boost"
git submodule --quiet update --init --recursive --jobs 8

### Checkout eigen3
git clone -b "3.3.7" https://github.com/eigenteam/eigen-git-mirror.git "$RootDir/eigen3"

### Checkout eigenpy and its submodules
git clone -b "v2.4.3" https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
Set-Location -Path "$RootDir/eigenpy"
git submodule --quiet update --init --recursive --jobs 8

### Checkout tinyxml (robotology fork for cmake compatibility)
git clone -b "master" https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"

### Checkout console_bridge
git clone -b "0.4.4" https://github.com/ros/console_bridge.git "$RootDir/console_bridge"

### Checkout urdfdom_headers
git clone -b "1.0.5" https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"

### Checkout urdfdom
git clone -b "1.0.4" https://github.com/ros/urdfdom.git "$RootDir/urdfdom"

### Checkout pinocchio and its submodules (sbarthelemy fork for windows compatibility - based on 2.1.11)
git clone -b "v2.4.7" https://github.com/stack-of-tasks/pinocchio.git "$RootDir/pinocchio"
Set-Location -Path "$RootDir/pinocchio"
git submodule --quiet update --init --recursive --jobs 8

################################### Build and install boost ############################################

# How to properly detect custom install of Boost library:
# - if Boost_NO_BOOST_CMAKE is TRUE:
#   * Set the cmake cache variable CMAKE_PREFIX_PATH
#   * Set the environment variable Boost_DIR
# - if Boost_NO_BOOST_CMAKE is FALSE:
#   * Set the cmake cache variable BOOST_ROOT and Boost_INCLUDE_DIR

### Patch /boost/python/operators.hpp (or /libs/python/include/boost/python/operators.hpp on github) to avoid conflicts with msvc
$LineNumbers = @(22, 371)
$Contents = Get-Content "$RootDir/boost/libs/python/include/boost/python/operators.hpp"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumbers[0] -eq $n) {
'// Workaround msvc iso646.h
#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __GCCXML__
#if defined(or)
#   pragma push_macro("or")
#   pragma push_macro("xor")
#   pragma push_macro("and")
#   undef or
#   undef xor
#   undef and
#endif
#endif
#endif'
} elseif ($LineNumbers[1] -eq $n) {
'// Workaround msvc iso646.h
#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __GCCXML__
#if defined(or)
#   pragma pop_macro("or")
#   pragma pop_macro("and")
#   pragma pop_macro("xor")
#endif
#endif
#endif'
} ; $_ ; $n++} | `
Out-File -Encoding ASCII "$RootDir/boost/libs/python/include/boost/python/operators.hpp"
Set-PSDebug -Trace 1

### Build and install the build tool b2 (build-ception !)
Set-Location -Path "$RootDir/boost"
./bootstrap.bat --prefix="$InstallDir"

### Build and install and install boost (Replace -d0 option by -d1 to check compilation errors)
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
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_PKGCONFIG=ON `
      -DCMAKE_CXX_FLAGS="/bigobj"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

################################### Build and install eigenpy ##########################################

### Remove line 73 of boost.cmake to disable library type enforced SHARED
$LineNumber = 73
$Contents = Get-Content "$RootDir/eigenpy/cmake/boost.cmake"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -ne $n) {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/eigenpy/cmake/boost.cmake"
Set-PSDebug -Trace 1

### Must patch /CMakefile.txt to disable library type enforced SHARED
$Contents = Get-Content "$RootDir/eigenpy/CMakeLists.txt"
($Contents -replace 'SHARED ','') | Out-File -Encoding ASCII "$RootDir/eigenpy/CMakeLists.txt"

### Build eigenpy
if (-not (Test-Path -PathType Container "$RootDir/eigenpy/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigenpy/build"
}
Set-Location -Path "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
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
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################## Build and install console_bridge ########################################

### Must remove lines 107 and 114 of CMakefile.txt `if (NOT MSVC) ... endif()`
$LineNumbers = @(107, 114)
$Contents = Get-Content "$RootDir/console_bridge/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if (-Not ($LineNumbers -Contains $n)) {$_} ; $n++} | `
Out-File -Encoding ASCII "$RootDir/console_bridge/CMakeLists.txt"
Set-PSDebug -Trace 1

###
if (-not (Test-Path -PathType Container "$RootDir/console_bridge/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/console_bridge/build"
}
Set-Location -Path "$RootDir/console_bridge/build"
cmake "$RootDir/console_bridge" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

############################## Build and install urdfdom_headers ######################################

### Must remove lines 51 and 56 of CMakefile.txt `if (NOT MSVC) ... endif()`
$LineNumbers = @(51, 56)
$Contents = Get-Content "$RootDir/urdfdom_headers/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if (-Not ($LineNumbers -Contains $n)) {$_} ; $n++} | `
Out-File -Encoding ASCII "$RootDir/urdfdom_headers/CMakeLists.txt"
Set-PSDebug -Trace 1

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom_headers/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom_headers/build"
}
Set-Location -Path "$RootDir/urdfdom_headers/build"
cmake -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj" "$RootDir/urdfdom_headers"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

################################# Build and install urdfdom ###########################################

### Patch line 71 of CMakeLists.txt to add TinyXML dependency to cmake configuration files generator
$LineNumber = 71
$Contents = Get-Content "$RootDir/urdfdom/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'set(PKG_DEPENDS urdfdom_headers console_bridge TinyXML)'
} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/urdfdom/CMakeLists.txt"
Set-PSDebug -Trace 1

### Patch line 81 of CMakeLists.txt to add TinyXML dependency to pkgconfig files generator
$LineNumber = 81
$Contents = Get-Content "$RootDir/urdfdom/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'set(PKG_URDF_LIBS "-lurdfdom_sensor -lurdfdom_model_state -lurdfdom_model -lurdfdom_world -ltinyxml")'
} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/urdfdom/CMakeLists.txt"
Set-PSDebug -Trace 1

### Must remove lines 78 and 86 of CMakefile.txt `if (NOT MSVC) ... endif()`
$LineNumbers = @(78, 86)
$Contents = Get-Content "$RootDir/urdfdom/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if (-Not ($LineNumbers -Contains $n)) {$_} ; $n++} | `
Out-File -Encoding ASCII "$RootDir/urdfdom/CMakeLists.txt"
Set-PSDebug -Trace 1

### Must patch /urdf_parser/CMakeLists.txt to disable library type enforced SHARED
(Get-Content "$RootDir/urdfdom/urdf_parser/CMakeLists.txt").replace('SHARED ', '') | `
Out-File -Encoding ASCII "$RootDir/urdfdom/urdf_parser/CMakeLists.txt"

###
if (-not (Test-Path -PathType Container "$RootDir/urdfdom/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/urdfdom/build"
}
Set-Location -Path "$RootDir/urdfdom/build"
cmake "$RootDir/urdfdom" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBUILD_TESTING=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj -D_USE_MATH_DEFINES -DURDFDOM_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

################################ Build and install Pinocchio ##########################################

### Remove line 73 of boost.cmake to disable library type enforced SHARED
$LineNumber = 73
$Contents = Get-Content "$RootDir/pinocchio/cmake/boost.cmake"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -ne $n) {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/cmake/boost.cmake"
Set-PSDebug -Trace 1

### Must patch /src/CMakefile.txt to disable library type enforced SHARED
$Contents = Get-Content "$RootDir/pinocchio/src/CMakeLists.txt"
($Contents -replace 'SHARED ','') | Out-File -Encoding ASCII "$RootDir/pinocchio/src/CMakeLists.txt"

### Remove every std::vector bindings of native types, since it makes absolutely no sense to bind such ambiguous types
$configFiles = Get-ChildItem -Path "$RootDir/pinocchio/*" -Include *.hpp -Recurse
Set-PSDebug -Trace 0
Foreach ($file in $configFiles)
{
  (Get-Content $file.PSPath) | `
  Where-Object {$_ -notmatch 'StdVectorPythonVisitor<'} | `
  Set-Content $file.PSPath
}
Set-PSDebug -Trace 1

### Build and install pinocchio, finally !
if (-not (Test-Path -PathType Container "$RootDir/pinocchio/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/pinocchio/build"
}
Set-Location -Path "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DCMAKE_PREFIX_PATH="$InstallDir" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBUILD_WITH_COLLISION_SUPPORT=OFF -DBUILD_TESTING=OFF -DINSTALL_DOCUMENTATION=OFF `
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="/EHsc /bigobj /wd4068 $(
)     -D_USE_MATH_DEFINES -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC $(
)     -DURDFDOM_STATIC -DEIGENPY_STATIC -DPINOCCHIO_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/lib/boost_filesystem*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_serialization*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
