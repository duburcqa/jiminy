################################## Configure the environment ###########################################

### Enable stop-on-error and debug print mode
$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1

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
git clone -b "boost-1.71.0" https://github.com/boostorg/boost.git "$RootDir/boost"
Set-Location -Path "$RootDir/boost"
git submodule --quiet update --init --recursive --jobs 8

### Checkout eigen3
git clone -b "3.3.7" https://github.com/eigenteam/eigen-git-mirror.git "$RootDir/eigen3"

### Checkout eigenpy and its submodules
git clone -b "v2.1.2" https://github.com/stack-of-tasks/eigenpy.git "$RootDir/eigenpy"
Set-Location -Path "$RootDir/eigenpy"
git submodule --quiet update --init --recursive --jobs 8

### Checkout tinyxml (robotology fork for cmake compatibility)
git clone -b "master" https://github.com/robotology-dependencies/tinyxml.git "$RootDir/tinyxml"

### Checkout console_bridge
git clone -b "0.4.4" https://github.com/ros/console_bridge.git "$RootDir/console_bridge"

### Checkout urdfdom_headers
git clone -b "1.0.3" https://github.com/ros/urdfdom_headers.git "$RootDir/urdfdom_headers"

### Checkout urdfdom
git clone -b "1.0.3" https://github.com/ros/urdfdom.git "$RootDir/urdfdom"

### Checkout pinocchio and its submodules (sbarthelemy fork for windows compatibility - based on 2.1.11)
git clone -b "sbarth_action" https://github.com/sbarthelemy/pinocchio.git "$RootDir/pinocchio"
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
         --without-wave --without-contract --without-graph --without-regex `
         --without-mpi --without-coroutine --without-fiber --without-context `
         --without-timer --without-chrono --without-atomic --without-graph_parallel `
         --without-type_erasure --without-container --without-exception --without-locale `
         --without-log --without-program_options --without-random --without-iostreams `
         --build-type=minimal toolset=msvc-14.2 variant=$BuildTypeB2 threading=multi --layout=system `
         architecture=x86 address-model=64 link=shared runtime-link=shared install -q -d0 -j2

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

###
if (-not (Test-Path -PathType Container "$RootDir/eigenpy/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/eigenpy/build"
}
Set-Location -Path "$RootDir/eigenpy/build"
cmake "$RootDir/eigenpy" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBoost_USE_STATIC_LIBS=OFF -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBUILD_TESTING=OFF `
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Must patch line 18 of $InstallDir/lib/pkgconfig/eigenpy.pc because if the list of library includes in ill-formated on Windows.
#   The pkconfig config file is generated by the cmake submodule, which is the same for pinocchio itself.
$LineNumber = 18
$Contents = Get-Content "$InstallDir/lib/pkgconfig/eigenpy.pc"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {'Libs:'} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$InstallDir/lib/pkgconfig/eigenpy.pc"
Set-PSDebug -Trace 1

### Must replace "Program Files" by "PROGRA~1" and "Program Files (x86)" by "PROGRA~2" to avoid having spaces in paths...
(Get-Content "$InstallDir/lib/pkgconfig/eigenpy.pc").replace("Program Files (x86)", "PROGRA~2") | `
Set-Content "$InstallDir/lib/pkgconfig/eigenpy.pc"
(Get-Content "$InstallDir/lib/pkgconfig/eigenpy.pc").replace("Program Files", "PROGRA~1") | `
Set-Content "$InstallDir/lib/pkgconfig/eigenpy.pc"

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/bin/eigenpy.dll" `
          -Destination "$InstallDir/lib/site-packages/eigenpy"
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
Set-Content "$RootDir/urdfdom/urdf_parser/CMakeLists.txt"

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

### Must add line after 337 (387 v2.1.2 eigenpy) of cmake/python.cmake to remove disk prefix in target names and shorten its name to less than 50 chars
$LineNumber = 337
$Contents = Get-Content "$RootDir/pinocchio/cmake/python.cmake"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'string(LENGTH ${FILE_TARGET_NAME} FILE_TARGET_LENGTH)
math(EXPR FILE_TARGET_START "${FILE_TARGET_LENGTH}-50")
if(${FILE_TARGET_START} LESS 3)
  set(FILE_TARGET_START 3)
endif()
string(SUBSTRING ${FILE_TARGET_NAME} ${FILE_TARGET_START} -1 FILE_TARGET_NAME)
'} ; $_ ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/cmake/python.cmake"
Set-PSDebug -Trace 1

### Replace line 170 of /CMakeLists.txt to include link directory of python and remove defined include headers directory
$LineNumber = 170
$Contents = Get-Content "$RootDir/pinocchio/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'link_directories("${PYTHON_LIBRARY_DIRS}")'
} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/CMakeLists.txt"
Set-PSDebug -Trace 1

### Add line at 129 to manually link eigenpy and urdfdom since pkgconfig has been patched to not do it automatically because it was generating errors...
$LineNumber = 129
$Contents = Get-Content "$RootDir/pinocchio/bindings/python/CMakeLists.txt"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'TARGET_LINK_LIBRARIES(${PYWRAP} "${CMAKE_INSTALL_PREFIX}/lib/eigenpy.lib")
TARGET_LINK_LIBRARIES(${PYWRAP} "${CMAKE_INSTALL_PREFIX}/lib/urdfdom_model.lib")
TARGET_LINK_LIBRARIES(${PYWRAP} "${CMAKE_INSTALL_PREFIX}/lib/tinyxml.lib")
TARGET_LINK_LIBRARIES(${PYWRAP} "${CMAKE_INSTALL_PREFIX}/lib/console_bridge.lib")
'
} ; $_ ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/bindings/python/CMakeLists.txt"
Set-PSDebug -Trace 1

### For some reason, the preprocessor directive `PINOCCHIO_EIGEN_PLAIN_TYPE((...))` is not properly generated. Expending it manually
#   in src/algorithm/joint-configuration.hpp and src/algorithm/joint-configuration.hxx
$FileNames = @("$RootDir/pinocchio/src/algorithm/joint-configuration.hpp", "$RootDir/pinocchio/src/algorithm/joint-configuration.hxx")
$DirectiveOrign = 'PINOCCHIO_EIGEN_PLAIN_TYPE((typename ModelTpl<Scalar,Options,JointCollectionTpl>::ConfigVectorType))'
$DirectiveAfter = 'Eigen::internal::plain_matrix_type< typename pinocchio::helper::argument_type<void(typename ModelTpl<Scalar,Options,JointCollectionTpl>::ConfigVectorType)>::type >::type'
Foreach ($file in $FileNames)
{
  (Get-Content $file).replace($DirectiveOrign, $DirectiveAfter) | Set-Content $file
}

### C-style overloading disambiguation is not working properly with MSVC when it requires double template substitution.
#   Must patch line at 31 of bindings/python/algorithm/expose-geometry.cpp
$LineNumber = 31
$Contents = Get-Content "$RootDir/pinocchio/bindings/python/algorithm/expose-geometry.cpp"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'(void (*)(GeometryModel &, const Eigen::MatrixBase<Vector3d> &))&setGeometryMeshScales<Vector3d>,'
} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/bindings/python/algorithm/expose-geometry.cpp"
Set-PSDebug -Trace 1

### Patch line 285 (289 v2.3.1 pinocchio) of /src/algorithm/model.hxx to fix dot placed after the closing double quote by mistake, not supported by MSVC.
$LineNumber = 285
$Contents = Get-Content "$RootDir/pinocchio/src/algorithm/model.hxx"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'"The number of joints to lock is greater than the total of joints in the reduced_model.");'
} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/src/algorithm/model.hxx"
Set-PSDebug -Trace 1

### Patch the files using the 'not' operator, not supported by MSVC without "#include <iso646.h>" or "#include <ciso646>",  by '!'.
(Get-Content "$RootDir/pinocchio/bindings/python/module.cpp").replace('if(not ', 'if(!') | `
Set-Content "$RootDir/pinocchio/bindings/python/module.cpp"
(Get-Content "$RootDir/pinocchio/src/algorithm/center-of-mass.hxx").replace('if(not ', 'if(!') | `
Set-Content "$RootDir/pinocchio/src/algorithm/center-of-mass.hxx"

### C-style overloading disambiguation is not working properly with MSVC.
#   Must patch lines at 35 of bindings/python/multibody/joint/joint-derived.hpp
$LineNumber = 35
$Contents = Get-Content "$RootDir/pinocchio/bindings/python/multibody/joint/joint-derived.hpp"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'.def("setIndexes", bp::make_function(
                    (void (JointModelDerived::*)(JointIndex, int, int))&JointModelDerived::setIndexes,
                    bp::default_call_policies(),
                    boost::mpl::vector<void, JointModelDerived, JointIndex, int, int>()))'
} else {$_} ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/bindings/python/multibody/joint/joint-derived.hpp"
Set-PSDebug -Trace 1

### Patch /bindings/python/module.cpp to add PY_ARRAY_UNIQUE_SYMBOL
$LineNumber = 5
$Contents = Get-Content "$RootDir/pinocchio/bindings/python/module.cpp"
Set-PSDebug -Trace 0
$Contents | Foreach {$n=1}{if ($LineNumber -eq $n) {
'#define PY_ARRAY_UNIQUE_SYMBOL EIGENPY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#define NO_IMPORT_ARRAY'
} ; $_ ; $n++ } | `
Out-File -Encoding ASCII "$RootDir/pinocchio/bindings/python/module.cpp"
Set-PSDebug -Trace 1

### Build and install pinocchio, finally !
if (-not (Test-Path -PathType Container "$RootDir/pinocchio/build")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/pinocchio/build"
}
Set-Location -Path "$RootDir/pinocchio/build"
cmake "$RootDir/pinocchio" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_CXX_STANDARD=11 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBoost_USE_STATIC_LIBS=OFF  -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" `
      -DBUILD_WITH_LUA_SUPPORT=OFF -DBUILD_WITH_COLLISION_SUPPORT=OFF -DBUILD_TESTING=OFF `
      -DBUILD_WITH_URDF_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON `
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj -D_USE_MATH_DEFINES -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC -DURDFDOM_STATIC"
cmake --build . --target install --config "${Env:BUILD_TYPE}" --parallel 2

### Fix wrong Python library dll naming convention for Windows
$PYTHON_EXT_SUFFIX = ( python -c "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))" )
Remove-Item -Force -Path "$InstallDir/lib/site-packages/pinocchio/pinocchio_pywrap.lib"
Rename-Item -Force -Path "$InstallDir/lib/site-packages/pinocchio/pinocchio_pywrap.dll" `
                   -NewName "$InstallDir/lib/site-packages/pinocchio/libpinocchio_pywrap${PYTHON_EXT_SUFFIX}"

### Embedded the required dynamic library in the package folder
Copy-Item -Path "$InstallDir/bin/eigenpy.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_filesystem*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_serialization*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$InstallDir/lib/site-packages/pinocchio"
