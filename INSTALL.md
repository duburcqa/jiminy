# Easy-install on Ubuntu 18+

## Jiminy

### Dependencies installation

There is no requirement to install `jiminy_py` on linux if one does not want to build it.

### Install Jiminy Python package

The project is available on PyPi and therefore can be install easily using `pip>=20.3`.

```bash
python3 -m pip install --prefer-binary jiminy-py[meshcat,plot]
```

## Gym Jiminy

### Dependencies installation

#### Pytorch>=1.13 with GPU support dependencies

Nowadays, it is straightforward to install CUDA. You are responsible for doing so since it depends on the OS.

Once done, `Pytorch` can be installed following the official "getting started" instructions: <https://pytorch.org/get-started/locally/>

#### (optional) stable_baselines3

Installing the Python packages `stable_baselines3==0.9` is required to run some of the provided examples, though it is not required to use gym_jiminy.

```bash
python3 -m pip install stable-baselines3[extra]==0.9
```

#### (optional) tianshou

Installing the Python packages `tianshou==0.3.0` is required to run some of the provided examples, though it is not required to use gym_jiminy.

```bash
python3 -m pip install tianshou==0.3.0
```

### Install Gym Jiminy learning Python package

```bash
python -m pip install --prefer-binary gym-jiminy[all]
```

# Building from source

## Excluding dependencies on Ubuntu 18+

First, one must install the pre-compiled libraries of the dependencies. Most of them are available on `robotpkg` APT repository. Just run the bash script to install them automatically for Ubuntu 18 and upward. It should be straightforward to adapt it to any other distribution for which `robotpkg` is available.

```bash
sudo env "PATH=$PATH" ./build_tools/easy_install_deps_ubuntu.sh
```

You are now ready to build and install Jiminy itself.

```bash
RootDir=".... The location of jiminy repository ...."

BUILD_TYPE="Release"
InstallDir="$RootDir/install"

mkdir "$RootDir/build" "$InstallDir"
cd "$RootDir/build"
cmake "$RootDir" -DCMAKE_INSTALL_PREFIX="$InstallDir" \
      -DBoost_NO_SYSTEM_PATHS=OFF \
      -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON -DBUILD_PYTHON_INTERFACE=ON \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2
```

## Including dependencies on Linux-based OS

### Prerequisites

```bash
sudo apt install -y gnupg curl wget build-essential cmake doxygen graphviz
python -m pip install "numpy>=1.16,<1.22"
```

### Jiminy dependencies build and install

Just run the bash script already available.

```bash
BUILD_TYPE="Release" ./build_tools/build_install_deps_unix.sh
```

### Build Procedure

```bash
RootDir=".... The location of jiminy repository ...."
PythonExe=".... Your Python executable, for instance $(which python3) ...."

BuildType="Release"
InstallDir="$RootDir/install"

unset Boost_ROOT

mkdir "$RootDir/build"
cd "$RootDir/build"
cmake "$RootDir" -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBoost_USE_STATIC_LIBS=OFF -DPYTHON_EXECUTABLE="$PythonExe" \
      -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON -DBUILD_PYTHON_INTERFACE=ON \
      -DCMAKE_BUILD_TYPE="$BuildType"
make install -j2
```

## Including dependencies on Windows 10+

### Prerequisites

You have to preinstall by yourself the (free) MSVC 2019 toolchain.

Then, install `numpy` and `wheel`.

```powershell
python -m pip install wheel "numpy>=1.16,<1.22"
```

### Jiminy dependencies build and install

Now you can simply run the powershell script already available.

```powershell
$env:BUILD_TYPE = "Release"
& './build_tools/build_install_deps_windows.ps1'
```

### Build Procedure

You are finally ready to build Jiminy itself.

```powershell
$RootDir = ".... The location of jiminy repository ...."

$env:BUILD_TYPE = "Release"
$RootDir = $RootDir -replace '\\', '/'
$InstallDir = "$RootDir/install"

if (Test-Path env:Boost_ROOT) {
  Remove-Item env:Boost_ROOT
}

if (-not (Test-Path -PathType Container $RootDir/build)) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/build"
}
Set-Location -Path $RootDir/build
cmake "$RootDir" -G "Visual Studio 16 2019" -T "v142" -DCMAKE_GENERATOR_PLATFORM=x64 `
      -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_MODULE_PATH="$InstallDir" `
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" `
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE `
      -DBoost_USE_STATIC_LIBS=OFF `
      -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON -DBUILD_PYTHON_INTERFACE=ON `
      -DCMAKE_CXX_FLAGS="-DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC -DBOOST_CORE_USE_GENERIC_CMATH"
cmake --build . --target all --config "${env:BUILD_TYPE}" --parallel 8

if (-not (Test-Path -PathType Container "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core"
}
Copy-Item -Path "$InstallDir/lib/boost_numpy*.dll" `
          -Destination "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core"
Copy-Item -Path "$InstallDir/lib/site-packages/*" `
          -Destination "$RootDir/build/PyPi/jiminy_py/src/jiminy_py" -Recurse

cmake --build . --target install --config "${env:BUILD_TYPE}"
```
