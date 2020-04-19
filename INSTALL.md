# Easy-install procedure on Ubuntu 14/16/18/19, and Debian 8/9

## Jiminy dependencies installation


There is not requirement to install `jiminy_py` on linux if one does not want to build it. Nevertheless, this package does not provide the backend viewer `gepetto-gui` (still, the backend `meshcat` is available).

The first step to install `gepetto-gui` is to setup the APT repository `robotpkg` to have access to compiled binaries.

```bash
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt update
```

Once done, it is straightforward to install the required package;

```bash
sudo apt install -y robotpkg-gepetto-viewer=4.4.0 robotpkg-py27-qt4-gepetto-viewer-corba=5.1.2 robotpkg-py27-omniorbpy
```

### Gym Jiminy dependencies (Python 3 only)

#### Tensorflow>=1.13 with GPU support dependencies (Cuda 10.1 and CuDNN 7.6)

Amazing tutorial: <https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070>

#### Open AI Gym along with some toy models

```bash
pip install gym[atari,box2d,classic_control]
```

#### Open AI Gym Stable-Baseline

```bash
pip install stable-baselines[mpi]
```

#### Coach dependencies

```bash
sudo apt install -y python-opencv
sudo apt install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev
pip install rl_coach
```

## Install Jiminy Python package

The project is available on PyPi and therefore can be install easily using `pip`.
```
python -m pip install jiminy-py
```

## Install Jiminy learning Python package

```
python -m pip install gym-jiminy
```

## (optional) Build Jiminy from source

First, one must install the pre-compiled libraries of the dependencies. Most of them are available on `robotpkg` APT repository. Just run the bash script to install them automatically.

```bash
sudo ./jiminy/build_tools/easy_install_deps_ubuntu18.sh
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
      -DCMAKE_CXX_FLAGS="-isystem/usr/include/eigen3" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2
```

# Easy-install procedure on Windows (Python 3 only)

## Jiminy dependencies installation

Install `python3` 3.6/3.7/3.8 (available on [Microsoft store](https://www.microsoft.com/en-us/p/python-38/9mssztt1n39l)), and [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

### Gym Jiminy dependencies

#### Tensorflow>=1.13 with GPU support dependencies (Cuda 10.1 and CuDNN 7.6)

See this tutorial: <https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781>

## Install Jiminy Python package

The project is available on PyPi and therefore can be install easily using `pip`.
```
python -m pip install jiminy-py
```

## Install Jiminy learning Python package

```
python -m pip install gym-jiminy
```

___


# Building from source on Linux

## Prerequisites

```bash
sudo apt install -y gnupg curl wget build-essential cmake doxygen graphviz
python -m pip install numpy
```

## Jiminy dependencies build and install

Just run the bash script already available.

```bash
BUILD_TYPE="Release" ./build_tools/build_install_deps_linux.sh
```

## Build Procedure

```bash
RootDir=".... The location of jiminy repository ...."
PythonVer=".... Your version X.Y of Python, for instance 3.8 ...."

BUILD_TYPE="Release"
InstallDir="$RootDir/install"

unset Boost_ROOT

mkdir "$RootDir/build"
cd "$RootDir/build"
cmake "$RootDir" -DCMAKE_INSTALL_PREFIX="$InstallDir" -DCMAKE_PREFIX_PATH="$InstallDir" \
      -DBOOST_ROOT="$InstallDir" -DBoost_INCLUDE_DIR="$InstallDir/include" \
      -DBoost_NO_SYSTEM_PATHS=TRUE -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBoost_USE_STATIC_LIBS=OFF -DPYTHON_REQUIRED_VERSION="$PythonVer" \
      -DBUILD_TESTING=ON -DBUILD_EXAMPLES=ON -DBUILD_PYTHON_INTERFACE=ON \
      -DCMAKE_CXX_FLAGS="-DURDFDOM_STATIC" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make install -j2

mkdir -p "$HOME/.local/lib/python${PythonVer}/site-packages"
echo "$InstallDir/lib/python${PythonVer}/site-packages" \
> "$HOME/.local/lib/python${PythonVer}/site-packages/user_site.pth"
```

___


# Building from source on Windows

## Prerequisites

You have to preinstall by yourself the (free) MSVC 2019 toolchain, `chocolatey` and `python`.

Then, install `Numpy` and `Pkg-Config` using
```pwsh
choco install pkgconfiglite -y
python -m pip install numpy wheel
```

## Jiminy dependencies build and install

Now you can simply run the powershell script already available.

```pwsh
$Env:BUILD_TYPE = "Release"
& './build_tools/build_install_deps_windows.ps1'
```

## Build Procedure

You are finally ready to build Jiminy itself
```pwsh
$RootDir = ".... The location of jiminy repository ...."

$Env:BUILD_TYPE = "Release"
$RootDir = $RootDir -replace '\\', '/'
$InstallDir = "$RootDir/install"

if (Test-Path Env:/Boost_ROOT) {
  Remove-Item Env:/Boost_ROOT
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
      -DCMAKE_CXX_FLAGS="/EHsc /bigobj -D_USE_MATH_DEFINES -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC -DURDFDOM_STATIC"
cmake --build . --config "${Env:BUILD_TYPE}" --parallel 2

if (-not (Test-Path -PathType Container "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core")) {
  New-Item -ItemType "directory" -Force -Path "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core"
}
Copy-Item -Path "$InstallDir/lib/boost_numpy*.dll" `
          -Destination "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core"
Copy-Item -Path "$InstallDir/lib/boost_python*.dll" `
          -Destination "$RootDir/build/PyPi/jiminy_py/src/jiminy_py/core"
Copy-Item -Path "$InstallDir/lib/site-packages/*" `
          -Destination "$RootDir/build/PyPi/jiminy_py/src/jiminy_py" -Recurse

cmake --build . --target INSTALL --config "${Env:BUILD_TYPE}" --parallel 2
```
