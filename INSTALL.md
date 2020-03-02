# Easy-install procedure on Linux

## Jiminy dependencies

### Automatic installation

Just run the setup script already available.

```bash
sudo ./jiminy/build_tools/setup.sh
```

## Manual installation

### Boost Python library

```bash
sudo apt install -y libboost-all-dev
```

### Robotpkg

#### Ubuntu 18.04 Bionic

##### Add the repository

```bash
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt update
```

##### Install some C++ libraries and binaries

```bash
sudo apt install -y robotpkg-urdfdom=0.3.0r2 robotpkg-urdfdom-headers=0.3.0 \
                    robotpkg-gepetto-viewer=4.4.0
```

##### Install some Python packages (Python 2.7 only)

```bash
sudo apt install -y robotpkg-py27-qt4-gepetto-viewer-corba=5.1.2 robotpkg-py27-omniorbpy \
                    robotpkg-py27-eigenpy robotpkg-py27-pinocchio
```

##### Install some Python packages (Python 3 only)

```bash
sudo apt install -y robotpkg-py36-qt4-gepetto-viewer-corba=5.1.2 robotpkg-py36-omniorbpy \
                    robotpkg-py36-eigenpy robotpkg-py36-pinocchio
```

#### Other distributions

Robotpkg is also available for Ubuntu 14.04, Ubuntu 16.04, Ubuntu 19.04, Debian 8, and Debian 9.

One can install Python bindings for Pinocchio using Conda:
```bash
conda install pinocchio --channel conda-forge
```

Yet, it is not helpful for compiling C++ code with Pinocchio dependency. If so, then one must compile it from sources. For Debian 10, please follow the advanced instruction in the next section.

### Python dependencies

#### Installation procedure (Python 3 only)

```bash
sudo apt install -y python3-tk
```

## Jiminy learning dependencies (Python 3 only)

### Tensorflow>=1.13 with GPU support dependencies (Cuda 10.1 and CuDNN 7.6)
Amazing tutorial: https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070

### Open AI Gym along with some toy models
```bash
pip install gym[atari,box2d,classic_control]
```

### Open AI Gym Stable-Baseline
```bash
pip install stable-baselines[mpi]
```

### Coach dependencies
```bash
sudo apt install -y python-opencv
sudo apt install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev
pip install rl_coach
```

## Install Jiminy Python package

The project is available on Pypi and therefore can be install easily using `pip`.
```bash
pip install jiminy-py
```

## Install Jiminy learning Python package

```bash
pip install gym-jiminy
```

___


# Building from source on Linux

## Prerequisites

```bash
sudo apt install -y libeigen3-dev doxygen libtinyxml-dev cmake git
```

### Custom Boost version

If for some reasons you need a specific version of Boost, use the following installation procedure:

```bash
wget http://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 && \
sudo tar --bzip2 -xf boost_1_65_1.tar.bz2 && \
cd boost_1_65_1 && sudo ./bootstrap.sh && \
sudo ./b2 --architecture=x86 --address-model=64 -j8 install && \
cd .. && rm boost_1_65_1.tar.bz2
```

If using Conda venv for Python, do not forget to update Python location in `boost_1_65_1/project-config.jam`. For instance:

```bash
import python ;
if ! [ python.configured ]
{
    using python : 3.5 : /home/aduburcq/anaconda3/envs/py35 : /home/aduburcq/anaconda3/envs/py35/include/python3.5m/ ;
}
```

At least for Boost 1.65.1, it is necessary to [fix Boost Numpy](https://github.com/boostorg/python/pull/218/commits/0fce0e589353d772ceda4d493b147138406b22fd). Update `wrap_import_array` method in `boost_1_65_1/libs/python/src/numpy/numpy.cpp`:
```bash
static void * wrap_import_array()
{
  import_array();
  return NULL;
}
```

Finally, add the following lines in your `.bashrc`:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

In addition, make sure to be using `numpy>=1.16.0`.

### Install Eigenpy

```bash
git clone --recursive https://github.com/stack-of-tasks/eigenpy.git && \
cd eigenpy && mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DPYTHON_EXECUTABLE:FILEPATH=/home/aduburcq/anaconda3/envs/py35/bin/python \
         -DBOOST_ROOT:PATHNAME=/usr/local/ && \
sudo make install -j8 && \
cd .. && rm -f eigenpy
```

### Install Urdfdom 0.3.0 (soversion 0.2.0)

```bash
git clone git://github.com/ros/console_bridge.git && \
cd console_bridge && git checkout 0.2.7 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local && \
sudo make install -j8 && \
cd .. && rm -f console_bridge
```

```bash
git clone git://github.com/ros/urdfdom_headers.git && \
cd urdfdom_headers && git checkout 0.3.0 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local && \
sudo make install -j8 && \
cd .. && rm -f urdfdom_headers
```

```bash
git clone git://github.com/ros/urdfdom.git && \
cd urdfdom && git checkout 0.3.0 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local && \
sudo make install -j8 && \
cd .. && rm -f urdfdom
```

At least for Urdfdom 0.3.0, it is necessary to fix Python Array API initialization and code Python 2.7 only:
update `DuckTypedFactory` method in `urdf_parser_py/src/urdf_parser_py/xml_reflection/core.py` to replace `except Exception, e` by `except Exception as e:`, and
`install` call in `urdf_parser_py/CMakeLists.txt` to remove `--install-layout deb`.

### Install Pinocchio v2.1.10 (lower versions are not compiling properly)

```bash
git clone --recursive https://github.com/stack-of-tasks/pinocchio && \
cd pinocchio && git checkout v2.1.10 && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DPYTHON_EXECUTABLE:FILEPATH=/home/aduburcq/anaconda3/envs/py35/bin/python \
         -DBOOST_ROOT:PATHNAME=/usr/local/ \
         -DPROJECT_URL="http://github.com/stack-of-tasks/pinocchio" && \
sudo make install -j8 && \
cd .. && rm -f pinocchio
```

### Configure .bashrc

Add the following lines:

```bash
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/pythonXXX/site-packages:$PYTHONPATH
```

## Build Procedure

```bash
mkdir jiminy && cd jiminy
cmake $HOME/src/jiminy -DCMAKE_INSTALL_PREFIX=$HOME/install
make && make install
```

___


# Building from source on Windows

## Prerequisites

You have to preinstall by yourself the `MSVC 14.0` toolchain (automated build script only compatible with this exact version so far), `chocolatey` and `python3`. Then, install `Numpy` and `Pkg-Config` using

```pwsh
    choco install pkgconfiglite -y
    python -m pip install numpy
```

## Automatic dependency installation

Now you can simply run the setup script already available.

```pwsh
    & './jiminy/build_tools/setup.ps1'
```

## Build Procedure

You are finally ready to build Jiminy itself
```pwsh
    $RootDir = ...
    $InstallDir = "$RootDir\install"
    $BuildType = "Release" # Must be the same flag as in setup.ps1 if used to compile the dependencies

    if (Test-Path Env:/Boost_ROOT) {
        Remove-Item Env:/Boost_ROOT
    }

    if (-not (Test-Path -PathType Container $RootDir\jiminy\build)) {
        New-Item -ItemType "directory" -Force -Path "$RootDir\jiminy\build"
    }
    Set-Location -Path $RootDir\jiminy\build
    $CmakeModulePath = "$InstallDir/share/eigen3/cmake/;$InstallDir/lib/cmake/eigenpy/;$InstallDir/CMake/" -replace '\\', '/'
    cmake -G "Visual Studio 15" -T "v140" -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_INSTALL_PREFIX="$InstallDir" `
                                          -DCMAKE_MODULE_PATH="$CmakeModulePath" -DBUILD_TESTING=OFF `
                                          -DCMAKE_CXX_FLAGS="/EHsc /bigobj -DBOOST_ALL_NO_LIB -DBOOST_LIB_DIAGNOSTIC" $RootDir\jiminy
    cmake --build . --target INSTALL --config "$BuildType" --parallel 2
```
