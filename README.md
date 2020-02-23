# Jiminy simulator

## Description

Jiminy is an open-source C++ simulator of poly-articulated systems, under the first restriction that the contact with the ground can be reduced to a dynamic set of points and the second restriction that the collisions between bodies or the environment can be neglected.

It is built upon [Pinocchio](https://github.com/stack-of-tasks/pinocchio), which is an open-source implementing highly efficient Rigid Body Algorithms for poly-articulated systems. It is used to handle low-level physics calculations related to the system, while the effect of the environment on it is handled by Jiminy itself. The integration of time is based on the open-source library [Boost Odeint](https://github.com/boostorg/odeint).

The visualisation relies on the open-source client [Gepetto-Viewer](https://github.com/Gepetto/gepetto-viewer), which is based on `CORBA` and `omniORB` at low-level, or alternatively [Meshcat](https://github.com/rdeits/meshcat-python), which is a remotely-controllable web-based visualizer especially well suited to Jupyter notebook running on remote servers as one can display directly in a Jupyter cell. It is possible to do real-time visual rendering and to replay a simulation afterward.

The data of the simulation can be exported in CSV or compressed binary format, or read directly from the RAM memory to avoid any disk access.

Python2.7 and Python3 bindings have been written using the open-source library [Boost Python](https://github.com/boostorg/python).

**The Doxygen documentation is available on [Github.io](https://wandercraft.github.io/jiminy/) and locally in `docs/index.html`. **

Thanks to Jan-Lukas Wynen for [Doxygen That Style](https://github.com/jl-wynen/that_style).

## Dependencies

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

##### Linux libraries and binaries

```bash
sudo apt install -y robotpkg-urdfdom=0.3.0r2 robotpkg-urdfdom-headers=0.3.0 \
                    robotpkg-gepetto-viewer=4.4.0
```

##### [Python 2.7 only] Installation procedure

```bash
sudo apt install -y robotpkg-py27-qt4-gepetto-viewer-corba=5.1.2 robotpkg-py27-omniorbpy \
                    robotpkg-py27-eigenpy robotpkg-py27-pinocchio
```

##### [Python 3.6 only] Installation procedure

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

Yet, it is not helpful for compiling C++ code with Pinocchio dependency. If so, then one must compile it from sources. For Debian 10, please follow [these instructions](./Installation.md).

### Python dependencies

#### [Python 3 only] Installation procedure

```bash
sudo apt install -y python3-tk
```

## Easy-install procedure

The project is available on Pypi and therefore can be install easily using `pip` (only binaries for Linux and Python3.6 are available so far).
```bash
pip install jiminy-py
```

# Jiminy learning

## Description

The Machine Learning library [Open AI Gym](https://github.com/openai/gym) is fully supported. Abstract environments and examples for toy models are available. Note that Python3 is not a requirement to use openAI Gym. Nevertheless, most Machine Learning Python packages that implements many standard reinforcement learning algorithms only support Python3,  such as [openAI Gym Baseline](https://github.com/hill-a/stable-baselines), which is based on the open-source Machine Learning framework [Tensorflow](https://github.com/tensorflow/tensorflow) for level-level computation.

## Dependencies [Python 3 only]

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

## Easy-install procedure

The project is available on Pypi and therefore can be install easily using `pip` (only binaries for Linux and Python3.6 are available so far).
```bash
pip install gym-jiminy
```
