# Jiminy simulator

## Description

Jiminy is an open-source C++ simulator of poly-articulated systems, under the first restriction that the contact with the ground can be reduced to a dynamic set of points and the second restriction that the collisions between bodies or the environment can be neglected.

It is built upon [Pinocchio](https://github.com/stack-of-tasks/pinocchio), which is an open-source implementing highly efficient Rigid Body Algorithms for poly-articulated systems. It is used to handle low-level physics calculations related to the system, while the effect of the environment on it is handled by Jiminy itself. The integration of time is based on the open-source library [Boost Odeint](https://github.com/boostorg/odeint).

The visualisation relies on the open-source client [Gepetto-Viewer](https://github.com/Gepetto/gepetto-viewer), which is based on CORBA and omniORB at low-level. It is possible to do real-time visual rendering and to replay a simulation afterward.

The data of the simulation can be exported in CSV format, or directely read from the RAM memory to avoid any disk acces.

Python2.7 and Python3 bindings have been written using the open-source library [Boost Python](https://github.com/boostorg/python).

**The Doxygen documentation is available on [Github.io](https://wandercraft.github.io/jiminy/) and locally in `docs/index.html`. **

Thanks to Jan-Lukas Wynen for [Doxygen That Style](https://github.com/jl-wynen/that_style).

## Dependencies

### Robotpkg

#### Add the repository
```bash
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub bionic robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" && \
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt update
```

#### [Python 2.7 only] Installation procedure
```bash
sudo apt install -y robotpkg-py27-pinocchio robotpkg-py27-qt4-gepetto-viewer-corba
```

#### [Python 3.6 only] Installation procedure
```bash
sudo apt install -y robotpkg-py36-pinocchio robotpkg-py36-qt4-gepetto-viewer-corba
```

### Python dependencies

#### [Python 3.6 only] Installation procedure
```bash
sudo apt install -y python3-tk
```

## Installation Procedure

```bash
mkdir jiminy && cd jiminy
cmake ~/wdc_workspace/src/jiminy -DCMAKE_INSTALL_PREFIX=/install
make && make install
```

# Jiminy learning

## Description

The Machine Learning library [Open AI Gym](https://github.com/openai/gym) is fully supported. Abstract environments and examples for toy models are available. Note that Python3 is not a requirement to use openAI Gym. Nevertheless, most Machine Learning Python packages that implements many standard reinforcement learning algorithms only support Python3,  such as [openAI Gym Baseline](https://github.com/hill-a/stable-baselines), which is based on the open-source Machine Learning framework [Tensorflow](https://github.com/tensorflow/tensorflow) for level-level computation.

## Dependencies [Python 3.6 only]

### Tensorflow>=1.13 with GPU support dependencies (Cuda 10.1 and CuDNN 7.6)
Amazing tutorial: https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070

### Open AI Gym along with some toy models
```bash
pip install gym[atari,box2d,classic_control]
```

### Open AI Gym Stable-Baseline
```bash
pip install gym[atari,box2d,classic_control]
```

### Coach dependencies
```bash
sudo apt install -y python-opencv
sudo apt install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev
pip install rl_coach
```
