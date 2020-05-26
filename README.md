# Jiminy simulator

## Description

Jiminy is an open-source C++ simulator of poly-articulated systems, under the first restriction that the contact with the ground can be reduced to a dynamic set of points and the second restriction that the collisions between bodies or the environment can be neglected.

It is built upon [Pinocchio](https://github.com/stack-of-tasks/pinocchio), which is an open-source implementing highly efficient Rigid Body Algorithms for poly-articulated systems. It is used to handle low-level physics calculations related to the system, while the effect of the environment on it is handled by Jiminy itself. The integration of time is based on the open-source library [Boost Odeint](https://github.com/boostorg/odeint).

The visualisation relies on the open-source client [Gepetto-Viewer](https://github.com/Gepetto/gepetto-viewer), which is based on `CORBA` and `omniORB` at low-level, or alternatively [Meshcat](https://github.com/rdeits/meshcat-python), which is a remotely-controllable web-based visualizer especially well suited to Jupyter notebook running on remote servers as one can display directly in a Jupyter cell. It is possible to do real-time visual rendering and to replay a simulation afterward.

The data of the simulation can be exported in CSV, raw binary format, or read directly from the RAM memory to avoid any disk access. The complete list of features, development status, and changelog are available on the [wiki](https://github.com/Wandercraft/jiminy/wiki).

Python bindings have been written using the open-source library [Boost Python](https://github.com/boostorg/python). It supports both Python2 and Python3, yet it is recommended to use Python3 over Python2 since support will be dropped in the future.

**The Doxygen documentation is available on [Github.io](https://wandercraft.github.io/jiminy/) and locally in `docs/index.html`. **

Thanks to Jan-Lukas Wynen for [Doxygen That Style](https://github.com/jl-wynen/that_style).

## Demo

<img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_plot_log.png" alt="" width="430"/> <img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_viewer_open.png" alt="" width="430"/>

___

# Jiminy learning

## Description

The Machine Learning library [Open AI Gym](https://github.com/openai/gym) is fully supported. Abstract environments and a few for toy models are provided: a [cartpole](https://gym.openai.com/envs/CartPole-v1/), an [acrobot](https://gym.openai.com/envs/Acrobot-v1/), and a [pendulum](https://gym.openai.com/envs/Pendulum-v0/).

Gym Jiminy is only compatible with Python3. Although Python3 is not required to use openAI Gym strictly speaking, most of the Reinforcement Learning packages implementing standard algorithms does not support Python2. For instance, [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3), [RL Coach](https://github.com/NervanaSystems/coach), [Tianshou](https://github.com/thu-ml/tianshou), or [Rllib](https://github.com/ray-project/ray). `RL Coach` leverages the open-source Machine Learning framework [Tensorflow](https://github.com/tensorflow/tensorflow) as backend, `Stable Baselines 3` and  `Tianshou` use its counterpart [Pytorch](https://pytorch.org/), and `Rllib` supports both. Note that `Stable Baselines 3`, `Tianshou` and `Rllib` are compatible with Linux, Mac OS and Windows.

A few learning examples are provided. Most of them rely on those packages, but one implements the DQN algorithm from scratch using Pytorch.

## Demo

<img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_tensorboard_cartpole.png" alt="" width="860"/>

<img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_learning_acrobot.gif" alt="" width="430"/> <img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_learning_cartpole.gif" alt="" width="430"/>

___

# Getting started

Jiminy is compatible with Linux and Windows It supports both Python2.7 and Python3.6+. Jiminy is distributed on PyPi for Python 3.6/3.7/3.8 on Linux and Windows, and can be installed using `pip`. Furthermore, helper scripts to built the dependencies from source on Windows and Linux are available. The complete installation instructions are available [here](./INSTALL.md).
