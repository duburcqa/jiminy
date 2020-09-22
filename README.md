# Jiminy simulator

## Description

Jiminy is a fast and lightweight open-source simulator for poly-articulated systems. It was built with two ideas in mind:

 - **provide a fast yet physically accurate simulator for robotics research.** Jiminy is built around
 [Pinocchio](https://github.com/stack-of-tasks/pinocchio), an open-source fast and efficient kinematics and
 dynamics library. Jiminy thus uses minimal coordinates and Lagrangian dynamics to simulate an articulated
 system: this makes Jiminy as close as numerically possible to an analytical solution, without the risk of
 joint violation.

- **build an efficient and flexible plateform for machine learning in robotics.** Beside a strong focus on
 performance to answer machine learning's need for numerous simulations, Jiminy is natively integrated in
 the `gym` framework. Furthermore, Jiminy enables easy modification of many parameters of the system, required for robust learning. From sensor bias to modification of mass and inertia, body length, or gravity itself, many aspects of the simulation can be easily modified to provider richer exploration.


Here are some of the key features of Jiminy:

### General

 - Simulation of multi-body systems using minimal coordinates and Lagrangian dynamics.

 - Fully binded in python, and designed with machine learning in mind, with a `gym` plugin.

 - Easy to install: a simple `pip install jiminy_py` is all that is needed to get you started !

 - 3D visualisation using either [Gepetto-Viewer](https://github.com/Gepetto/gepetto-viewer) for desktop
 view, or [Meshcat](https://github.com/rdeits/MeshCat.jl) for integration in web browsers, inclusing jupyter notebooks.

 - Available for both Linux and Windows platform.

### Physics

 - Support contact and collision, using either a fixed set of contact points, or a collision mesh, and
 a spring-damper reaction force.

 - Able to simulate multiple articulated systems simulatneously, interacting with each other.

 - Support of compliant joints with spring-damper dynamics.

 - Support of simple geometrical constraints on the system.

 - Simulate both continuous or discrete-time controller, with possibly different controller and sensor
 update frequencies.

A more complete list of features, development status, and changelog are available on the [wiki](https://github.com/Wandercraft/jiminy/wiki).

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
