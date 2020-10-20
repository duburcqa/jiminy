<div align="center">
  <a href="#"><img width="400px" height="auto" src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_logo.svg"></a>
</div>

____


Jiminy is a fast and lightweight cross-platform open-source simulator for poly-articulated systems. It was built with two ideas in mind:

- **provide a fast yet physically accurate simulator for robotics research.**

Jiminy is built around [Pinocchio](https://github.com/stack-of-tasks/pinocchio), an open-source fast and efficient kinematics and dynamics library. Jiminy thus uses minimal coordinates and Lagrangian dynamics to simulate an articulated system: this makes Jiminy as close as numerically possible to an analytical solution, without the risk of joint violation.

- **build an efficient and flexible platform for machine learning in robotics.**

Beside a strong focus on performance to answer machine learning's need for running computationally demanding distributed simulations, Jiminy offers convenience tools for learning via a dedicated module [Gym-Jiminy](#gym-jiminy). It is fully compliant with `gym` standard API and provides an highly customizable wrapper to interface any robotics system with state-of-the-art learning frameworks.

## Key features

### General

- Simulation of multi-body systems using minimal coordinates and Lagrangian dynamics.
- Comprehensive API for computing dynamic quantities and their derivatives, exposing and extending Pinocchio.
- C++ core with full python bindings, providing frontend API parity between both languages.
- Designed with machine learning in mind, with seemless wrapping of robots in `gym` environments using one-liners. Jiminy provides both the physical engine and the robot model (including sensors) required for learning.
- Easy to install: `pip` is all that is needed to [get you started](#getting-started) !
- Dedicated integration in jupyter notebook working out-of-the-box - including 3D rendering using [Meshcat](https://github.com/rdeits/MeshCat.jl). This facilitates working on remote headless environnement such as machine learning clusters.
- Rich simulation log output, easily customizable for recording, introspection and debugging. The simulation log is made available in RAM directly for fast access, and can be exported as CSV or binary data.
- Available for both Linux and Windows platform.

### Physics

- Support contact and collision with the ground, using either a fixed set of contact points or collision meshes and primitives, through spring-damper reaction forces and friction model.
- Able to simulate multiple articulated systems simultaneously, interacting with each other, to support use cases such as multi-agent reinforcement learning or swarm robotics.
- Support of compliant joints with spring-damper dynamics, to model joint elasticity, a common phenomenon particularly in legged robotics.
- Simulate both continuous or discrete-time controller, with possibly different controller and sensor update frequencies.

A more complete list of features, development status, and changelog are available on the [wiki](https://github.com/Wandercraft/jiminy/wiki).

**The Doxygen documentation is available on [Github.io](https://wandercraft.github.io/jiminy/) and locally in `docs/index.html`.**

## Gym Jiminy

Gym Jiminy is an interface between Jiminy simulator and reinforcement learning frameworks. It is fully compliant with now standard [Open AI Gym](https://github.com/openai/gym) API. Additionally, it offers a generic and easily configurable learning environment for learning locomotion tasks, with minimal intervention from the user, who usually only needs to provide the robot's [URDF](https://wiki.ros.org/urdf) file. Furthermore, Gym Jiminy enables easy modification of many aspects of the simulation to provide richer exploration and ensure robust learning. This ranges from external perturbation forces to sensor noise and bias, including randomization of masses and inertias, ground friction model or even gravity itself. Note that learning can
easily be done on any high-level dynamics features, or restricted to mock sensor data for end-to-end learning.

Gym is cross-platform and compatible out-of-the-box with most Reinforcement Learning frameworks implementing standard algorithms. For instance, [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3), [RL Coach](https://github.com/NervanaSystems/coach), [Tianshou](https://github.com/thu-ml/tianshou), or [Rllib](https://github.com/ray-project/ray). `RL Coach` leverages the open-source Machine Learning framework [Tensorflow](https://github.com/tensorflow/tensorflow) as backend, `Stable Baselines 3` and  `Tianshou` use its counterpart [Pytorch](https://pytorch.org/), and `Rllib` supports both. A few learning examples relying on those packages are also provided.

Pre-configured environments for some well-known toys models and reference robotics platforms are provided: [cartpole](https://gym.openai.com/envs/CartPole-v1/), [acrobot](https://gym.openai.com/envs/Acrobot-v1/), [pendulum](https://gym.openai.com/envs/Pendulum-v0/), [ANYmal](https://www.anymal-research.org/#getting-started), and [Atlas](https://www.bostondynamics.com/atlas).

## Demo

<a href="./examples/tutorial.ipynb"> <img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_plot_log.png" alt="" width="49.7%"/> <img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_viewer_open.png" alt="" width="49.7%"/>
<img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_tensorboard_cartpole.png" alt="" width="100%"/>
<img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_learning_acrobot.gif" alt="" width="49.7%"/> <img src="https://raw.github.com/Wandercraft/jiminy/readme/jiminy_learning_cartpole.gif" alt="" width="49.7%"/> </a>

## Getting started

Jiminy is compatible with Linux and Windows and supports Python3.6+. Jiminy is distributed on PyPi for Python 3.6/3.7/3.8 on Linux and Windows, and can be installed using `pip`:

```bash
python -m pip install jiminy_py
```

Detailed installation instructions, including building from source, are available [here](./INSTALL.md).
