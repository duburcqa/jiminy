<div align="center">
  <a href="#"><img width="400px" height="auto" src="https://raw.github.com/duburcqa/jiminy/readme/jiminy_logo.svg"></a>
</div>

____


Jiminy is a fast and portable cross-platform open-source simulator for poly-articulated systems. It was built with two ideas in mind:

- **provide a fast yet physically accurate simulator for robotics research.**

Jiminy is built around [Pinocchio](https://github.com/stack-of-tasks/pinocchio), an open-source fast and efficient kinematics and dynamics library. Jiminy thus uses minimal coordinates and Lagrangian dynamics to simulate an articulated system: this makes Jiminy as close as numerically possible to an analytical solution, without the risk of joint violation.

- **build an efficient and flexible platform for machine learning in robotics.**

Beside a strong focus on performance to answer machine learning's need for running computationally demanding distributed simulations, Jiminy offers convenience tools for learning via a dedicated module [Gym-Jiminy](#gym-jiminy). It is fully compliant with `gym` standard API and provides a highly customizable wrapper to interface any robotics system with state-of-the-art learning frameworks.

## Key features

### General

- Simulation of multi-body systems using minimal coordinates and Lagrangian dynamics.
- Comprehensive API for computing dynamic quantities and their derivatives, exposing and extending Pinocchio.
- C++ core with full python bindings, providing frontend API parity between both languages.
- Designed with machine learning in mind, with seamless wrapping of robots as [OpenAI Gym](https://github.com/openai/gym) environments using one-liners. Jiminy provides both the physical engine and the robot model (including sensors) required for learning.
- Rich simulation log output, easily customizable for recording, introspection and debugging. The simulation log is made available in RAM directly for fast access, and can be exported in raw binary, CSV or [HDF5](https://portal.hdfgroup.org/display/HDF5/Introduction+to+HDF5) format.
- Dedicated integration in Google Colab, Jupyter Lab, and VSCode working out-of-the-box - including interactive 3D viewer based on [Meshcat](https://github.com/rdeits/MeshCat.jl). This facilitates working on remote environments.
- Cross-platform offscreen rendering capability, without requiring X-server, based on [Panda3d](https://github.com/panda3d/panda3d).
- Easy to install: `pip` is all that is needed to [get you started](#getting-started) ! Support Linux, Mac and Windows platforms.

### Physics

- Provide both classical phenomenological force-level spring-damper contact model and constraint solver based on maximum energy dissipation principle.
- Support contact and collision with the ground, using either a fixed set of contact points or collision meshes and primitives.
- Able to simulate multiple articulated systems simultaneously, interacting with each other, to support use cases such as multi-agent reinforcement learning or swarm robotics.
- Support of compliant joints with force-based spring-damper dynamics, to model joint elasticity, a common phenomenon particularly in legged robotics.
- Simulate both continuous or discrete-time controller, with possibly different controller and sensor update frequencies.

A more complete list of features is available on the [wiki](https://github.com/duburcqa/jiminy/wiki).

**The documentation is available on [Github.io](https://duburcqa.github.io/jiminy/), or locally in `docs/html/index.html` if built from source.**

## Gym Jiminy

Gym Jiminy is an interface between Jiminy simulator and reinforcement learning frameworks. It is fully compliant with now standard [Open AI Gym](https://github.com/openai/gym) API. Additionally, it offers a generic and easily configurable learning environment for learning locomotion tasks, with minimal intervention from the user, who usually only needs to provide the robot's [URDF](https://wiki.ros.org/urdf) file. Furthermore, Gym Jiminy enables easy modification of many aspects of the simulation to provide richer exploration and ensure robust learning. This ranges from external perturbation forces to sensor noise and bias, including randomization of masses and inertias, ground friction model or even gravity itself. Note that learning can
easily be done on any high-level dynamics features, or restricted to mock sensor data for end-to-end learning.

Gym is cross-platform and compatible with most Reinforcement Learning frameworks implementing standard algorithms. For instance, [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3), [Tianshou](https://github.com/thu-ml/tianshou), or [Rllib](https://github.com/ray-project/ray). Stable Baselines 3 and Tianshou use its counterpart [Pytorch](https://pytorch.org/), and Rllib supports both. A few learning examples relying on those packages are also provided.

Pre-configured environments for some well-known toys models and reference robotics platforms are provided: [cartpole](https://gym.openai.com/envs/CartPole-v1/), [acrobot](https://gym.openai.com/envs/Acrobot-v1/), [pendulum](https://gym.openai.com/envs/Pendulum-v0/), [Ant](https://gym.openai.com/envs/Ant-v2/), [ANYmal](https://www.anymal-research.org/#getting-started), and [Cassie](https://www.agilityrobotics.com/robots#cassie), and [Atlas](https://www.bostondynamics.com/atlas).

## Demo

<a href="./examples/python/tutorial.ipynb">
<p align="middle">
  <img src="https://raw.github.com/duburcqa/jiminy/readme/jiminy_plot_log.png" alt="" width="49.0%"/>
  <img src="https://raw.github.com/duburcqa/jiminy/readme/jiminy_viewer_open.png" alt="" width="49.0%"/>
  <img src="https://raw.github.com/duburcqa/jiminy/readme/jiminy_tensorboard_cartpole.png" alt="" width="98.5%"/>
  <img src="https://raw.github.com/duburcqa/jiminy/readme/jiminy_learning_ant.gif" alt="" width="32.5%"/>
  <img src="https://raw.github.com/duburcqa/jiminy/readme/cassie.png" alt="" width="32.5%"/>
  <img src="https://raw.github.com/duburcqa/jiminy/readme/atlas.png" alt="" width="32.5%"/>
</p>
</a>

## Getting started

Jiminy and Gym Jiminy support Linux, Mac and Windows, and is compatible with Python3.6+. Pre-compiled binaries are distributed on PyPi for Python 3.6/3.7/3.8/3.9. They can be installed using `pip`:

```bash
# For installing Jiminy
python -m pip install jiminy_py

# For installing Gym Jiminy
python -m pip install gym_jiminy[all]
```

Detailed installation instructions, including building from source, are available [here](./INSTALL.md).
