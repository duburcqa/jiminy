import os
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import jiminy_py.core as jiminy
from jiminy_py.viewer import Viewer

import pinocchio as pin


# Get script directory
MODULE_DIR = os.path.dirname(__file__)

# Create a gym environment for a simple cube
urdf_path = f"{MODULE_DIR}/../../jiminy_py/unit_py/data/sphere_primitive.urdf"

# Instantiate the robots
robot1 = jiminy.Robot()
robot1.initialize(urdf_path, has_freeflyer=True)
robot2 = jiminy.Robot()
robot2.initialize(urdf_path, has_freeflyer=True)

# Instantiate a multi-robot engine
engine = jiminy.EngineMultiRobot()
engine.add_system("robot1", robot1)
engine.add_system("robot2", robot2)

# Define coupling force
stiffness = np.array([0.2, 0.5, 1.0, 0.2, 0.3, 0.6])
damping = np.array([0.0, 0.7, 0.3, 0.5, 0.8, 1.1])
alpha = 0.5
engine.register_viscoelastic_force_coupling(
    "robot1", "robot2", "root_joint", "root_joint", 
    stiffness, damping, alpha)

# Remove gravity
engine_options = engine.get_options()
engine_options["stepper"]["odeSolver"] = 'runge_kutta_dopri5'
engine_options["stepper"]["tolAbs"] = 1e-10
engine_options["stepper"]["tolRel"] = 1e-10
engine_options["stepper"]["dtMax"] = 0.01
engine_options["world"]["gravity"][:] = 0.0
engine.set_options(engine_options)

# Initialize the system
np.random.seed(3)
q_init, v_init = {}, {}
q_init["robot2"] = np.concatenate((
    1.0 * np.random.rand(3),
    pin.Quaternion(pin.exp3(0.5 * np.random.rand(3))).coeffs()
))
q_init["robot1"] = np.concatenate((
    - q_init["robot2"][:3],
    pin.Quaternion(q_init["robot2"][3:]).conjugate().coeffs()
))
v_init["robot2"] = np.concatenate((
    0.8 * np.random.rand(3),
    0.6 * np.random.rand(3)
))
v_init["robot1"] = - v_init["robot2"]
engine.start(q_init, v_init)

# Extract the interacting frames
oMf1, oMf2 = robot1.pinocchio_data.oMf[1], robot2.pinocchio_data.oMf[1]

# Add markers
Viewer.close()
views = [Viewer(system.robot) for system in engine.systems]
Viewer._backend_obj.gui.show_floor(False)
views[0].add_marker(
    "root_joint_1", "frame",
    [oMf1.translation, oMf1.rotation],
    color="black",
    scale=1)
views[0].add_marker(
    "root_joint_2", "frame",
    [oMf2.translation, oMf2.rotation],
    color="red", scale=1)

# Run the simulation while extracting the coupling force
dt = 0.01
forces_ref = [system_state.f_external[1]
              for system_state in engine.systems_states.values()]
forces, kinetic_momentum, energy_robots, energy_spring = [], [], [], []
try:
    for i in range(10000):
        for view in views:
            view.refresh()
        engine.step(dt)
        forces.append([
            (oMf * f_ext).vector
            for oMf, f_ext in zip((oMf1, oMf2), forces_ref)])
        energy_robots.append([
            0.5 * robot.pinocchio_model.inertias[1].vtiv(
                robot.pinocchio_data.v[1])
            for robot in (robot1, robot2)])
        quatRef12 = oMf1.rotation @ pin.exp3(
            alpha * pin.log3(oMf1.rotation.T @ oMf2.rotation))
        energy_spring.append(
            0.5 * np.dot(stiffness[:3], (quatRef12.T @ (oMf2.translation - oMf1.translation)) ** 2) +
            0.5 * np.dot(stiffness[3:], pin.log3(oMf1.rotation.T @ oMf2.rotation) ** 2)
        )
        kinetic_momentum.append([(
            oMf * (
                robot.pinocchio_model.inertias[1] * robot.pinocchio_data.v[1])
            ).vector for oMf, robot in zip((oMf1, oMf2), (robot1, robot2))])
        # assert sum(energy_robots[-1]) < 1.02 * (energy_spring[0] + sum(energy_robots[0]))
except AssertionError as e:
    logging.exception(e)
except Exception:
    pass
engine.stop()
print(engine.stepper_state)

# Plot spatial kinematic momentum
fig, axes = plt.subplots(2, 3)
fig.suptitle("Kinematic Momentum")
for i, e in enumerate(2 * ("X", "Y", "Z")):
    ax = axes.flat[i]
    ax.set_title(f"{'Linear' if i < 3 else 'Angular'} {e}")
    for j in range(2):
        ax.plot(
            np.asarray(kinetic_momentum)[:, j, i], f"C{j}-",
            label=f"robot {j} - {e}")
    ax.plot(
        np.sum(np.asarray(kinetic_momentum)[..., i], axis=1), "k--",
        label=f"total - {e}")
    ax.legend()
plt.show()

# Plot spatial forces
forces_kin = np.diff(np.asarray(kinetic_momentum), axis=0) / dt
fig, axes = plt.subplots(2, 3)
fig.suptitle("Forces")
for i, e in enumerate(2 * ("X", "Y", "Z")):
    ax = axes.flat[i]
    ax.set_title(f"{'Linear' if i < 3 else 'Angular'} {e}")
    for j in range(2):
        ax.plot(
            np.asarray(forces)[:, j, i], f"C{j}-",
            label=f"robot {j} - true - {e}")
        ax.plot(
            forces_kin[:, j, i], f"C{j}--",
            label=f"robot {j} - diff - {e}")
    ax.legend()
plt.show()

# Plot energy
plt.figure()
plt.plot(np.asarray(energy_robots))
plt.plot(energy_spring)
plt.plot(np.asarray(energy_spring) +
         np.sum(np.asarray(energy_robots), axis=1), "k--")
plt.show()
