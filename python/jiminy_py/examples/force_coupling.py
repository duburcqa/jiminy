import numpy as np
import matplotlib.pyplot as plt

import pinocchio as pin

import jiminy_py.core as jiminy
from jiminy_py.viewer import Viewer


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

# * Check single translation axis (Z) + random pose:
# force symmetry + rest + same z translation
# * Check single rotation axis (Z) + random pose:
# force symmetry + no linear force + rest + same z orientation
# * Check homogeneous all axis translation + random pose:
# force symmetry + no angular force + rest + same position
stiffness = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
damping = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
engine.register_viscoelastic_force_coupling(
    "robot1", "robot2", "root_joint", "root_joint", stiffness, damping)
# engine.register_viscoelastic_directional_force_coupling(
#     "robot1", "robot2", "root_joint", "root_joint", 9.0, 6.0)

# Remove gravity
engine_options = engine.get_options()
engine_options["world"]["gravity"][:] = 0.0
engine.set_options(engine_options)

# Compute the initial condition
q_init, v_init = {}, {}
np.random.seed(4)
for i, (name, _) in enumerate(zip(engine.systems_names, engine.systems)):
    if i > - 1:
        q_init[name] = np.concatenate((
            1.0 * np.random.rand(3),
            pin.Quaternion(pin.exp3(1.0 * np.random.rand(3))).coeffs()
            ))
        v_init[name] = np.concatenate((
            5.0 * np.random.rand(3), 5.0 * np.random.rand(3)))
    else:
        q_init["robot2"] = - q_init["robot1"]
        v_init["robot2"] = - v_init["robot1"]
engine.start(q_init, v_init)

# Add markers
oMf1 = robot1.pinocchio_data.oMf[1]
oMf2 = robot2.pinocchio_data.oMf[1]

Viewer.close()
views = [Viewer(system.robot) for system in engine.systems]
Viewer._backend_obj.gui.show_floor(False)
views[0].add_marker(
    "root_joint_1", "frame",
    [oMf1.translation, oMf1.rotation],
    color="black", scale=1)
views[1].add_marker(
    "root_joint_2", "frame",
    [oMf2.translation, oMf2.rotation],
    color="red", scale=1)

# Run the simulation while extracting the coupling force
forces_ref = [system_state.f_external[1].vector
              for system_state in engine.systems_states.values()]
forces_val = [[], []]
try:
    for _ in range(1000):
        engine.step(0.01)
        for i, force_ref in enumerate(forces_ref):
            forces_val[i].append(force_ref.copy())
        for view in views:
            view.refresh()
        # print("===")
except Exception:
    pass
engine.stop()
print(engine.stepper_state)

# Show the results
for i in range(len(forces_val)):
    forces_val[i] = np.stack(forces_val[i], axis=0)
plt.figure()
for forces_i, style in zip(forces_val, ("-", "--")):
    plt.plot(forces_i[:, :3], style)
plt.show()
