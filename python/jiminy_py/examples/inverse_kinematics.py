from math import radians

import numpy as np

from jiminy_py.simulator import Simulator
from jiminy_py.dynamics import update_quantities

# pinocchio must be imported after jiminy_py since in reality it is a submodule
import pinocchio as pin


# Instantiate the simulation
mesh_folder = "/home/duburcqa/Downloads/droid/robot"
urdf_path = "/home/duburcqa/Downloads/droid/robot/droid_description/urdf/left_leg/robot.urdf"

simu = Simulator.build(urdf_path,
                       mesh_path=mesh_folder,
                       has_freeflyer=False,
                       viewer_backend="panda3d")

# Extract some proxies for convenience
robot = simu.robot
model = robot.pinocchio_model
data = robot.pinocchio_data
SO3 = pin.liegroups.SO3()

# Delete contact points since there is no ground in this case
robot.remove_contact_points([])

# Display the system
simu.render(display_dcm=False, display_f_external=False)
simu.viewer._backend_obj.show_floor(False)

# Use pinnochio model to get the neutral position joint positions
q_init = pin.neutral(robot.pinocchio_model)
print(q_init)

# Create a zero vector for the initial velocity.
# The dimensions of the velocity vector is the number of DOF.
v_init = np.zeros(robot.nv)

# Get reference to foot position.
foot_id = model.getFrameId('foot')
pose_foot = data.oMf[foot_id]

# Add marker in viewer to visualize foot position.
# It will be updated automatically when calling 'render'.
simu.viewer.add_marker('foot',
                       'frame',
                       pose_foot,
                       scale=0.02,
                       remove_if_exists=True)

# Display the robot in different configurations.
# it doesn't matter is a simulation is already running or not.
q_display = np.array([radians(20.0), 0.0, radians(90.0)])
simu.viewer.display(q_display)

# Run a simulation of 10s and replay the result
simu.start(q_init=q_init, v_init=v_init)
simu.step(10.0)
simu.stop()
simu.replay()

# Add viscous friction
for name in robot.motors_names:
    motor = robot.get_motor(name)
    option = motor.get_options()
    option["enableFriction"] = True
    option["frictionViscousPositive"] = -0.1  # Nm/rad/s
    option["frictionViscousNegative"] = -0.1
    motor.set_options(option)

# Run a new simulation
simu.start(q_init=q_init, v_init=v_init)
simu.step(10.0)
simu.stop()
simu.replay()

# Inverse kinematics basic algorithm.
# The goal is to input XYZ frame coordinate of the foot relative to base link
# and output the corresponding joint angles.

# Define target position and orientation.
# It does not matter if it is not reachable.
x, y, z = 0.2, -0.4, 0.0
roll, pitch, yaw = radians(-45), radians(0), radians(90)

# Algorithms parameters
enable_orientation = False
step = 0.01
threshold = 1.0e-4  # meters
max_iterations = 2000
damping = 1e-8

# Gather the position and orientation in transformation object
rot_ref = pin.rpy.rpyToMatrix(np.array([roll, pitch, yaw]))
pose_foot_ref = pin.SE3(rot_ref, np.array([x, y, z]))

# Add marker for target position
simu.viewer.add_marker('target',
                       'frame',
                       pose_foot_ref,
                       # color="red",
                       scale=0.02,
                       remove_if_exists=True)

# Visualize initial position
simu.viewer.display(q_init)

# Perform naive gradient descent.
# Constraints such as joint bounds are ignored for now.
q = q_init.copy()
for i in range(max_iterations):
    # Backup previous configuration for error computation
    q_prev = q.copy()

    # Compute quantities of the robot based on system state.
    # Here we skip many of them because we focus on frame positions.
    update_quantities(robot,
                      q,
                      update_com=False,
                      update_energy=False,
                      update_jacobian=True,
                      update_collisions=False,
                      use_theoretical_model=False)

    # Evaluate the jacobian of foot at foot origin but aligned with world frame
    jac_foot = pin.getFrameJacobian(
        model, data, foot_id, pin.LOCAL_WORLD_ALIGNED)
    if not enable_orientation:
        jac_foot = jac_foot[:3]

    # Compute the pose error between the target and current pose
    error_position_foot = pose_foot.translation - pose_foot_ref.translation
    if enable_orientation:
        error_orientation_foot = pose_foot.rotation @ SO3.difference(
            pin.Quaternion(pose_foot_ref.rotation).coeffs(),
            pin.Quaternion(pose_foot.rotation).coeffs())
        error_pose_foot = np.concatenate((
            error_position_foot, error_orientation_foot))
    else:
        error_pose_foot = error_position_foot

    # Compute the error in joint space.
    # Solve the linear system: $\Delta p_\text{foot} = J_\text{foot} q$
    error_joint_position = jac_foot.T @ np.linalg.solve(
        jac_foot @ jac_foot.T + damping * np.eye(len(error_pose_foot)),
        error_pose_foot)

    # Update current conifguration.
    # Do a newton step in the direction of the gradient.
    q = pin.integrate(model, q, - step * error_joint_position)

    # Make sure q is within joint bounds
    q = np.clip(q, robot.position_limit_lower, robot.position_limit_upper)

    # Display periodically the current configuration of the robot and error
    if i % 20:
        simu.render(display_dcm=False, display_f_external=False)
        print("error:", np.linalg.norm(error_pose_foot))

    # Check if the terminal condition is reached.
    # It stops as soon as the error is not reducing anymore.
    if (np.abs(q_prev - q) < threshold).all():
        print(f"Solution found in {i} iterations!")
        break

# Extract the final configuration and rendering it
q_result = q
simu.render(display_dcm=False, display_f_external=False)
