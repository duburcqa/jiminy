import os

import numpy as np
import matplotlib.pyplot as plt

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator

import pinocchio as pin


# Get script directory
MODULE_DIR = os.path.dirname(__file__)


if __name__ == '__main__':
    # Create a gym environment for a simple cube
    urdf_path = f"{MODULE_DIR}/../../jiminy_py/unit_py/data/sphere_primitive.urdf"
    simulator = Simulator.build(
        urdf_path, has_freeflyer=True, hardware_path="")
    robot = simulator.robot

    # Disable constraint solver regularization
    engine_options = simulator.engine.get_options()
    engine_options["constraints"]["regularization"] = 0.0
    simulator.engine.set_options(engine_options)

    # Continuous sensor and controller update
    engine_options = simulator.engine.get_options()
    engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
    engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
    simulator.engine.set_options(engine_options)

    # Add fixed frame constraint
    constraint = jiminy.FrameConstraint(
        "MassBody", [True, True, True, True, True, True])
    robot.add_constraint("MassBody", constraint)
    constraint.baumgarte_freq = 1.0

    # Add IMU to the robot
    imu_sensor = jiminy.ImuSensor("MassBody")
    robot.attach_sensor(imu_sensor)
    imu_sensor.initialize("MassBody")

    # Sample the initial state
    qpos = pin.neutral(robot.pinocchio_model)
    qvel = np.zeros(robot.nv)

    # Run a simulation
    delta = []
    simulator.reset()
    simulator.start(qpos, qvel)
    for i in range(700):
        delta_t = simulator.stepper_state.t % (1.1 / constraint.baumgarte_freq)
        if min(delta_t, 1.1 / constraint.baumgarte_freq - delta_t) < 1e-6:
            constraint.reference_transform = pin.SE3.Random()
        transform = robot.pinocchio_data.oMf[constraint.frame_index]
        ref_transform = constraint.reference_transform
        delta.append(np.concatenate((
            transform.translation - ref_transform.translation,
            pin.log3(transform.rotation @ ref_transform.rotation.T))))
        simulator.step(step_dt=0.01)
    simulator.stop()
    delta = np.stack(delta, axis=0)

    # Replay the simulation
    simulator.render(display_dcm=False)
    simulator.viewer._backend_obj.gui.show_floor(False)
    simulator.viewer.add_marker(
        "MassBody", "frame", pose=robot.pinocchio_data.oMf[1])
    simulator.replay(enable_travelling=False)

    # Plot the simulation data
    simulator.plot()

    # Check that the error is strictly decreasing, with the expected rate
    plt.figure()
    plt.plot(delta[:, :3], '-')
    plt.plot(delta[:, 3:], '--')
    plt.show()
