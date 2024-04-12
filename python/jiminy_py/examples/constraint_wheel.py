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
    urdf_path = f"{MODULE_DIR}/../../jiminy_py/unit_py/data/wheel.urdf"
    simulator = Simulator.build(
        urdf_path, has_freeflyer=True, hardware_path="")

    # Disable constraint solver regularization
    engine_options = simulator.get_options()
    engine_options["constraints"]["regularization"] = 0.0

    # Continuous sensor and controller update
    engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
    engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0

    # Configure integrator
    engine_options['stepper']['odeSolver'] = 'runge_kutta_dopri5'
    engine_options['stepper']['dtMax'] = 1.0e-3
    simulator.set_options(engine_options)

    # Add fixed frame constraint
    constraint = jiminy.WheelConstraint(
        "MassBody", 0.5, np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    simulator.robot.add_constraint("MassBody", constraint)
    constraint.baumgarte_freq = 20.0

    # Register external forces

    # simulator.register_impulse_force(
    #     "MassBody", 0.5, 1e-3, np.array([15.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # simulator.register_impulse_force(
    #     "MassBody", 1.5, 1e-3, np.array([0.0, 15.0, 0.0, 0.0, 0.0, 0.0]))

    simulator.register_impulse_force(
        "MassBody", 0.5, 1e-3, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 8.0]))
    simulator.register_impulse_force(
        "MassBody", 1.5, 1e-3, np.array([0.0, 15.0, 0.0, 0.0, 0.0, 0.0]))

    # simulator.register_impulse_force(
    #     "MassBody", 0.5, 1e-3, np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # simulator.register_impulse_force(
    #     "MassBody", 0.5, 1e-3, np.array([0.0, 8.0, 0.0, 0.0, 0.0, 0.0]))

    # Sample the initial state
    qpos = pin.neutral(simulator.robot.pinocchio_model)
    qpos[2] = 0.5
    qvel = np.zeros(simulator.robot.nv)

    # Run a simulation
    simulator.simulate(10.0, qpos, qvel)

    # Replay the simulation
    simulator.replay(enable_travelling=False, speed_ratio=1.0)
