import os
from itertools import starmap

import numpy as np

from jiminy_py.simulator import Simulator
from jiminy_py.core import random_tile_ground, sum_heightmaps, merge_heightmaps

import pinocchio as pin


# Get script directory
MODULE_DIR = os.path.dirname(__file__)

# Set hyperpaameters of the ground profile
TILE_SIZE = [np.array([4.0, 4.0]),
             np.array([100.0, 0.05]),
             np.array([0.05, 100.0])]
TILE_HEIGHT_MAX = [1.0, 0.05, 0.05]
TILE_INTERP_DELTA = [np.array([0.5, 1.0]),
                     np.array([0.01, 0.01]),
                     np.array([0.01, 0.01])]
TILE_SPARSITY = [1, 8, 8]
TILE_ORIENTATION = [0.0, np.pi / 4.0, 0.0]
TILE_SEED = range(3)

if __name__ == '__main__':
    # Create a gym environment for a simple cube
    urdf_path = f"{MODULE_DIR}/../../jiminy_py/unit_py/data/box_collision_mesh.urdf"
    simulator = Simulator.build(urdf_path, has_freeflyer=True)

    # Enable constraint contact model
    engine_options = simulator.get_options()
    engine_options['contacts']['model'] = 'constraint'
    engine_options['contacts']['stabilizationFreq'] = 20.0
    engine_options["constraints"]['regularization'] = 0.0

    # Configure integrator
    engine_options["telemetry"]['logInternalStepperSteps'] = True
    engine_options['stepper']['odeSolver'] = 'runge_kutta_dopri'
    engine_options['stepper']['dtMax'] = 1.0e-3

    # Set the ground contact options
    engine_options['contacts']['friction'] = 1.0
    engine_options['contacts']['torsion'] = 0.0
    simulator.set_options(engine_options)

    # Generate random ground profile
    ground_params = list(starmap(random_tile_ground, zip(
        TILE_SIZE, TILE_HEIGHT_MAX, TILE_INTERP_DELTA, TILE_SPARSITY,
        TILE_ORIENTATION, TILE_SEED)))
    engine_options["world"]["groundProfile"] = sum_heightmaps([
        ground_params[0], merge_heightmaps(ground_params[1:])])
    simulator.set_options(engine_options)

    # Sample the initial state
    qpos = pin.neutral(simulator.robot.pinocchio_model)
    qvel = np.zeros(simulator.robot.nv)
    qpos[2] += 1.5
    qvel[0] = 2.0
    qvel[3] = 1.0
    qvel[5] = 2.0

    # Run a simulation
    simulator.simulate(5.0, qpos, qvel)

    # Replay the simulation
    simulator.replay(enable_travelling=False, display_contact_frames=True)
