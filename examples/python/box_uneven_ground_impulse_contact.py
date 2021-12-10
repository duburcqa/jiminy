import os
from itertools import starmap

import numpy as np

from jiminy_py.simulator import Simulator
from jiminy_py.core import random_tile_ground, sum_heightmap, merge_heightmap
from gym_jiminy.common.envs import BaseJiminyEnv


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


# Create a gym environment for a simple cube
urdf_path = f"{MODULE_DIR}/../../unit_py/data/box_collision_mesh.urdf"
env = BaseJiminyEnv(Simulator.build(
    urdf_path, has_freeflyer=True), step_dt=0.01)

# Enable constraint contact model
engine_options = env.engine.get_options()
engine_options['contacts']['model'] = 'constraint'

# Configure integrator
engine_options['stepper']['odeSolver'] = 'euler_explicit'
engine_options['stepper']['dtMax'] = 1.0e-3

# Generate random ground profile
ground_params = list(starmap(random_tile_ground, zip(
    TILE_SIZE, TILE_HEIGHT_MAX, TILE_INTERP_DELTA, TILE_SPARSITY,
    TILE_ORIENTATION, TILE_SEED)))
engine_options["world"]["groundProfile"] = sum_heightmap([
    ground_params[0], merge_heightmap(ground_params[1:])])
env.engine.set_options(engine_options)

# Monkey-patch the initial state sampling function
sample_state_orig = env._sample_state

def sample_state():
    qpos, qvel = env._neutral(), np.zeros(env.robot.nv)
    qpos[2] += 1.5
    qvel[0] = 2.0
    qvel[3] = 1.0
    qvel[5] = 2.0
    return qpos, qvel

env._sample_state = sample_state

# Run a simulation
engine_options['contacts']['friction'] = 1.0
engine_options['contacts']['torsion'] = 0.0
env.engine.set_options(engine_options)

env.reset()
for _ in range(500):
    env.step()
env.stop()

# Replay the simulation
env.replay(enable_travelling=False, display_contact_frames=True)
