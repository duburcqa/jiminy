""" TODO: Write documentation.
"""
import os
import numpy as np
from pkg_resources import resource_filename

from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.controllers import PDController
from gym_jiminy.common.pipeline import build_pipeline


# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0
# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1
# Stepper update period (:float [s])
STEP_DT = 1.0e-3

# PID proportional gains (one per actuated joint)
PID_KP = np.array([1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0,
                   1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0])
# PID derivative gains (one per actuated joint)
PID_KD = np.array([0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
                   0.003, 0.003, 0.003, 0.003, 0.003, 0.003])

# Reward weight for each individual component that can be optimized
REWARD_MIXTURE = {
    'direction': 0.0,
    'energy': 0.0,
    'done': 1.0
}
# Standard deviation ratio of each individual origin of randomness
STD_RATIO = {
    'model': 0.0,
    'ground': 0.0,
    'sensors': 0.0,
    'disturbance': 0.0,
}


class ANYmalJiminyEnv(WalkerJiminyEnv):
    """ TODO: Write documentation.
    """
    def __init__(self, debug: bool = False, **kwargs):
        """ TODO: Write documentation.
        """
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/quadrupedal_robots/anymal")
        urdf_path = os.path.join(data_root_dir, "anymal.urdf")

        # Initialize the walker environment
        super().__init__(
            urdf_path=urdf_path,
            mesh_path=data_root_dir,
            avoid_instable_collisions=True,
            debug=debug,
            **{**dict(
                simu_duration_max=SIMULATION_DURATION,
                step_dt=STEP_DT,
                reward_mixture=REWARD_MIXTURE,
                std_ratio=STD_RATIO),
                **kwargs})


ANYmalPDControlJiminyEnv = build_pipeline(**{
    'env_config': {
        'env_class': ANYmalJiminyEnv
    },
    'blocks_config': [{
        'block_class': PDController,
        'block_kwargs': {
            'update_ratio': HLC_TO_LLC_RATIO,
            'pid_kp': PID_KP,
            'pid_kd': PID_KD
        },
        'wrapper_kwargs': {
            'augment_observation': False
        }}
    ]
})
