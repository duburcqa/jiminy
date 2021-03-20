import os
from pkg_resources import resource_filename

from gym_jiminy.common.envs import WalkerJiminyEnv


# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0
# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1
# Stepper update period (:float [s])
STEP_DT = 1.0e-3

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


class SpotmicroJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs):
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/quadrupedal_robots/spotmicro")
        urdf_path = os.path.join(data_root_dir, "spotmicro.urdf")

        # Initialize the walker environment
        super().__init__(**{**dict(
            urdf_path=urdf_path,
            mesh_path=data_root_dir,
            simu_duration_max=SIMULATION_DURATION,
            step_dt=STEP_DT,
            reward_mixture=REWARD_MIXTURE,
            std_ratio=STD_RATIO,
            avoid_instable_collisions=False,
            debug=debug), **kwargs})
