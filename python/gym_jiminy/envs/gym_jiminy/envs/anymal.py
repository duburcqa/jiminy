import os
import numpy as np
from pkg_resources import resource_filename

from ..common.env_locomotion import WalkerJiminyEnv, WalkerPDControlJiminyEnv


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
    """
    @brief    TODO
    """
    def __init__(self, debug: bool = False, **kwargs):
        """
        @brief    TODO
        """
        # Get the urdf and mesh paths
        data_root_dir = os.path.join(
            resource_filename('gym_jiminy.envs', 'data'),
            "quadrupedal_robots/anymal")
        urdf_path = os.path.join(data_root_dir, "anymal.urdf")

        # Initialize the walker environment
        super().__init__(
            urdf_path=urdf_path,
            mesh_path=data_root_dir,
            simu_duration_max=SIMULATION_DURATION,
            dt=STEP_DT,
            reward_mixture=REWARD_MIXTURE,
            std_ratio=STD_RATIO,
            avoid_instable_collisions=False,
            debug=debug,
            **kwargs)

    def _refresh_observation_space(self) -> None:
        self.observation_space = self._get_state_space()

    def _fetch_obs(self) -> None:
        return np.concatenate(self._state)

    def _is_done(self):
        return super()._is_done()

    def _compute_reward(self):
        return super()._compute_reward()


class ANYmalPDControlJiminyEnv(ANYmalJiminyEnv, WalkerPDControlJiminyEnv):
    """
    @brief    TODO
    """
    def __init__(self, hlc_to_llc_ratio: int = HLC_TO_LLC_RATIO, **kwargs):
        """
        @brief    TODO
        """
        super().__init__(
            hlc_to_llc_ratio=hlc_to_llc_ratio, pid_kp=PID_KP, pid_kd=PID_KD,
            **kwargs)
