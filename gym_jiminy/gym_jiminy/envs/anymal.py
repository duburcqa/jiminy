## @file
import os
import numpy as np
from pkg_resources import resource_filename

from ..common.env_locomotion import WalkerJiminyEnv, WalkerPDControlJiminyEnv


SIMULATION_DURATION = 20.0  # ([float] s) Default simulation duration
HLC_TO_LLC_RATIO = 1        # ([int]  NA) Ratio between the High-level neural network PID target update and Low-level PID torque update
STEP_DT = 0.02              # ([float] s) Stepper update period

PID_KP = np.array([1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0,
                   1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0])
PID_KD = np.array([0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
                   0.003, 0.003, 0.003, 0.003, 0.003, 0.003])

REWARD_MIXTURE = {
    'direction': 0.0,
    'energy': 0.0,
    'done': 1.0
}
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
            debug=debug,
            **kwargs)

    def _refresh_observation_space(self) -> None:
        super()._refresh_observation_space()
        self.observation_space = self.observation_space['state']
        self._observation = self._observation['state']

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
    def __init__(self,
                 hlc_to_llc_ratio: int = HLC_TO_LLC_RATIO,
                 debug: bool = False):
        """
        @brief    TODO
        """
        super().__init__(debug,
            hlc_to_llc_ratio=hlc_to_llc_ratio, pid_kp=PID_KP, pid_kd=PID_KD)
