## @file
import os
import numpy as np
from pkg_resources import resource_filename

from pinocchio import neutral

from ..common.env_locomotion import WalkerJiminyEnv, WalkerPDControlJiminyEnv


DEFAULT_SAGITTAL_HIP_ANGLE = 0.2

SIMULATION_DURATION = 20.0  # (s) Default simulation duration
HLC_TO_LLC_RATIO = 1  # (NA)
ENGINE_DT = 1.0e-3  # (s) Stepper update period

PID_KP = np.array([1000.0, 12000.0, 1000.0,                          # Back: [X, Y, Z]
                   100.0, 100.0, 100.0, 100.0, 500.0, 10.0, 10.0,    # Left arm: [ElX, ElY, MwX, ShX, ShZ, UwY, LwY]
                   1000.0, 1500.0, 4000.0, 4000.0, 8000.0, 1000.0,   # Left leg: [KnY, AkX, HpY, HpX, AkY, HpZ]
                   1000.0,                                           # Neck: [Y]
                   100.0, 100.0, 100.0, 100.0, 500.0, 10.0, 10.0,    # Right arm: [ElX, ElY, MwX, ShX, ShZ, UwY, LwY]
                   1000.0, 1500.0, 4000.0, 4000.0, 8000.0, 1000.0])  # Right leg: [KnY, AkX, HpY, HpX, AkY, HpZ]
PID_KD = np.array([0.01, 0.01, 0.01,                                 # Back: [X, Y, Z]
                   0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,         # Left arm: [ElX, ElY, MwX, ShX, ShZ, UwY, LwY]
                   0.01, 0.002, 0.002, 0.002, 0.002, 0.01,            # Left leg: [KnY, AkX, HpY, HpX, AkY, HpZ]
                   0.01,                                             # Neck: [Y]
                   0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,         # Right arm: [ElX, ElY, MwX, ShX, ShZ, UwY, LwY]
                   0.00, 0.002, 0.002, 0.002, 0.002, 0.01])           # Right leg: [KnY, AkX, HpY, HpX, AkY, HpZ]

REWARD_MIXTURE = {
    'direction': 0.0,
    'energy': 0.0,
    'done': 1.0
}
REWARD_STD_RATIO = {
    'model': 0.0,
    'ground': 0.0,
    'sensors': 0.0,
    'disturbance': 0.0,
}


class AtlasJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs):
        # Get the urdf and mesh paths
        data_root_dir = os.path.join(
            resource_filename('gym_jiminy.envs', 'data'),
            "bipedal_robots/atlas")
        urdf_path = os.path.join(data_root_dir, "atlas_v5.urdf")

        # Initialize the walker environment
        super().__init__(
            urdf_path=urdf_path,
            mesh_path=data_root_dir,
            simu_duration_max=SIMULATION_DURATION,
            dt=ENGINE_DT,
            reward_mixture=REWARD_MIXTURE,
            std_ratio=REWARD_STD_RATIO,
            debug=debug,
            **kwargs)

    def _neutral(self):
        def joint_position_idx(joint_name):
            joint_idx = self.robot.pinocchio_model.getJointId(joint_name)
            return self.robot.pinocchio_model.joints[joint_idx].idx_q

        qpos = neutral(self.robot.pinocchio_model)
        qpos[joint_position_idx('back_bky')] = DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('l_arm_elx')] = DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('l_arm_shx')] = - np.pi / 2 + 1e-6
        qpos[joint_position_idx('l_arm_shz')] = np.pi / 4 - 1e-6
        qpos[joint_position_idx('l_arm_ely')] = np.pi / 4 + np.pi / 2
        qpos[joint_position_idx('r_arm_elx')] = - DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('r_arm_shx')] = np.pi / 2 - 1e-6
        qpos[joint_position_idx('r_arm_shz')] = - np.pi / 4 + 1e-6
        qpos[joint_position_idx('r_arm_ely')] = np.pi / 4 + np.pi / 2

        return qpos

    def _update_obs(self, obs):
        super()._update_obs(obs)

    def _is_done(self):
        return super()._is_done()

    def _compute_reward(self):
        return super()._compute_reward()


class AtlasPDControlJiminyEnv(AtlasJiminyEnv, WalkerPDControlJiminyEnv):
    def __init__(self,
                 hlc_to_llc_ratio: int = HLC_TO_LLC_RATIO,
                 debug: bool = False):
        super().__init__(debug,
            hlc_to_llc_ratio=hlc_to_llc_ratio, pid_kp=PID_KP, pid_kd=PID_KD)
