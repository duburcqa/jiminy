import os
import numpy as np
from pkg_resources import resource_filename

from pinocchio import neutral

from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.controllers import PDController
from gym_jiminy.common.pipeline import build_pipeline


# Sagittal hip angle of neutral configuration (:float [rad])
DEFAULT_SAGITTAL_HIP_ANGLE = 0.2

# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0
# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1
# Stepper update period (:float [s])
STEP_DT = 1.0e-3

# PID proportional gains (one per actuated joint)
PID_KP = np.array([
    # Back: [X, Y, Z]
    4000.0, 12000.0, 1000.0,
    # Left arm: [ElX, ElY, ShX, ShZ, WrX, WrY, WrY2]
    200.0, 100.0, 100.0, 500.0, 100.0, 10.0, 10.0,
    # Left leg: [AkX, AkY, HpZ, HpX, HpY, KnY]
    1500.0, 10000.0, 4000.0, 4000.0, 1000.0, 1000.0,
    # Neck: [Y]
    1000.0,
    # Right arm: [ElX, ElY, ShX, ShZ, WrX, WrY, WrY2]
    200.0, 100.0, 100.0, 500.0, 100.0, 10.0, 10.0,
    # Right leg: [AkX, AkY, HpZ, HpX, HpY, KnY]
    1500.0, 10000.0, 4000.0, 4000.0, 1000.0, 1000.0])
# PID derivative gains (one per actuated joint)
PID_KD = np.array([
    # Back: [X, Y, Z]
    0.08, 0.02, 0.01,
    # Left arm: [ElX, ElY, ShX, ShZ, WrX, WrY, WrY2]
    0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Left leg: [AkX, AkY, HpZ, HpX, HpY, KnY]
    0.002, 0.02, 0.002, 0.002, 0.01, 0.01,
    # Neck: [Y]
    0.01,
    # Right arm: [ElX, ElY, ShX, ShZ, WrX, WrY, WrY2]
    0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    # Right leg: [AkX, AkY, HpZ, HpX, HpY, KnY]
    0.002, 0.02, 0.002, 0.002, 0.01, 0.01])


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


class AtlasJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs):
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/bipedal_robots/atlas")
        urdf_path = os.path.join(data_root_dir, "atlas_v4.urdf")

        # Initialize the walker environment
        super().__init__(**{**dict(
            urdf_path=urdf_path,
            mesh_path=data_root_dir,
            simu_duration_max=SIMULATION_DURATION,
            step_dt=STEP_DT,
            reward_mixture=REWARD_MIXTURE,
            std_ratio=STD_RATIO,
            avoid_instable_collisions=True,
            debug=debug), **kwargs})

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


AtlasPDControlJiminyEnv = build_pipeline(**{
    'env_config': {
        'env_class': AtlasJiminyEnv
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
