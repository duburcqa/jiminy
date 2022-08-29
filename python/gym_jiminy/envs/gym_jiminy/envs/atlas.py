import os
import numpy as np
from pathlib import Path
from pkg_resources import resource_filename
from typing import Any

from jiminy_py.core import build_models_from_urdf, build_reduced_models, Robot
from jiminy_py.robot import load_hardware_description_file, BaseJiminyRobot
from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.controllers import PDController
from gym_jiminy.common.pipeline import build_pipeline
from gym_jiminy.toolbox.math import ConvexHull

from pinocchio import neutral


# Sagittal hip angle of neutral configuration (:float [rad])
DEFAULT_SAGITTAL_HIP_ANGLE = 0.2

# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0

# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1

# Stepper update period (:float [s])
STEP_DT = 0.04

# PID proportional gains (one per actuated joint)
PID_REDUCED_KP = np.array([
    # Left leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    5000.0, 5000.0, 8000.0, 4000.0, 8000.0, 5000.0,
    # Right leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    5000.0, 5000.0, 8000.0, 4000.0, 8000.0, 5000.0])
PID_REDUCED_KD = np.array([
    # Left leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    0.02, 0.01, 0.015, 0.01, 0.015, 0.01,
    # Right leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    0.02, 0.01, 0.015, 0.01, 0.015, 0.01])

PID_FULL_KP = np.array([
    # Neck: [Y]
    1000.0,
    # Back: [Z, Y, X]
    5000.0, 8000.0, 5000.0,
    # Left arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    500.0, 100.0, 200.0, 500.0, 10.0, 100.0, 10.0,
    # Right arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    500.0, 100.0, 200.0, 500.0, 10.0, 100.0, 10.0,
    # Lower body motors
    *PID_REDUCED_KP])
PID_FULL_KD = np.array([
    # Neck: [Y]
    0.01,
    # Back: [Z, Y, X]
    0.01, 0.015, 0.02,
    # Left arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02,
    # Right arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02,
    # Lower body motors
    *PID_REDUCED_KD])

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


def _cleanup_contact_points(env: "AtlasJiminyEnv") -> None:
    contact_frames_idx = env.robot.contact_frames_idx
    contact_frames_names = env.robot.contact_frames_names
    num_contacts = int(len(env.robot.contact_frames_idx) // 2)
    for contact_slice in (slice(num_contacts), slice(num_contacts, None)):
        contact_positions = np.stack([
            env.robot.pinocchio_data.oMf[frame_idx].translation
            for frame_idx in contact_frames_idx[contact_slice]], axis=0)
        contact_bottom_idx = np.argsort(
            contact_positions[:, 2])[:int(num_contacts//2)]
        convex_hull = ConvexHull(contact_positions[contact_bottom_idx, :2])
        env.robot.remove_contact_points([
            contact_frames_names[contact_slice][i]
            for i in set(range(num_contacts)).difference(
                contact_bottom_idx[convex_hull._vertex_indices])])


class AtlasJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs: Any) -> None:
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/bipedal_robots/atlas")
        urdf_path = os.path.join(data_root_dir, "atlas_v4.urdf")

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

        # Remove irrelevant contact points
        _cleanup_contact_points(self)

    def _neutral(self):
        def joint_position_idx(joint_name):
            joint_idx = self.robot.pinocchio_model.getJointId(joint_name)
            return self.robot.pinocchio_model.joints[joint_idx].idx_q

        qpos = neutral(self.robot.pinocchio_model)
        qpos[joint_position_idx('back_bky')] = DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('l_arm_elx')] = DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('l_arm_shx')] = - np.pi / 2.0
        qpos[joint_position_idx('l_arm_shz')] = np.pi / 4.0
        qpos[joint_position_idx('l_arm_ely')] = np.pi / 4.0 + np.pi / 2.0
        qpos[joint_position_idx('r_arm_elx')] = - DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('r_arm_shx')] = np.pi / 2.0
        qpos[joint_position_idx('r_arm_shz')] = - np.pi / 4.0
        qpos[joint_position_idx('r_arm_ely')] = np.pi / 4.0 + np.pi / 2.0

        return qpos


class AtlasReducedJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs: Any) -> None:
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/bipedal_robots/atlas")
        urdf_path = os.path.join(data_root_dir, "atlas_v4.urdf")

        # Load the full models
        pinocchio_model, collision_model, visual_model = \
            build_models_from_urdf(urdf_path,
                                   has_freeflyer=True,
                                   build_visual_model=True,
                                   mesh_package_dirs=[data_root_dir])

        # Generate the reference configuration
        def joint_position_idx(joint_name):
            joint_idx = pinocchio_model.getJointId(joint_name)
            return pinocchio_model.joints[joint_idx].idx_q

        qpos = neutral(pinocchio_model)
        qpos[joint_position_idx('back_bky')] = DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('l_arm_elx')] = DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('l_arm_shx')] = - np.pi / 2.0
        qpos[joint_position_idx('l_arm_shz')] = np.pi / 4.0
        qpos[joint_position_idx('l_arm_ely')] = np.pi / 4.0 + np.pi / 2.0
        qpos[joint_position_idx('r_arm_elx')] = - DEFAULT_SAGITTAL_HIP_ANGLE
        qpos[joint_position_idx('r_arm_shx')] = np.pi / 2.0
        qpos[joint_position_idx('r_arm_shz')] = - np.pi / 4.0
        qpos[joint_position_idx('r_arm_ely')] = np.pi / 4.0 + np.pi / 2.0

        # Build the reduced models
        joint_locked_indices = [
            pinocchio_model.getJointId(joint_name)
            for joint_name in pinocchio_model.names[2:]
            if "_leg_" not in joint_name]
        pinocchio_model, collision_model, visual_model = build_reduced_models(
            pinocchio_model,
            collision_model,
            visual_model,
            joint_locked_indices,
            qpos)

        # Build the robot and load the hardware
        robot = BaseJiminyRobot()
        Robot.initialize(robot, pinocchio_model, collision_model, visual_model)
        robot._urdf_path_orig = urdf_path
        hardware_path = str(Path(urdf_path).with_suffix('')) + '_hardware.toml'
        load_hardware_description_file(
            robot,
            hardware_path,
            avoid_instable_collisions=True,
            verbose=debug)

        # Initialize the walker environment
        super().__init__(
            robot=robot,
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

        # Remove irrelevant contact points
        _cleanup_contact_points(self)


AtlasPDControlJiminyEnv = build_pipeline(**{
    'env_config': {
        'env_class': AtlasJiminyEnv
    },
    'blocks_config': [{
        'block_class': PDController,
        'block_kwargs': {
            'update_ratio': HLC_TO_LLC_RATIO,
            'pid_kp': PID_FULL_KP,
            'pid_kd': PID_FULL_KD
        },
        'wrapper_kwargs': {
            'augment_observation': False
        }}
    ]
})


AtlasReducedPDControlJiminyEnv = build_pipeline(**{
    'env_config': {
        'env_class': AtlasReducedJiminyEnv
    },
    'blocks_config': [{
        'block_class': PDController,
        'block_kwargs': {
            'update_ratio': HLC_TO_LLC_RATIO,
            'pid_kp': PID_REDUCED_KP,
            'pid_kd': PID_REDUCED_KD
        },
        'wrapper_kwargs': {
            'augment_observation': False
        }}
    ]
})
