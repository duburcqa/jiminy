import os
import math
import numpy as np
from pathlib import Path
from pkg_resources import resource_filename

from jiminy_py.core import (joint_t,
                            get_joint_type,
                            build_models_from_urdf,
                            build_reduced_models,
                            DistanceConstraint,
                            Robot)
from jiminy_py.robot import load_hardware_description_file, BaseJiminyRobot
from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.controllers import PDController
from gym_jiminy.common.pipeline import build_pipeline

from pinocchio import neutral, SE3


# Parameters of neutral configuration
DEFAULT_SAGITTAL_HIP_ANGLE = 23.0 / 180.0 * math.pi
DEFAULT_KNEE_ANGLE = -65.0 / 180.0 * math.pi
DEFAULT_ANKLE_ANGLE = 80.0 / 180.0 * math.pi
DEFAULT_TOE_ANGLE = -90.0 / 180.0 * math.pi

# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0

# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1

# Stepper update period (:float [s])
STEP_DT = 0.04

# PID proportional gains (one per actuated joint)
PID_KP = np.array([50.0, 50.0, 50.0, 80.0, 8.0,
                   50.0, 50.0, 50.0, 80.0, 8.0])
# PID derivative gains (one per actuated joint)
PID_KD = np.array([0.01, 0.02, 0.02, 0.03, 0.02,
                   0.01, 0.02, 0.02, 0.03, 0.02])

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


class CassieJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs):
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/bipedal_robots/cassie")
        urdf_path = os.path.join(data_root_dir, "cassie.urdf")

        # Load the full models
        pinocchio_model, collision_model, visual_model = \
            build_models_from_urdf(urdf_path,
                                   has_freeflyer=True,
                                   build_visual_model=True,
                                   mesh_package_dirs=[data_root_dir])

        # Fix passive rotary joints with spring.
        # Alternatively, it would be more realistic to model them using the
        # internal dynamics of the controller to add spring forces, but it
        # would slow down the simulation.
        qpos = neutral(pinocchio_model)
        joint_locked_indices = [
            pinocchio_model.getJointId(joint_name)
            for joint_name in ("knee_to_shin_right", "knee_to_shin_left")]
        pinocchio_model, collision_model, visual_model = build_reduced_models(
            pinocchio_model, collision_model, visual_model,
            joint_locked_indices, qpos)

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

        # Add missing pushrod close kinematic chain constraint
        M_pushrod_tarsus_right = SE3(
            np.eye(3), np.array([-0.12, 0.03, -0.005]))
        M_pushrod_hip_right = SE3(
            np.eye(3), np.array([0.0, 0.0, -0.045]))
        self.robot.add_frame(
            "right_pushrod_tarsus", "right_tarsus", M_pushrod_tarsus_right)
        self.robot.add_frame(
            "right_pushrod_hip", "hip_flexion_right", M_pushrod_hip_right)
        pushrod_right = DistanceConstraint(
            "right_pushrod_tarsus", "right_pushrod_hip", 0.5)
        pushrod_right.baumgarte_freq = 2.0
        self.robot.add_constraint("pushrod_right", pushrod_right)
        M_pushrod_tarsus_left = SE3(
            np.eye(3), np.array([-0.12, 0.03, 0.005]))
        M_pushrod_hip_left = SE3(
            np.eye(3), np.array([0.0, 0.0, 0.045]))
        self.robot.add_frame(
            "left_pushrod_tarsus", "left_tarsus", M_pushrod_tarsus_left)
        self.robot.add_frame(
            "left_pushrod_hip", "hip_flexion_left", M_pushrod_hip_left)
        pushrod_left = DistanceConstraint(
            "left_pushrod_tarsus", "left_pushrod_hip", 0.5)
        pushrod_left.baumgarte_freq = 2.0
        self.robot.add_constraint("pushrod_left", pushrod_left)

        # Remove irrelevant contact points
        self.robot.remove_contact_points([
            name for name in self.robot.contact_frames_names
            if int(name.split("_")[-1]) in (0, 1, 4, 5)])

    def _neutral(self):
        def set_joint_rotary_position(joint_name, q_full, theta):
            joint_idx = self.robot.pinocchio_model.getJointId(joint_name)
            joint = self.robot.pinocchio_model.joints[joint_idx]
            joint_type = get_joint_type(
                self.robot.pinocchio_model, joint_idx)
            if joint_type == joint_t.ROTARY_UNBOUNDED:
                q_joint = np.array([math.cos(theta), math.sin(theta)])
            else:
                q_joint = theta
            q_full[joint.idx_q + np.arange(joint.nq)] = q_joint

        qpos = neutral(self.robot.pinocchio_model)
        for s in ['left', 'right']:
            set_joint_rotary_position(
                f'hip_flexion_{s}', qpos, DEFAULT_SAGITTAL_HIP_ANGLE)
            set_joint_rotary_position(
                f'knee_joint_{s}', qpos, DEFAULT_KNEE_ANGLE)
            set_joint_rotary_position(
                f'ankle_joint_{s}', qpos, DEFAULT_ANKLE_ANGLE)
            set_joint_rotary_position(
                f'toe_joint_{s}', qpos, DEFAULT_TOE_ANGLE)

        return qpos


CassiePDControlJiminyEnv = build_pipeline(**{
    'env_config': {
        'env_class': CassieJiminyEnv
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
