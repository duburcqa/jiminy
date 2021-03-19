import os
import math
import numpy as np
from pkg_resources import resource_filename

from jiminy_py.core import (
    DistanceConstraint, JointConstraint, get_joint_type, joint_t)

from pinocchio import neutral, SE3

from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.controllers import PDController
from gym_jiminy.common.pipeline import build_pipeline


# Parameters of neutral configuration
DEFAULT_SAGITTAL_HIP_ANGLE = 30.0 / 180.0 * math.pi
DEFAULT_KNEE_ANGLE = -65.0 / 180.0 * math.pi
DEFAULT_ANKLE_ANGLE = 80.0 / 180.0 * math.pi
DEFAULT_TOE_ANGLE = -95.0 / 180.0 * math.pi

# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0
# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1
# Stepper update period (:float [s])
STEP_DT = 0.01

# PID proportional gains (one per actuated joint)
PID_KP = np.array([50.0, 50.0, 50.0, 100.0, 10.0,
                   50.0, 50.0, 50.0, 100.0, 10.0])
# PID derivative gains (one per actuated joint)
PID_KD = np.array([0.01, 0.02, 0.02, 0.05, 0.08,
                   0.01, 0.02, 0.02, 0.05, 0.08])

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
            "right_pushrod_tarsus", "right_pushrod_hip", 0.5012)
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
            "left_pushrod_tarsus", "left_pushrod_hip", 0.5012)
        self.robot.add_constraint("pushrod_left", pushrod_left)

        # Replace knee to shin spring by fixed joint constraint
        right_spring_knee_to_shin = JointConstraint("knee_to_shin_right")
        self.robot.add_constraint(
            "right_spring_knee_to_shin", right_spring_knee_to_shin)
        left_spring_knee_to_shin = JointConstraint("knee_to_shin_left")
        self.robot.add_constraint(
            "left_spring_knee_to_shin", left_spring_knee_to_shin)

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
