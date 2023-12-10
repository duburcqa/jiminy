import os
import sys
import math
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

from jiminy_py.core import (JointModelType,
                            get_joint_type,
                            build_models_from_urdf,
                            DistanceConstraint,
                            Robot)
from jiminy_py.robot import load_hardware_description_file, BaseJiminyRobot
from pinocchio import neutral, SE3, buildReducedModel

from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.blocks import PDController, MahonyFilter
from gym_jiminy.common.utils import build_pipeline

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


# Parameters of neutral configuration
NEUTRAL_FRONTAL_HIP_ANGLE = 20.0 / 180.0 * math.pi
NEUTRAL_SAGITTAL_HIP_ANGLE = 5.7 / 180.0 * math.pi
NEUTRAL_SAGITTAL_SHOULDER_ANGLE = 45.0 / 180.0 * math.pi
NEUTRAL_SAGITTAL_ELBOW_ANGLE = 68.0 / 180.0 * math.pi

# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0

# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1

# Stepper update period (:float [s])
STEP_DT = 0.04

# PID proportional gains (one per actuated joint)
PD_KP = (
    # Left leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    80.0, 100.0, 100.0, 100.0, 80.0, 80.0,
    # Left arm: [ShX, ShY, ShZ, ElY]
    50.0, 50.0, 50.0, 50.0,
    # Right leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    80.0, 100.0, 100.0, 100.0, 80.0, 80.0,
    # Right arm: [ShX, ShY, ShZ, ElY]
    50.0, 50.0, 50.0, 50.0)
# PID derivative gains (one per actuated joint)
PD_KD = (
    # Left leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    0.015, 0.02, 0.02, 0.02, 0.015, 0.015,
    # Left arm: [ShX, ShY, ShZ, ElY]
    0.01, 0.01, 0.01, 0.01,
    # Right leg: [HpX, HpZ, HpY, KnY, AkY, AkX]
    0.015, 0.02, 0.02, 0.02, 0.015, 0.015,
    # Right arm: [ShX, ShY, ShZ, ElY]
    0.01, 0.01, 0.01, 0.01)

# Mahony filter proportional and derivative gains
MAHONY_KP = 1.0
MAHONY_KI = 0.1

# Reward weight for each individual component that can be optimized
REWARD_MIXTURE = {
    'direction': 0.0,
    'energy': 0.0,
    'survival': 1.0
}
# Standard deviation ratio of each individual origin of randomness
STD_RATIO = {
    'model': 0.0,
    'ground': 0.0,
    'sensors': 0.0,
    'disturbance': 0.0,
}


class DigitJiminyEnv(WalkerJiminyEnv):
    """ TODO: Write documentation.
    """
    def __init__(self, debug: bool = False, **kwargs: Any) -> None:
        """
        :param debug: Whether the debug mode must be enabled.
                      See `BaseJiminyEnv` constructor for details.
        :param kwargs: Keyword arguments to forward to `Simulator` and
                       `WalkerJiminyEnv` constructors.
        """
        # Get the urdf and mesh paths
        data_dir = str(files("gym_jiminy.envs") / "data/bipedal_robots/digit")
        urdf_path = os.path.join(data_dir, "digit.urdf")

        # Load the full models
        pinocchio_model, collision_model, visual_model = \
            build_models_from_urdf(urdf_path,
                                   has_freeflyer=True,
                                   build_visual_model=True,
                                   mesh_package_dirs=[data_dir])

        # Fix passive rotary joints with spring.
        # Alternatively, it would be more realistic to model them using the
        # internal dynamics of the controller to add spring forces, but it
        # would slow down the simulation.
        qpos = neutral(pinocchio_model)
        joint_locked_indices = [
            pinocchio_model.getJointId(joint_name)
            for joint_name in ("shin_to_tarsus_right", "shin_to_tarsus_left")]
        pinocchio_model, (collision_model, visual_model) = buildReducedModel(
            pinocchio_model, [collision_model, visual_model],
            joint_locked_indices, qpos)

        # Build the robot and load the hardware
        robot = BaseJiminyRobot()
        Robot.initialize(robot, pinocchio_model, collision_model, visual_model)
        robot._urdf_path_orig = urdf_path  # type: ignore[attr-defined]
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
            mesh_path_dir=data_dir,
            avoid_instable_collisions=True,
            debug=debug,
            **{**dict(
                simulation_duration_max=SIMULATION_DURATION,
                step_dt=STEP_DT,
                reward_mixture=REWARD_MIXTURE,
                std_ratio=STD_RATIO),
                **kwargs})

        # Add missing pushrod close kinematic chain constraints
        M_pushrod_tarsus_right = SE3(
            np.eye(3), np.array([-0.11, 0.0, 0.0]))
        M_pushrod_hip_right = SE3(
            np.eye(3), np.array([0.0, 0.0, 0.046]))
        self.robot.add_frame(
            "right_pushrod_tarsus", "right_tarsus", M_pushrod_tarsus_right)
        self.robot.add_frame(
            "right_pushrod_hip", "hip_flexion_right", M_pushrod_hip_right)
        pushrod_right = DistanceConstraint(
            "right_pushrod_tarsus", "right_pushrod_hip")
        pushrod_right.baumgarte_freq = 20.0
        self.robot.add_constraint("pushrod_right", pushrod_right)
        M_pushrod_tarsus_left = SE3(
            np.eye(3), np.array([-0.11, 0.0, 0.0]))
        M_pushrod_hip_left = SE3(
            np.eye(3), np.array([0.0, 0.0, 0.046]))
        self.robot.add_frame(
            "left_pushrod_tarsus", "left_tarsus", M_pushrod_tarsus_left)
        self.robot.add_frame(
            "left_pushrod_hip", "hip_flexion_left", M_pushrod_hip_left)
        pushrod_left = DistanceConstraint(
            "left_pushrod_tarsus", "left_pushrod_hip")
        pushrod_left.baumgarte_freq = 20.0
        self.robot.add_constraint("pushrod_left", pushrod_left)

        # Remove irrelevant contact points
        self.robot.remove_contact_points([
            name for name in self.robot.contact_frames_names
            if int(name.split("_")[-1]) in (2, 3, 6, 7)])

    def _neutral(self) -> np.ndarray:
        def set_joint_rotary_position(joint_name: str, theta: float) -> None:
            """Helper to set the configuration of a 1-DoF revolute joint.
            """
            joint_idx = self.robot.pinocchio_model.getJointId(joint_name)
            joint = self.robot.pinocchio_model.joints[joint_idx]
            joint_type = get_joint_type(
                self.robot.pinocchio_model, joint_idx)
            q_joint: Union[Sequence[float], float]
            if joint_type == JointModelType.ROTARY_UNBOUNDED:
                q_joint = (math.cos(theta), math.sin(theta))
            else:
                q_joint = theta
            qpos[joint.idx_q + np.arange(joint.nq)] = q_joint

        qpos = neutral(self.robot.pinocchio_model)
        for s in ('left', 'right'):
            sign = 1.0 if s == 'left' else -1.0
            set_joint_rotary_position(f"hip_abduction_{s}",
                                      sign * NEUTRAL_FRONTAL_HIP_ANGLE)
            set_joint_rotary_position(f"hip_flexion_{s}",
                                      sign * NEUTRAL_SAGITTAL_HIP_ANGLE)
            set_joint_rotary_position(f"shoulder_pitch_joint_{s}",
                                      sign * NEUTRAL_SAGITTAL_SHOULDER_ANGLE)
            set_joint_rotary_position(f"elbow_joint_{s}",
                                      sign * NEUTRAL_SAGITTAL_ELBOW_ANGLE)
        return qpos


DigitPDControlJiminyEnv = build_pipeline(
    env_config=dict(
        cls=DigitJiminyEnv
    ),
    layers_config=[
        dict(
            block=dict(
                cls=PDController,
                kwargs=dict(
                    update_ratio=HLC_TO_LLC_RATIO,
                    order=1,
                    kp=PD_KP,
                    kd=PD_KD,
                    target_position_margin=0.0,
                    target_velocity_limit=float("inf")
                )
            ),
            wrapper=dict(
                kwargs=dict(
                    augment_observation=False
                )
            )
        ), dict(
            block=dict(
                cls=MahonyFilter,
                kwargs=dict(
                    update_ratio=1,
                    kp=MAHONY_KP,
                    ki=MAHONY_KI,
                )
            )
        )
    ]
)
