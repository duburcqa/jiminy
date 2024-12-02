""" TODO: Write documentation.
"""
import os
from pathlib import Path
from importlib.resources import files
from typing import Any

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.robot import load_hardware_description_file
from jiminy_py.viewer.viewer import DEFAULT_CAMERA_XYZRPY_REL
from pinocchio import neutral, buildReducedModel

from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.blocks import (MotorSafetyLimit,
                                      PDController,
                                      PDAdapter,
                                      MahonyFilter)
from gym_jiminy.common.utils import build_pipeline
from gym_jiminy.toolbox.math import ConvexHull2D


# Sagittal hip angle of neutral configuration (:float [rad])
NEUTRAL_SAGITTAL_HIP_ANGLE = 0.2

# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0

# Stepper update period (:float [s])
STEP_DT = 0.04

# Motor safety to avoid violent motions
MOTOR_POSITION_MARGIN = 0.02
MOTOR_VELOCITY_SAFE_GAIN = 0.15
MOTOR_VELOCITY_MAX = 4.0
MOTOR_ACCELERATION_MAX = 30.0

# PID proportional gains (one per actuated joint)
PD_REDUCED_KP = (
    # Left leg: [HpZ, HpX, HpY, KnY, AkY, AkX]
    5000.0, 5000.0, 8000.0, 4000.0, 8000.0, 5000.0,
    # Right leg: [HpZ, HpX, HpY, KnY, AkY, AkX]
    5000.0, 5000.0, 8000.0, 4000.0, 8000.0, 5000.0)
PD_REDUCED_KD = (
    # Left leg: [HpZ, HpX, HpY, KnY, AkY, AkX]
    0.01, 0.02, 0.02, 0.01, 0.025, 0.01,
    # Right leg: [HpZ, HpX, HpY, KnY, AkY, AkX]
    0.01, 0.02, 0.02, 0.01, 0.025, 0.01)

# PID derivative gains (one per actuated joint)
PD_FULL_KP = (
    # Back: [Z, Y, X]
    5000.0, 8000.0, 5000.0,
    # Left arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    500.0, 100.0, 200.0, 500.0, 10.0, 100.0, 10.0,
    # Neck: [Y]
    100.0,
    # Right arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    500.0, 100.0, 200.0, 500.0, 10.0, 100.0, 10.0,
    # Lower body motors
    *PD_REDUCED_KP)
PD_FULL_KD = (
    # Back: [Z, Y, X]
    0.01, 0.015, 0.02,
    # Left arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02,
    # Neck: [Y]
    0.01,
    # Right arm: [ShZ, ShX, ElY, ElX, WrY, WrX, WrY2]
    0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02,
    # Lower body motors
    *PD_REDUCED_KD)

# Mahony filter proportional and derivative gains
# See: https://cas.mines-paristech.fr/~petit/papers/ral22/main.pdf
MAHONY_KP = 0.75
MAHONY_KI = 0.057

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


def _cleanup_contact_points(env: WalkerJiminyEnv) -> None:
    contact_frame_indices = env.robot.contact_frame_indices
    contact_frame_names = env.robot.contact_frame_names
    num_contacts = len(env.robot.contact_frame_indices) // 2
    for contact_slice in (slice(num_contacts), slice(num_contacts, None)):
        contact_positions = np.stack([
            env.robot.pinocchio_data.oMf[frame_index].translation
            for frame_index in contact_frame_indices[contact_slice]
            ], axis=0)
        contact_bottom_index = np.argsort(
            contact_positions[:, 2])[:(num_contacts // 2)]
        convex_hull = ConvexHull2D(contact_positions[contact_bottom_index, :2])
        env.robot.remove_contact_points([
            contact_frame_names[contact_slice][i]
            for i in set(range(num_contacts)).difference(
                contact_bottom_index[convex_hull.indices])])


class AtlasJiminyEnv(WalkerJiminyEnv):
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
        data_dir = str(files("gym_jiminy.envs") / "data/bipedal_robots/atlas")
        urdf_path = os.path.join(data_dir, "atlas.urdf")

        # Override default camera pose to change the reference frame
        kwargs.setdefault("viewer_kwargs", {}).setdefault(
            "camera_pose", (*DEFAULT_CAMERA_XYZRPY_REL, 'utorso'))

        # Initialize the walker environment
        super().__init__(
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

        # Remove irrelevant contact points
        _cleanup_contact_points(self)

    def _neutral(self) -> np.ndarray:
        def joint_position_index(joint_name: str) -> int:
            """Helper to get the start index from a joint index.
            """
            joint_index = self.robot.pinocchio_model.getJointId(joint_name)
            return self.robot.pinocchio_model.joints[joint_index].idx_q

        qpos = neutral(self.robot.pinocchio_model)
        qpos[joint_position_index('back_bky')] = NEUTRAL_SAGITTAL_HIP_ANGLE
        qpos[joint_position_index('l_arm_elx')] = NEUTRAL_SAGITTAL_HIP_ANGLE
        qpos[joint_position_index('l_arm_shx')] = - np.pi / 2.0
        qpos[joint_position_index('l_arm_shz')] = np.pi / 4.0
        qpos[joint_position_index('l_arm_ely')] = np.pi / 4.0 + np.pi / 2.0
        qpos[joint_position_index('r_arm_elx')] = - NEUTRAL_SAGITTAL_HIP_ANGLE
        qpos[joint_position_index('r_arm_shx')] = np.pi / 2.0
        qpos[joint_position_index('r_arm_shz')] = - np.pi / 4.0
        qpos[joint_position_index('r_arm_ely')] = np.pi / 4.0 + np.pi / 2.0

        return qpos


class AtlasReducedJiminyEnv(WalkerJiminyEnv):
    def __init__(self, debug: bool = False, **kwargs: Any) -> None:
        # Get the urdf and mesh paths
        data_dir = str(files("gym_jiminy.envs") / "data/bipedal_robots/atlas")
        urdf_path = os.path.join(data_dir, "atlas.urdf")

        # Load the full models
        pinocchio_model, collision_model, visual_model = (
            jiminy.build_models_from_urdf(urdf_path,
                                          has_freeflyer=True,
                                          build_visual_model=True,
                                          mesh_package_dirs=[data_dir]))

        # Generate the reference configuration
        def joint_position_index(joint_name: str) -> int:
            """Helper to get the start index from a joint index.
            """
            nonlocal pinocchio_model
            joint_index = pinocchio_model.getJointId(joint_name)
            return pinocchio_model.joints[joint_index].idx_q

        qpos = neutral(pinocchio_model)
        qpos[joint_position_index('back_bky')] = NEUTRAL_SAGITTAL_HIP_ANGLE
        qpos[joint_position_index('l_arm_elx')] = NEUTRAL_SAGITTAL_HIP_ANGLE
        qpos[joint_position_index('l_arm_shx')] = - np.pi / 2.0
        qpos[joint_position_index('l_arm_shz')] = np.pi / 4.0
        qpos[joint_position_index('l_arm_ely')] = np.pi / 4.0 + np.pi / 2.0
        qpos[joint_position_index('r_arm_elx')] = - NEUTRAL_SAGITTAL_HIP_ANGLE
        qpos[joint_position_index('r_arm_shx')] = np.pi / 2.0
        qpos[joint_position_index('r_arm_shz')] = - np.pi / 4.0
        qpos[joint_position_index('r_arm_ely')] = np.pi / 4.0 + np.pi / 2.0

        # Build the reduced models
        joint_locked_indices = [
            pinocchio_model.getJointId(joint_name)
            for joint_name in pinocchio_model.names[2:]
            if "_leg_" not in joint_name]
        pinocchio_model, (collision_model, visual_model) = buildReducedModel(
            pinocchio_model, [collision_model, visual_model],
            joint_locked_indices, qpos)

        # Build the robot and load the hardware
        robot = jiminy.Robot()
        robot.initialize(pinocchio_model, collision_model, visual_model)
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
                config_path=str(
                    Path(urdf_path).with_suffix('')) + '_options.toml',
                simulation_duration_max=SIMULATION_DURATION,
                step_dt=STEP_DT,
                reward_mixture=REWARD_MIXTURE,
                std_ratio=STD_RATIO),
                **kwargs})

        # Remove irrelevant contact points
        _cleanup_contact_points(self)


AtlasPDControlJiminyEnv = build_pipeline(
    env_config=dict(
        cls=AtlasJiminyEnv
    ),
    layers_config=[
        dict(
            block=dict(
                cls=MotorSafetyLimit,
                kwargs=dict(
                    kp=1.0 / MOTOR_POSITION_MARGIN,
                    kd=MOTOR_VELOCITY_SAFE_GAIN,
                    soft_position_margin=0.0,
                    soft_velocity_max=MOTOR_VELOCITY_MAX,
                )
            ),
        ), dict(
            block=dict(
                cls=PDController,
                kwargs=dict(
                    update_ratio=1,
                    kp=PD_FULL_KP,
                    kd=PD_FULL_KD,
                    joint_position_margin=0.0,
                    joint_velocity_limit=MOTOR_VELOCITY_MAX,
                    joint_acceleration_limit=MOTOR_ACCELERATION_MAX
                )
            ),
            wrapper=dict(
                kwargs=dict(
                    augment_observation=False
                )
            )
        ), dict(
            block=dict(
                cls=PDAdapter,
                kwargs=dict(
                    update_ratio=-1,
                    order=1,
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

AtlasReducedPDControlJiminyEnv = build_pipeline(
    env_config=dict(
        cls=AtlasReducedJiminyEnv
    ),
    layers_config=[
        dict(
            block=dict(
                cls=MotorSafetyLimit,
                kwargs=dict(
                    kp=1.0 / MOTOR_POSITION_MARGIN,
                    kd=1.0 / MOTOR_VELOCITY_MAX,
                    soft_position_margin=0.0,
                    soft_velocity_max=MOTOR_VELOCITY_MAX
                )
            ),
        ), dict(
            block=dict(
                cls=PDController,
                kwargs=dict(
                    update_ratio=1,
                    kp=PD_REDUCED_KP,
                    kd=PD_REDUCED_KD,
                    joint_position_margin=0.0,
                    joint_velocity_limit=MOTOR_VELOCITY_MAX,
                    joint_acceleration_limit=MOTOR_ACCELERATION_MAX
                )
            ),
            wrapper=dict(
                kwargs=dict(
                    augment_observation=False
                )
            )
        ), dict(
            block=dict(
                cls=PDAdapter,
                kwargs=dict(
                    update_ratio=-1,
                    order=1,
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
