""" TODO: Write documentation.
"""
import os
import sys
from typing import Any

from gym_jiminy.common.envs import WalkerJiminyEnv
from gym_jiminy.common.blocks import PDController, MahonyFilter
from gym_jiminy.common.utils import build_pipeline

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


# Default simulation duration (:float [s])
SIMULATION_DURATION = 20.0
# Ratio between the High-level neural network PID target update and Low-level
# PID torque update (:int [NA])
HLC_TO_LLC_RATIO = 1
# Stepper update period (:float [s])
STEP_DT = 0.04

# PID proportional gains (one per actuated joint)
PD_KP = (1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0,
         1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0)
# PID derivative gains (one per actuated joint)
PD_KD = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
         0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

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


class ANYmalJiminyEnv(WalkerJiminyEnv):
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
        data_dir = str(
            files("gym_jiminy.envs") / "data/quadrupedal_robots/anymal")
        urdf_path = os.path.join(data_dir, "anymal.urdf")

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


ANYmalPDControlJiminyEnv = build_pipeline(
    env_config=dict(
        cls=ANYmalJiminyEnv
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
