
""" TODO: Write documentation.
"""

import os
import sys
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
from pinocchio import (Quaternion,
                       FrameType,
                       neutral,
                       normalize,
                       framesForwardKinematics)

from jiminy_py.simulator import Simulator
from gym_jiminy.common.bases import InfoType, EngineObsType
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


# Stepper update period
STEP_DT = 0.05


class AntEnv(BaseJiminyEnv[np.ndarray, np.ndarray]):
    """ TODO: Write documentation.
    """

    def __init__(self, debug: bool = False, **kwargs: Any) -> None:
        """
        :param debug: Whether the debug mode must be enabled.
                      See `BaseJiminyEnv` constructor for details.
        :param kwargs: Keyword arguments to forward to `Simulator` and
                       `BaseJiminyEnv` constructors.
        """
        # Get the urdf and mesh paths
        data_dir = str(files("gym_jiminy.envs") / "data/toys_models/ant")
        urdf_path = os.path.join(data_dir, "ant.urdf")
        hardware_path = os.path.join(data_dir, "ant_hardware.toml")
        config_path = os.path.join(data_dir, "ant_options.toml")

        # Configure the backend simulator
        simulator = Simulator.build(
            urdf_path, hardware_path, data_dir,
            has_freeflyer=True, use_theoretical_model=False,
            config_path=config_path, debug=debug, **kwargs)

        # Get the list of independant bodies (not connected via fixed joint)
        self.bodies_idx = [0]  # World is part of bodies list
        for i, frame in enumerate(simulator.robot.pinocchio_model.frames):
            if frame.type == FrameType.BODY:
                frame_prev = simulator.robot.pinocchio_model.frames[
                    frame.previousFrame]
                if frame_prev.type != FrameType.FIXED_JOINT:
                    self.bodies_idx.append(i)

        # Observation chunks proxy for fast access
        self.obs_chunks: List[np.ndarray] = []
        self.obs_chunks_sizes: List[Tuple[int, int]] = []

        # Previous torso position along x-axis in world frame
        self.xpos_prev = 0.0

        # Initialize base class
        super().__init__(
            simulator=simulator,
            debug=debug,
            **{**dict(
                step_dt=STEP_DT,
                enforce_bounded_spaces=False),
                **kwargs})

    def _neutral(self) -> np.ndarray:
        """ TODO: Write documentation.
        """
        def joint_position_idx(joint_name: str) -> int:
            joint_idx = self.robot.pinocchio_model.getJointId(joint_name)
            return self.robot.pinocchio_model.joints[joint_idx].idx_q

        qpos = neutral(self.robot.pinocchio_model)
        qpos[2] = 0.75
        qpos[joint_position_idx('ankle_1')] = 1.0
        qpos[joint_position_idx('ankle_2')] = -1.0
        qpos[joint_position_idx('ankle_3')] = -1.0
        qpos[joint_position_idx('ankle_4')] = 1.0

        return qpos

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """ TODO: Write documentation.
        """
        # Add noise on top of neutral configuration
        qpos = self._neutral()
        qpos += sample(scale=0.1, shape=(self.robot.nq,), rg=self.np_random)
        qpos = normalize(self.robot.pinocchio_model, qpos)

        # Make sure it does not go through the ground
        framesForwardKinematics(
            self.robot.pinocchio_model, self.robot.pinocchio_data, qpos)
        dist_rlt = self.robot.collision_data.distanceResults
        qpos[2] -= min(0.0, *[dist_req.min_distance for dist_req in dist_rlt])

        # Zero mean normally distributed initial velocity
        qvel = sample(
            dist='normal', scale=0.1, shape=(self.robot.nv,),
            rg=self.np_random)

        return qpos, qvel

    def _initialize_observation_space(self) -> None:
        """ TODO: Write documentation.

        The observation space comprises:

            - robot configuration vector (absolute position (x, y) excluded),
            - robot velocity vector,
            - flatten external forces applied on each bodies in world frame
              at robot center of mass.
        """
        # http://www.mujoco.org/book/APIreference.html#mjData

        position_space, velocity_space = self._get_agent_state_space().values()
        assert isinstance(position_space, gym.spaces.Box)
        assert isinstance(velocity_space, gym.spaces.Box)

        low = np.concatenate([
            np.full_like(position_space.low[2:], -np.inf),
            np.full_like(velocity_space.low, -np.inf),
            np.full(len(self.bodies_idx) * 6, -1.0)
        ])
        high = np.concatenate([
            np.full_like(position_space.high[2:], np.inf),
            np.full_like(velocity_space.high, np.inf),
            np.full(len(self.bodies_idx) * 6, 1.0)
        ])

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float64)

    def refresh_observation(self, measurement: EngineObsType) -> None:
        # TODO: Do not rely on anything else than `measurement` to compute the
        # observation, as anything else is not reliable.

        if not self.is_simulation_running:
            # Initialize observation chunks
            self.obs_chunks = [
                self._system_state_q[2:],
                self._system_state_v,
                *[f.vector for f in self.system_state.f_external]
            ]

            # Initialize observation chunks sizes
            self.obs_chunks_sizes = []
            idx_start = 0
            for obs in self.obs_chunks:
                idx_end = idx_start + len(obs)
                self.obs_chunks_sizes.append((idx_start, idx_end))
                idx_start = idx_end

            # Initialize previous torso position
            self.xpos_prev = self._system_state_q[0]

        # Update observation buffer
        assert isinstance(self.observation_space, gym.spaces.Box)
        for obs, size in zip(self.obs_chunks, self.obs_chunks_sizes):
            obs_idx = slice(*size)
            low = self.observation_space.low[obs_idx]
            high = self.observation_space.high[obs_idx]
            obs.clip(low, high, out=self.observation[obs_idx])

        # Transform observed linear velocity to be in world frame
        self.observation[slice(*self.obs_chunks_sizes[1])][:3] = \
            Quaternion(self._system_state_q[3:7]) * self.obs_chunks[1][:3]

    def has_terminated(self) -> Tuple[bool, bool]:
        """ TODO: Write documentation.
        """
        # Call base implementation
        terminated, truncated = super().has_terminated()

        # Check if the agent is jumping far too high or stuck on its back
        zpos = self._system_state_q[2]
        if 1.0 < zpos or zpos < 0.2:
            truncated = True

        return terminated, truncated

    def compute_reward(self,
                       terminated: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        """ TODO: Write documentation.
        """
        # Initialize total reward
        reward = 0.0

        # Compute forward velocity reward
        xpos = self._system_state_q[0]
        forward_reward = (xpos - self.xpos_prev) / self.step_dt

        ctrl_cost = 0.5 * np.square(self.action).sum()

        f_ext_idx = slice(self.obs_chunks_sizes[2][0],
                          self.obs_chunks_sizes[-1][1])
        f_ext = self.observation[f_ext_idx]
        contact_cost = 0.5 * 1e-3 * np.square(f_ext).sum()

        survive_reward = 1.0 if not terminated else 0.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        info.update({
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': survive_reward
        })

        # Update previous torso forward position buffer
        self.xpos_prev = xpos

        return reward
