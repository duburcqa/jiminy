
""" TODO: Write documentation.
"""

import os
from pkg_resources import resource_filename
from typing import Tuple, Dict, Any

import gym
import numpy as np
from pinocchio import Quaternion, neutral, normalize, FrameType

from jiminy_py.simulator import Simulator
from jiminy_py.dynamics import update_quantities
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample


# Stepper update period
STEP_DT = 0.05


class AntEnv(BaseJiminyEnv):
    """ TODO: Write documentation.
    """

    def __init__(self, debug: bool = False, **kwargs) -> None:
        """ TODO: Write documentation.
        """
        # Get the urdf and mesh paths
        data_root_dir = resource_filename(
            "gym_jiminy.envs", "data/toys_models/ant")
        urdf_path = os.path.join(data_root_dir, "ant.urdf")
        hardware_path = os.path.join(data_root_dir, "ant_hardware.toml")
        config_path = os.path.join(data_root_dir, "ant_options.toml")

        # Configure the backend simulator
        simulator = Simulator.build(
            urdf_path, hardware_path, data_root_dir,
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
        self.obs_chunks = []
        self.obs_chunks_sizes = []

        # Previous torso position along x-axis in world frame
        self.xpos_prev = 0.0

        # Initialize base class
        super().__init__(**{**dict(
            simulator=simulator,
            step_dt=STEP_DT,
            enforce_bounded_spaces=False,
            debug=debug), **kwargs})

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
        qpos += sample(scale=0.1, shape=(self.robot.nq,), rg=self.rg)
        qpos = normalize(self.robot.pinocchio_model, qpos)

        # Make sure it does not go through the ground
        update_quantities(self.robot, qpos, use_theoretical_model=False)
        dist_rlt = self.robot.collision_data.distanceResults
        qpos[2] -= min(0.0, *[dist_req.min_distance for dist_req in dist_rlt])

        # Zero mean normally distributed initial velocity
        qvel = sample(
            dist='normal', scale=0.1, shape=(self.robot.nv,), rg=self.rg)

        return qpos, qvel

    def _refresh_observation_space(self) -> None:
        """ TODO: Write documentation.

        The observation space comprises:
            - robot configuration vector (absolute position (x, y) excluded),
            - robot velocity vector,
            - flatten external forces applied on each bodies in world frame
              at robot center of mass.
        """
        # http://www.mujoco.org/book/APIreference.html#mjData

        state_space = self._get_state_space()

        low = np.concatenate([
            np.full_like(state_space['Q'].low[2:], -np.infty),
            np.full_like(state_space['V'].low, -np.infty),
            np.full(len(self.bodies_idx) * 6, -1.0)
        ])
        high = np.concatenate([
            np.full_like(state_space['Q'].high[2:], np.infty),
            np.full_like(state_space['V'].high, np.infty),
            np.full(len(self.bodies_idx) * 6, 1.0)
        ])

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float64)

    def refresh_observation(self) -> None:
        if not self.simulator.is_simulation_running:
            # Initialize observation chunks
            self.obs_chunks = [
                self.system_state.q[2:],
                self.system_state.v,
                *[f.vector for f in self.system_state.f_external]
            ]

            # Initialize observation chunks sizes
            self.obs_chunks_sizes = []
            idx_start = 0
            for obs in self.obs_chunks:
                idx_end = idx_start + len(obs)
                self.obs_chunks_sizes.append([idx_start, idx_end])
                idx_start = idx_end

            # Initialize previous torso position
            self.xpos_prev = self.system_state.q[0]

        # Update observation buffer
        for obs, size in zip(self.obs_chunks, self.obs_chunks_sizes):
            obs_idx = slice(*size)
            low = self.observation_space.low[obs_idx]
            high = self.observation_space.high[obs_idx]
            self._observation[obs_idx] = np.clip(obs, low, high)

        # Transform observed linear velocity to be in world frame
        self._observation[slice(*self.obs_chunks_sizes[1])][:3] = \
            Quaternion(self.system_state.q[3:7]) * self.obs_chunks[1][:3]

    def is_done(self) -> bool:
        """ TODO: Write documentation.
        """
        zpos = self.system_state.q[2]
        not_done = zpos >= 0.2 and zpos <= 1.0
        return not not_done

    def compute_reward(self,  # type: ignore[override]
                       *, info: Dict[str, Any]) -> float:
        """ TODO: Write documentation.
        """
        # pylint: disable=arguments-differ

        # Initialize total reward
        reward = 0.0

        # Compute forward velocity reward
        xpos = self.system_state.q[0]
        forward_reward = (xpos - self.xpos_prev) / self.step_dt

        ctrl_cost = 0.5 * np.square(self._action).sum()

        f_ext_idx = slice(self.obs_chunks_sizes[2][0],
                          self.obs_chunks_sizes[-1][1])
        f_ext = self._observation[f_ext_idx]
        contact_cost = 0.5 * 1e-3 * np.square(f_ext).sum()

        survive_reward = 1.0

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
