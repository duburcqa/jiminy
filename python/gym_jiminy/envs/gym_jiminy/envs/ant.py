
""" TODO: Write documentation.
"""

import os
import sys
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import pinocchio as pin

from jiminy_py.simulator import Simulator
from gym_jiminy.common.bases import InfoType, EngineObsType
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample, copyto, squared_norm_2

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


# Stepper update period
STEP_DT = 0.05


class AntJiminyEnv(BaseJiminyEnv[np.ndarray, np.ndarray]):
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
            has_freeflyer=True, config_path=config_path, debug=debug, **kwargs)

        # Get the list of independent bodies (not connected via fixed joint)
        self.body_indices = [0]  # World is part of bodies list
        for i, frame in enumerate(simulator.robot.pinocchio_model.frames):
            if frame.type == pin.FrameType.BODY:
                frame_prev = simulator.robot.pinocchio_model.frames[
                    frame.previousFrame]
                if frame_prev.type != pin.FrameType.FIXED_JOINT:
                    self.body_indices.append(i)

        # Previous torso position along x-axis in world frame
        self._xpos_prev = 0.0

        # Define observation slices proxy for fast access.
        # Note that they will be initialized in `_initialize_buffers`.
        self._obs_slices: Tuple[np.ndarray, ...] = ()

        # Define base orientation and external forces proxies for fast access
        self._base_rot = np.array([])
        self._f_external: Tuple[np.ndarray, ...] = ()

        # Theoretical state of the robot
        self._q_th = np.array([])
        self._v_th = np.array([])

        # Initialize base class
        super().__init__(
            simulator=simulator,
            debug=debug,
            **{**dict(
                step_dt=STEP_DT,
                enforce_bounded_spaces=False),
                **kwargs})

    def _neutral(self) -> np.ndarray:
        """Returns a neutral valid configuration for the agent.

        This configuration is statically stable on flat ground. The four legs
        are all in contact with the ground in the same configuration.
        """
        def joint_position_index(joint_name: str) -> int:
            joint_index = self.robot.pinocchio_model.getJointId(joint_name)
            return self.robot.pinocchio_model.joints[joint_index].idx_q

        qpos = pin.neutral(self.robot.pinocchio_model)
        qpos[2] = 0.75
        qpos[joint_position_index('ankle_1')] = 1.0
        qpos[joint_position_index('ankle_2')] = -1.0
        qpos[joint_position_index('ankle_3')] = -1.0
        qpos[joint_position_index('ankle_4')] = 1.0

        return qpos

    def _sample_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a randomized yet valid configuration and velocity for the
        robot.

        Randomness is added on top of the neutral configuration of the robot,
        including the floating base and all revolute joints. On top of that,
        the initial vertical height of the robot is randomized separately. The
        initial velocity completely random without any particular pattern.
        """
        # Add noise on top of neutral configuration
        qpos = self._neutral()
        qpos += sample(scale=0.1, shape=(self.robot.nq,), rg=self.np_random)
        qpos = pin.normalize(self.robot.pinocchio_model, qpos)

        # Make sure it does not go through the ground
        pin.framesForwardKinematics(
            self.robot.pinocchio_model, self.robot.pinocchio_data, qpos)
        dist_rlt = self.robot.collision_data.distanceResults
        qpos[2] -= min(0.0, *[dist_req.min_distance for dist_req in dist_rlt])

        # Zero mean normally distributed initial velocity
        qvel = sample(dist='normal',
                      scale=0.1,
                      shape=(self.robot.nv,),
                      rg=self.np_random)

        return qpos, qvel

    def _initialize_observation_space(self) -> None:
        """Configure the observation of the environment.

        The observation space comprises:

            * theoretical configuration (absolute position (x, y) excluded),
            * theoretical velocity (with base linear velocity in world frame),
            * flattened external forces applied on each body in local frame,
              ie centered at their respective center of mass.
        """
        # http://www.mujoco.org/book/APIreference.html#mjData

        position_space, velocity_space = self._get_agent_state_space().values()
        assert isinstance(position_space, gym.spaces.Box)
        assert isinstance(velocity_space, gym.spaces.Box)

        low = np.concatenate([
            np.full_like(position_space.low[2:], -np.inf),
            np.full_like(velocity_space.low, -np.inf),
            np.full(len(self.body_indices) * 6, -1.0)
        ])
        high = np.concatenate([
            np.full_like(position_space.high[2:], np.inf),
            np.full_like(velocity_space.high, np.inf),
            np.full(len(self.body_indices) * 6, 1.0)
        ])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float64)

    def _initialize_buffers(self) -> None:
        # Extract observation from the robot state.
        # Note that this is only reliable with using a fixed step integrator.
        engine_options = self.simulator.get_options()
        if engine_options['stepper']['odeSolver'] in ('runge_kutta_dopri5',):
            raise ValueError(
                "This environment does not support adaptive step integrators. "
                "Please use either 'euler_explicit' or 'runge_kutta_4'.")

        # Initialize the base orientation as a rotation matrix
        self._base_rot = self.robot.pinocchio_data.oMf[1].rotation

        # Initialize vector of external forces
        self._f_external = tuple(
            self.robot_state.f_external[joint_index].vector
            for joint_index in self.robot.mechanical_joint_indices)

        # Refresh buffers manually to initialize them early
        self._refresh_buffers()

        # Re-initialize observation slices.
        # Note that the base linear velocity is isolated as it will be computed
        # on the fly and therefore updated separately.
        obs_slices = []
        obs_index_first = 0
        for data in (
                self._q_th[2:],
                self._v_th[:3],
                self._v_th[3:],
                *self._f_external):
            obs_index_last = obs_index_first + len(data)
            obs_slices.append(self.observation[obs_index_first:obs_index_last])
            obs_index_first = obs_index_last
        self._obs_slices = (obs_slices[0], *obs_slices[2:], obs_slices[1])

        # Initialize previous torso position along x-axis
        self._xpos_prev = self._robot_state_q[0]

    def _refresh_buffers(self) -> None:
        self._q_th = self.robot.get_theoretical_position_from_extended(
            self._robot_state_q)
        self._v_th = self.robot.get_theoretical_velocity_from_extended(
            self._robot_state_v)

    def refresh_observation(self, measurement: EngineObsType) -> None:
        # Update observation
        copyto(self._obs_slices[:-1], (
            self._q_th[2:], self._v_th[3:], *self._f_external))

        # Transform observed linear velocity to be in world frame
        self._obs_slices[-1][:] = self._base_rot @ self._robot_state_v[:3]

        # Clip observation in-place to make sure it is not out of bounds
        assert isinstance(self.observation_space, gym.spaces.Box)
        np.clip(self.observation,
                self.observation_space.low,
                self.observation_space.high,
                out=self.observation)

    def has_terminated(self) -> Tuple[bool, bool]:
        """Determine whether the episode is over.

        It adds one extra truncation criterion on top of the one defined in the
        base implementation. More precisely, the vertical height of the
        floating base of the robot must not exceed 0.2m.
        """
        # Call base implementation
        terminated, truncated = super().has_terminated()

        # Check if the agent is jumping far too high or stuck on its back
        zpos = self._robot_state_q[2]
        if 1.0 < zpos or zpos < 0.2:
            truncated = True

        return terminated, truncated

    def compute_reward(self,
                       terminated: bool,
                       truncated: bool,
                       info: InfoType) -> float:
        """Compute the reward related to a specific control block.

        # TODO: Write documentation.
        """
        # Initialize total reward
        reward = 0.0

        # Compute forward velocity reward
        xpos = self._robot_state_q[0]
        forward_reward = (xpos - self._xpos_prev) / self.step_dt

        ctrl_cost = 0.5 * np.square(self.action).sum()

        contact_cost = 0.5 * 1e-3 * sum(map(squared_norm_2, self._f_external))

        survive_reward = 1.0 if not terminated else 0.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        info.update({
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': survive_reward
        })

        # Update previous torso forward position buffer
        self._xpos_prev = xpos

        return reward
