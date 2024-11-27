"""This module implements to classic toy model environment "ant" using Jiminy
simulator for integrating the rigid-body dynamics.

The agent is a basic 3D quadruped. This dynamics is very basic and not
realistic from a robotic standpoint. Its main advantage is that training a
policy for a given task is extremely rapidly compared to a real quadrupedal
robot such as Anymal. Still, it is complex enough for prototyping learning of
advanced motions or locomotion tasks, as well as model-based observers and
controllers usually intended for robotic applications.
"""
import os
from importlib.resources import files
from typing import Any, Tuple, Sequence

import gymnasium as gym
import numpy as np
import pinocchio as pin

from jiminy_py.core import array_copyto  # pylint: disable=no-name-in-module
from jiminy_py.simulator import Simulator
from gym_jiminy.common.bases import InfoType, EngineObsType
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import sample, get_robot_state_space


# Stepper update period
STEP_DT = 0.05


class AntJiminyEnv(BaseJiminyEnv[np.ndarray, np.ndarray]):
    """The agent is a 3D quadruped consisting of a torso (freeflyer) with 4
    legs attached to it, where each leg has two body parts. The goal is to move
    forward as fast as possible by applying torque to the 8 joints.

    .. seealso::
        See original `ant` environment implementation in Gymnasium:
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant.py
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
        model = simulator.robot.pinocchio_model_th

        # Previous torso position along x-axis in world frame
        self._xpos_prev = 0.0

        # Define base orientation and external forces proxies for fast access
        self._base_rot = np.array([])
        self._f_external: Sequence[np.ndarray] = ()

        # Initialize base class
        super().__init__(
            simulator=simulator,
            debug=debug,
            **{**dict(
                step_dt=STEP_DT),
                **kwargs})

        # Define observation slices proxy for fast access.
        # Note that it is impossible to extract expected observation data from
        # true sensor measurements. Indeed, the total external forces applied
        # on each joint cannot be measured by any type of sensor that Jiminy
        # provides since such sensor does not exit in reality. Because of this
        # limitation, the observation will be based on the current robot state,
        # which is only available once a simulation is running.
        model = self.robot.pinocchio_model_th
        obs_slice_indices = (0, *np.cumsum((
            model.nq - 2, 3, model.nv - 3, *((6,) * (model.njoints - 1)))))
        self._obs_slices: Tuple[np.ndarray, ...] = tuple(
            self.observation[obs_slice_indices[i]:obs_slice_indices[i + 1]]
            for i in range(len(obs_slice_indices) - 1))

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

        state_space = get_robot_state_space(self.robot)
        position_space, velocity_space = state_space["q"], state_space["v"]
        assert isinstance(position_space, gym.spaces.Box)
        assert isinstance(velocity_space, gym.spaces.Box)

        low = np.concatenate([
            np.full_like(position_space.low[2:], -np.inf),
            np.full_like(velocity_space.low, -np.inf),
            np.full((self.robot.pinocchio_model_th.njoints - 1) * 6, -1.0)
        ])
        high = np.concatenate([
            np.full_like(position_space.high[2:], np.inf),
            np.full_like(velocity_space.high, np.inf),
            np.full((self.robot.pinocchio_model_th.njoints - 1) * 6, 1.0)
        ])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float64)

    def _initialize_buffers(self) -> None:
        # Make sure observe update is discrete-time
        if self.observe_dt <= 0.0:
            raise ValueError(
                "This environment does not support time-continuous update.")

        # Initialize previous torso position along x-axis
        self._xpos_prev = self._robot_state_q[0]

        # Initialize the base orientation as a rotation matrix
        self._base_rot = self.robot.pinocchio_data.oMf[1].rotation

        # Initialize vector of external forces
        self._f_external = tuple(
            self.robot_state.f_external[joint_index].vector
            for joint_index in (1, *self.robot.mechanical_joint_indices))

    def refresh_observation(self, measurement: EngineObsType) -> None:
        # Define some proxies for convenience
        q, v = self._robot_state_q, self._robot_state_v

        # Compute the theoretical state of the robot
        q_th = self.robot.get_theoretical_position_from_extended(q)
        v_th = self.robot.get_theoretical_velocity_from_extended(v)

        # Linear velocity of the freeflyer in world frame
        ff_vel_lin_world = self._base_rot @ v_th[:3]

        # Update observation from robot state
        array_copyto(self._obs_slices[0], q_th[2:])
        array_copyto(self._obs_slices[1], ff_vel_lin_world)
        array_copyto(self._obs_slices[2], v_th[3:])
        for i, obs_slice in enumerate(self._obs_slices[3:]):
            array_copyto(obs_slice, self._f_external[i])

        # Clip observation in-place to make sure it is not out of bounds
        assert isinstance(self.observation_space, gym.spaces.Box)
        np.clip(self.observation,
                self.observation_space.low,
                self.observation_space.high,
                out=self.observation)

    def has_terminated(self, info: InfoType) -> Tuple[bool, bool]:
        """Determine whether the episode is over.

        It adds one extra truncation criterion on top of the one defined in the
        base implementation. More precisely, the vertical height of the
        floating base of the robot must not exceed 0.2m.

        :param info: Dictionary of extra information for monitoring.

        :returns: terminated and truncated flags.
        """
        # Call base implementation
        terminated, truncated = super().has_terminated(info)

        # Check if the agent is jumping far too high or stuck on its back
        zpos = self._robot_state_q[2]
        if 1.0 < zpos or zpos < 0.2:
            truncated = True

        return terminated, truncated

    def compute_reward(self, terminated: bool, info: InfoType) -> float:
        """Compute reward at current episode state.

        The reward is defined as the sum of several individual components:
            * forward_reward: Positive reward corresponding to the crow-fly
              velocity of the torso along x-axis, ie the difference between the
              initial and final position of the torso over the current step
              divided by its duration.
            * survive_reward: Constant positive reward equal to 1.0 as long as
              no termination condition has been triggered.
            * ctrl_cost: Negative reward to penalize power consumption, defined
              as the L^2-norm of the action vector weighted by 0.5.
            * contact_cost: Negative reward to penalize jerky, violent motions,
              defined as the aggregated L^2-norm of the external forces applied
              on all bodies weighted by 5e-4.

        The value of each individual reward is added to `info` for monitoring.

        :returns: Aggregated reward.
        """
        # Initialize total reward
        reward = 0.0

        # Compute all reward components
        xpos = self._robot_state_q[0]
        forward_reward = (xpos - self._xpos_prev) / self.step_dt
        survive_reward = 1.0 if not terminated else 0.0
        ctrl_cost = 0.5 * np.dot(self.action, self.action)
        contact_cost = 5e-4 * sum((
            np.dot(f_ext, f_ext) for f_ext in self._f_external))

        # Compute total reward
        reward = forward_reward + survive_reward - ctrl_cost - contact_cost

        # Keep track of all individual review components for monitoring
        info.update({
            'reward_survive': survive_reward,
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
        })

        # Update previous torso forward position buffer
        self._xpos_prev = xpos

        return reward
