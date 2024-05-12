""" TODO: Write documentation
"""
import unittest

import numpy as np

import gymnasium as gym
import jiminy_py
import pinocchio as pin
from jiminy_py.log import extract_trajectory_from_log

from gym_jiminy.common.compositions import (
    OdometryVelocityReward, SurviveReward, AdditiveMixtureReward)


class Rewards(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_average_odometry_velocity(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)
        env.stop()
        trajectory = extract_trajectory_from_log(env.log_data)
        env.quantities.add_trajectory("reference", trajectory)
        env.quantities.select_trajectory("reference")

        reward = OdometryVelocityReward(env, cutoff=0.3)
        quantity_odom_vel_true = reward.quantity.requirements['value_left']
        quantity_odom_vel_ref = reward.quantity.requirements['value_left']

        env.reset(seed=0)
        base_pose_prev = env.robot_state.q[:7].copy()
        _, _, terminated, _, _ = env.step(env.action)
        base_pose = env.robot_state.q[:7].copy()

        se3 = pin.liegroups.SE3()
        base_pose_diff = pin.LieGroup.difference(
            se3, base_pose_prev, base_pose)
        base_velocity_mean_local =  base_pose_diff / env.step_dt
        base_pose_mean = pin.LieGroup.integrate(
            se3, base_pose_prev, 0.5 * base_pose_diff)
        rot_mat = pin.Quaternion(base_pose_mean[-4:]).matrix()
        base_velocity_mean_world = np.concatenate((
            rot_mat @ base_velocity_mean_local[:3],
            rot_mat @ base_velocity_mean_local[3:]))
        np.testing.assert_allclose(
            quantity_odom_vel_true.requirements['data'].data,
            base_velocity_mean_world)
        base_odom_velocity = base_velocity_mean_world[[0, 1, 5]]
        np.testing.assert_allclose(
            quantity_odom_vel_true.data, base_odom_velocity)
        gamma = - np.log(0.01) / reward.cutoff ** 2
        value = np.exp(- gamma * np.sum((
            quantity_odom_vel_true.data - quantity_odom_vel_ref.data) ** 2))
        np.testing.assert_allclose(reward(terminated, {}), value)

    def test_mixture(self):
        env = gym.make("gym_jiminy.envs:atlas")

        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)
        env.stop()
        trajectory = extract_trajectory_from_log(env.log_data)
        env.quantities.add_trajectory("reference", trajectory)
        env.quantities.select_trajectory("reference")

        reward_odometry = OdometryVelocityReward(env, cutoff=0.3)
        reward_survive = SurviveReward(env)
        reward_sum = AdditiveMixtureReward(
            env,
            "reward_total",
            components=(reward_odometry, reward_survive),
            weights=(0.5, 0.2))
        reward_sum_normalized = AdditiveMixtureReward(
            env,
            "reward_total",
            components=(reward_odometry, reward_survive),
            weights=(0.7, 0.3))

        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(env.action)

        assert len(reward_sum_normalized.components) == 2
        assert reward_sum_normalized.is_terminal == False
        assert reward_sum_normalized.is_normalized
        assert not reward_sum.is_normalized
        assert reward_sum(terminated, {}) == (
            0.5 * reward_odometry(terminated, {}) +
            0.2 * reward_survive(terminated, {}))
