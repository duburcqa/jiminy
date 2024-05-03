""" TODO: Write documentation
"""
import unittest

import numpy as np
import gymnasium as gym
import jiminy_py
import pinocchio as pin

from gym_jiminy.common.rewards import OdometryVelocityReward


class Rewards(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_average_odometry_velocity(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        reward_fun = OdometryVelocityReward(
            env, np.array([0.0, 0.0, 0.0]), cutoff=0.3)

        env.reset(seed=0)
        base_pose_prev = env.robot_state.q[:7].copy()
        _, _, terminated, _, info = env.step(env.action)
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
            reward_fun.quantity.requirements['data'].data,
            base_velocity_mean_world)
        base_odom_velocity = base_velocity_mean_world[[0, 1, 5]]
        np.testing.assert_allclose(
            reward_fun.quantity.data, base_odom_velocity)
        gamma = - np.log(0.01) / reward_fun.cutoff ** 2
        reward = np.exp(- gamma * np.sum(base_odom_velocity ** 2))
        np.testing.assert_allclose(
            reward_fun(terminated, info), reward)
