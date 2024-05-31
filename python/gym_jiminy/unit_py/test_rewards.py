""" TODO: Write documentation
"""
import gc
import unittest

import numpy as np

import gymnasium as gym
from jiminy_py.log import extract_trajectory_from_log

from gym_jiminy.common.compositions import (
    TrackingActuatedJointPositionsReward,
    TrackingBaseOdometryVelocityReward,
    TrackingBaseHeightReward,
    TrackingCapturePointReward,
    TrackingFootPositionsReward,
    TrackingFootOrientationsReward,
    SurviveReward,
    MinimizeAngularMomentumReward,
    AdditiveMixtureReward)


class Rewards(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        self.env = gym.make("gym_jiminy.envs:atlas")

        self.env.reset(seed=1)
        action = self.env.action_space.sample()
        for _ in range(10):
            self.env.step(action)
        self.env.stop()

        trajectory = extract_trajectory_from_log(self.env.log_data)
        self.env.quantities.add_trajectory("reference", trajectory)
        self.env.quantities.select_trajectory("reference")

    def test_deletion(self):
        assert len(self.env.quantities.registry) == 0
        reward_survive = TrackingActuatedJointPositionsReward(
            self.env, cutoff=1.0)
        assert len(self.env.quantities.registry) > 0
        del reward_survive
        gc.collect()
        assert len(self.env.quantities.registry) == 0

    def test_tracking(self):
        for reward_class, cutoff in (
                (TrackingBaseOdometryVelocityReward, 20.0),
                (TrackingActuatedJointPositionsReward, 20.0),
                (TrackingBaseHeightReward, 0.1),
                (TrackingCapturePointReward, 0.5),
                (TrackingFootPositionsReward, 1.0),
                (TrackingFootOrientationsReward, 2.0)) * 20:
            reward = reward_class(self.env, cutoff=cutoff)
            quantity_true = reward.quantity.requirements['value_left']
            quantity_ref = reward.quantity.requirements['value_right']

            self.env.reset(seed=0)
            action = self.env.action_space.sample()
            for _ in range(5):
                self.env.step(action)
            _, _, terminated, _, _ = self.env.step(self.env.action)

            with np.testing.assert_raises(AssertionError):
                np.testing.assert_allclose(
                    quantity_true.get(), quantity_ref.get())

            if isinstance(reward, TrackingBaseHeightReward):
                np.testing.assert_allclose(
                    quantity_true.get(), self.env.robot_state.q[2])

            gamma = - np.log(0.01) / cutoff ** 2
            value = np.exp(- gamma * np.sum((reward.quantity.op(
                quantity_true.get(), quantity_ref.get())) ** 2))
            np.testing.assert_allclose(reward(terminated, {}), value)

            del reward

    def test_mixture(self):
        reward_odometry = TrackingBaseOdometryVelocityReward(
            self.env, cutoff=0.3)
        reward_survive = SurviveReward(self.env)
        reward_sum = AdditiveMixtureReward(
            self.env,
            "reward_total",
            components=(reward_odometry, reward_survive),
            weights=(0.5, 0.2))
        reward_sum_normalized = AdditiveMixtureReward(
            self.env,
            "reward_total",
            components=(reward_odometry, reward_survive),
            weights=(0.7, 0.3))

        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        _, _, terminated, _, _ = self.env.step(action)

        assert len(reward_sum_normalized.components) == 2
        assert reward_sum_normalized.is_terminal == False
        assert reward_sum_normalized.is_normalized
        assert not reward_sum.is_normalized
        assert reward_sum(terminated, {}) == (
            0.5 * reward_odometry(terminated, {}) +
            0.2 * reward_survive(terminated, {}))
