""" TODO: Write documentation
"""
import gc
import unittest

import numpy as np

import gymnasium as gym
from jiminy_py.log import extract_trajectory_from_log

from gym_jiminy.common.compositions import (
    CUTOFF_ESP,
    TrackingActuatedJointPositionsReward,
    TrackingBaseOdometryVelocityReward,
    TrackingBaseHeightReward,
    TrackingCapturePointReward,
    TrackingFootPositionsReward,
    TrackingFootOrientationsReward,
    TrackingFootForceDistributionReward,
    MinimizeFrictionReward,
    SurviveReward,
    AdditiveMixtureReward)
from gym_jiminy.toolbox.compositions import (
    tanh_normalization,
    MaximizeStability)


class Rewards(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        self.env = gym.make("gym_jiminy.envs:atlas-pid", debug=False)

        self.env.eval()
        self.env.reset(seed=1)
        action = 0.5 * self.env.action_space.sample()
        for _ in range(10):
            self.env.step(action)
        self.env.stop()
        trajectory = extract_trajectory_from_log(self.env.log_data)
        self.env.train()

        self.env.quantities.add_trajectory("reference", trajectory)
        self.env.quantities.select_trajectory("reference")

    def test_deletion(self):
        """ TODO: Write documentation
        """
        assert len(self.env.quantities.registry) == 0
        reward_survive = TrackingActuatedJointPositionsReward(
            self.env, cutoff=1.0)
        assert len(self.env.quantities.registry) > 0
        del reward_survive
        gc.collect()
        assert len(self.env.quantities.registry) == 0

    def test_tracking(self):
        """ TODO: Write documentation
        """
        for reward_class, cutoff in (
                (TrackingBaseOdometryVelocityReward, 20.0),
                (TrackingActuatedJointPositionsReward, 10.0),
                (TrackingBaseHeightReward, 0.1),
                (TrackingCapturePointReward, 0.2),
                (TrackingFootPositionsReward, 1.0),
                (TrackingFootOrientationsReward, 2.0),
                (TrackingFootForceDistributionReward, 2.0)):
            reward = reward_class(self.env, cutoff=cutoff)
            quantity = reward.quantity

            self.env.reset(seed=0)
            action = 0.5 * self.env.action_space.sample()
            for _ in range(5):
                self.env.step(action)
            _, _, terminated, _, _ = self.env.step(self.env.action)

            with np.testing.assert_raises(AssertionError):
                np.testing.assert_allclose(
                    quantity.value_left, quantity.value_right)

            if isinstance(reward, TrackingBaseHeightReward):
                np.testing.assert_allclose(
                    quantity.value_left, self.env.robot_state.q[2])

            gamma = - np.log(CUTOFF_ESP) / cutoff ** 2
            value = np.exp(- gamma * np.sum((quantity.op(
                quantity.value_left, quantity.value_right)) ** 2))
            assert value > 0.01
            np.testing.assert_allclose(reward(terminated, {}), value)

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

    def test_stability(self):
        """ TODO: Write documentation
        """
        CUTOFF_INNER, CUTOFF_OUTER = 0.1, 0.5
        reward_stability = MaximizeStability(
            self.env, cutoff_inner=0.1, cutoff_outer=0.5)
        quantity = reward_stability.quantity

        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        _, _, terminated, _, _ = self.env.step(action)

        dist = quantity.support_polygon.get_distance_to_point(quantity.zmp)
        value = tanh_normalization(dist.item(), -CUTOFF_INNER, CUTOFF_OUTER)
        np.testing.assert_allclose(tanh_normalization(
            -CUTOFF_INNER, -CUTOFF_INNER, CUTOFF_OUTER), 1.0 - CUTOFF_ESP)
        np.testing.assert_allclose(tanh_normalization(
            CUTOFF_OUTER, -CUTOFF_INNER, CUTOFF_OUTER), CUTOFF_ESP)
        np.testing.assert_allclose(reward_stability(terminated, {}), value)

    def test_friction(self):
        """ TODO: Write documentation
        """
        CUTOFF = 0.5
        reward_friction = MinimizeFrictionReward(self.env, cutoff=CUTOFF)
        quantity = reward_friction.quantity

        self.env.reset(seed=0)
        _, _, terminated, _, _ = self.env.step(self.env.action)
        force_tangential_rel = quantity.get()
        force_tangential_rel_norm = np.sum(np.square(force_tangential_rel))

        gamma = - np.log(CUTOFF_ESP) / CUTOFF ** 2
        value = np.exp(- gamma * force_tangential_rel_norm)
        assert value > 0.01
        np.testing.assert_allclose(reward_friction(terminated, {}), value)
