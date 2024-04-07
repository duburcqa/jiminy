""" TODO: Write documentation
"""
import unittest

import numpy as np
import gymnasium as gym
import jiminy_py
import pinocchio as pin

from gym_jiminy.common.bases import QuantityManager
from gym_jiminy.common.quantities import (
    EulerAnglesFrame, CenterOfMass, ZeroMomentPoint)


class Quantities(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_shared_cache(self):
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset()

        quantity_manager = QuantityManager(
            env.simulator,
            {
                "acom": (CenterOfMass, dict(kinematic_level=pin.ACCELERATION)),
                "com": (CenterOfMass, {}),
                "zmp": (ZeroMomentPoint, {}),
            })
        quantities = quantity_manager.quantities

        assert len(quantity_manager._caches)
        assert len(quantity_manager._quantities_all) == 4

        zmp_0 = quantity_manager.zmp.copy()
        assert quantities["com"]._cache._has_value
        assert not quantities["acom"]._cache._has_value
        assert not quantities["com"]._is_initialized
        assert quantities["zmp"].requirements["com"]._is_initialized

        env.step(env.action)
        zmp_1 = quantity_manager["zmp"].copy()
        assert np.all(zmp_0 == zmp_1)
        quantity_manager.clear()
        assert quantities["zmp"].requirements["com"]._is_initialized
        assert not quantities["com"]._cache._has_value
        zmp_1 = quantity_manager.zmp.copy()
        assert np.any(zmp_0 != zmp_1)

        env.step(env.action)
        quantity_manager.reset()
        assert not quantities["zmp"].requirements["com"]._is_initialized
        zmp_2 = quantity_manager.zmp.copy()
        assert np.any(zmp_1 != zmp_2)

    def test_dynamic_batching(self):
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset()
        env.step(env.action)

        quantity_manager = QuantityManager(
            env.simulator,
            {
                "rpy_0": (EulerAnglesFrame, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                "rpy_1": (EulerAnglesFrame, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                "rpy_2": (EulerAnglesFrame, dict(
                    frame_name=env.robot.pinocchio_model.frames[-1].name)),
            })
        quantities = quantity_manager.quantities

        rpy_0 = quantity_manager.rpy_0.copy()
        assert len(quantities['rpy_0'].requirements['data'].frame_names) == 1
        assert np.all(rpy_0 == quantity_manager.rpy_1)
        rpy_2 = quantity_manager.rpy_2.copy()
        assert np.any(rpy_0 != rpy_2)
        assert len(quantities['rpy_2'].requirements['data'].frame_names) == 2

        env.step(env.action)
        quantity_manager.reset()
        rpy_0_next = quantity_manager.rpy_0
        assert np.any(rpy_0 != rpy_0_next)
        assert len(quantities['rpy_2'].requirements['data'].frame_names) == 2

        quantity_manager.reset(reset_tracking=True)
        assert np.all(rpy_0_next == quantity_manager.rpy_0)
        assert len(quantities['rpy_0'].requirements['data'].frame_names) == 1
