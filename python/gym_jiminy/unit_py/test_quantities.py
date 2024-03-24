""" TODO: Write documentation
"""
import unittest

import numpy as np
import gymnasium as gym
import jiminy_py
import pinocchio as pin

from gym_jiminy.common.bases import QuantityManager
from gym_jiminy.common.quantities import CenterOfMass, ZeroMomentPoint


class Quantities(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_quantity_manager(self):
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset()

        quantity_manager = QuantityManager(
            env.simulator,
            {
                "acom": (CenterOfMass, dict(
                    kinematic_level=pin.KinematicLevel.ACCELERATION)),
                "com": (CenterOfMass, {}),
                "zmp": (ZeroMomentPoint, {}),
            })
        quantities = quantity_manager.quantities

        assert len(quantity_manager._caches)
        assert len(quantity_manager._quantities_all) == 4

        zmp_0 = quantity_manager.zmp.copy()
        assert quantities["com"]._cache.has_value()
        assert not quantities["acom"]._cache.has_value()
        assert not quantities["com"]._is_initialized
        assert quantities["zmp"].requirements["com"]._is_initialized

        env.step(env.action)
        zmp_1 = quantity_manager["zmp"].copy()
        assert np.all(zmp_0 == zmp_1)
        quantity_manager.clear()
        assert quantities["zmp"].requirements["com"]._is_initialized
        assert not quantities["com"]._cache.has_value()
        zmp_1 = quantity_manager.zmp.copy()
        assert np.any(zmp_0 != zmp_1)

        env.step(env.action)
        quantity_manager.reset()
        assert not quantities["zmp"].requirements["com"]._is_initialized
        zmp_2 = quantity_manager.zmp.copy()
        assert np.any(zmp_1 != zmp_2)
