""" TODO: Write documentation
"""
import gc
import unittest

import numpy as np
import gymnasium as gym
import jiminy_py
import pinocchio as pin

from gym_jiminy.common.quantities import (
    QuantityManager, FrameEulerAngles, FrameXYZQuat,
    AverageFrameSpatialVelocity, CenterOfMass, ZeroMomentPoint)


class Quantities(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_shared_cache(self):
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset()

        quantity_manager = QuantityManager(env)
        for name, cls, kwargs in (
                ("acom", CenterOfMass, dict(kinematic_level=pin.ACCELERATION)),
                ("com", CenterOfMass, {}),
                ("zmp", ZeroMomentPoint, {})):
            quantity_manager[name] = (cls, kwargs)
        quantities = quantity_manager.registry

        assert len(quantity_manager) == 3
        assert len(quantities["zmp"].cache.owners) == 1
        assert len(quantities["com"].cache.owners) == 2

        zmp_0 = quantity_manager.zmp.copy()
        assert quantities["com"].cache.has_value
        assert not quantities["acom"].cache.has_value
        assert not quantities["com"]._is_initialized
        assert quantities["zmp"].requirements["com"]._is_initialized

        env.step(env.action)
        zmp_1 = quantity_manager["zmp"].copy()
        assert np.all(zmp_0 == zmp_1)
        quantity_manager.clear()
        assert quantities["zmp"].requirements["com"]._is_initialized
        assert not quantities["com"].cache.has_value
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

        quantity_manager = QuantityManager(env)
        for name, cls, kwargs in (
                ("xyzquat_0", FrameXYZQuat, dict(
                    frame_name=env.robot.pinocchio_model.frames[2].name)),
                ("rpy_0", FrameEulerAngles, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                ("rpy_1", FrameEulerAngles, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                ("rpy_2", FrameEulerAngles, dict(
                    frame_name=env.robot.pinocchio_model.frames[-1].name))):
            quantity_manager[name] = (cls, kwargs)
        quantities = quantity_manager.registry

        xyzquat_0 =  quantity_manager.xyzquat_0.copy()
        rpy_0 = quantity_manager.rpy_0.copy()
        assert len(quantities['rpy_0'].requirements['data'].frame_names) == 1
        assert np.all(rpy_0 == quantity_manager.rpy_1)
        rpy_2 = quantity_manager.rpy_2.copy()
        assert np.any(rpy_0 != rpy_2)
        assert len(quantities['rpy_2'].requirements['data'].frame_names) == 2

        env.step(env.action)
        quantity_manager.reset()
        rpy_0_next = quantity_manager.rpy_0
        xyzquat_0_next =  quantity_manager.xyzquat_0.copy()
        assert np.any(rpy_0 != rpy_0_next)
        assert np.any(xyzquat_0 != xyzquat_0_next)
        assert len(quantities['rpy_2'].requirements['data'].frame_names) == 2

        assert len(quantities['rpy_1'].requirements['data'].cache.owners) == 3
        del quantity_manager['rpy_2']
        gc.collect()
        assert len(quantities['rpy_1'].requirements['data'].cache.owners) == 2
        quantity_manager.rpy_1
        assert len(quantities['rpy_1'].requirements['data'].frame_names) == 1

        quantity_manager.reset(reset_tracking=True)
        assert np.all(rpy_0_next == quantity_manager.rpy_0)
        assert np.all(xyzquat_0_next == quantity_manager.xyzquat_0)
        assert len(quantities['rpy_0'].requirements['data'].frame_names) == 1

    def test_discard(self):
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset()

        quantity_manager = QuantityManager(env)
        for name, cls, kwargs in (
                ("rpy_0", FrameEulerAngles, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                ("rpy_1", FrameEulerAngles, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                ("rpy_2", FrameEulerAngles, dict(
                    frame_name=env.robot.pinocchio_model.frames[-1].name))):
            quantity_manager[name] = (cls, kwargs)
        quantities = quantity_manager.registry

        assert len(quantities['rpy_1'].cache.owners) == 2
        assert len(quantities['rpy_2'].requirements['data'].cache.owners) == 3

        del quantity_manager['rpy_0']
        gc.collect()
        assert len(quantities['rpy_1'].cache.owners) == 1
        assert len(quantities['rpy_2'].requirements['data'].cache.owners) == 2

        del quantity_manager['rpy_1']
        gc.collect()
        assert len(quantities['rpy_2'].requirements['data'].cache.owners) == 1

        del quantity_manager['rpy_2']
        gc.collect()
        for cache in quantity_manager._caches.values():
            assert len(cache.owners) == 0

    def test_env(self):
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["com"] = (CenterOfMass, {})

        env.reset(seed=0)
        com_0 = env.quantities["com"].copy()
        env.step(env.action)
        assert np.all(com_0 != env.quantities["com"])
        env.reset(seed=0)
        assert np.all(com_0 == env.quantities["com"])

    def test_stack(self):
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset()

        env.quantities["v_avg"] = (
            AverageFrameSpatialVelocity,
            dict(frame_name=env.robot.pinocchio_model.frames[1].name))

        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.quantities["v_avg"]
        env.step(env.action)
        v_avg = env.quantities["v_avg"].copy()
        env.step(env.action)
        assert np.all(v_avg != env.quantities["v_avg"])
