""" TODO: Write documentation
"""
import gc
import unittest

import numpy as np
import gymnasium as gym

import jiminy_py
from jiminy_py.log import extract_trajectory_from_log
import pinocchio as pin

from gym_jiminy.common.bases import QuantityEvalMode, DatasetTrajectoryQuantity
from gym_jiminy.common.quantities import (
    QuantityManager,
    FrameEulerAngles,
    FrameXYZQuat,
    MaskedQuantity,
    AverageFrameSpatialVelocity,
    AverageOdometryVelocity,
    ActuatedJointPositions,
    CenterOfMass,
    ZeroMomentPoint)


class Quantities(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_shared_cache(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)

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

        env.step(env.action_space.sample())
        zmp_1 = quantity_manager["zmp"].copy()
        assert np.all(zmp_0 == zmp_1)
        quantity_manager.clear()
        assert quantities["zmp"].requirements["com"]._is_initialized
        assert not quantities["com"].cache.has_value
        zmp_1 = quantity_manager.zmp.copy()
        assert np.any(zmp_0 != zmp_1)

        env.step(env.action_space.sample())
        quantity_manager.reset()
        assert not quantities["zmp"].requirements["com"]._is_initialized
        zmp_2 = quantity_manager.zmp.copy()
        assert np.any(zmp_1 != zmp_2)

    def test_dynamic_batching(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)
        env.step(env.action_space.sample())

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

        env.step(env.action_space.sample())
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
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)

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
        for (cls, _), cache in quantity_manager._caches.items():
            assert len(cache.owners) == (cls is DatasetTrajectoryQuantity)

    def test_env(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["zmp"] = (ZeroMomentPoint, {})

        env.reset(seed=0)
        zmp_0 = env.quantities["zmp"].copy()
        env.step(env.action_space.sample())
        assert np.all(zmp_0 != env.quantities["zmp"])
        env.reset(seed=0)
        assert np.all(zmp_0 == env.quantities["zmp"])

    def test_stack(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)

        quantity_cls = AverageFrameSpatialVelocity
        quantity_kwargs = dict(
            frame_name=env.robot.pinocchio_model.frames[1].name)
        env.quantities["v_avg"] = (quantity_cls, quantity_kwargs)

        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.quantities["v_avg"]

        env.step(env.action_space.sample())
        v_avg = env.quantities["v_avg"].copy()
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        assert np.all(v_avg != env.quantities["v_avg"])

    def test_masked(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)
        env.step(env.action_space.sample())

        # 1. From non-slice-able indices
        env.quantities["v_masked"] = (MaskedQuantity, dict(
            quantity=(FrameXYZQuat, dict(frame_name="root_joint")),
            key=(0, 1, 5)))
        quantity = env.quantities.registry["v_masked"]
        assert not quantity._slices
        np.testing.assert_allclose(
            env.quantities["v_masked"], quantity.data[[0, 1, 5]])
        del env.quantities["v_masked"]

        # 2. From boolean mask
        env.quantities["v_masked"] = (MaskedQuantity, dict(
            quantity=(FrameXYZQuat, dict(frame_name="root_joint")),
            key=(True, True, False, False, False, True)))
        quantity = env.quantities.registry["v_masked"]
        np.testing.assert_allclose(
            env.quantities["v_masked"], quantity.data[[0, 1, 5]])
        del env.quantities["v_masked"]

        # 3. From slice-able indices
        env.quantities["v_masked"] = (MaskedQuantity, dict(
            quantity=(FrameXYZQuat, dict(frame_name="root_joint")),
            key=(0, 2, 4)))
        quantity = env.quantities.registry["v_masked"]
        assert len(quantity._slices) == 1 and quantity._slices[0] == slice(0, 5, 2)
        np.testing.assert_allclose(
            env.quantities["v_masked"], quantity.data[[0, 2, 4]])

    def test_true_vs_reference(self):
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["zmp"] = (
            ZeroMomentPoint, dict(mode=QuantityEvalMode.TRUE))
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)
        zmp_0 = env.quantities["zmp"].copy()
        env.stop()

        trajectory = extract_trajectory_from_log(env.log_data)
        env.quantities["zmp_ref"] = (
            ZeroMomentPoint, dict(mode=QuantityEvalMode.REFERENCE))

        with self.assertRaises(RuntimeError):
            env.reset(seed=0)

        env.quantities.add_trajectory("reference", trajectory)
        env.quantities.select_trajectory("reference")

        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample() * 0.05)
        assert np.all(zmp_0 != env.quantities["zmp"])
        np.testing.assert_allclose(zmp_0, env.quantities["zmp_ref"])

    def test_average_odometry_velocity(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["odometry_velocity"] = (
            AverageOdometryVelocity, dict(mode=QuantityEvalMode.TRUE))
        quantity = env.quantities.registry["odometry_velocity"]

        env.reset(seed=0)
        base_pose_prev = env.robot_state.q[:7].copy()
        env.step(env.action_space.sample())
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
            quantity.requirements['data'].data, base_velocity_mean_world)
        base_odom_velocity = base_velocity_mean_world[[0, 1, 5]]
        np.testing.assert_allclose(
            env.quantities["odometry_velocity"], base_odom_velocity)
