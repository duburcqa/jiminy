# mypy: disable-error-code="no-untyped-def, var-annotated"
""" TODO: Write documentation
"""
import sys
import math
import unittest

import numpy as np
import gymnasium as gym

from jiminy_py.dynamics import update_quantities
from jiminy_py.log import extract_trajectory_from_log
import pinocchio as pin

from gym_jiminy.common.utils import (
    matrix_to_quat, quat_average, quat_to_matrix, quat_to_yaw,
    remove_yaw_from_quat)
from gym_jiminy.common.bases import QuantityEvalMode, DatasetTrajectoryQuantity
from gym_jiminy.common.quantities import (
    EnergyGenerationMode,
    OrientationType,
    QuantityManager,
    StackedQuantity,
    MaskedQuantity,
    MultiFrameMeanXYZQuat,
    MultiFrameOrientation,
    MultiFootMeanOdometryPose,
    MultiFootRelativeXYZQuat,
    MultiFrameCollisionDetection,
    MultiActuatedJointKinematic,
    MultiContactNormalizedSpatialForce,
    MultiFootNormalizedForceVertical,
    FrameOrientation,
    FrameXYZQuat,
    FrameSpatialAverageVelocity,
    BaseOdometryAverageVelocity,
    BaseRelativeHeight,
    AverageBaseMomentum,
    AverageMechanicalPowerConsumption,
    CenterOfMass,
    CapturePoint,
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
        assert quantities["com"].cache.sm_state == 2
        assert quantities["acom"].cache.sm_state == 0
        is_initialized_all = [
            owner._is_initialized for owner in quantities["com"].cache.owners]
        assert len(is_initialized_all) == 2
        assert len(set(is_initialized_all)) == 2

        env.step(env.action_space.sample())
        zmp_1 = quantity_manager["zmp"].copy()
        assert np.all(zmp_0 == zmp_1)
        quantity_manager.clear()
        is_initialized_all = [
            owner._is_initialized for owner in quantities["com"].cache.owners]
        assert any(is_initialized_all)
        assert quantities["com"].cache.sm_state == 1
        zmp_1 = quantity_manager.zmp.copy()
        assert np.any(zmp_0 != zmp_1)

        env.step(env.action_space.sample())
        quantity_manager.reset()
        assert quantities["com"].cache.sm_state == 0
        is_initialized_all = [
            owner._is_initialized for owner in quantities["com"].cache.owners]
        assert not any(is_initialized_all)
        zmp_2 = quantity_manager.zmp.copy()
        assert np.any(zmp_1 != zmp_2)

    def test_dynamic_batching(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)
        env.step(env.action_space.sample())

        frame_names = [
            frame.name for frame in env.robot.pinocchio_model.frames]

        quantity_manager = QuantityManager(env)
        for name, cls, kwargs in (
                ("xyzquat_0", FrameXYZQuat, dict(
                    frame_name=frame_names[2])),
                ("rpy_0", FrameOrientation, dict(
                    frame_name=frame_names[1],
                    type=OrientationType.EULER)),
                ("rpy_1", FrameOrientation, dict(
                    frame_name=frame_names[1],
                    type=OrientationType.EULER)),
                ("rpy_2", FrameOrientation, dict(
                    frame_name=frame_names[-1],
                    type=OrientationType.EULER)),
                ("rpy_batch_0", MultiFrameOrientation, dict(  # Intersection
                    frame_names=(frame_names[-3], frame_names[1]),
                    type=OrientationType.EULER)),
                ("rpy_batch_1", MultiFrameOrientation, dict(  # Inclusion
                    frame_names=(frame_names[1], frame_names[-1]),
                    type=OrientationType.EULER)),
                ("rpy_batch_2", MultiFrameOrientation, dict(  # Disjoint
                    frame_names=(frame_names[1], frame_names[-4]),
                    type=OrientationType.EULER)),
                ("rot_mat_batch", MultiFrameOrientation, dict(
                    frame_names=(frame_names[1], frame_names[-1]),
                    type=OrientationType.MATRIX)),
                ("quat_batch", MultiFrameOrientation, dict(
                    frame_names=(frame_names[1], frame_names[-4]),
                    type=OrientationType.QUATERNION))):
            quantity_manager[name] = (cls, kwargs)
        quantities = quantity_manager.registry

        xyzquat_0 = quantity_manager.xyzquat_0.copy()
        rpy_0 = quantity_manager.rpy_0.copy()
        assert len(quantities['rpy_0'].data.cache.owners[0].frame_names) == 1
        assert np.all(rpy_0 == quantity_manager.rpy_1)
        rpy_2 = quantity_manager.rpy_2.copy()
        assert np.any(rpy_0 != rpy_2)
        assert len(quantities['rpy_2'].data.cache.owners[0].frame_names) == 2
        assert tuple(quantity_manager.rpy_batch_0.shape) == (3, 2)
        assert len(quantities['rpy_batch_0'].data.cache.owners[0].frame_names) == 3
        quantity_manager.rpy_batch_1
        assert len(quantities['rpy_batch_1'].data.cache.owners[0].frame_names) == 3
        quantity_manager.rpy_batch_2
        assert len(quantities['rpy_batch_2'].data.cache.owners[0].frame_names) == 5
        assert tuple(quantity_manager.rot_mat_batch.shape) == (3, 3, 2)
        assert tuple(quantity_manager.quat_batch.shape) == (4, 2)
        assert len(quantities['quat_batch'].data.rot_mat_map.cache.owners[0].frame_names) == 8

        env.step(env.action_space.sample())
        quantity_manager.reset()
        rpy_0_next = quantity_manager.rpy_0
        xyzquat_0_next = quantity_manager.xyzquat_0.copy()
        assert np.any(rpy_0 != rpy_0_next)
        assert np.any(xyzquat_0 != xyzquat_0_next)
        assert len(quantities['rpy_2'].data.cache.owners[0].frame_names) == 5

        assert len(quantities['rpy_1'].data.cache.owners) == 6
        del quantity_manager['rpy_2']
        assert len(quantities['rpy_1'].data.cache.owners) == 5
        quantity_manager.rpy_1
        assert len(quantities['rpy_1'].data.cache.owners[0].frame_names) == 1

        quantity_manager.reset(reset_tracking=True)
        assert np.all(rpy_0_next == quantity_manager.rpy_0)
        assert np.all(xyzquat_0_next == quantity_manager.xyzquat_0)
        assert len(quantities['rpy_0'].data.cache.owners[0].frame_names) == 1

    def test_discard(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)

        quantity_manager = QuantityManager(env)
        for name, cls, kwargs in (
                ("rpy_0", FrameOrientation, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                ("rpy_1", FrameOrientation, dict(
                    frame_name=env.robot.pinocchio_model.frames[1].name)),
                ("rpy_2", FrameOrientation, dict(
                    frame_name=env.robot.pinocchio_model.frames[-1].name))):
            quantity_manager[name] = (cls, kwargs)
        quantities = quantity_manager.registry

        assert len(quantities['rpy_1'].cache.owners) == 2
        assert len(quantities['rpy_2'].data.cache.owners) == 3

        del quantity_manager['rpy_0']
        assert len(quantities['rpy_1'].cache.owners) == 1
        assert len(quantities['rpy_2'].data.cache.owners) == 2

        del quantity_manager['rpy_1']
        assert len(quantities['rpy_2'].data.cache.owners) == 1

        del quantity_manager['rpy_2']
        for (cls, _), cache in quantity_manager._caches:
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

    def test_stack_auto_refresh(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)

        quantity_cls = FrameSpatialAverageVelocity
        quantity_kwargs = dict(
            frame_name=env.robot.pinocchio_model.frames[1].name)
        env.quantities["v_avg"] = (quantity_cls, quantity_kwargs)
        env.quantities["v_avg_2"] = (quantity_cls, quantity_kwargs)

        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.quantities["v_avg"]

        env.step(env.action_space.sample())
        v_avg = env.quantities["v_avg"].copy()
        env.step(env.action_space.sample())
        del env.quantities["v_avg_2"]
        env.step(env.action_space.sample())
        assert np.all(v_avg != env.quantities["v_avg"])

    def test_stack_api(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        for max_stack, as_array, mode in (
                (None, False, "slice"),
                (3, False, "slice"),
                (3, True, "slice"),
                (3, False, "zeros"),
                (3, True, "zeros")):
            quantity_creator = (StackedQuantity, dict(
                quantity=(MultiFootRelativeXYZQuat, {}),
                max_stack=max_stack or sys.maxsize,
                as_array=as_array,
                mode=mode))
            env.quantities["xyzquat_stack"] = quantity_creator
            env.reset(seed=0)

            value = env.quantities["xyzquat_stack"]
            if as_array:
                assert isinstance(value, np.ndarray)
            else:
                assert isinstance(value, list)
            for i in range(1, (max_stack or 5) + 2):
                num_stack = max_stack or i
                if mode == "slice":
                    num_stack = min(i, num_stack)
                value = env.quantities["xyzquat_stack"]
                if as_array:
                    assert value.shape[-1] == num_stack
                    if mode == "zeros":
                        np.testing.assert_allclose(value[..., :-i], 0.0)
                else:
                    assert len(value) == num_stack
                    if mode == "zeros":
                        np.testing.assert_allclose(value[:-i], 0.0)
                env.step(env.action)

            del env.quantities["xyzquat_stack"]

    def test_masked(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")
        env.reset(seed=0)
        env.step(env.action_space.sample())

        # 1. From non-slice-able indices
        env.quantities["v_masked"] = (MaskedQuantity, dict(
            quantity=(FrameXYZQuat, dict(frame_name="root_joint")),
            keys=(0, 1, 5)))
        quantity = env.quantities.registry["v_masked"]
        assert not quantity._slices
        value = quantity.quantity.get()
        np.testing.assert_allclose(
            env.quantities["v_masked"], value[[0, 1, 5]])
        del env.quantities["v_masked"]

        # 2. From boolean mask
        env.quantities["v_masked"] = (MaskedQuantity, dict(
            quantity=(FrameXYZQuat, dict(frame_name="root_joint")),
            keys=(True, True, False, False, False, True)))
        quantity = env.quantities.registry["v_masked"]
        value = quantity.quantity.get()
        np.testing.assert_allclose(
            env.quantities["v_masked"], value[[0, 1, 5]])
        del env.quantities["v_masked"]

        # 3. From slice-able indices
        env.quantities["v_masked"] = (MaskedQuantity, dict(
            quantity=(FrameXYZQuat, dict(frame_name="root_joint")),
            keys=(0, 2, 4)))
        quantity = env.quantities.registry["v_masked"]
        assert len(quantity._slices) == 1 and (
            quantity._slices[0] == slice(0, 5, 2))
        value = quantity.quantity.get()
        np.testing.assert_allclose(
            env.quantities["v_masked"], value[[0, 2, 4]])

    def test_true_vs_reference(self):
        env = gym.make("gym_jiminy.envs:atlas", debug=False)
        env.eval()

        frame_names = [
            frame.name for frame in env.robot.pinocchio_model.frames]

        for quantity_creator in (
                lambda mode: (ZeroMomentPoint, dict(mode=mode)),
                lambda mode: (FrameOrientation, dict(
                    type=OrientationType.MATRIX,
                    frame_name=frame_names[1],
                    mode=mode)),
                lambda mode: (FrameXYZQuat, dict(
                    frame_name=frame_names[2],
                    mode=mode)),
                lambda mode: (MultiFrameMeanXYZQuat, dict(
                    frame_names=tuple(frame_names[i] for i in (1, 3, -2)),
                    mode=mode)),
                lambda mode: (MultiFootMeanOdometryPose, dict(
                    mode=mode)),
                lambda mode: (FrameSpatialAverageVelocity, dict(
                    frame_name=frame_names[1],
                    mode=mode)),
                lambda mode: (BaseOdometryAverageVelocity, dict(
                    mode=mode)),
                lambda mode: (MultiActuatedJointKinematic, dict(
                    kinematic_level=pin.KinematicLevel.POSITION,
                    is_motor_side=False,
                    mode=mode)),
                lambda mode: (MultiActuatedJointKinematic, dict(
                    kinematic_level=pin.KinematicLevel.VELOCITY,
                    is_motor_side=True,
                    mode=mode)),
                lambda mode: (CenterOfMass, dict(
                    kinematic_level=pin.KinematicLevel.ACCELERATION,
                    mode=mode)),
                lambda mode: (CapturePoint, dict(
                    mode=mode)),
                lambda mode: (ZeroMomentPoint, dict(
                    mode=mode))
                ):
            env.quantities["true"] = quantity_creator(QuantityEvalMode.TRUE)

            values = []
            env.reset(seed=0)
            for _ in range(10):
                env.step(env.action)
                values.append(env.quantities["true"].copy())
            env.stop()
            trajectory = extract_trajectory_from_log(env.log_data)

            env.quantities["ref"] = quantity_creator(
                QuantityEvalMode.REFERENCE)

            # No trajectory has been selected
            with self.assertRaises(RuntimeError):
                env.reset(seed=0)
                env.quantities["ref"]

            env.quantities.trajectory_dataset.add("reference", trajectory)
            env.quantities.trajectory_dataset.select("reference")

            env.reset(seed=0)
            for value in values:
                env.step(env.action_space.sample() * 0.05)
                with np.testing.assert_raises(AssertionError):
                    np.testing.assert_allclose(value, env.quantities["true"])
                np.testing.assert_allclose(value, env.quantities["ref"])
            env.stop()

            env.quantities.trajectory_dataset.discard("reference")
            del env.quantities["true"]
            del env.quantities["ref"]

    def test_average_odometry_velocity(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["odometry_velocity"] = (
            BaseOdometryAverageVelocity, dict(
                mode=QuantityEvalMode.TRUE))
        quantity = env.quantities.registry["odometry_velocity"]

        env.reset(seed=0)
        base_pose_prev = env.robot_state.q[:7].copy()
        env.step(env.action_space.sample())
        base_pose = env.robot_state.q[:7].copy()

        se3 = pin.liegroups.SE3()
        base_pose_diff = se3.difference(base_pose_prev, base_pose)
        base_velocity_mean_local =  base_pose_diff / env.step_dt
        base_pose_mean = se3.integrate(base_pose_prev, 0.5 * base_pose_diff)
        rot_mat = quat_to_matrix(remove_yaw_from_quat(base_pose_mean[-4:]))
        base_velocity_mean_world = np.concatenate((
            rot_mat @ base_velocity_mean_local[:3],
            rot_mat @ base_velocity_mean_local[3:]))

        np.testing.assert_allclose(
            quantity.data.quantity.get(), base_velocity_mean_world)
        base_odom_velocity = base_velocity_mean_world[[0, 1, 5]]
        np.testing.assert_allclose(
            env.quantities["odometry_velocity"], base_odom_velocity)

    def test_average_momentum(self):
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["base_momentum"] = (
            AverageBaseMomentum, dict(mode=QuantityEvalMode.TRUE))

        env.reset(seed=0)
        base_pose_prev = env.robot_state.q[:7].copy()
        env.step(env.action_space.sample())
        base_pose = env.robot_state.q[:7].copy()

        se3 = pin.liegroups.SE3()
        base_pose_diff = se3.difference(base_pose_prev, base_pose)
        base_velocity_mean_local =  base_pose_diff / env.step_dt
        base_pose_mean = se3.integrate(base_pose_prev, 0.5 * base_pose_diff)
        I = env.robot.pinocchio_model.inertias[1].inertia
        rot_mat = quat_to_matrix(remove_yaw_from_quat(base_pose_mean[-4:]))
        angular_momentum = rot_mat @ (I @ base_velocity_mean_local[3:])

        np.testing.assert_allclose(
            env.quantities["base_momentum"], angular_momentum)

    def test_actuated_joints_kinematic(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:cassie")

        for level in (
                pin.KinematicLevel.POSITION,
                pin.KinematicLevel.VELOCITY,
                pin.KinematicLevel.ACCELERATION):
            env.quantities[f"joint_{level}"] = (
                MultiActuatedJointKinematic, dict(
                    kinematic_level=level,
                    is_motor_side=False,
                    mode=QuantityEvalMode.TRUE))
            if level < 2:
                env.quantities[f"motor_{level}"] = (
                    MultiActuatedJointKinematic, dict(
                        kinematic_level=level,
                        is_motor_side=True,
                        mode=QuantityEvalMode.TRUE))

            env.reset(seed=0)
            env.step(env.action_space.sample())

            kinematic_indices = []
            for motor in env.robot.motors:
                joint = env.robot.pinocchio_model.joints[motor.joint_index]
                if level == pin.KinematicLevel.POSITION:
                    kin_first, kin_last = joint.idx_q, joint.idx_q + joint.nq
                else:
                    kin_first, kin_last = joint.idx_v, joint.idx_v + joint.nv
                kinematic_indices += range(kin_first, kin_last)
            if level == pin.KinematicLevel.POSITION:
                joint_value = env.robot_state.q[kinematic_indices]
            elif level == pin.KinematicLevel.VELOCITY:
                joint_value = env.robot_state.v[kinematic_indices]
            else:
                joint_value = env.robot_state.a[kinematic_indices]
            encoder_data = env.robot.sensor_measurements["EncoderSensor"]

            np.testing.assert_allclose(
                env.quantities[f"joint_{level}"], joint_value)
            if level < 2:
                np.testing.assert_allclose(
                    env.quantities[f"motor_{level}"], encoder_data[level])

    def test_capture_point(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        update_quantities(
            env.robot,
            pin.neutral(env.robot.pinocchio_model_th),
            update_dynamics=True,
            update_centroidal=True,
            update_energy=False,
            update_jacobian=False,
            update_collisions=False,
            use_theoretical_model=True)
        min_height = min(
            oMf.translation[2] for oMf in env.robot.pinocchio_data_th.oMf)
        gravity = abs(env.robot.pinocchio_model.gravity.linear[2])
        robot_height = env.robot.pinocchio_data_th.com[0][2] - min_height
        omega = math.sqrt(gravity / robot_height)

        env.quantities["dcm"] = (CapturePoint, dict(
            reference_frame=pin.LOCAL_WORLD_ALIGNED,
            mode=QuantityEvalMode.TRUE))
        quantity = env.quantities.registry["dcm"]

        env.reset(seed=0)
        env.step(env.action_space.sample())

        com_position = env.robot.pinocchio_data.com[0]
        np.testing.assert_allclose(quantity.com_position.get(), com_position)
        com_velocity = env.robot.pinocchio_data.vcom[0]
        np.testing.assert_allclose(quantity.com_velocity.get(), com_velocity)
        np.testing.assert_allclose(
            env.quantities["dcm"],
            com_position[:2] + com_velocity[:2] / omega)

    def test_mean_pose(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        frame_names = [
            frame.name for frame in env.robot.pinocchio_model.frames]

        env.quantities["mean_pose"] = (
            MultiFrameMeanXYZQuat, dict(
                frame_names=frame_names[:5],
                mode=QuantityEvalMode.TRUE))

        env.reset(seed=0)
        env.step(env.action_space.sample())

        pos = np.mean(np.stack([
            oMf.translation for oMf in env.robot.pinocchio_data.oMf
            ][:5], axis=-1), axis=-1)
        quat = quat_average(np.stack([
            matrix_to_quat(oMf.rotation)
            for oMf in env.robot.pinocchio_data.oMf][:5], axis=-1))
        if quat[-1] < 0.0:
            quat *= -1

        value = env.quantities["mean_pose"]
        if value[-1] < 0.0:
            value[-4:] *= -1

        np.testing.assert_allclose(value, np.concatenate((pos, quat)))

    def test_foot_odometry_pose(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["foot_odom_pose"] = (MultiFootMeanOdometryPose, {})

        env.reset(seed=0)
        env.step(env.action_space.sample())

        foot_left_index, foot_right_index = (
            env.robot.pinocchio_model.getFrameId(name)
            for name in ("l_foot", "r_foot"))
        foot_left_pose, foot_right_pose = (
            env.robot.pinocchio_data.oMf[frame_index]
            for frame_index in (foot_left_index, foot_right_index))

        mean_pos = (foot_left_pose.translation[:2] +
                    foot_right_pose.translation[:2]) / 2.0
        mean_yaw = quat_to_yaw(quat_average(np.stack(tuple(map(matrix_to_quat,
            (foot_left_pose.rotation, foot_right_pose.rotation))), axis=-1)))
        value = env.quantities["foot_odom_pose"]

        np.testing.assert_allclose(value, np.array((*mean_pos, mean_yaw)))

    def test_foot_relative_pose(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["foot_rel_poses"] = (MultiFootRelativeXYZQuat, {})

        env.reset(seed=0)
        env.step(env.action_space.sample())

        foot_poses = []
        for frame_name in ("l_foot", "r_foot"):
            frame_index = env.robot.pinocchio_model.getFrameId(frame_name)
            foot_poses.append(env.robot.pinocchio_data.oMf[frame_index])
        pos_feet = np.stack([
            foot_pose.translation for foot_pose in foot_poses], axis=-1)
        quat_feet = np.stack([
            matrix_to_quat(foot_pose.rotation)
            for foot_pose in foot_poses], axis=-1)

        pos_mean = np.mean(pos_feet, axis=-1, keepdims=True)
        rot_mean = quat_to_matrix(quat_average(quat_feet))
        pos_rel = rot_mean.T @ (pos_feet - pos_mean)
        quat_rel = np.stack([
            matrix_to_quat(rot_mean.T @ foot_pose.rotation)
            for foot_pose in foot_poses], axis=-1)
        quat_rel[-4:] *= np.sign(quat_rel[-1])

        value = env.quantities["foot_rel_poses"].copy()
        value[-4:] *= np.sign(value[-1])

        np.testing.assert_allclose(value[:3], pos_rel[:, :-1])
        np.testing.assert_allclose(value[-4:], quat_rel[:, :-1])

    def test_contact_spatial_forces(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas")

        env.quantities["force_spatial_rel"] = (
            MultiContactNormalizedSpatialForce, {})

        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)

        gravity = abs(env.robot.pinocchio_model.gravity.linear[2])
        robot_weight = env.robot.pinocchio_data.mass[0] * gravity
        force_spatial_rel = np.stack([np.concatenate(
            (constraint.lambda_c[:3], np.zeros((2,)), constraint.lambda_c[[3]])
            ) for constraint in env.robot.constraints.contact_frames.values()],
            axis=-1) / robot_weight
        np.testing.assert_allclose(
            force_spatial_rel, env.quantities["force_spatial_rel"])

    def test_foot_vertical_forces(self):
        """ TODO: Write documentation
        """
        env = gym.make("gym_jiminy.envs:atlas-pid")

        env.quantities["force_vertical_rel"] = (
            MultiFootNormalizedForceVertical, {})

        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)

        gravity = abs(env.robot.pinocchio_model.gravity.linear[2])
        robot_weight = env.robot.pinocchio_data.mass[0] * gravity
        force_vertical_rel = np.empty((2,))
        for i, frame_name in enumerate(("l_foot", "r_foot")):
            frame_index = env.robot.pinocchio_model.getFrameId(frame_name)
            frame = env.robot.pinocchio_model.frames[frame_index]
            transform = env.robot.pinocchio_data.oMf[frame_index]
            f_external = env.robot_state.f_external[frame.parent]
            f_z_world = np.dot(transform.rotation[2], f_external.linear)
            force_vertical_rel[i] = f_z_world / robot_weight
        np.testing.assert_allclose(
            force_vertical_rel, env.quantities["force_vertical_rel"])
        np.testing.assert_allclose(np.sum(force_vertical_rel), 1.0, atol=1e-3)

    def test_base_height(self):
        env = gym.make("gym_jiminy.envs:atlas-pid")

        env.quantities["base_height"] = (BaseRelativeHeight, {})

        env.reset(seed=0)
        action = env.action_space.sample()
        for _ in range(10):
            env.step(action)

        value = env.quantities["base_height"]
        base_z = env.robot.pinocchio_data.oMf[1].translation[[2]]
        contacts_z = []
        for constraint in env.robot.constraints.contact_frames.values():
            frame_index = constraint.frame_index
            frame_pos = env.robot.pinocchio_data.oMf[frame_index]
            contacts_z.append(frame_pos.translation[[2]])
        np.testing.assert_allclose(base_z - np.min(contacts_z), value)

    def test_frames_collision(self):
        env = gym.make("gym_jiminy.envs:atlas-pid", step_dt=0.01)

        env.quantities["frames_collision"] = (
            MultiFrameCollisionDetection, dict(
                frame_names=("l_foot", "r_foot"),
                security_margin=0.0))

        motor_names = [motor.name for motor in env.robot.motors]
        left_motor_index = motor_names.index('l_leg_hpx')
        right_motor_index = motor_names.index('r_leg_hpx')
        action = np.zeros((len(motor_names),))
        action[[left_motor_index, right_motor_index]] = -0.5, 0.5

        env.robot.remove_contact_points([])
        env.eval()
        env.reset(seed=0)
        assert not env.quantities["frames_collision"]
        for _ in range(20):
            env.step(action)
            if env.quantities["frames_collision"]:
                break
        else:
            raise AssertionError("No collision detected.")

    def test_power_consumption(self):
        env = gym.make("gym_jiminy.envs:cassie")

        for mode in (
                EnergyGenerationMode.CHARGE,
                EnergyGenerationMode.LOST_EACH,
                EnergyGenerationMode.LOST_GLOBAL,
                EnergyGenerationMode.PENALIZE):
            env.quantities["mean_power_consumption"] = (
                AverageMechanicalPowerConsumption, dict(
                    horizon=0.2,
                    generator_mode=mode))
            quantity = env.quantities.registry["mean_power_consumption"]
            env.reset(seed=0)

            total_power_stack = [0.0,]
            encoder_data = env.robot.sensor_measurements["EncoderSensor"]
            _, motor_velocities = encoder_data
            for _ in range(8):
                motor_efforts = 0.1 * env.action_space.sample()
                env.step(motor_efforts)

                motor_powers = motor_efforts * motor_velocities
                if mode == EnergyGenerationMode.CHARGE:
                    total_power = np.sum(motor_powers)
                elif mode == EnergyGenerationMode.LOST_EACH:
                    total_power = np.sum(np.maximum(motor_powers, 0.0))
                elif mode == EnergyGenerationMode.LOST_GLOBAL:
                    total_power = max(np.sum(motor_powers), 0.0)
                else:
                    total_power = np.sum(np.abs(motor_powers))
                total_power_stack.append(total_power)
                mean_total_power = np.mean(
                    total_power_stack[-quantity.max_stack:])

                value = quantity.total_power_stack.get()
                np.testing.assert_allclose(total_power, value[-1])
                np.testing.assert_allclose(mean_total_power, quantity.get())

            del env.quantities["mean_power_consumption"]
