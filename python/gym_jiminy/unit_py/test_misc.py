# mypy: disable-error-code="no-untyped-def, var-annotated"
""" TODO: Write documentation
"""
import os
import shutil
import tempfile
import unittest
from importlib.resources import files

import numpy as np
import gymnasium as gym

from jiminy_py.core import EncoderSensor, ImuSensor, EffortSensor, ForceSensor
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import Viewer
from jiminy_py.viewer.replay import (
    extract_replay_data_from_log, play_trajectories)
import pinocchio as pin

from gym_jiminy.common.utils import math


class Miscellaneous(unittest.TestCase):
    def test_play_log_data(self):
        """ TODO: Write documentation.
        """
        # Instantiate the environment
        env = gym.make("gym_jiminy.envs:atlas", debug=False)
        env.eval()

        # Run a few steps
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)
        env.stop()

        # Generate temporary video file
        fd, video_path = tempfile.mkstemp(prefix=f"atlas_", suffix=".mp4")
        os.close(fd)

        # Play trajectory
        Viewer.close()
        trajectory, update_hook, kwargs = extract_replay_data_from_log(
            env.log_data)
        viewer, = play_trajectories(trajectory,
                                    update_hook,
                                    backend="panda3d-sync",
                                    record_video_path=video_path,
                                    display_contacts=True,
                                    display_f_external=True,
                                    verbose=False)
        Viewer.close()

        # Check the external forces has been updated
        np.testing.assert_allclose(
            tuple(f_ext.vector for f_ext in env.robot_state.f_external[1:]),
            tuple(f_ext.vector for f_ext in viewer.f_external))

        # Check that sensor data has been updated
        for key, data in env.robot.sensor_measurements.items():
            np.testing.assert_allclose(
                data, trajectory.robot.sensor_measurements[key])

    def test_default_hardware(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Copy original Atlas URDF and meshes
            data_dir = files("gym_jiminy.envs") / "data/bipedal_robots/atlas"
            urdf_path = os.path.join(tmpdirname, "atlas.urdf")
            shutil.copy2(str(data_dir / "atlas.urdf"), urdf_path)
            shutil.copytree(
                str(data_dir / "meshes"), os.path.join(tmpdirname, "meshes"))

            # Build simulator with default hardware
            simulator = Simulator.build(urdf_path)

            # Check that all mechanical joints are actuated
            assert simulator.robot.nmotors == len(
                simulator.robot.mechanical_joint_names)

            # Check that all mechanical joints have encoder and effort sensors
            assert len(simulator.robot.sensors[EncoderSensor.type]) == 30
            assert len(simulator.robot.sensors[EffortSensor.type]) == 30

            # Check that the root joint has an IMU sensor
            assert len(simulator.robot.sensors[ImuSensor.type]) == 1
            sensor = simulator.robot.sensors[ImuSensor.type][0]
            assert sensor.frame_name == (
                simulator.robot.pinocchio_model.frames[2].name)

            # Check that the each foot has a force sensor
            assert len(simulator.robot.sensors[ForceSensor.type]) == 2

    def test_math(self):
        for batch_size in ((3, 7, 2), (10,), ()):
            # SO3 inverse exponential map / apply
            size = int(np.prod(batch_size))
            quat = np.random.rand(4, *batch_size)
            quat /= np.linalg.norm(quat, axis=0)
            pos = np.random.rand(3, *batch_size)
            omega = math.log3(quat)
            pos_rel = math.quat_apply(quat, pos)
            for quat_i, omega_i, pos_i, pos_rel_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        quat, omega, pos, pos_rel))):
                np.testing.assert_allclose(
                    omega_i, pin.log3(pin.Quaternion(quat_i).matrix()))
                np.testing.assert_allclose(
                    pos_rel_i, pin.Quaternion(quat_i) * pos_i)
            np.testing.assert_allclose(
                np.zeros(3), math.log3(np.array([0.0, 0.0, 0.0, 1.0])))

            # SO3 exponential map
            omega = np.random.rand(3, *batch_size)
            quat = math.exp3(omega)
            for quat_i, omega_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        quat, omega))):
                np.testing.assert_allclose(
                    quat_i, pin.Quaternion(pin.exp3(omega_i)).coeffs())
            np.testing.assert_allclose(
                np.array([0.0, 0.0, 0.0, 1.0]), math.exp3(np.zeros(3)))

            # SE3 inverse exponential map
            xyzquat = np.random.rand(7, *batch_size)
            xyzquat[-4:] /= np.linalg.norm(xyzquat[-4:], axis=0)
            v_spatial = math.log6(xyzquat)
            for xyzquat_i, v_spatial_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        xyzquat, v_spatial))):
                np.testing.assert_allclose(v_spatial_i, pin.log6(pin.SE3(
                    pin.Quaternion(xyzquat_i[-4:]).matrix(), xyzquat_i[:3])))
            np.testing.assert_allclose(np.zeros(6), math.log6(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])))

            # SE3 exponential map
            v_spatial = np.random.rand(6, *batch_size)
            xyzquat = math.exp6(v_spatial)
            for xyzquat_i, v_spatial_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        xyzquat, v_spatial))):
                np.testing.assert_allclose(
                    xyzquat_i[:3], pin.exp6(v_spatial_i).translation)
                np.testing.assert_allclose(xyzquat_i[-4:], pin.Quaternion(
                    pin.exp6(v_spatial_i).rotation).coeffs())
            np.testing.assert_allclose(np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), math.exp6(np.zeros(6)))

            # SO3 difference / multiply
            so3 = pin.liegroups.SO3()
            quat_1 = np.random.rand(4, *batch_size)
            quat_1 /= np.linalg.norm(quat_1, axis=0)
            quat_2 = np.random.rand(4, *batch_size)
            quat_2 /= np.linalg.norm(quat_2, axis=0)
            omega = math.quat_difference(quat_1, quat_2)
            quat_12 = math.quat_multiply(quat_1, quat_2)
            for quat_1_i, quat_2_i, quat_12_i, omega_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        quat_1, quat_2, quat_12, omega))):
                np.testing.assert_allclose(
                    omega_i, so3.difference(quat_1_i, quat_2_i))
                np.testing.assert_allclose(quat_12_i, (pin.Quaternion(
                    quat_1_i) * pin.Quaternion(quat_2_i)).coeffs())

            # SE3 difference
            se3 = pin.liegroups.SE3()
            xyzquat_1 = np.random.rand(7, *batch_size)
            xyzquat_1[-4:] /= np.linalg.norm(xyzquat_1[-4:], axis=0)
            xyzquat_2 = np.random.rand(7, *batch_size)
            xyzquat_2[-4:] /= np.linalg.norm(xyzquat_2[-4:], axis=0)
            v_spatial = math.xyzquat_difference(xyzquat_1, xyzquat_2)
            for xyzquat_1_i, xyzquat_2_i, v_spatial_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        xyzquat_1, xyzquat_2, v_spatial))):
                np.testing.assert_allclose(
                    v_spatial_i, se3.difference(xyzquat_1_i, xyzquat_2_i))

            # Conversions
            quat = np.random.rand(4, *batch_size)
            quat /= np.linalg.norm(quat, axis=0)
            rpy = math.quat_to_rpy(quat)
            rot_mat = math.quat_to_matrix(quat)
            np.testing.assert_allclose(math.quat_to_yaw(quat), rpy[-1])
            for quat_i, rpy_i, rot_mat_i in zip(*(
                    data.reshape((-1, size)).T for data in (
                        quat, rpy, rot_mat))):
                np.testing.assert_allclose(
                    rot_mat_i.reshape((3, 3)), pin.Quaternion(quat_i).matrix())
                np.testing.assert_allclose(
                    rpy_i, pin.rpy.matrixToRpy(rot_mat_i.reshape((3, 3))))
            np.testing.assert_allclose(math.rpy_to_quat(rpy), quat)
            np.testing.assert_allclose(
                math.matrix_to_quat(rot_mat), quat, atol=1e-9)
