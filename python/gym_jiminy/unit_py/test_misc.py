""" TODO: Write documentation
"""
import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import gymnasium as gym

from jiminy_py.core import EncoderSensor, ImuSensor, EffortSensor, ForceSensor
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import Viewer
from jiminy_py.viewer.replay import (
    extract_replay_data_from_log, play_trajectories)

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


class Miscellaneous(unittest.TestCase):
    def test_play_log_data(self):
        """ TODO: Write documentation.
        """
        # Instantiate the environment
        env = gym.make("gym_jiminy.envs:atlas", debug=True)

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
                                    display_f_external=True)
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
