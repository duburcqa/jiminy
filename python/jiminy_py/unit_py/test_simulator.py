"""This file aims at verifying the sanity of the physics and the integration
method of jiminy on simple mass.
"""
import os
import tempfile
import unittest

import numpy as np

from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator

from jiminy_py.log import read_log, build_robot_from_log
from jiminy_py.viewer.replay import play_logs_data


class SimulatorTest(unittest.TestCase):
    def test_single_robot_simulation(self):
        '''
        Test simulation with a single robot.
        '''
        # Set initial condition and simulation duration
        np.random.seed(0)
        q0, v0 = np.random.rand(2), np.random.rand(2)
        t_end = 0.5

        # Define URDF path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_root_dir = os.path.join(current_dir, "data")
        urdf_path = os.path.join(data_root_dir, "double_pendulum.urdf")

        # Create robot.
        # The existing hardware file should be loaded automatically.
        robot = BaseJiminyRobot()
        robot.initialize(urdf_path, has_freeflyer=False)

        # Create simulator.
        # The existing options file should be loaded automatically.
        simulator = Simulator(robot, viewer_backend="panda3d-sync")

        # Test all instances of viewer start.
        simulator.start(q0, v0)
        simulator.render(return_rgb_array=True)
        simulator.stop()
        simulator.start(q0, v0)
        simulator.stop()
        simulator.close()

        # Run the simulation and write log
        log_path = os.path.join(
            tempfile.gettempdir(),
            f"log_{next(tempfile._get_candidate_names())}.hdf5")
        simulator.simulate(
            t_end, q0, v0, log_path=log_path, show_progress_bar=False)
        self.assertTrue(os.path.isfile(log_path))

        # Test: viewer
        video_path = os.path.join(
            tempfile.gettempdir(),
            f"video_{next(tempfile._get_candidate_names())}.mp4")
        simulator.replay(record_video_path=video_path, verbose=False)
        self.assertTrue(os.path.isfile(video_path))

        # Test: log reading
        log_data = read_log(log_path)
        robot = build_robot_from_log(log_data)

        # Test: replay from log
        video_path = os.path.join(
            tempfile.gettempdir(),
            f"video_{next(tempfile._get_candidate_names())}.mp4")
        play_logs_data(
            robot, log_data, record_video_path=video_path, verbose=False)
        self.assertTrue(os.path.isfile(video_path))


if __name__ == '__main__':
    unittest.main()
