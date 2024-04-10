"""This file aims at verifying the sanity of the physics and the integration
method of jiminy on simple mass.
"""
import os
import tempfile
import unittest

import numpy as np

from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator

from jiminy_py.log import (read_log,
                           extract_trajectory_from_log,
                           extract_trajectories_from_log)
from jiminy_py.viewer.replay import play_logs_data


class SimulatorTest(unittest.TestCase):
    def test_single_robot_simulation(self):
        """Test simulation with a single robot.
        """
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
        simulator = Simulator(robot, viewer_kwargs={"backend": "panda3d-sync"})

        # Check synchronous rendering:
        # simulation started before display and the other way around.
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
        trajectory = extract_trajectory_from_log(log_data, robot)

        final_log_states = trajectory['evolution_robot'][-1]

        np.testing.assert_array_almost_equal(
            simulator.robot_state.q, final_log_states.q, decimal=10)
        np.testing.assert_array_almost_equal(
            simulator.robot_state.v, final_log_states.v, decimal=10)

        # Test: replay from log
        video_path = os.path.join(
            tempfile.gettempdir(),
            f"video_{next(tempfile._get_candidate_names())}.mp4")
        play_logs_data(
            robot, log_data, record_video_path=video_path, verbose=False)
        self.assertTrue(os.path.isfile(video_path))

    def test_double_robot_simulation(self):
        """Test simulation with two robots.
        """
        robot1_name = "robot1"
        robot2_name = "robot2"
        # Set initial condition and simulation duration
        np.random.seed(0)
        q0 = {robot1_name : np.random.rand(2), robot2_name : np.random.rand(2)}
        v0 = {robot1_name : np.random.rand(2), robot2_name : np.random.rand(2)}
        t_end = 0.5

        # Define URDF path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_root_dir = os.path.join(current_dir, "data")
        urdf_path = os.path.join(data_root_dir, "double_pendulum.urdf")

        # Create robot1.
        # The existing hardware file should be loaded automatically.
        robot1 = BaseJiminyRobot(robot1_name)
        robot1.initialize(urdf_path, has_freeflyer=False)

        # Create simulator.
        # The existing options file should be loaded automatically.
        simulator = Simulator(robot1, viewer_kwargs={"backend": "panda3d-sync"})

        # Add robot2 to the simulation
        simulator.add_robot(robot2_name, urdf_path, has_freeflyer=False)

        # Check synchronous rendering:
        # simulation started before display and the other way around.
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
        with self.assertRaises(NotImplementedError):
            simulator.replay(verbose=False)

        # Test: log reading
        log_data = read_log(log_path)
        trajectories = extract_trajectories_from_log(log_data)

        trajectory_1, trajectory_2 = (
            trajectories[robot.name]['evolution_robot'][-1]
            for robot in simulator.robots)
        robot_states_1, robot_states_2 = simulator.robot_states

        np.testing.assert_array_almost_equal(
            robot_states_1.q, trajectory_1.q, decimal=10)
        np.testing.assert_array_almost_equal(
            robot_states_1.v, trajectory_1.v, decimal=10)
        np.testing.assert_array_almost_equal(
            robot_states_2.q, trajectory_2.q, decimal=10)
        np.testing.assert_array_almost_equal(
            robot_states_2.v, trajectory_2.v, decimal=10)


if __name__ == '__main__':
    unittest.main()
