"""This file aims at verifying the sanity of the physics and the integration
method of jiminy on simple mass.
"""
import os
import tempfile
import unittest

import numpy as np

from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator

from jiminy_py.log import read_log, build_robots_from_log
from jiminy_py.viewer.replay import play_logs_data, play_logs_files

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

        # Delete existing simulator once done
        simulator.close()

        # # Test: log reading
        # log_data, log_constants = read_log(log_path)
        # robot = build_robots_from_log(log_constants)

        # # Test: replay from log
        # video_path = os.path.join(
        #     tempfile.gettempdir(),
        #     f"video_{next(tempfile._get_candidate_names())}.mp4")
        # play_logs_data(
        #     robot, log_data, record_video_path=video_path, verbose=False, close_backend=True)
        # self.assertTrue(os.path.isfile(video_path))

        # # Test: replay from file
        # play_logs_files(
        #     log_path, record_video_path=video_path, verbose=False, close_backend=True)
    
    def test_single_robot_simulation_with_multi_config(self):
        '''
        Test simulation with a single robot but initialized as in multi robots
        simulations.
        '''
        # Set initial condition and simulation duration
        np.random.seed(0)
        name1 = 'robot1'
        q0 = {name1 : np.random.rand(2)}
        v0 = {name1 : np.random.rand(2)}
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
        simulator = Simulator(robot, viewer_backend="panda3d-sync", system_name=name1)

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

        # Delete existing simulator once done
        simulator.close()

        # # Test: log reading
        # log_data, log_constants = read_log(log_path)
        # robot = build_robots_from_log(log_constants)

        # # Test: replay from log
        # video_path = os.path.join(
        #     tempfile.gettempdir(),
        #     f"video_{next(tempfile._get_candidate_names())}.mp4")
        # play_logs_data(
        #     robot, log_data, record_video_path=video_path, verbose=False, close_backend=True)
        # self.assertTrue(os.path.isfile(video_path))

        # # Test: replay from file
        # play_logs_files(
        #     log_path, record_video_path=video_path, verbose=False, close_backend=True)
    
    def test_multi_robot_simulation(self):
        '''
        Test simulation with two robots.
        '''
        # Set initial condition and simulation duration
        np.random.seed(0)
        name1 = 'robot1'
        name2 = 'robot2'
        q0 = {name1 : np.random.rand(2), name2 : np.random.rand(2)}
        v0 = {name1 : np.random.rand(2), name2 : np.random.rand(2)}
        t_end = 0.5

        # Define URDF path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_root_dir = os.path.join(current_dir, "data")
        urdf_path = os.path.join(data_root_dir, "double_pendulum.urdf")

        # Create robots
        # The existing hardware file should be loaded automatically.
        robot1 = BaseJiminyRobot()
        robot1.initialize(urdf_path, has_freeflyer=False)

        robot2 = BaseJiminyRobot()
        robot2.initialize(urdf_path, has_freeflyer=False)

        # Create simulator.
        # The existing options file should be loaded automatically.
        simulator = Simulator(robot1, viewer_backend="panda3d-sync", system_name=name1)
        simulator.engine.add_system(name2, robot2)

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

        # Delete existing simulator once done
        simulator.close()

        # # Test: log reading
        # log_data, log_constants = read_log(log_path)
        # robots = build_robots_from_log(log_constants)

        # # Test: replay from log
        # video_path = os.path.join(
        #     tempfile.gettempdir(),
        #     f"video_{next(tempfile._get_candidate_names())}.mp4")
        # play_logs_data(
        #     robots, log_data, record_video_path=video_path, verbose=False, close_backend=True)
        # self.assertTrue(os.path.isfile(video_path))

        # # Test: replay from file
        # play_logs_files(
        #     log_path, record_video_path=video_path, verbose=False, close_backend=True)

if __name__ == '__main__':
    unittest.main()
