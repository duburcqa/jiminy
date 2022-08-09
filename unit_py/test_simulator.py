"""This file aims at verifying the sanity of the physics and the integration
method of jiminy on simple mass.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator

from jiminy_py.log import read_log, build_robot_from_log_constants
from jiminy_py.viewer.replay import play_logs_data


# Source files
SIMULATION_DURATION = 0.5


class SimulatorTest(unittest.TestCase):
    def setUp(self):
        # Model specification
        self.body_name = 'MassBody'

        # Define the parameters of the contact dynamics
        self.k_contact = 1.0e6
        self.nu_contact = 2.0e3
        self.friction = 2.0
        self.transtion_vel = 5.0e-2
        self.dtMax = 1.0e-5

    def test_single_robot_simulation(self):
        '''
        Test simulation with a single robot.
        '''
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
        simulator = Simulator(robot)

        # Run simulation
        q0 = 0.1 * np.random.rand(2)
        v0 = np.random.rand(2)
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Run the simulation and write log
            log_path = f"{tmpdirname}/log.hdf5"
            simulator.simulate(SIMULATION_DURATION, q0, v0, log_path=log_path)

            # Test: viewer
            video_path_1 = f"{tmpdirname}/video_1.mp4"
            simulator.replay(record_video_path=video_path_1)
            assert video_path_1.is_file()

            # Test: log reading
            log_data, log_constants = read_log(log_path)
            robot = build_robot_from_log_constants(log_constants)

            # Test: replay from log
            video_path_2 = f"{tmpdirname}/video_2.mp4"
            play_logs_data(robot,
                           log_data,
                           record_video_path=video_path_2)
            assert video_path_2.is_file()


if __name__ == '__main__':
    unittest.main()
