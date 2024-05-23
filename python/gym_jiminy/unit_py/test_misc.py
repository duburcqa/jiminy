""" TODO: Write documentation
"""
import unittest

import numpy as np
import gymnasium as gym

from jiminy_py.viewer import Viewer
from jiminy_py.viewer.replay import (
    extract_replay_data_from_log, play_trajectories)


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

        # Play trajectory
        Viewer.close()
        trajectory, update_hook, kwargs = extract_replay_data_from_log(
            env.log_data)
        viewer, = play_trajectories(trajectory,
                                    update_hook,
                                    backend="panda3d-sync",
                                    display_contacts=True,
                                    display_f_external=True)

        # Check the external forces has been updated
        np.testing.assert_allclose(
            env.robot_state.f_external, trajectory.states[-1].f_external)

        # Check that sensor data has been updated
        for key, data in env.robot.sensor_measurements.items():
            np.testing.assert_allclose(
                data, trajectory.robot.sensor_measurements[key])
