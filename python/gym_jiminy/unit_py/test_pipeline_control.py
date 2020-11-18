""" TODO: Write documentation
"""
import os
import warnings
import unittest
import numpy as np
import matplotlib.pyplot as plt

from jiminy_py.core import EncoderSensor as encoder

from gym_jiminy.envs import AtlasPDControlJiminyEnv


class PipelineControlAtlas(unittest.TestCase):
    def setUp(self):
        """ TODO: Write documentation
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.env = AtlasPDControlJiminyEnv()

    def test_pid_standing(self):
        """ TODO: Write documentation
        """
        # Reset the environment
        obs_init = self.env.reset()

        # The initial target corresponds to the initial joints state, so that
        # the robot stand-still.
        encoder_data = obs_init['sensors'][encoder.type]
        action_init = dict(zip(
            encoder.fieldnames,
            encoder_data[:, self.env.controller.motor_to_encoder]))
        # Run the simulation during 5s
        for _ in range(5000):
            self.env.step(action_init)

        # Get the final posture of the robot as an RGB array
        rgb_array = self.env.render(mode='rgb_array')
        # plt.imsave("atlas_standing_meshcat.png", rgb_array)

        # Check that the final posture matches the expected one.
        robot_name = self.env.robot.pinocchio_model.name
        img_name = '_'.join((robot_name, "standing_meshcat.png"))
        img_fullpath = os.path.join(os.path.dirname(__file__), img_name)
        rgba_array_rel_orig = plt.imread(img_fullpath)
        rgb_array_abs_orig = (
            rgba_array_rel_orig[..., :3] * 255).astype(np.uint8)
        img_diff = np.mean(np.abs(rgb_array - rgb_array_abs_orig))
        self.assertTrue(img_diff < 0.1)

        # Get the simulation log
        log_data, _ = self.env.get_log()

        # Check that joint velocity target is zero
        controller_fieldnames = self.env.controller.get_fieldnames()
        target_pos_name = controller_fieldnames[encoder.fieldnames[1]][0]
        log_name = '.'.join((
            'HighLevelController', self.env.controller_name, target_pos_name))
        data = log_data[log_name]
        self.assertTrue(np.all(np.abs(data) < 1e-9))

        # Check that the velocity is close to zero at the end
        velocity_mes = np.stack([
            log_data['.'.join((encoder.type, name, encoder.fieldnames[1]))]
            for name in self.env.robot.sensors_names[encoder.type]], axis=-1)
        self.assertTrue(np.all(np.abs(velocity_mes[-1000:]) < 1e-3))
