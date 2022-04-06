""" TODO: Write documentation
"""
import os
import io
import base64
import warnings
import unittest
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from jiminy_py.core import EncoderSensor as encoder
from jiminy_py.viewer import Viewer

from gym_jiminy.envs import AtlasPDControlJiminyEnv, CassiePDControlJiminyEnv


IMAGE_DIFF_THRESHOLD = 1.0


class PipelineControl(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        """ TODO: Write documentation
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def _test_pid_standing(self):
        """ TODO: Write documentation
        """
        # Reset the environment
        obs_init = self.env.reset()

        # Compute the initial target, so that the robot stand-still.
        # In practice, it corresponds to the initial joints state.
        encoder_data = obs_init['sensors'][encoder.type]
        action_init = {}
        action_init['Q'], action_init['V'] = encoder_data[
            :, self.env.controller.motor_to_encoder]

        # Run the simulation
        while self.env.stepper_state.t < 19.0:
            self.env.step(action_init)

        # Get the final posture of the robot as an RGB array
        rgb_array = self.env.render(
            mode='rgb_array', display_com=False, display_contacts=False)

        # Check that the final posture matches the expected one
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        img_prefix = '_'.join((
            self.env.robot.name, "standing", self.env.viewer.backend, "*"))
        img_diff = np.inf
        for img_fullpath in glob(os.path.join(data_dir, img_prefix)):
            try:
                rgba_array_rel_orig = plt.imread(img_fullpath)
            except FileNotFoundError:
                break
            rgb_array_abs_orig = (
                rgba_array_rel_orig[..., :3] * 255).astype(np.uint8)
            try:
                img_diff = np.mean(np.abs(rgb_array - rgb_array_abs_orig))
            except ValueError:
                pass
            if img_diff < IMAGE_DIFF_THRESHOLD:
                break
        if img_diff > IMAGE_DIFF_THRESHOLD:
            img_obj = Image.fromarray(rgb_array)
            raw_bytes = io.BytesIO()
            img_obj.save(raw_bytes, "PNG")
            raw_bytes.seek(0)
            print(f"{self.env.robot.name} - {self.env.viewer.backend}:",
                  base64.b64encode(raw_bytes.read()))
        self.assertTrue(img_diff < IMAGE_DIFF_THRESHOLD)

        # Get the simulation log
        log_data = self.env.log_data

        # Check that the joint velocity target is zero
        time = log_data["Global.Time"]
        velocity_target = np.stack([
            log_data['.'.join((
                'HighLevelController', self.env.controller_name, name))]
            for name in self.env.controller.get_fieldnames()['V']], axis=-1)
        self.assertTrue(np.all(
            np.abs(velocity_target[time > time[-1] - 1.0]) < 1.0e-9))

        # Check that the whole-body robot velocity is close to zero at the end
        velocity_mes = np.stack([
            log_data['.'.join(('HighLevelController', name))]
            for name in self.env.robot.logfile_velocity_headers], axis=-1)
        self.assertTrue(np.all(
            np.abs(velocity_mes[time > time[-1] - 1.0]) < 1.0e-3))

    def test_pid_standing(self):
        for backend in ['meshcat', 'panda3d']:
            for Env in [AtlasPDControlJiminyEnv, CassiePDControlJiminyEnv]:
                self.env = Env(debug=False, viewer_backend=backend)
                self._test_pid_standing()
                Viewer.close()
