""" TODO: Write documentation
"""
import os
import warnings
import unittest
import numpy as np
import matplotlib.pyplot as plt

from jiminy_py.core import EncoderSensor as encoder
from jiminy_py.viewer import Viewer

from gym_jiminy.envs import AtlasPDControlJiminyEnv
from gym_jiminy.envs import CassiePDControlJiminyEnv


class PipelineControlAtlas(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        """ TODO: Write documentation
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def _test_pid_standing(self):
        """ TODO: Write documentation
        """
        # Check that it is not possible to get simulation log at this point
        self.assertRaises(RuntimeError, self.env.get_log)

        # Reset the environment
        def configure_telemetry() -> None:
            nonlocal self
            engine_options = self.env.simulator.engine.get_options()
            engine_options['telemetry']['enableVelocity'] = True
            self.env.simulator.engine.set_options(engine_options)

        obs_init = self.env.reset(controller_hook=configure_telemetry)

        # Compute the initial target, so that the robot stand-still.
        # In practice, it corresponds to the initial joints state.
        encoder_data = obs_init['sensors'][encoder.type]
        action_init = {}
        action_init['Q'], action_init['V'] = encoder_data[
            :, self.env.controller.motor_to_encoder]

        # Run the simulation during 5s
        while self.env.stepper_state.t < 5.0:
            self.env.step(action_init)

        # Get the final posture of the robot as an RGB array
        rgb_array = self.env.render(mode='rgb_array')

        # Check that the final posture matches the expected one.
        robot_name = self.env.robot.name
        i = 0
        img_diff = np.inf
        while img_diff > 0.1:
            img_name = '_'.join((
                robot_name, f"standing_{self.env.viewer.backend}_{i}.png"))
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            img_fullpath = os.path.join(data_dir, img_name)
            # plt.imsave(img_fullpath, rgb_array)
            try:
                rgba_array_rel_orig = plt.imread(img_fullpath)
            except FileNotFoundError:
                break
            rgb_array_abs_orig = (
                rgba_array_rel_orig[..., :3] * 255).astype(np.uint8)
            img_diff = np.mean(np.abs(rgb_array - rgb_array_abs_orig))
            i += 1
        self.assertTrue(img_diff < 0.1)

        # Get the simulation log
        log_data, _ = self.env.get_log()

        # Check that the joint velocity target is zero
        time = log_data["Global.Time"]
        velocity_target = np.stack([
            log_data['.'.join((
                'HighLevelController', self.env.controller_name, name))]
            for name in self.env.controller.get_fieldnames()['V']], axis=-1)
        self.assertTrue(np.all(np.abs(velocity_target[time > 4.0]) < 1e-9))

        # Check that the whole-body robot velocity is close to zero at the end
        velocity_mes = np.stack([
            log_data['.'.join(('HighLevelController', name))]
            for name in self.env.robot.logfile_velocity_headers], axis=-1)
        self.assertTrue(np.all(np.abs(velocity_mes[time > 4.0]) < 1e-3))

    def test_pid_standing(self):
        for Env in [AtlasPDControlJiminyEnv, CassiePDControlJiminyEnv]:
            self.env = Env(debug=False)
            self._test_pid_standing()
            Viewer.close()
