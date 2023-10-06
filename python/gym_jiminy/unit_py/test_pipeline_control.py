""" TODO: Write documentation
"""
import os
import io
import base64
import logging
import unittest
from glob import glob
from tempfile import mkstemp

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from jiminy_py.core import ImuSensor as imu
from jiminy_py.viewer import Viewer

import pinocchio as pin

from gym_jiminy.envs import (
    AtlasPDControlJiminyEnv, CassiePDControlJiminyEnv, DigitPDControlJiminyEnv)
from gym_jiminy.common.blocks import PDController


IMAGE_DIFF_THRESHOLD = 5.0

# Small tolerance for numerical equality
TOLERANCE = 1.0e-6


class PipelineControl(unittest.TestCase):
    """ TODO: Write documentation
    """
    def _test_pid_standing(self):
        """ TODO: Write documentation
        """
        # Reset the environment
        self.env.reset()

        # Zero target motors velocities, so that the robot stands still
        action = np.zeros(self.env.robot.nmotors)

        # Run the simulation
        while self.env.stepper_state.t < 19.0:
            self.env.step(action)

        # Export figure
        fd, pdf_path = mkstemp(prefix="plot_", suffix=".pdf")
        os.close(fd)
        self.env.plot(pdf_path=pdf_path)

        # Get the final posture of the robot as an RGB array
        rgb_array = self.env.render()

        # Check that the final posture matches the expected one
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        img_prefix = '_'.join((self.env.robot.name, "standing", "*"))
        img_min_diff = np.inf
        for img_fullpath in glob(os.path.join(data_dir, img_prefix)):
            rgba_array_rel_ref = plt.imread(img_fullpath)
            rgb_array_ref = (
                rgba_array_rel_ref[..., :3] * 255).astype(np.uint8)
            try:
                img_diff = np.mean(np.abs(rgb_array - rgb_array_ref))
            except ValueError:
                logging.exception(
                    "Impossible to compare captured frame with ref image "
                    "'{img_fullpath}', likely because of shape mismatch.")
                continue
            img_min_diff = min(img_min_diff, img_diff)
            if img_min_diff < IMAGE_DIFF_THRESHOLD:
                break
        else:
            img_obj = Image.fromarray(rgb_array)
            raw_bytes = io.BytesIO()
            img_obj.save(raw_bytes, "PNG")
            raw_bytes.seek(0)
            print(f"{self.env.robot.name} - {self.env.viewer.backend}:",
                  base64.b64encode(raw_bytes.read()))
        self.assertLessEqual(img_min_diff, IMAGE_DIFF_THRESHOLD)

        # Get the simulation log
        log_vars = self.env.log_data["variables"]

        # Check that the joint velocity target is zero
        time = log_vars["Global.Time"]
        velocity_target = np.stack([
            log_vars['.'.join((
                'HighLevelController', self.env.controller.name, name))]
            for name in self.env.controller.fieldnames], axis=-1)
        self.assertTrue(np.all(
            np.abs(velocity_target[time > time[-1] - 1.0]) < 1.0e-9))

        # Check that the whole-body robot velocity is close to zero at the end
        velocity_mes = np.stack([
            log_vars['.'.join(('HighLevelController', name))]
            for name in self.env.robot.log_fieldnames_velocity], axis=-1)
        self.assertTrue(np.all(
            np.abs(velocity_mes[time > time[-1] - 1.0]) < 1.0e-3))

    def test_pid_standing(self):
        for backend in ('panda3d-sync', 'meshcat'):
            for Env in (AtlasPDControlJiminyEnv,
                        CassiePDControlJiminyEnv,
                        DigitPDControlJiminyEnv):
                self.env = Env(
                    debug=True,
                    render_mode='rgb_array',
                    viewer_kwargs=dict(
                        backend=backend,
                        width=500,
                        height=500,
                        display_com=False,
                        display_dcm=False,
                        display_contacts=False,
                    )
                )
                self._test_pid_standing()
                Viewer.close()

    def test_mahony_filter(self):
        """Check consistency between IMU orientation estimation provided by
        Mahony filter and ground truth for moderately slow motions.
        """
        # Instantiate and reset the environment
        env = AtlasPDControlJiminyEnv()
        env.reset()

        # Define a constant action that move the upper-body in all directions
        robot = env.robot
        action = np.zeros((robot.nmotors,))
        for name, value in (
                ('back_bkz', 0.4), ('back_bky', 0.08), ('back_bkx', 0.08)):
            action[robot.get_motor(name).idx] = value

        # Extract proxies for convenience
        sensor = robot.get_sensor(imu.type, robot.sensors_names[imu.type][0])
        imu_rot = robot.pinocchio_data.oMf[sensor.frame_idx].rotation

        # Check that the estimate IMU orientation is accurate over the episode
        for i in range(200):
            env.step(action * (1 - 2 * ((i // 50) % 2)))
            rpy_true = pin.rpy.matrixToRpy(imu_rot)
            rpy_est = pin.rpy.matrixToRpy(
                pin.Quaternion(env.observer.observation).matrix())
            self.assertTrue(np.allclose(rpy_true, rpy_est, atol=0.01))
        env.stop()

    def test_pid_controller(self):
        """ TODO: Write documentation.
        """
        # Instantiate the environment and run a simulation with random action
        env = AtlasPDControlJiminyEnv()
        env.reset()
        env.unwrapped._height_neutral = float("-inf")
        while env.stepper_state.t < 2.0:
            env.step(0.2 * env.action_space.sample())

        # Extract the target position and velocity of a single motor
        controller = env.env.controller
        assert isinstance(controller, PDController) and controller.order == 1
        ctrl_name = controller.name
        n_motors = len(controller.fieldnames)
        pos = env.log_data["variables"][".".join((
            "HighLevelController", ctrl_name, "state", str(n_motors - 1)))]
        vel = env.log_data["variables"][".".join((
            "HighLevelController", ctrl_name, controller.fieldnames[-1]))]

        # Make sure that the position and velocity targets are consistent
        self.assertTrue(np.allclose(
            np.diff(pos) / controller.control_dt, vel[1:], atol=TOLERANCE))

        # Make sure that the position targets are within bounds.
        # No such guarantee exists for higher-order derivatives.
        robot = env.robot
        pos_min = robot.position_limit_lower[robot.motors_position_idx[-1]]
        pos_max = robot.position_limit_upper[robot.motors_position_idx[-1]]
        self.assertTrue(np.all(np.logical_and(pos_min < pos, pos < pos_max)))
