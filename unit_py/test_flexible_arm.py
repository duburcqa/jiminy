"""
@brief This file aims at verifying the sanity of the physics and the
       integration method of jiminy on simple models.
"""
import os
import io
import base64
import unittest
import tempfile

import numpy as np
from PIL import Image

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import Viewer, play_logs_files

from utilities import load_urdf_default


# Small tolerance for numerical equality
TOLERANCE = 1.0e-5
IMAGE_DIFF_THRESHOLD = 0.0


class SimulateFlexibleArm(unittest.TestCase):
    """
    @brief Simulate the motion of a pendulum, comparing against python
           integration.
    """
    def setUp(self):
        # Load URDF, create model.
        self.urdf_name = 'flexible_arm.urdf'
        self.motors_names = ["base_to_link1"]
        robot = load_urdf_default(
            self.urdf_name, self.motors_names, use_temporay_urdf=True)

        # Camera pose
        self.camera_xyzrpy = ([0.0, -2.0, 0.0], [np.pi/2, 0.0, 0])

        # Instantiate and initialize the controller
        controller = jiminy.ControllerFunctor()
        controller.initialize(robot)

        # Create a simulator using this robot and controller
        self.simulator = Simulator(robot, controller)

    def tearDown(self):
        Viewer.close()

    def _read_write_replay_log(self, format):
        # Set initial condition and simulation duration
        q0, v0 = np.array([0.]), np.array([0.])
        t_end = 4.0

        # Generate temporary log file
        ext = format if format != "binary" else "data"
        fd, log_path = tempfile.mkstemp(
            prefix=f"{self.urdf_name}_", suffix=f"_log.{ext}")
        os.close(fd)

        # Run the simulation
        self.simulator.simulate(
            t_end, q0, v0, is_state_theoretical=True, log_path=log_path,
            show_progress_bar=False)

        # Generate temporary video file
        fd, video_path = tempfile.mkstemp(
            prefix=f"{self.urdf_name}_", suffix=f"_video.mp4")
        os.close(fd)

        # Record the result
        viewer, *_ = play_logs_files(log_path,
                                     delete_robot_on_close=True,
                                     camera_xyzrpy=self.camera_xyzrpy,
                                     record_video_path=video_path,
                                     verbose=False)
        viewer.close()

        return True

    def test_write_replay_standalone_log(self):
        """
        @brief Check if reading/writing standalone log file is working.
        """
        # Configure log file to be standalone
        engine_options = self.simulator.engine.get_options()
        engine_options['telemetry']['isPersistent'] = True
        self.simulator.engine.set_options(engine_options)

        # Specify joint flexibility parameters
        model_options = self.simulator.robot.get_model_options()
        model_options['dynamics']['enableFlexibleModel'] = True
        model_options['dynamics']['flexibilityConfig'] = [{
            'frameName': f"link{i}_to_link{i+1}",
            'stiffness': np.zeros(3),
            'damping': np.zeros(3),
            'inertia': np.array([1.0, 1.0, 0.0])
        } for i in range(1,4)]
        self.simulator.robot.set_model_options(model_options)

        # Check both HDF5 and binary log formats
        for format in ('hdf5', 'binary'):
            self.assertTrue(self._read_write_replay_log(format))

        # Make sure the scene is empty now
        self.assertEqual(len(Viewer._backend_robot_names), 0)

    def test_rigid_vs_flex_at_frame(self):
        """
        @brief Test if the result is the same with and without flexibility
        if the inertia is extremely large.
        """
        # Set initial condition and simulation duration
        q0, v0 = np.array([0.]), np.array([0.])
        t_end = 4.0

        # Run a first simulation without flexibility
        self.simulator.simulate(
            t_end, q0, v0, is_state_theoretical=True, show_progress_bar=False)

        # Extract the final configuration
        q_rigid = self.simulator.system_state.q.copy()

        # Render the scene
        img_rigid = self.simulator.render(
            return_rgb_array=True,
            camera_xyzrpy=self.camera_xyzrpy)

        # Check different flexibility ordering
        q_flex, img_flex, pnc_model_flex, visual_model_flex = [], [], [], []
        for ord in [(1, 2, 3), (3, 2, 1), (3, 1, 2), (1, 2, 3)]:
            # Specify joint flexibility parameters
            model_options = self.simulator.robot.get_model_options()
            model_options['dynamics']['enableFlexibleModel'] = True
            model_options['dynamics']['flexibilityConfig'] = [{
                'frameName': f"link{i}_to_link{i+1}",
                'stiffness': np.zeros(3),
                'damping': np.zeros(3),
                'inertia': np.full(3, fill_value=1e6)
            } for i in ord]
            self.simulator.robot.set_model_options(model_options)

            # Serialize the model
            pnc_model_flex.append(
                self.simulator.robot.pinocchio_model.copy())
            visual_model_flex.append(
                self.simulator.robot.visual_model.copy())

            # Launch the simulation
            self.simulator.simulate(
                t_end, q0, v0, is_state_theoretical=True,
                show_progress_bar=False)

            # Extract the final configuration
            q_flex.append(
                self.simulator.robot.get_rigid_configuration_from_flexible(
                    self.simulator.system_state.q))

            # Render the scene
            img_flex.append(self.simulator.render(
                return_rgb_array=True, camera_xyzrpy=self.camera_xyzrpy))

        # Compare the final results
        for q in q_flex:
            self.assertTrue(np.allclose(q, q_rigid, atol=TOLERANCE))
        for i, img in enumerate(img_flex):
            img_diff = np.mean(np.abs(img - img_rigid))
            if img_diff > IMAGE_DIFF_THRESHOLD:
                for name, img_tmp in zip(
                        ("rigid", f"flex {i}"), (img, img_rigid)):
                    img_obj = Image.fromarray(img_tmp)
                    raw_bytes = io.BytesIO()
                    img_obj.save(raw_bytes, "PNG")
                    raw_bytes.seek(0)
                    print(f"{name}:", base64.b64encode(raw_bytes.read()))
            self.assertLessEqual(img_diff, IMAGE_DIFF_THRESHOLD)

        # Check that the flexible models are identicals
        for i in range(len(visual_model_flex) - 1):
            self.assertEqual(visual_model_flex[i], visual_model_flex[i+1])
        for i in range(len(pnc_model_flex) - 1):
            for I1, I2 in zip(
                    pnc_model_flex[i].inertias, pnc_model_flex[i+1].inertias):
                self.assertTrue(np.allclose(I1.toDynamicParameters(),
                                            I2.toDynamicParameters(),
                                            atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
