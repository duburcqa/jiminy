"""
@brief This file aims at verifying the sanity of the physics and the
       integration method of jiminy on simple models.
"""
import os
import io
import base64
import unittest
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator
from jiminy_py.viewer import Viewer, play_logs_files

from utilities import load_urdf_default


# Small tolerance for numerical equality
TOLERANCE = 1.0e-5
IMAGE_DIFF_THRESHOLD = 0.0

# Model parameters
MASS_SEGMENTS = 0.1
INERTIA_SEGMENTS = 0.001
LENGTH_SEGMENTS = 0.01
N_FLEXIBILITY = 40


def generate_flexible_arm(mass_segments: float,
                          inertia_segments: float,
                          length_segments: float,
                          n_segments: int,
                          urdf_path: str) -> None:
    """Helper function for procedural generation of robot arm with
    variable number of deformation points.
    """
    robot = ET.Element("robot", name="flexible_arm")

    ET.SubElement(robot, "link", name="base")

    for i in range(n_segments):
        link = ET.SubElement(robot, "link", name=f"link{i}")
        visual = ET.SubElement(link, "visual")
        ET.SubElement(
            visual, "origin", xyz=f"{length_segments/2} 0 0", rpy="0 0 0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", size=f"{length_segments} 0.025 0.01")
        material = ET.SubElement(visual, "material", name="")
        ET.SubElement(material, "color", rgba="0 0 0 1")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(
            inertial, "origin", xyz=f"{length_segments/2} 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass_segments}")
        ET.SubElement(
            inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0",
            izz=f"{inertia_segments}")

    motor = ET.SubElement(
        robot, "joint", name="base_to_link0", type="revolute")
    ET.SubElement(motor, "parent", link="base")
    ET.SubElement(motor, "child", link="link0")
    ET.SubElement(motor, "origin", xyz="0 0 0", rpy=f"{np.pi/2} 0 0")
    ET.SubElement(motor, "axis", xyz="0 0 1")
    ET.SubElement(
        motor, "limit", effort="100.0", lower=f"{-np.pi}", upper=f"{np.pi}",
        velocity="10.0")

    for i in range(1, n_segments):
        joint = ET.SubElement(
            robot, "joint", name=f"link{i-1}_to_link{i}", type="fixed")
        ET.SubElement(joint, "parent", link=f"link{i-1}")
        ET.SubElement(joint, "child", link=f"link{i}")
        ET.SubElement(
            joint, "origin", xyz=f"{length_segments} 0 0", rpy="0 0 0")

    tree = ET.ElementTree(robot)
    tree.write(urdf_path)


class SimulateFlexibleArm(unittest.TestCase):
    """Simulate the motion of a pendulum, comparing against python integration.
    """
    def setUp(self):
        # Create temporary urdf file
        fd, urdf_path = tempfile.mkstemp(
            prefix="flexible_arm_", suffix=".urdf")
        os.close(fd)

        # Procedural model generation
        generate_flexible_arm(
            MASS_SEGMENTS, INERTIA_SEGMENTS, LENGTH_SEGMENTS,
            N_FLEXIBILITY + 1, urdf_path)

        # Load URDF, create model.
        self.motors_names = ["base_to_link0"]
        robot = load_urdf_default(urdf_path, self.motors_names)

        # Remove temporary file
        os.remove(urdf_path)

        # Instantiate and initialize the controller
        controller = jiminy.ControllerFunctor()
        controller.initialize(robot)

        # Create a simulator using this robot and controller
        self.simulator = Simulator(
            robot,
            controller,
            viewer_kwargs=dict(
                camera_xyzrpy=((0.0, -2.0, 0.0), (np.pi/2, 0.0, 0.0))
            ))

    def tearDown(self):
        Viewer.close()

    def _read_write_replay_log(self, log_format):
        # Set initial condition and simulation duration
        q0, v0 = np.array([0.]), np.array([0.])
        t_end = 4.0

        # Generate temporary log file
        ext = log_format if log_format != "binary" else "data"
        fd, log_path = tempfile.mkstemp(
            prefix=f"{self.simulator.robot.name}_", suffix=f".{ext}")
        os.close(fd)

        # Run the simulation
        self.simulator.simulate(
            t_end, q0, v0, is_state_theoretical=True, log_path=log_path,
            show_progress_bar=False)

        # Generate temporary video file
        fd, video_path = tempfile.mkstemp(
            prefix=f"{self.simulator.robot.name}_", suffix=".mp4")
        os.close(fd)

        # Record the result
        viewer, *_ = play_logs_files(
            log_path,
            delete_robot_on_close=True,
            **self.simulator.viewer_kwargs,
            record_video_path=video_path,
            verbose=False)
        viewer.close()

        # Remove temporary log and video file
        os.remove(log_path)
        os.remove(video_path)

        return True

    def test_write_replay_standalone_log(self):
        """Check if reading/writing standalone log file is working.
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
            'stiffness': 10.0 * np.ones(3),
            'damping': 0.2 * np.ones(3),
            'inertia': np.array([1.0, 1.0, 1.0e-3])
        } for i in range(N_FLEXIBILITY)]
        self.simulator.robot.set_model_options(model_options)

        # Check both HDF5 and binary log formats
        for log_format in ('hdf5', 'binary'):
            self.assertTrue(self._read_write_replay_log(log_format))

        # Make sure the scene is empty now
        self.assertEqual(len(Viewer._backend_robot_names), 0)

    def test_rigid_vs_flex_at_frame(self):
        """Test if the result is the same with and without flexibility if the
        inertia is extremely large.
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
        img_rigid = self.simulator.render(return_rgb_array=True)

        # Check different flexibility ordering
        q_flex, img_flex, pnc_model_flex, visual_model_flex = [], [], [], []
        for order in (range(N_FLEXIBILITY), range(N_FLEXIBILITY)[::-1]):
            # Specify joint flexibility parameters
            model_options = self.simulator.robot.get_model_options()
            model_options['dynamics']['enableFlexibleModel'] = True
            model_options['dynamics']['flexibilityConfig'] = [{
                'frameName': f"link{i}_to_link{i+1}",
                'stiffness': np.zeros(3),
                'damping': np.zeros(3),
                'inertia': np.full(3, fill_value=1e6)
            } for i in order]
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
            img_flex.append(self.simulator.render(return_rgb_array=True))

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

        # Check that the flexible models are identical
        for i in range(len(visual_model_flex) - 1):
            for geom1, geom2 in zip(
                    visual_model_flex[i].geometryObjects,
                    visual_model_flex[i+1].geometryObjects):
                self.assertEqual(geom1.parentJoint, geom2.parentJoint)
                self.assertEqual(geom1.parentFrame, geom2.parentFrame)
                self.assertTrue(np.allclose(geom1.placement.homogeneous,
                                            geom2.placement.homogeneous,
                                            atol=TOLERANCE))
        for i in range(len(pnc_model_flex) - 1):
            for I1, I2 in zip(
                    pnc_model_flex[i].inertias, pnc_model_flex[i+1].inertias):
                self.assertTrue(np.allclose(I1.toDynamicParameters(),
                                            I2.toDynamicParameters(),
                                            atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
