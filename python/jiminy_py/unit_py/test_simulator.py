""" TODO: Write documentation.
"""
import os
import tempfile
import unittest

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.robot import BaseJiminyRobot
from jiminy_py.simulator import Simulator

from jiminy_py.log import (read_log,
                           extract_variables_from_log,
                           extract_trajectory_from_log,
                           extract_trajectories_from_log)
from jiminy_py.viewer.replay import play_logs_data


class SimulatorTest(unittest.TestCase):
    def test_consistency_velocity_acceleration(self):
        """Check that the acceleration is the derivative of the velocity.
        """
        # Define URDF path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_root_dir = os.path.join(current_dir, "data")
        urdf_path = os.path.join(data_root_dir, "double_pendulum.urdf")

        # Instantiate the robot
        motor_joint_names = ("PendulumJoint", "SecondPendulumJoint",)
        robot = jiminy.Robot()
        robot.initialize(urdf_path, has_freeflyer=False)
        for joint_name in motor_joint_names:
            motor = jiminy.SimpleMotor(joint_name)
            robot.attach_motor(motor)
            motor.initialize(joint_name)

            options = motor.get_options()
            options["enableBacklash"] = True
            options["enableArmature"] = True
            options["backlash"] = 0.05
            options["armature"] = 3.0
            motor.set_options(options)

        for fname in ("PendulumMass", "SecondPendulumMass"):
            imu = jiminy.ImuSensor(fname)
            robot.attach_sensor(imu)
            imu.initialize(fname)

        # Define a PD controller with fixed target position
        class Controller(jiminy.BaseController):
            def compute_command(self, t, q, v, command):
                target = np.array([1.5, 0.0])
                command[:] = -5000 * ((q[::2] - target) + 0.07 * v[::2])

        robot.controller = Controller()

        # Instantiate the engine
        engine = jiminy.Engine()
        engine.add_robot(robot)

        # Configuration the simulation
        engine_options = engine.get_options()
        engine_options["stepper"]["odeSolver"] = "euler_explicit"
        engine_options["stepper"]["dtMax"] = 1e-3
        engine_options["stepper"]["sensorUpdatePeriod"] = 0.0
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine_options['contacts']['model'] = "constraint"

        # Run multiple simulations with different options
        for control_dt in (1e-3, 0.0):
            # Update the controller update period
            engine_options["stepper"]["controllerUpdatePeriod"] = control_dt
            engine.set_options(engine_options)

            # Run the simulation
            q0 = np.array([1.45, 0.0, 0.0, 0.0])
            v0 = np.zeros(robot.pinocchio_model.nv)
            tf = 5.0
            engine.simulate(tf, q0, v0)

            # Compare the finite difference of velocity with acceleration
            log_vars = engine.log_data["variables"]
            time = log_vars["Global.Time"]
            time_filter = np.array([*(time[1:] != time[:-1]), False])
            velocities = np.stack(extract_variables_from_log(
                log_vars, robot.log_velocity_fieldnames), axis=0
                )[:, time_filter]
            diff_velocities = np.diff(
                velocities, axis=1) / np.diff(time[time_filter])
            accelerations = np.stack(extract_variables_from_log(
                log_vars, robot.log_acceleration_fieldnames), axis=0
                )[:, time_filter]
            np.testing.assert_allclose(
                diff_velocities, accelerations[:, :-1], atol=1e-12, rtol=0.0)

            # Check that IMU accelerations match gravity at rest
            for imu_name in robot.sensor_names[jiminy.ImuSensor.type]:
                log_imu_name = ".".join((jiminy.ImuSensor.type, imu_name))
                imu_data = np.stack(extract_variables_from_log(
                    log_vars, jiminy.ImuSensor.fieldnames, log_imu_name
                    ), axis=0)
                imu_gyro, imu_accel = np.split(imu_data, 2)
                imu_gyro_norm = np.linalg.norm(imu_gyro, axis=0)
                imu_accel_norm = np.linalg.norm(imu_accel, axis=0)
                is_static = (time > 1.0) & (imu_gyro_norm < 1e-10)
                np.testing.assert_allclose(
                    imu_accel_norm[is_static], 9.81, atol=1e-6, rtol=0.0)

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
        trajectory = extract_trajectory_from_log(log_data)

        final_log_states = trajectory.states[-1]

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
        robot1_name, robot2_name = "robot1", "robot2"
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
            trajectories[robot.name].states[-1]
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
