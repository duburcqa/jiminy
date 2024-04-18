"""
@brief This file aims at verifying the sanity of the physics and the
       integration method of jiminy on simple models.
"""
import unittest
import numpy as np
import scipy
from scipy.interpolate import interp1d
from typing import Union, Dict, Tuple, Sequence

import jiminy_py.core as jiminy
from pinocchio import Quaternion, log3, exp3
from pinocchio.rpy import matrixToRpy, rpyToMatrix

from utilities import (
    load_urdf_default,
    setup_controller_and_engine,
    integrate_dynamics,
    neutral_state,
    simulate_and_get_state_evolution)

# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1.0e-7


class SimulateSimplePendulum(unittest.TestCase):
    """Simulate the motion of a pendulum, comparing against python integration.
    """
    def setUp(self):
        # Load URDF, create model.
        self.urdf_name = "simple_pendulum.urdf"
        self.motor_names = ["PendulumJoint"]
        self.robot = load_urdf_default(self.urdf_name, self.motor_names)

        # Add IMU to the robot
        self.imu_sensor = jiminy.ImuSensor("PendulumLink")
        self.robot.attach_sensor(self.imu_sensor)
        self.imu_sensor.initialize("PendulumLink")

        # System dynamics: get length and inertia
        axis_str = self.robot.pinocchio_model.joints[1].shortname()[-1]
        self.axis = np.zeros(3)
        self.axis[ord(axis_str) - ord('X')] = 1.0
        self.l = abs(self.robot.pinocchio_model.inertias[1].lever[2])
        self.m = self.robot.pinocchio_model.inertias[1].mass
        self.g = self.robot.pinocchio_model.gravity.linear[2]
        self.I = self.m * self.l ** 2

    @staticmethod
    def _simulate_and_get_imu_data_evolution(
            engine: jiminy.Engine,
            tf: float,
            x0: Union[Dict[str, np.ndarray], np.ndarray],
            split: bool = False) -> Union[
                Sequence[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Simulate the dynamics of the system and retrieve the imu sensor
        evolution over time.

        :param engine: List of time instant at which to evaluate the solution.
        :param tf: Duration of the simulation.
        :param x0: Initial state of the system.
        :param split: Whether to return quat, gyro, accel separately or as a
                      unique vector.

        :returns: Pair containing first the sequence of time and second the
        associated sequence of IMU measurements as a 2D array each line
        corresponds to a given time.
        """
        # Run simulation
        if isinstance(x0, np.ndarray):
            q0, v0 = x0[:engine.robots[0].nq], x0[-engine.robots[0].nv:]
        else:
            q0, v0 = {}, {}
            for robot in engine.robots:
                q0[robot.name] = x0[robot.name][:robot.nq]
                v0[robot.name] = x0[robot.name][-robot.nv:]

        engine.simulate(tf, q0, v0)

        # Get log data
        log_vars = engine.log_data["variables"]

        # Extract state evolution over time
        time = log_vars['Global.Time']
        imu_jiminy = np.stack([
            log_vars['.'.join((jiminy.ImuSensor.type, 'PendulumLink', field))]
            for field in jiminy.ImuSensor.fieldnames], axis=-1)

        # Split IMU data if requested
        if split:
            gyro_jiminy, accel_jiminy = np.split(imu_jiminy, 2, axis=-1)
            return time, gyro_jiminy, accel_jiminy

        return time, imu_jiminy

    def test_armature(self):
        """Verify the dynamics of the system when adding rotor inertia.
        """
        # Configure the robot: set rotor inertia
        J = 0.1
        robot_options = self.robot.get_options()
        robot_options["motors"]["PendulumJoint"]['enableArmature'] = True
        robot_options["motors"]["PendulumJoint"]['armature'] = J
        self.robot.set_options(robot_options)

        # Dynamics: simulate a spring of stiffness k
        k_spring = 500.0
        def spring_force(t, q, v, sensor_measurements, u_custom):
            u_custom[:] = - k_spring * q

        # Initialize the controller and setup the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, internal_dynamics=spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine.set_options(engine_options)

        # Run simulation and extract log data
        x0 = np.array([0.1, 0.0])
        tf = 2.0
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)

        # Analytical solution: a simple mass on a spring
        I_eq = self.I + J
        A = np.array([[             0.0, 1.0],
                      [-k_spring / I_eq, 0.0]])
        x_analytical = np.stack([
            scipy.linalg.expm(A * t).dot(x0) for t in time], axis=0)

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_pendulum_integration(self):
        """Compare pendulum motion, as simulated by Jiminy, against an
        equivalent simulation done in python.

        Since we don't have a simple analytical expression for the solution of
        a (nonlinear) pendulum motion, we perform the simulation in Python,
        with the same integrator, and compare both results.
        """
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        setup_controller_and_engine(engine, self.robot)

        # Run simulation and extract log data
        x0 = np.array([0.1, 0.0])
        tf = 2.0
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)

        # Pendulum dynamics
        def dynamics(t, x):
            return np.array([x[1], self.g / self.l * np.sin(x[0])])

        # Integrate this non-linear dynamics
        x_rk_python = integrate_dynamics(time, x0, dynamics)

        # Compare the numerical and numerical integration of analytical model
        # using Scipy
        self.assertTrue(np.allclose(x_jiminy, x_rk_python, atol=TOLERANCE))

    def test_backlash(self):
        """Test adding a backlash to a pendulum, and make sure that:
        - while 'inside' the backlash, the pendulum behaves like no motor is
          present
        - once the backlash limit is reached, the pendulum behaves like a
          single body, with rotor and body inertia summed up.
        """
        # Configure the robot: add rotor inertia and backlash
        J = 1.0
        BACKLASH = 1.1
        robot_options = self.robot.get_options()
        robot_options["motors"]["PendulumJoint"]['enableArmature'] = True
        robot_options["motors"]["PendulumJoint"]['armature'] = J
        robot_options["motors"]["PendulumJoint"]["enableBacklash"] = True
        robot_options["motors"]["PendulumJoint"]["backlash"] = 2 * BACKLASH
        self.robot.set_options(robot_options)

        TAU = 5.0
        def ControllerConstant(t, q, v, sensor_measurements, command):
            command[:] = - TAU

        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, compute_command=ControllerConstant)

        engine_options = engine.get_options()
        engine_options["constraints"]["regularization"] = 0.0
        engine.set_options(engine_options)

        # Run simulation and extract log data
        x0 = np.array([0.0, 0.1, 0.0, 0.0])
        tf = 5.0
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)

        # Now we compare the two phases of the motion:
        #  - first phase: before impact: both bodies move independently
        #  - second phase: after impact: both bodies move as one (a tolerance
        #    of 0.4s is applied to make sure all rebounds are gone)
        t_impact = np.sqrt(BACKLASH / (TAU / J) * 2)
        t1, t2 = np.searchsorted(time, [t_impact - 0.02, t_impact + 0.4])
        time_phase_1 = time[:t1]
        x_jiminy_phase_1 = x_jiminy[:t1]
        def dynamics(t, x):
            # The angle of the pendulum mass is actually x[0] + x[1], not x[1]
            return np.array([x[2],
                             x[3],
                             -TAU / J,
                             self.g / self.l * np.sin(x[0] + x[1]) + TAU / J])
        x_rk_python = integrate_dynamics(time_phase_1, x0, dynamics)
        self.assertTrue(np.allclose(
            x_jiminy_phase_1, x_rk_python, atol=TOLERANCE))

        time_phase_2 = time[t2:] - time[t2]
        x_jiminy_phase_2 = x_jiminy[t2:]

        I_total = self.m * self.l ** 2 + J
        G = self.m * self.g * self.l / I_total
        def dynamics(t, x):
            acc = G * np.sin(x[0] + x[1]) - TAU / I_total
            return np.array([x[2], x[3], acc, 0.0])
        x_rk_python = integrate_dynamics(
            time_phase_2, x_jiminy_phase_2[0], dynamics)
        self.assertTrue(np.allclose(
            x_jiminy_phase_2, x_rk_python, atol=TOLERANCE))

        # Simulate with more damping on the constraint to prevent bouncing.
        # Note that this needs to be changed after the simulation start.
        engine_options["stepper"]["dtMax"] = 0.001
        engine.set_options(engine_options)
        engine.start(x0[:self.robot.nq], x0[-self.robot.nv:])
        self.robot.constraints.bounds_joints["PendulumJointBacklash"].kp = 1e5
        self.robot.constraints.bounds_joints["PendulumJointBacklash"].kd = 1e3
        engine.step(t_impact + 1.0)
        log_vars = engine.log_data["variables"]
        time = log_vars['Global.Time']
        q_jiminy = np.stack([
            log_vars[field] for field in self.robot.log_position_fieldnames
            ], axis=-1)
        v_jiminy = np.stack([
            log_vars[field] for field in self.robot.log_velocity_fieldnames
            ], axis=-1)
        x_jiminy = np.concatenate((q_jiminy, v_jiminy), axis=-1)

        # Ignoring some uncertainty at impact time due to the pendulum motion
        t2 = np.searchsorted(time, t_impact + 0.20)
        time_phase_2 = time[t2:] - time[t2]
        x_jiminy_phase_2 = x_jiminy[t2:]
        x_rk_python = integrate_dynamics(
            time_phase_2, x_jiminy_phase_2[0], dynamics)
        self.assertTrue(np.allclose(
            x_jiminy_phase_2, x_rk_python, atol=TOLERANCE))

    def test_imu_sensor(self):
        """Test IMU sensor on pendulum motion.

        .. note::
            Since we don't have a simple analytical expression for the solution
            of a (nonlinear) pendulum motion, we perform the simulation in
            python, with the same integrator.

        .. warning::
            The actual expected solution of the pendulum motion is used to
            compute the expected IMU data, instead of the result of the
            simulation done by jiminy itself. So this test is checking at the
            same time that the result of the simulation matches the solution,
            and that the sensor IMU data are valid. Though it is redundant, it
            validates that an IMU mounted on a pendulum gives the signal one
            would expect from an IMU on a pendulum, which is what a user would
            expect. Moreover, this test is indirectly checking that the
            acceleration computed by jiminy is valid.
        """
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        setup_controller_and_engine(engine, self.robot)

        # Run simulation and extract log data
        x0 = np.array([0.1, 0.1])
        tf = 2.0
        time, gyro_jiminy, accel_jiminy = \
            SimulateSimplePendulum._simulate_and_get_imu_data_evolution(
                engine, tf, x0, split=True)

        # Pendulum dynamics
        def dynamics(t: float, x: np.ndarray) -> np.ndarray:
            return np.stack(
                (x[..., 1], self.g / self.l * np.sin(x[..., 0])), axis=-1)

        # Integrate this non-linear dynamics
        x_rk_python = integrate_dynamics(time, x0, dynamics)

        # Compute sensor acceleration, i.e. acceleration in polar coordinates
        theta, dtheta = x_rk_python[:, 0], x_rk_python[:, 1]

        # Acceleration: to resolve algebraic loop (current acceleration is
        # function of input which itself is function of sensor signal, sensor
        # data is computed using q_t, v_t, a_t
        ddtheta = dynamics(0.0, x_rk_python)[:, 1]

        expected_accel = np.stack([
            - self.l * ddtheta + self.g * np.sin(theta),
            np.zeros_like(theta),
            self.l * dtheta ** 2 - self.g * np.cos(theta)], axis=-1)
        expected_gyro = np.stack([
            np.zeros_like(theta),
            dtheta,
            np.zeros_like(theta)], axis=-1)

        # Compare sensor signal, ignoring first iterations that correspond to
        # system initialization
        self.assertTrue(np.allclose(
            expected_gyro, gyro_jiminy, atol=TOLERANCE))
        self.assertTrue(np.allclose(
            expected_accel, accel_jiminy, atol=TOLERANCE))

    def test_sensor_delay(self):
        """Test sensor delay for an IMU sensor on a simple pendulum.
        """
        # Configure the IMU
        imu_options = self.imu_sensor.get_options()
        imu_options['delayInterpolationOrder'] = 0
        imu_options['delay'] = 0.0
        self.imu_sensor.set_options(imu_options)

        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        setup_controller_and_engine(engine, self.robot)

        # Configure the engine: No gravity + Continuous time simulation
        engine_options = engine.get_options()
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
        engine.set_options(engine_options)

        # Run simulation and extract imu data
        x0 = np.array([0.1, 0.0])
        tf = 2.0
        time, imu_jiminy = \
            SimulateSimplePendulum._simulate_and_get_imu_data_evolution(
                engine, tf, x0, split=False)

        # Deduce shifted imu data
        imu_jiminy_shifted_0 = interp1d(
            time, imu_jiminy, kind='zero',
            bounds_error=False, fill_value=imu_jiminy[0], axis=0
        )(time + 1e-10 - 1.0e-2)
        imu_jiminy_shifted_1 = interp1d(
            time, imu_jiminy,
            kind='linear', bounds_error=False, fill_value=imu_jiminy[0], axis=0
        )(time - 5.0e-3)

        # Configure the IMU
        imu_options = self.imu_sensor.get_options()
        imu_options['delayInterpolationOrder'] = 0
        imu_options['delay'] = 1.0e-2
        self.imu_sensor.set_options(imu_options)

        # Run simulation and extract imu data
        time, imu_jiminy_delayed_0 = \
            SimulateSimplePendulum._simulate_and_get_imu_data_evolution(
                engine, tf, x0, split=False)

        # Configure the IMU
        imu_options = self.imu_sensor.get_options()
        imu_options['delayInterpolationOrder'] = 1
        imu_options['delay'] = 5.0e-3
        self.imu_sensor.set_options(imu_options)

        # Run simulation
        time, imu_jiminy_delayed_1 = \
            SimulateSimplePendulum._simulate_and_get_imu_data_evolution(
                engine, tf, x0, split=False)

        # Compare sensor signals
        np.testing.assert_allclose(imu_jiminy_delayed_0, imu_jiminy_shifted_0)
        np.testing.assert_allclose(imu_jiminy_delayed_1, imu_jiminy_shifted_1)

    def test_sensor_noise_bias(self):
        """Test sensor noise and bias for an IMU sensor on a simple pendulum
        in static pose.
        """
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        setup_controller_and_engine(engine, self.robot)

        # Configure the IMU
        imu_options = self.imu_sensor.get_options()
        imu_options['noiseStd'] = np.array([0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
        imu_options['bias'] = np.array([0.1, 0.2, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7])
        self.imu_sensor.set_options(imu_options)

        # Run simulation and extract log data
        x0 = np.array([0.0, 0.0])
        tf = 200.0
        _, gyro_jiminy, accel_jiminy = \
            SimulateSimplePendulum._simulate_and_get_imu_data_evolution(
                engine, tf, x0, split=True)

        # Estimate the quaternion bias
        # Because the IMU rotation is identity, the resulting rotation will
        # simply be R_b and thus directly recover the unbiased sensor data.
        quat_rot_bias = exp3(imu_options['bias'][:3])
        accel_mean = np.mean(
            accel_jiminy, axis=0) - quat_rot_bias.T @ imu_options['bias'][-3:]
        acc = accel_mean / np.linalg.norm(accel_mean)
        axis = np.array((acc[1], -acc[0], 0.0))
        s = np.sqrt(2 * (1 + acc[2]))
        quat_bias = np.stack((*(axis / s), s / 2))
        quat_axis_bias = log3(Quaternion(quat_bias).matrix())

        # Remove sensor rotation bias from gyro / accel data
        gyro_jiminy = np.vstack([quat_rot_bias @ v for v in gyro_jiminy])
        accel_jiminy = np.vstack([quat_rot_bias @ v for v in accel_jiminy])

        # Estimate the gyroscope and accelerometer noise and bias
        gyro_bias = np.mean(gyro_jiminy, axis=0)
        accel_bias = np.mean(accel_jiminy, axis=0) - np.array([0.0, 0.0, 9.81])
        gyro_std = np.std(gyro_jiminy, axis=0)
        accel_std = np.std(accel_jiminy, axis=0)

        # Compare estimated sensor noise and bias with the configuration
        self.assertTrue(np.allclose(
            imu_options['bias'][:2], quat_axis_bias[:2], atol=1.0e-2))
        self.assertTrue(np.allclose(
            imu_options['bias'][3:-3], gyro_bias, atol=1.0e-2))
        self.assertTrue(np.allclose(
            imu_options['bias'][-3:], accel_bias, atol=1.0e-2))
        self.assertTrue(np.allclose(
            imu_options['noiseStd'][:3], gyro_std, atol=1.0e-2))
        self.assertTrue(np.allclose(
            imu_options['noiseStd'][-3:], accel_std, atol=1.0e-2))

    def test_pendulum_force_impulse(self):
        """Validate the impulse-momentum theorem

        The analytical expression for the solution is exact for impulse of
        force that are perfect dirac functions.
        """
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        setup_controller_and_engine(engine, self.robot)

        # Analytical solution
        def sys(t):
            q = 0.0
            v = 0.0
            for i, force in enumerate(F_register):
                if t > force["t"]:
                    pos = self.l * np.array([
                        -np.cos(q - np.pi / 2), 0.0, np.sin(q - np.pi / 2)])
                    n = pos / np.linalg.norm(pos)
                    d = np.cross(self.axis, n)
                    F_proj = force["F"][:3].T.dot(d)
                    v_delta = ((F_proj + force["F"][4] / self.l) * min(
                        force["dt"], t - force["t"])) / self.m
                    if (i < len(F_register) - 1):
                        q += (v + v_delta) * max(
                            0, min(t, F_register[i + 1]["t"]) -
                            (force["t"] + force["dt"]))
                    else:
                        q += (v + v_delta) * max(
                            0, t - force["t"] + force["dt"])
                    q += (v + v_delta/2) * min(
                        force["dt"], t - force["t"])
                    v += v_delta
                else:
                    break
            return np.array([q, v])

        # Register a set of impulse forces
        np.random.seed(0)
        F_register = [{"t": 0.0, "dt": 2.0e-3,
                       "F": np.array([1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0])},
                      {"t": 0.1, "dt": 1.0e-3,
                       "F": np.array([0.0, 1.0e3, 0.0, 0.0, 0.0, 0.0])},
                      {"t": 0.2, "dt": 2.0e-5,
                       "F": np.array([-1.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])},
                      {"t": 0.2, "dt": 2.0e-4,
                       "F": np.array([0.0, 0.0, 1.0e4, 0.0, 0.0, 0.0])},
                      {"t": 0.4, "dt": 1.0e-5,
                       "F": np.array([0.0, 0.0, 0.0, 0.0, 2.0e4, 0.0])},
                      {"t": 0.4, "dt": 1.0e-5,
                       "F": np.array([1.0e3, 1.0e4, 3.0e4, 0.0, 0.0, 0.0])},
                      {"t": 0.6, "dt": 1.0e-6,
                       "F": (2.0 * (np.random.rand(6) - 0.5)) * 4.0e6},
                      {"t": 0.8, "dt": 2.0e-6,
                       "F": np.array([0.0, 0.0, 2.0e5, 0.0, 0.0, 0.0])}]
        for force in F_register:
            engine.register_impulse_force(
                "", "PendulumLink", force["t"], force["dt"], force["F"])

        # Configure the engine: No gravity + Continuous time simulation
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine.set_options(engine_options)

        # Run simulation and extract some information from log data
        x0 = np.array([0.0, 0.0])
        tf = 1.0
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)

        # Compute the associated analytical solution
        x_analytical = np.stack([sys(t) for t in time], axis=0)

        # Check if t = t_start / t_end were breakpoints.
        # Note that the accuracy for the log is 1us.
        t_break_err = np.concatenate([np.array([
                min(abs(f["t"] - time)),
                min(abs(f["t"] + f["dt"] - time))])
            for f in F_register])
        self.assertTrue(np.allclose(t_break_err, 0.0, atol=TOLERANCE))

        # This test has a specific tolerance because the analytical solution is
        # an approximation since in practice, the external force is not
        # constant over its whole application duration but rather depends on
        # the orientation of the pole. For simplicity, the effect of the
        # impulse forces is assumed to be constant. As a result, the tolerance
        # cannot be tighter.
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=1e-6))

        # Configure the engine: No gravity + Discrete time simulation
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine.set_options(engine_options)

        # Configure the engine: Continuous time simulation
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
        engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
        engine.set_options(engine_options)

        # Run simulation
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)

        # Compute the associated analytical solution
        x_analytical = np.stack([sys(t) for t in time], axis=0)

        # Check if t = t_start / t_end were breakpoints
        t_break_err = np.concatenate([np.array([
                min(abs(f["t"] - time)),
                min(abs(f["t"] + f["dt"] - time))])
            for f in F_register])
        self.assertTrue(np.allclose(t_break_err, 0.0, atol=TOLERANCE))

        # Compare the numerical and analytical solution
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=1e-6))

    def test_flexibility_armature(self):
        """Test the addition of a flexibility in the system.

        This test asserts that, by adding a flexibility and a rotor inertia,
        the output is 'sufficiently close' to a SEA system:

        .. seealso::
            See 'note_on_flexibility_model.pdf' for more information as to why
            this is not a true equality.
        """
        # Physical parameters: rotor inertia, spring stiffness and damping.
        J = 0.1
        k = 20.0
        nu = 0.1

        # Enable flexibility
        model_options = self.robot.get_model_options()
        model_options["dynamics"]["enableFlexibility"] = True
        model_options["dynamics"]["flexibilityConfig"] = [{
            'frameName': "PendulumJoint",
            'stiffness': k * np.ones(3),
            'damping': nu * np.ones(3),
            'inertia': 1e-5 * np.ones(3)
        }]
        self.robot.set_model_options(model_options)

        # Enable rotor inertia
        robot_options = self.robot.get_options()
        robot_options["motors"]["PendulumJoint"]['enableArmature'] = True
        robot_options["motors"]["PendulumJoint"]['armature'] = J
        self.robot.set_options(robot_options)

        # Create an engine: PD controller on motor and no internal dynamics
        k_control, nu_control = 100.0, 1.0
        def compute_command(t, q, v, sensor_measurements, command):
            command[:] = - k_control * q[4] - nu_control * v[3]

        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, compute_command=compute_command)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine.set_options(engine_options)

        # Run simulation and extract some information from log data.
        # Note that to avoid having to handle angle conversion, start with an
        # initial velocity for the output mass.
        v_init = 0.1
        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, v_init, 0.0, 0.0])
        tf = 10.0
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)

        # Convert quaternion to RPY
        x_jiminy = np.stack([
            np.concatenate((
                matrixToRpy(Quaternion(x[:4][:, np.newaxis]).matrix()).astype(
                    x.dtype, copy=False),
                x[4:]
            )) for x in x_jiminy
        ], axis=0)

        # First, check that there was no motion other than along the Y axis.
        self.assertTrue(np.allclose(x_jiminy[:, [0, 2, 4, 6]], 0))

        # Now let's group x_jiminy to match the analytical system:
        # flexibility angle, pendulum angle, flexibility velocity, pendulum
        # velocity.
        x_jiminy_extract = x_jiminy[:, [1, 3, 5, 7]]

        # Simulate the system: a perfect SEA system.
        A = np.array([[                 0.0,            0.0,                   1.0,             0.0],
                      [                 0.0,            0.0,                   0.0,             1.0],
                      [-k * (1 / self.I + 1 / J),  k_control / J, -nu * (1 / self.I + 1 / J),  nu_control / J],
                      [                    k / J, -k_control / J,                     nu / J, -nu_control / J]])
        x_analytical = np.stack([
            scipy.linalg.expm(A * t).dot(x_jiminy_extract[0]) for t in time
            ], axis=0)

        # This test has a specific tolerance because we know the dynamics don't
        # exactly match: they are however very close, since the inertia of the
        # flexible element is negligible before I.
        self.assertTrue(np.allclose(
            x_jiminy_extract, x_analytical, atol=1e-4))

    def test_fixed_body_constraint_armature(self):
        """Test fixed body constraint together with rotor inertia.
        """
        # Create robot with freeflyer, set rotor inertia.
        robot = load_urdf_default(
            self.urdf_name, self.motor_names, has_freeflyer=True)

        # Enable rotor inertia
        J = 0.1
        robot_options = robot.get_options()
        robot_options["motors"]["PendulumJoint"]['enableArmature'] = True
        robot_options["motors"]["PendulumJoint"]['armature'] = J
        robot.set_options(robot_options)

        # Set fixed body constraint.
        freeflyer_constraint = jiminy.FrameConstraint("world")
        robot.add_constraint("world", freeflyer_constraint)

        # Create an engine: simulate a spring internal dynamics
        k_spring = 500
        def spring_force(t, q, v, sensor_measurements, u_custom):
            u_custom[:] = - k_spring * q[-1]

        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, robot, internal_dynamics=spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine_options["constraints"]["regularization"] = 0.0
        engine.set_options(engine_options)

        # Run simulation and extract some information from log data
        x0 = np.array([0.1, 0.0])
        qInit, vInit = neutral_state(robot, split=True)
        qInit[-1], vInit[-1] = x0
        xInit = np.concatenate((qInit, vInit))
        tf = 2.0
        time, q_jiminy, v_jiminy = simulate_and_get_state_evolution(
            engine, tf, xInit, split=True)

        # Analytical solution: dynamics should be unmodified by
        # the constraint, so we have a simple mass on a spring.
        I_eq = self.I + J
        A = np.array([[               0, 1],
                      [-k_spring / I_eq, 0]])
        x_analytical = np.stack([
            scipy.linalg.expm(A * t).dot(x0) for t in time], axis=0)

        self.assertTrue(np.allclose(
            q_jiminy[:, :-1], qInit[:-1], atol=TOLERANCE))
        self.assertTrue(np.allclose(
            v_jiminy[:, :-1], vInit[:-1], atol=TOLERANCE))
        self.assertTrue(np.allclose(
            q_jiminy[:, -1], x_analytical[:, 0], atol=TOLERANCE))
        self.assertTrue(np.allclose(
            v_jiminy[:, -1], x_analytical[:, 1], atol=TOLERANCE))


    def test_flexibility_api(self):
        """
        @brief Test the addition and disabling of a flexibility in the system.

        @details This function only tests that the flexibility API works, but
                 performs no validation of the physics behind it.
        """

        # Enable flexibility
        model_options = self.robot.get_model_options()
        model_options["dynamics"]["enableFlexibility"] = True
        model_options["dynamics"]["flexibilityConfig"] = [{
            'frameName': "PendulumJoint",
            'stiffness': np.ones(3),
            'damping': np.ones(3),
            'inertia': np.ones(3)
        }]
        self.robot.set_model_options(model_options)
        self.assertTrue(self.robot.flexibility_joint_indices == [1])

        engine = jiminy.Engine()
        engine.add_robot(self.robot)

        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        tf = 0.1
        simulate_and_get_state_evolution(engine, tf, x0, split=False)

        self.assertTrue(self.robot.flexibility_joint_indices == [1])

        # Disable flexibility
        model_options = self.robot.get_model_options()
        model_options["dynamics"]["enableFlexibility"] = False
        self.robot.set_model_options(model_options)
        self.assertTrue(self.robot.flexibility_joint_indices == [])
        x0 = np.array([0.0, 0.0])
        simulate_and_get_state_evolution(engine, tf, x0, split=False)
        self.assertTrue(self.robot.flexibility_joint_indices == [])

        # Re-enable flexibility
        model_options = self.robot.get_model_options()
        model_options["dynamics"]["enableFlexibility"] = True
        self.robot.set_model_options(model_options)
        self.assertTrue(self.robot.flexibility_joint_indices == [1])

        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        simulate_and_get_state_evolution(engine, tf, x0, split=False)

        self.assertTrue(self.robot.flexibility_joint_indices == [1])

        # Test empty flexibility list
        model_options = self.robot.get_model_options()
        model_options["dynamics"]["enableFlexibility"] = True
        model_options["dynamics"]["flexibilityConfig"] = []
        self.robot.set_model_options(model_options)
        self.assertTrue(self.robot.flexibility_joint_indices == [])

        x0 = np.array([0.0, 0.0])
        simulate_and_get_state_evolution(engine, tf, x0, split=False)

        self.assertTrue(self.robot.flexibility_joint_indices == [])

if __name__ == '__main__':
    unittest.main()
