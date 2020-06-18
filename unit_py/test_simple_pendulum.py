# This file aims at verifying the sanity of the physics and the integration method of
# jiminy on simple models.
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import ode
from scipy.interpolate import interp1d

from jiminy_py import core as jiminy
from pinocchio import Quaternion, log3, exp3
from pinocchio.rpy import matrixToRpy, rpyToMatrix

from utilities import load_urdf_default, integrate_dynamics

# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1.0e-7


class SimulateSimplePendulum(unittest.TestCase):
    """
    @brief Simulate the motion of a pendulum, comparing against python integration.
    """
    def setUp(self):
        # Load URDF, create model.
        self.urdf_path = "data/simple_pendulum.urdf"

        # Create the jiminy model

        # Instanciate model and engine
        self.robot = load_urdf_default(self.urdf_path, ["PendulumJoint"])

    def test_rotor_inertia(self):
        """
        @brief Verify the dynamics of the system when adding  rotor inertia.
        """
        # No controller
        def computeCommand(t, q, v, sensors_data, u):
            u[:] = 0.0

        # Dynamics: simulate a spring of stiffness k
        k_spring = 500
        def internalDynamics(t, q, v, sensors_data, u):
            u[:] = - k_spring * q[:]

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)

        # Set rotor inertia
        J = 0.1
        motor_options = self.robot.get_motors_options()
        motor_options["PendulumJoint"]['enableRotorInertia'] = True
        motor_options["PendulumJoint"]['rotorInertia'] = J
        self.robot.set_motors_options(motor_options)

        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6) # Turn off gravity
        engine.set_options(engine_options)

        x0 = np.array([0.1, 0.0])
        tf = 2.0

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Analytical solution: a simple mass on a spring
        pnc_model = self.robot.pinocchio_model_th
        I = pnc_model.inertias[1].mass * pnc_model.inertias[1].lever[2] ** 2

        # Write system dynamics
        I_eq = I + J
        A = np.array([[               0, 1],
                      [-k_spring / I_eq, 0]])
        x_analytical = np.stack([expm(A * t).dot(x0) for t in time], axis=0)

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_pendulum_integration(self):
        """
        @brief   Compare pendulum motion, as simulated by Jiminy, against an
                 equivalent simulation done in python.

        @details Since we don't have a simple analytical expression for the solution
                 of a (nonlinear) pendulum motion, we perform the simulation in
                 python, with the same integrator, and compare both results.
        """
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        x0 = np.array([0.1, 0.0])
        tf = 2.0

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # System dynamics: get length and inertia
        l = -self.robot.pinocchio_model_th.inertias[1].lever[2]
        g = self.robot.pinocchio_model.gravity.linear[2]

        # Pendulum dynamics
        def dynamics(t, x):
            return np.array([x[1], g / l * np.sin(x[0])])

        # Integrate this non-linear dynamics
        x_rk_python = integrate_dynamics(time, x0, dynamics)

        # Compare the numerical and numerical integration of analytical model using scipy
        self.assertTrue(np.allclose(x_jiminy, x_rk_python, atol=TOLERANCE))

    def test_imu_sensor(self):
        """
        @brief   Test IMU sensor on pendulum motion.

        @details Note that the actual expected solution of the pendulum motion is
                 used to compute the expected IMU data, instead of the result of
                 the simulation done by jiminy itself. So this test is checking at
                 the same time that the result of the simulation matches the
                 solution, and that the sensor IMU data are valid. Though it is
                 redundant, it validates that an IMU mounted on a pendulum gives
                 the signal one would expect from an IMU on a pendulum, which is
                 what a user would expect. Moreover, Jiminy output log does not
                 feature the acceleration - to this test is indirectly checking
                 that the acceleration computed by jiminy is valid.

        @remark  Since we don't have a simple analytical expression for the
                 solution of a (nonlinear) pendulum motion, we perform the
                 simulation in python, with the same integrator.
        """
        # Add IMU
        imu_sensor = jiminy.ImuSensor("PendulumLink")
        self.robot.attach_sensor(imu_sensor)
        imu_sensor.initialize("PendulumLink")

        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        x0 = np.array([0.1, 0.1])
        tf = 2.0

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        quat_jiminy = np.stack([
            log_data['PendulumLink.Quat' + s] for s in ['x', 'y', 'z', 'w']
        ], axis=-1)
        gyro_jiminy = np.stack([
            log_data['PendulumLink.Gyro' + s] for s in ['x', 'y', 'z']
        ], axis=-1)
        accel_jiminy = np.stack([
            log_data['PendulumLink.Accel' + s] for s in ['x', 'y', 'z']
        ], axis=-1)

        # System dynamics: get length and inertia
        l = -self.robot.pinocchio_model_th.inertias[1].lever[2]
        g = self.robot.pinocchio_model.gravity.linear[2]

        # Pendulum dynamics
        def dynamics(t, x):
            return np.stack([x[..., 1], g / l * np.sin(x[..., 0])], axis=-1)

        # Integrate this non-linear dynamics
        x_rk_python = integrate_dynamics(time, x0, dynamics)

        # Compute sensor acceleration, i.e. acceleration in polar coordinates
        theta = x_rk_python[:, 0]
        dtheta = x_rk_python[:, 1]

        # Acceleration: to resolve algebraic loop (current acceleration is
        # function of input which itself is function of sensor signal, sensor
        # data is computed using q_t, v_t, a_(t-1)
        ddtheta = np.concatenate((np.zeros(1), dynamics(0.0, x_rk_python)[:-1, 1]))

        expected_accel = np.stack([- l * ddtheta + g * np.sin(theta),
                                   np.zeros_like(theta),
                                   l * dtheta ** 2 - g * np.cos(theta)], axis=-1)
        expected_gyro = np.stack([np.zeros_like(theta),
                                  dtheta,
                                  np.zeros_like(theta)], axis=-1)

        expected_quat = np.stack([
            Quaternion(rpyToMatrix(np.array([0., t, 0.]))).coeffs()
            for t in theta
        ], axis=0)

        # Compare sensor signal, ignoring first iterations that correspond to system initialization
        self.assertTrue(np.allclose(expected_quat[2:, :], quat_jiminy[2:, :], atol=TOLERANCE))
        self.assertTrue(np.allclose(expected_gyro[2:, :], gyro_jiminy[2:, :], atol=TOLERANCE))
        self.assertTrue(np.allclose(expected_accel[2:, :], accel_jiminy[2:, :], atol=TOLERANCE))

    def test_sensor_delay(self):
        """
        @brief   Test sensor delay for an IMU sensor on a simple pendulum.
        """
        # Add IMU.
        imu_sensor = jiminy.ImuSensor("PendulumLink")
        self.robot.attach_sensor(imu_sensor)
        imu_sensor.initialize("PendulumLink")

        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        # Configure the engine: No gravity + Continuous time simulation
        engine_options = engine.get_options()
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
        engine.set_options(engine_options)

        x0 = np.array([0.1, 0.0])
        tf = 2.0

        # Configure the IMU
        imu_options = imu_sensor.get_options()
        imu_options['delayInterpolationOrder'] = 0
        imu_options['delay'] = 0.0
        imu_sensor.set_options(imu_options)

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        imu_jiminy = np.stack([
            log_data['PendulumLink.' + f] for f in jiminy.ImuSensor.fieldnames
        ], axis=-1)
        imu_jiminy_shifted_0 = interp1d(
            time, imu_jiminy, kind='zero',
            bounds_error=False, fill_value=imu_jiminy[0], axis=0
        )(time - 1.0e-2)
        imu_jiminy_shifted_1 = interp1d(
            time, imu_jiminy,
            kind='linear', bounds_error=False, fill_value=imu_jiminy[0], axis=0
        )(time - 1.0e-2)

        # Configure the IMU
        imu_options = imu_sensor.get_options()
        imu_options['delayInterpolationOrder'] = 0
        imu_options['delay'] = 1.0e-2
        imu_sensor.set_options(imu_options)

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        imu_jiminy_delayed_0 = np.stack([
            log_data['PendulumLink.' + f] for f in jiminy.ImuSensor.fieldnames
        ], axis=-1)

        # Configure the IMU
        imu_options = imu_sensor.get_options()
        imu_options['delayInterpolationOrder'] = 1
        imu_options['delay'] = 1.0e-2
        imu_sensor.set_options(imu_options)

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        imu_jiminy_delayed_1 = np.stack([
            log_data['PendulumLink.' + f] for f in jiminy.ImuSensor.fieldnames
        ], axis=-1)

        # Compare sensor signals
        self.assertTrue(np.mean(imu_jiminy_delayed_0 - imu_jiminy_shifted_0) < 1.0e-5)
        self.assertTrue(np.allclose(imu_jiminy_delayed_1, imu_jiminy_shifted_1, atol=TOLERANCE))

    def test_sensor_noise_bias(self):
        """
        @brief   Test sensor noise and bias for an IMU sensor on a simple pendulum in static pose.
        """
        # Add IMU.
        imu_sensor = jiminy.ImuSensor("PendulumLink")
        self.robot.attach_sensor(imu_sensor)
        imu_sensor.initialize("PendulumLink")

        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        x0 = np.array([0.0, 0.0])
        tf = 200.0

        # Configure the engine: No gravity
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine.set_options(engine_options)

        # Configure the IMU
        imu_options = imu_sensor.get_options()
        imu_options['noiseStd'] = np.linspace(0.0, 0.2, 9)
        imu_options['bias'] = np.linspace(0.0, 1.0, 9)
        imu_sensor.set_options(imu_options)

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        quat_jiminy = np.stack([
            log_data['PendulumLink.Quat' + s] for s in ['x', 'y', 'z', 'w']
        ], axis=-1)
        gyro_jiminy = np.stack([
            log_data['PendulumLink.Gyro' + s] for s in ['x', 'y', 'z']
        ], axis=-1)
        accel_jiminy = np.stack([
            log_data['PendulumLink.Accel' + s] for s in ['x', 'y', 'z']
        ], axis=-1)

        # Convert quaternion to a rotation vector.
        quat_axis = np.stack([log3(Quaternion(q[:, np.newaxis]).matrix())
                              for q in quat_jiminy], axis=0)

        # Estimate the quaternion noise and bias
        # Because the IMU rotation is identity, the resulting rotation will
        # simply be R_b R_noise. Since R_noise is a small rotation, we can
        # consider that the resulting rotation is simply the rotation resulting
        # from the sum of the rotation vector (this is only true at the first
        # order) and thus directly recover the unbiased sensor data.
        quat_axis_bias = np.mean(quat_axis, axis=0)
        quat_axis_std = np.std(quat_axis, axis=0)

        # Remove sensor rotation bias from gyro / accel data
        quat_rot_bias = exp3(quat_axis_bias)
        gyro_jiminy = np.vstack([quat_rot_bias @ v for v in gyro_jiminy])
        accel_jiminy = np.vstack([quat_rot_bias @ v for v in accel_jiminy])

        # Estimate the gyroscope and accelerometer noise and bias
        gyro_std = np.std(gyro_jiminy, axis=0)
        gyro_bias = np.mean(gyro_jiminy, axis=0)
        accel_std = np.std(accel_jiminy, axis=0)
        accel_bias = np.mean(accel_jiminy, axis=0)

        # Compare estimated sensor noise and bias with the configuration
        self.assertTrue(np.allclose(imu_options['noiseStd'][:3], quat_axis_std, atol=1.0e-2))
        self.assertTrue(np.allclose(imu_options['bias'][:3], quat_axis_bias, atol=1.0e-2))
        self.assertTrue(np.allclose(imu_options['noiseStd'][3:-3], gyro_std, atol=1.0e-2))
        self.assertTrue(np.allclose(imu_options['bias'][3:-3], gyro_bias, atol=1.0e-2))
        self.assertTrue(np.allclose(imu_options['noiseStd'][-3:], accel_std, atol=1.0e-2))
        self.assertTrue(np.allclose(imu_options['bias'][-3:], accel_bias, atol=1.0e-2))

    def test_pendulum_force_impulse(self):
        """
        @brief   Validate the impulse-momentum theorem

        @details The analytical expression for the solution is exact for
                 impulse of force that are perfect dirac functions.
        """
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        # Analytical solution
        pnc_model = self.robot.pinocchio_model_th
        mass = pnc_model.inertias[1].mass
        length = abs(pnc_model.inertias[1].lever[2])
        axis = np.array([0.0, 1.0, 0.0])
        def sys(t):
            q = 0.0
            v = 0.0
            for i in range(len(F_register)):
                if t > F_register[i]["t"]:
                    pos = length * np.array([-np.cos(q - np.pi / 2), 0.0, np.sin(q - np.pi / 2)])
                    n = pos / np.linalg.norm(pos)
                    d = np.cross(axis, n)
                    F_proj = F_register[i]["F"][:3].T.dot(d)
                    v_delta = ((F_proj + F_register[i]["F"][4] / length) * min(F_register[i]["dt"], t - F_register[i]["t"])) / mass
                    if (i < len(F_register) - 1):
                        q += (v + v_delta) * max(0, min(t, F_register[i+1]["t"]) - (F_register[i]["t"] + F_register[i]["dt"]))
                    else:
                        q += (v + v_delta) * max(0, t - F_register[i]["t"] + F_register[i]["dt"])
                    q += (v + v_delta/2) * min(F_register[i]["dt"], t - F_register[i]["t"])
                    v += v_delta
                else:
                    break
            return np.array([q, v])

        # Register a set of impulse forces
        np.random.seed(0)
        F_register = [{"t": 0.0, "dt": 2.0e-3, "F": np.array([1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0])},
                      {"t": 0.1, "dt": 1.0e-3, "F": np.array([0.0, 1.0e3, 0.0, 0.0, 0.0, 0.0])},
                      {"t": 0.2, "dt": 2.0e-5, "F": np.array([-1.0e5, 0.0, 0.0, 0.0, 0.0, 0.0])},
                      {"t": 0.2, "dt": 2.0e-4, "F": np.array([0.0, 0.0, 1.0e4, 0.0, 0.0, 0.0])},
                      {"t": 0.4, "dt": 1.0e-5, "F": np.array([0.0, 0.0, 0.0, 0.0, 2.0e4, 0.0])},
                      {"t": 0.4, "dt": 1.0e-5, "F": np.array([1.0e3, 1.0e4, 3.0e4, 0.0, 0.0, 0.0])},
                      {"t": 0.6, "dt": 1.0e-6, "F": (2.0 * (np.random.rand(6) - 0.5)) * 4.0e6},
                      {"t": 0.8, "dt": 2.0e-6, "F": np.array([0.0, 0.0, 2.0e5, 0.0, 0.0, 0.0])}]
        for f in F_register:
            engine.register_force_impulse("PendulumLink", f["t"], f["dt"], f["F"])

        # Set the initial state and simulation duration
        x0 = np.array([0.0, 0.0])
        tf = 1.0

        # Configure the engine: No gravity + Continuous time simulation
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine.set_options(engine_options)

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Compute the associated analytical solution
        x_analytical = np.stack([sys(t) for t in time], axis=0)

        # Check if t = t_start / t_end were breakpoints (the accuracy for the log is 1us)
        t_break_err = np.concatenate([np.array([min(abs(f["t"] - log_data['Global.Time'])),
                                                min(abs(f["t"] + f["dt"] - log_data['Global.Time']))])
                                      for f in F_register])
        self.assertTrue(np.allclose(t_break_err, 0.0, atol=1e-12))

        # This test has a specific tolerance because the analytical solution is an
        # approximation since in practice, the external force is not constant over
        # its whole application duration but rather depends on the orientation of
        # the pole. For simplicity, the effect of the impulse forces is assumed
        # to be constant. As a result, the tolerance cannot be tighter.
        TOLERANCE = 1e-6
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

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
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Compute the associated analytical solution
        x_analytical = np.stack([sys(t) for t in time], axis=0)

        # Check if t = t_start / t_end were breakpoints (the accuracy for the log is 1us)
        t_break_err = np.concatenate([np.array([min(abs(f["t"] - log_data['Global.Time'])),
                                                min(abs(f["t"] + f["dt"] - log_data['Global.Time']))])
                                      for f in F_register])
        self.assertTrue(np.allclose(t_break_err, 0.0, atol=1e-12))

         # Compare the numerical and analytical solution
        TOLERANCE = 1e-6
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_flexibility_rotor_inertia(self):
        """
        @brief Test the addition of a flexibility in the system.

        @details This test asserts that, by adding a flexibility and a rotor inertia,
                 the output is 'sufficiently close' to a SEA system:
                 see 'note_on_flexibli_model.pdf' for more information as to why this
                 is not a true equality.
        """
        # Controller: PD controller on motor.
        k_control = 100.0
        nu_control = 1.0
        def computeCommand(t, q, v, sensors_data, u):
            u[:] = -k_control * q[4] - nu_control * v[3]

        def internalDynamics(t, q, v, sensors_data, u):
            u[:] = 0.0

        # Physical parameters: rotor inertia, spring stiffness and damping.
        J = 0.1
        k = 20.0
        nu = 0.1

        # Enable flexibility
        model_options = self.robot.get_model_options()
        model_options["dynamics"]["enableFlexibleModel"] = True
        model_options["dynamics"]["flexibilityConfig"] = [{'jointName': "PendulumJoint",
                                                           'stiffness': k * np.ones(3),
                                                           'damping': nu * np.ones(3)}]
        self.robot.set_model_options(model_options)
        # Enable rotor inertia
        motor_options = self.robot.get_motors_options()
        motor_options["PendulumJoint"]['enableRotorInertia'] = True
        motor_options["PendulumJoint"]['rotorInertia'] = J
        self.robot.set_motors_options(motor_options)

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6) # Turn off gravity
        engine.set_options(engine_options)

        # To avoid having to handle angle conversion,
        # start with an initial velocity for the output mass.
        v_init = 0.1
        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, v_init, 0.0, 0.0])
        tf = 10.0

        # Run simulation
        engine.simulate(tf, x0)

        # Get log data
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Convert quaternion to RPY
        x_jiminy = np.stack([
            np.concatenate((
                matrixToRpy(Quaternion(x[:4][:, np.newaxis]).matrix()).astype(x.dtype, copy=False),
                x[4:]
            )) for x in x_jiminy
        ], axis=0)

        # First, check that there was no motion other than along the Y axis.
        self.assertTrue(np.allclose(x_jiminy[:, [0, 2, 4, 6]], 0))

        # Now let's group x_jiminy to match the analytical system:
        # flexibility angle, pendulum angle, flexibility velocity, pendulum velocity
        x_jiminy_extract = x_jiminy[:, [1, 3, 5, 7]]

        # And let's simulate the system: a perfect SEA system.
        pnc_model = self.robot.pinocchio_model_th
        I = pnc_model.inertias[1].mass * pnc_model.inertias[1].lever[2] ** 2

        # Write system dynamics
        A = np.array([[0,                 0,                1,                                  0],
                      [0,                 0,                0,                                  1],
                      [-k * (1 / I + 1 / J), k_control / J, -nu * (1 / I + 1 / J), nu_control / J],
                      [               k / J,-k_control / J,                nu / J,-nu_control / J]])
        x_analytical = np.stack([expm(A * t).dot(x_jiminy_extract[0]) for t in time], axis=0)

        # This test has a specific tolerance because we know the dynamics don't exactly
        # match: they are however very close, since the inertia of the flexible element
        # is negligible before I.
        TOLERANCE = 1e-4
        self.assertTrue(np.allclose(x_jiminy_extract, x_analytical, atol=TOLERANCE))


    def test_fixed_body_constraint_rotor_inertia(self):
        """
        @brief Test fixed body constraint together with rotor inertia.
        """
        # Create robot with freeflyer, set rotor inertia.
        self.robot = load_urdf_default(self.urdf_path, ["PendulumJoint"])
        J = 0.1
        motor_options = self.robot.get_motors_options()
        motor_options["PendulumJoint"]['enableRotorInertia'] = True
        motor_options["PendulumJoint"]['rotorInertia'] = J
        self.robot.set_motors_options(motor_options)

        # No controller
        def computeCommand(t, q, v, sensors_data, u):
            u[:] = 0.0

        # Dynamics: simulate a spring of stifness k
        k_spring = 500
        def internalDynamics(t, q, v, sensors_data, u):
            u[:] = - k_spring * q[:]

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)

        # Set fixed body constraint.
        freeflyer_constraint = jiminy.FixedFrameConstraint("world")
        self.robot.add_constraint("world", freeflyer_constraint)

        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6) # Turn off gravity
        engine.set_options(engine_options)

        x0 = np.array([0.1, 0.0])
        tf = 2.0
        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Analytical solution: dynamics should be unmodifed by
        # the constraint, so we have a simple mass on a spring.
        pnc_model = self.robot.pinocchio_model_th
        I = pnc_model.inertias[1].mass * pnc_model.inertias[1].lever[2] ** 2

        # Write system dynamics
        I_eq = I + J
        A = np.array([[               0, 1],
                      [-k_spring / I_eq, 0]])
        x_analytical = np.stack([expm(A * t).dot(x0) for t in time], axis=0)

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

if __name__ == '__main__':
    unittest.main()
