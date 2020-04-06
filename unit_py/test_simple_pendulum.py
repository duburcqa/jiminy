# This file aims at verifying the sanity of the physics and the integration method of
# jiminy on simple models.
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import ode

from jiminy_py import core as jiminy
from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy


# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class SimulateSimplePendulum(unittest.TestCase):
    '''
    @brief Simulate the motion of a pendulum, comparing against python integration.
    '''
    def setUp(self):
        # Load URDF, create model.
        urdf_path = "data/simple_pendulum.urdf"

        # Create the jiminy model

        # Instanciate model and engine
        self.robot = jiminy.Robot()
        self.robot.initialize(urdf_path, has_freeflyer=False)
        motor = jiminy.SimpleMotor("PendulumJoint")
        self.robot.attach_motor(motor)
        motor.initialize("PendulumJoint")

        # Configure model.
        model_options = self.robot.get_model_options()
        motor_options = self.robot.get_motors_options()
        model_options["joints"]["enablePositionLimit"] = False
        model_options["joints"]["enableVelocityLimit"] = False
        for m in motor_options:
            motor_options[m]['enableTorqueLimit'] = False
            motor_options[m]['enableRotorInertia'] = False
        self.robot.set_model_options(model_options)
        self.robot.set_motors_options(motor_options)

    def test_rotor_inertia(self):
        '''
        @brief Verify the dynamics of the system when adding  rotor inertia.
        '''
        # No controller
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = 0.0

        # Dynamics: simulate a spring of stifness k
        k_spring = 500
        def internalDynamics(t, q, v, sensor_data, u):
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
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Analytical solution: a simple mass on a spring.
        pnc_model = self.robot.pinocchio_model_th
        I = pnc_model.inertias[1].mass * pnc_model.inertias[1].lever[2] ** 2

        # Write system dynamics
        I_eq = I + J
        A = np.array([[               0, 1],
                      [-k_spring / I_eq, 0]])
        x_analytical = np.stack([expm(A * t) @ x0 for t in time], axis=0)

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_pendulum_integration(self):
        '''
        @brief   Compare pendulum motion, as simulated by Jiminy, against an
                 equivalent simulation done in python.

        @details Since we don't have a simple analytical expression for the solution
                 of a (nonlinear) pendulum motion, we perform the simulation in
                 python, with the same integrator, and compare both results.
        '''
        # Create an engine: no controller and no internal dynamics
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        x0 = np.array([0.1, 0.0])
        tf = 2.0

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # System dynamics: get length and inertia.
        l = -self.robot.pinocchio_model_th.inertias[1].lever[2]
        g = self.robot.pinocchio_model.gravity.linear[2]

        # Pendulum dynamics
        def dynamics(t, x):
            return np.array([x[1], g / l * np.sin(x[0])])

        # Integrate, using same Runge-Kutta integrator.
        solver = ode(dynamics)
        solver.set_initial_value(x0)
        solver.set_integrator("dopri5")
        x_rk_python = [x0]
        for t in time[1:]:
            solver.integrate(t)
            x_rk_python.append(solver.y)
        x_rk_python = np.stack(x_rk_python, axis=0)

        # Compare the numerical and numerical integration of analytical model using scipy
        self.assertTrue(np.allclose(x_jiminy, x_rk_python, atol=TOLERANCE))

    def test_pendulum_force_impulse(self):
        '''
        @brief   Validate the impulse-momentum theorem

        @details The analytical expression for the solution is exact for
                 impulse of force that are perfect dirac functions.
        '''
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
                    F_proj = F_register[i]["F"][:3].T @ d
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

         # Compare the numerical and analytical solution
        TOLERANCE = 1e-6
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_flexibility_rotor_inertia(self):
        '''
        @brief Test the addition of a flexibility in the system.

        @details This test asserts that, by adding a flexibility and a rotor inertia,
                 the output is 'sufficiently close' to a SEA system:
                 see 'note_on_flexibli_model.pdf' for more information as to why this
                 is not a true equality.
        '''
        # Controller: PD controller on motor.
        k_control = 100.0
        nu_control = 1.0
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = -k_control * q[4] - nu_control * v[3]

        def internalDynamics(t, q, v, sensor_data, u):
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
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
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
        x_analytical = np.stack([expm(A * t) @ x_jiminy_extract[0] for t in time], axis=0)

        # This test has a specific tolerance because we know the dynamics don't exactly
        # match: they are however very close, since the inertia of the flexible element
        # is negligible before I.
        TOLERANCE = 1e-4
        self.assertTrue(np.allclose(x_jiminy_extract, x_analytical, atol=TOLERANCE))

if __name__ == '__main__':
    unittest.main()
