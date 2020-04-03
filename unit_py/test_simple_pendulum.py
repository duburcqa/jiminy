# This file aims at verifying the sanity of the physics and the integration method of
# jiminy on simple models.
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import ode

from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
from jiminy_py import core as jiminy


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
        # No controller and no internal dynamics
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internalDynamics(t, q, v, sensor_data, u):
            u[:] = 0.0

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

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
        g = 9.81

        # Pendulum dynamics
        def dynamics(t, x):
            return np.array([x[1], - g / l * np.sin(x[0])])

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
        @brief   Compare pendulum motion, as simulated by Jiminy, against an
                 equivalent simulation done in python.

        @details Since we don't have a simple analytical expression for the solution
                 of a (nonlinear) pendulum motion, we perform the simulation in
                 python, with the same integrator, and compare both results.
        '''
        # No controller and no internal dynamics
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internalDynamics(t, q, v, sensor_data, u):
            u[:] = 0.0

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        # Configure the engine: No gravity + Continuous time simulation
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine.set_options(engine_options)

        x0 = np.array([0.0, 0.0])
        tf = 2.0

        # Run simulation
        engine.simulate(tf, x0)
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)



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
