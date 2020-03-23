# This file aims at verifying the sanity of the physics and the integration method of jiminy
# on simple models.
import unittest
import os
import numpy as np
import scipy.linalg
import scipy.integrate
from jiminy_py import core as jiminy
import pinocchio as pnc

# Unit test precision threshold.
# This tolerance is needed because we log time with a precision of 1us,
# whereas jiminy sometimes takes steps that are slightly off the desired
# frequency, leading to inconsistent times  in the log (with an error bounded
# by 1us, so it's not important in practice but it does hinder matching here).
TOLERANCE = 5e-4

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
        x_jiminy = np.array([log_data['HighLevelController.' + s] for s in self.robot.logfile_position_headers + self.robot.logfile_velocity_headers]).T

        # Analytical solution: a simple mass on a spring.
        I = self.robot.pinocchio_model_th.inertias[1].mass * self.robot.pinocchio_model_th.inertias[1].lever[2]**2

        # Write system dynamics
        I_eq = I + J
        A = np.array([[0,                1],
                      [-k_spring / I_eq, 0]])
        x_analytical = np.array([scipy.linalg.expm(A * t) @ x0 for t in time])

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol = TOLERANCE))

    def test_pendulum_integration(self):
        '''
        @brief Compare pendulum motion, as simulated by Jiminy, against an equivalent simulation done in python.

        @details Since we don't have a simple analytical expression for the solution of a (nonlinear) pendulum motion,
                 we perform the simulation in python, with the same integrator, and compare both results.
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
        x_jiminy = np.array([log_data['HighLevelController.' + s] for s in self.robot.logfile_position_headers + self.robot.logfile_velocity_headers]).T

        # System dynamics: get length and inertia.
        l = -self.robot.pinocchio_model_th.inertias[1].lever[2]
        g = 9.81
        # Pendulum dynamics
        def dynamics(t, x):
            return np.array([x[1], - g / l * np.sin(x[0])])
        # Integrate, using same Runge-Kutta integrator.
        solver = scipy.integrate.ode(dynamics)
        solver.set_initial_value(x0)
        solver.set_integrator("dopri5")
        x_rk_python = [x0]
        for t in time[1:]:
            solver.integrate(t)
            x_rk_python.append(solver.y)
        x_rk_python = np.array(x_rk_python)
        self.assertTrue(np.allclose(x_jiminy, x_rk_python, atol = TOLERANCE))

if __name__ == '__main__':
    unittest.main()
