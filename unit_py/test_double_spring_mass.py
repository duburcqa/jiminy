# This file aims at verifying the sanity of the physics and the integration method of jiminy
# on simple models.
import unittest
import os
import numpy as np
import scipy.linalg
from jiminy_py import core as jiminy


# Unit test precision threshold.
# This 'high' tolerance is needed because we log time with a precision of 1us, whereas jiminy sometimes
# takes steps that are slightly off the desired frequency, leading to inconsistant times
# in the log (with an error bounded by 1us, so it's not important in practice but it does hinder matching here).
TOLERANCE = 5e-4
class SimulateTwoMasses(unittest.TestCase):
    '''
    @brief Simulate the motion of two masses held together by a spring damper, and compare with the analytical solution.

    @details The system simulated can be represented as such:
                  k1   M1   k2   M2
             //| <><> |__| <><> |__|
    '''
    def setUp(self):
        # Load URDF, create model.
        urdf_path = "data/linear_two_masses.urdf"

        # Specify spring stiffness and damping for this simulation
        self.k = np.array([200, 20])
        self.nu = np.array([0.1, 0.2])
        # Define initial state and simulation duration
        self.x0 = np.array([0.1, -0.1, 0.0, 0.0])
        self.tf = 10.0

        # Create the jiminy model

        # Instanciate model and engine
        self.model = jiminy.Model()
        self.model.initialize(urdf_path, has_freeflyer=False)
        for joint_name in ["FirstJoint", "SecondJoint"]:
            motor = jiminy.SimpleMotor(joint_name)
            self.model.attach_motor(motor)
            motor.initialize(joint_name)

        # Configure model.
        model_options = self.model.get_model_options()
        motor_options = self.model.get_motors_options()
        model_options["joints"]["enablePositionLimit"] = False
        model_options["joints"]["enableVelocityLimit"] = False
        for m in motor_options:
            motor_options[m]['enableTorqueLimit'] = False
            motor_options[m]['enableRotorInertia'] = False
        self.model.set_model_options(model_options)
        self.model.set_motors_options(motor_options)

        # Comute the matrix representing this linear system for analytical computation
        m = np.array([self.model.pinocchio_model_th.inertias[i].mass  for i in range(1,3)])

        # Dynamics is linear: dX = A X
        I =  (1 / m[1] + 1 / m[0])
        self.A = np.array([[0,                 0,                1,                  0],
                           [0,                 0,                0,                  1],
                           [-self.k[0] / m[0], self.k[1] / m[0], -self.nu[0] / m[0], self.nu[1] / m[0]],
                           [self.k[0] / m[0],  -self.k[1] * I,   self.nu[0] / m[0],  -self.nu[1] * I]])

    def test_continuous_simulation(self):
        '''
        @brief Test simualtion of this system using a continuous controller.
        '''
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = - self.k * q - self.nu * v

        def internalDynamics(t, q, v, sensor_data, u):
            u[:] = 0.0

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.model)
        engine = jiminy.Engine()
        engine.initialize(self.model, controller)

        engine_options = engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine.set_options(engine_options)

        # Run simulation
        engine.simulate(self.tf, self.x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.array([log_data['HighLevelController.' + s] for s in self.model.logfile_position_headers + self.model.logfile_velocity_headers]).T

        # Compute analytical solution
        x_analytical = np.array([scipy.linalg.expm(self.A * t) @ self.x0 for t in time])

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol = TOLERANCE))

    def test_discrete_simulation(self):
        '''
        @brief Test simualtion of this system using internal dynamics + a discrete controller update.
        '''
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internalDynamics(t, q, v, sensor_data, u):
            u[:] = - self.k * q - self.nu * v

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.model)
        engine = jiminy.Engine()
        engine.initialize(self.model, controller)

        engine_options = engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1e-3
        engine_options["stepper"]["controllerUpdatePeriod"] = 1e-3
        engine.set_options(engine_options)

        # Run simulation
        engine.simulate(self.tf, self.x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.array([log_data['HighLevelController.' + s] for s in self.model.logfile_position_headers + self.model.logfile_velocity_headers]).T

        # Compute analytical solution
        x_analytical = np.array([scipy.linalg.expm(self.A * t) @ self.x0 for t in time])

        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol = TOLERANCE))

if __name__ == '__main__':
    unittest.main()
