# This file aims at verifying the sanity of the physics and the integration method of jiminy
# on simple models.
import unittest
import os
import numpy as np

from jiminy_py import core as jiminy


class SimulateSimplePendulum(unittest.TestCase):
    '''
    @brief Simulate the motion of a pendulum, comparing against python integration.
    '''
    def setUp(self):
        # Load URDF, create robot
        urdf_path = "data/simple_pendulum.urdf"

        # Create the jiminy robot

        # Instantiate robot and engine
        self.robot = jiminy.Robot()
        self.robot.initialize(urdf_path, has_freeflyer=False)
        motor = jiminy.SimpleMotor("PendulumJoint")
        self.robot.attach_motor(motor)
        motor.initialize("PendulumJoint")

        # Configure robot
        model_options = self.robot.get_model_options()
        motor_options = self.robot.get_motors_options()
        model_options["joints"]["enablePositionLimit"] = False
        model_options["joints"]["enableVelocityLimit"] = False
        for m in motor_options:
            motor_options[m]['enableTorqueLimit'] = False
            motor_options[m]['enableRotorInertia'] = False
        self.robot.set_model_options(model_options)
        self.robot.set_motors_options(motor_options)

    def test_flexiblility_rotor_inertia(self):
        '''
        @brief Test the addition of a flexibility in the model + rotor inertia to create a two-mass oscillating system.

        @details By adding a rotor inertia J and a flexibility, and removing gravity, we should get a linear system of two
                 masses of inertia (J, I) linked by a spring damper: this test verifies that jiminy matches the analytical system.
        '''
        # No controller and no internal dynamics
        def computeCommand(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internalDynamics(t, q, v, sensor_data, u):
            u[:] = 0.0

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)

        # Parameters: rotor inertia, spring stiffness and damping.
        J = 0.1

        # Enable rotor inertia
        motor_options = self.robot.get_motors_options()
        motor_options["PendulumJoint"]['enableRotorInertia'] = True
        motor_options["PendulumJoint"]['rotorInertia'] = J
        self.robot.set_motors_options(motor_options)

        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6) # Turn off gravity
        engine.set_options(engine_options)

        # TODO

if __name__ == '__main__':
    unittest.main()
