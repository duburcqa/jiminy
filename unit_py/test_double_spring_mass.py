# This file aims at verifying the sanity of the physics and the integration
# method of jiminy on simple models.
import unittest
import numpy as np
from scipy.linalg import expm

from jiminy_py import core as jiminy


# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class SimulateTwoMasses(unittest.TestCase):
    '''
    @brief Simulate the motion of two masses held together by a spring damper,
           and compare with the analytical solution.

    @details The system simulated can be represented as such:
                  k1   M1   k2   M2
             //| <><> |__| <><> |__|
    '''
    def setUp(self):
        # Load URDF, create robot.
        urdf_path = "data/linear_two_masses.urdf"

        # Specify spring stiffness and damping for this simulation
        self.k = np.array([200, 20])
        self.nu = np.array([0.1, 0.2])

        # Define initial state and simulation duration
        self.x0 = np.array([0.1, -0.1, 0.0, 0.0])
        self.tf = 10.0

        # Create the jiminy robot

        # Instantiate robot and engine
        self.robot = jiminy.Robot()
        self.robot.initialize(urdf_path, has_freeflyer=False)
        for joint_name in ["FirstJoint", "SecondJoint"]:
            motor = jiminy.SimpleMotor(joint_name)
            self.robot.attach_motor(motor)
            motor.initialize(joint_name)

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

        # Compute the matrix representing this linear system for analytical
        # computation
        m = np.stack([self.robot.pinocchio_model_th.inertias[i].mass
                      for i in range(1,3)], axis=0)

        # Dynamics is linear: dX = A X
        I =  (1 / m[1] + 1 / m[0])
        self.A = np.array([[                0,                0,                  1,                0],
                           [                0,                0,                  0,                1],
                           [-self.k[0] / m[0], self.k[1] / m[0], -self.nu[0] / m[0], self.nu[1] / m[0]],
                           [ self.k[0] / m[0],   -self.k[1] * I,  self.nu[0] / m[0],  -self.nu[1] * I]])

    def test_continuous_simulation(self):
        '''
        @brief Test simulation of this system using a continuous time controller.
        '''
        def compute_command(t, q, v, sensor_data, u):
            u[:] = - self.k * q - self.nu * v

        def internal_dynamics(t, q, v, sensor_data, u):
            u[:] = 0.0

        controller = jiminy.ControllerFunctor(compute_command, internal_dynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        engine_options = engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine.set_options(engine_options)

        # Run simulation
        engine.simulate(self.tf, self.x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Compute analytical solution
        x_analytical = np.stack([expm(self.A * t) @ self.x0 for t in time], axis=0)

        # Compare the numerical and analytical solutions
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_discrete_simulation(self):
        '''
        @brief Test simulation of this system using internal dynamics using a
               discrete time controller.
        '''
        def compute_command(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internal_dynamics(t, q, v, sensor_data, u):
            u[:] = - self.k * q - self.nu * v

        controller = jiminy.ControllerFunctor(compute_command, internal_dynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        engine_options = engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1e-3
        engine_options["stepper"]["controllerUpdatePeriod"] = 1e-3
        engine.set_options(engine_options)

        # Run simulation
        engine.simulate(self.tf, self.x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Compute analytical solution
        x_analytical = np.stack([expm(self.A * t) @ self.x0 for t in time], axis=0)

        # Compare the numerical and analytical solutions
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_external_force_profile(self):
        '''
        @brief Test adding an external force profile function to the system.
        '''
        # Set same springs as usual
        def compute_command(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internal_dynamics(t, q, v, sensor_data, u):
            u[:] = - self.k * q - self.nu * v

        controller = jiminy.ControllerFunctor(compute_command, internal_dynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        # Define external force: a spring linking the second mass to the origin.
        k_ext = 50
        def external_force(t, q, v, f):
            f[0] = - k_ext * (q[0] + q[1])
        engine.register_force_profile("SecondMass", external_force)

        # Run simulation
        engine.simulate(self.tf, self.x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Compute analytical solution
        # Add extra external force to second mass.
        m = self.robot.pinocchio_model_th.inertias[2].mass
        self.A[3, :] += np.array([-k_ext / m, -k_ext / m, 0, 0])
        x_analytical = np.stack([expm(self.A * t) @ self.x0 for t in time], axis=0)

        # Compare the numerical and analytical solutions
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_fixed_body_constraint(self):
        '''
        @brief Test kinematic constraint: fixed second mass with a constaint.
        '''
        # Set same spings as usual
        def compute_command(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internal_dynamics(t, q, v, sensor_data, u):
            u[:] = - self.k * q - self.nu * v
        controller = jiminy.ControllerFunctor(compute_command, internal_dynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        # Add a kinematic constraint on the second mass
        constraint = jiminy.FixedFrameConstraint("SecondMass")
        self.robot.add_constraint("fixMass", constraint)

        # Run simulation
        engine.simulate(self.tf, self.x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Compute analytical solution
        # The dynamics of the first mass is not changed, the acceleration of the second
        # mass is the opposite of that of the first mass to provide a constant
        # output position.
        self.A[3, :] = -self.A[2, :]
        x_analytical = np.stack([expm(self.A * t) @ self.x0 for t in time], axis=0)

        # Compare the numerical and analytical solutions
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_freeflyer_multiple_constraints(self):
        '''
        @brief Test having several constraints at once.
        @details This test features:
                     - a freeflyer with a fixed body constraint on the freeflyer
                    (this gives a non-trivial constraint to solve to effectively cancel the freeflyer)
                     - a fixed body constaint on the output mass.
        '''
        # Rebuild the model with a freeflyer.
        self.robot = jiminy.Robot()
        self.robot.initialize("data/linear_two_masses.urdf", has_freeflyer=True)
        for joint_name in ["FirstJoint", "SecondJoint"]:
            motor = jiminy.SimpleMotor(joint_name)
            self.robot.attach_motor(motor)
            motor.initialize(joint_name)

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

        # Set same spings as usual
        def compute_command(t, q, v, sensor_data, u):
            u[:] = 0.0

        def internal_dynamics(t, q, v, sensor_data, u):
            u[6:] = - self.k * q[7:] - self.nu * v[6:]

        controller = jiminy.ControllerFunctor(compute_command, internal_dynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)
        # Disable gravity.
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6) # Turn off gravity
        engine.set_options(engine_options)

        # Add a kinematic constraints.
        freeflyer_constraint = jiminy.FixedFrameConstraint("world")
        self.robot.add_constraint("world", freeflyer_constraint)
        fix_mass_constraint = jiminy.FixedFrameConstraint("SecondMass")
        self.robot.add_constraint("fixMass", fix_mass_constraint)

        # Initialize with zero freeflyer velocity...
        x_init = np.zeros(17)
        x_init[7:9] = self.x0[:2]
        x_init[-2:] = self.x0[2:]
        # ... and a "random" (but fixed) freeflyer quaternion
        x_init[:7] = np.array([0.5, -0.8, 42.0, 0.1, 0.5, -0.7, 0.1])
        x_init[3:7] /= np.linalg.norm(x_init[3:7])

        # Run simulation
        engine.simulate(self.tf, x_init)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                             for s in self.robot.logfile_position_headers + \
                                      self.robot.logfile_velocity_headers], axis=-1)

        # Verify that freeflyer hasn't moved.
        self.assertTrue(np.allclose(x_jiminy[:, 9:15], 0, atol=TOLERANCE))
        self.assertTrue(np.allclose(x_jiminy[:, :7], x_jiminy[0, :7], atol=TOLERANCE))

        # Compute analytical solution - the acceleration of the second mass should
        # be the opposite of that of the first.
        self.A[3, :] = -self.A[2, :]
        x_analytical = np.stack([expm(self.A * t) @ self.x0 for t in time], axis=0)

        # Compare the numerical and analytical solutions
        self.assertTrue(np.allclose(x_jiminy[:, [7,8,15,16]], x_analytical, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
