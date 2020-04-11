# This file aims at verifying the sanity of the physics and the integration
# method of jiminy on simple models.
import unittest
import numpy as np
from scipy.linalg import expm

from jiminy_py import core as jiminy

from utilities import load_urdf_default, integrate_dynamics

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
        self.urdf_path = "data/linear_two_masses.urdf"
        self.motor_names = ["FirstJoint", "SecondJoint"]
        self.robot = load_urdf_default(self.urdf_path, self.motor_names)

        # Specify spring stiffness and damping for this simulation
        self.k = np.array([200, 20])
        self.nu = np.array([0.1, 0.2])

        # Define initial state and simulation duration
        self.x0 = np.array([0.1, -0.1, 0.0, 0.0])
        self.tf = 10.0

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
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
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
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
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
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
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
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
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
        self.robot = load_urdf_default(self.urdf_path, self.motor_names, has_freeflyer = True)

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
        np.random.seed(42)
        x_init[:7] = np.random.rand(7)
        x_init[3:7] /= np.linalg.norm(x_init[3:7])

        # Run simulation
        engine.simulate(self.tf, x_init)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['.'.join(['HighLevelController', s])]
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

    def test_constraint_external_force(self):
        '''
        @brief Test support of external force applied with constraints.
        @details To provide a non-trivial test case with an external force non-colinear
                 to the constraints, simulate two masses oscillating, one along
                 the x axis and the other along the y axis, with a spring between
                 them (in diagonal). The system may be represented as such ((f) indicates fixed bodies)

                 [M_22 (f)]
                     ^
                     ^
                   [M_21]\<>\
                     ^     \<>\
                     ^       \<>\
                     ^         \<>\
                   [O (f)] <><> [M_11] <><> [M_12 (f)]

        '''
        # Build two robots with freeflyers, with a freeflyer and a fixed second body constraint.
        # Rebuild the model with a freeflyer.
        robots = []
        engine = jiminy.EngineMultiRobot()

        system_names = ['FirstSystem', 'SecondSystem']
        k = np.array([[100, 50], [80, 120]])
        nu = np.array([[0.2, 0.01], [0.05, 0.1]])
        k_cross = 100

        class Controllers():
            def __init__(self, k, nu):
                self.k = k
                self.nu = nu

            def compute_command(self, t, q, v, sensor_data, u):
                u[:] = 0

            def internal_dynamics(self, t, q, v, sensor_data, u):
                u[6:] = - self.k * q[7:] - self.nu * v[6:]

        for i in range(2):
            # Load robot.
            robots.append(load_urdf_default(self.urdf_path, self.motor_names, True))

            # Apply constraints.
            freeflyer_constraint = jiminy.FixedFrameConstraint("world")
            robots[i].add_constraint("world", freeflyer_constraint)
            fix_mass_constraint = jiminy.FixedFrameConstraint("SecondMass")
            robots[i].add_constraint("fixMass", fix_mass_constraint)

            # Create controller
            controller = Controllers(k[i, :], nu[i, :])
            controller = jiminy.ControllerFunctor(controller.compute_command, controller.internal_dynamics)
            controller.initialize(robots[i])

            # Add system to engine.
            engine.add_system(system_names[i], robots[i], controller)

        # Add coupling force.
        def coupling_force(t, q1, v1, q2, v2, f):
            # Putting a linear spring between both systems would actually
            # decouple them (the force applied on each system would be a function
            # of this system state only). So we use a nonlinear stiffness, proportional
            # to the square of the distance of both systems to the origin.
            dsquared = q1[7] ** 2 + q2[7] ** 2
            f[0] = - k_cross * (1 + dsquared) * q1[7]
            f[1] = k_cross * (1 + dsquared) * q2[7]

        engine.add_coupling_force(system_names[0], system_names[1], "FirstMass", "FirstMass", coupling_force)

        # Initialize the whole system.
        # First sytem: freeflyer at identity.
        x_init = {'FirstSystem': np.zeros(17)}
        x_init['FirstSystem'][7:9] = self.x0[:2]
        x_init['FirstSystem'][7:9] = self.x0[:2]
        x_init['FirstSystem'][-2:] = self.x0[2:]
        x_init['FirstSystem'][6] = 1.0
        # Second system: rotation by pi / 2 around Z to bring X axis to Y axis.
        x_init['SecondSystem'] = np.copy(x_init['FirstSystem'])
        x_init['SecondSystem'][5:7] = np.sqrt(2) / 2.0

        # Run simulation
        engine.simulate(self.tf, x_init)

        # Extract log data
        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = [np.stack([log_data['.'.join(['HighLevelController', system_names[i], s])]
                                for s in robots[i].logfile_position_headers + \
                                         robots[i].logfile_velocity_headers] , axis=-1)
                                            for i in range(len(system_names))]

        # Verify that both freeflyers didn't moved.
        for i in range(len(system_names)):
            self.assertTrue(np.allclose(x_jiminy[i][:, 9:15], 0.0, atol=TOLERANCE))
            self.assertTrue(np.allclose(x_jiminy[i][:, :7], x_jiminy[i][0, :7], atol=TOLERANCE))

        # Extract coordinates in a minimum state vector.
        x_jiminy_extract = np.hstack([x_jiminy[i][:, [7,8,15,16]]
                                        for i in range(len(system_names))])

        # Define dynamics of this system
        def system_dynamics(t, x):
            dx = np.zeros(8)
            # Velocity to position.
            dx[:2] = x[2:4]
            dx[4:6] = x[6:8]
            # Compute acceleration linked to the spring.
            for i in range(2):
                dx[2 + 4 * i] = -k[i, 0] * x[4 * i] - nu[i, 0] * x[2 + 4 * i] \
                                + k[i, 1] * x[1 + 4 * i] + nu[i, 1] * x[3 + 4 * i]

            # Coupling force between both system.
            dsquared = x[0] ** 2 + x[4] ** 2
            dx[2] += - k_cross * (1 + dsquared) * x[0]
            dx[6] += - k_cross * (1 + dsquared) * x[4]

            for i in range(2):
                # Devide forces by mass.
                m = robots[i].pinocchio_model_th.inertias[1].mass
                dx[2 + 4 * i] /= m
                # Second mass accelration should be opposite of the first.
                dx[3 + 4 * i] = -dx[2 + 4 * i]
            return dx

        x0 = np.hstack([x_init[key][[7, 8, 15, 16]] for key in x_init])
        x_python = integrate_dynamics(time, x0, system_dynamics)
        self.assertTrue(np.allclose(x_jiminy_extract, x_python, atol=TOLERANCE))

if __name__ == '__main__':
    unittest.main()
