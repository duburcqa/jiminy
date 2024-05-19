"""
@brief This file aims at verifying the sanity of the physics and the
       integration method of jiminy on simple models.
"""
import unittest
import numpy as np
import scipy

import jiminy_py.core as jiminy

from utilities import (
    load_urdf_default,
    setup_controller_and_engine,
    integrate_dynamics,
    simulate_and_get_state_evolution)

# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1.0e-7


class SimulateTwoMasses(unittest.TestCase):
    """Simulate the motion of two masses held together by a spring damper, and
    compare with the analytical solution.

    The system simulated can be represented as such:

                  k1   M1   k2   M2
             //| <><> |__| <><> |__|
    """
    def setUp(self):
        # Create the robot and load the URDF
        self.urdf_name = "linear_two_masses.urdf"
        self.motor_names = ("FirstJoint", "SecondJoint")
        self.robot = load_urdf_default(
            self.urdf_name, self.motor_names, has_freeflyer=False)

        # Specify spring stiffness and damping for this simulation
        self.k = np.array([200.0, 20.0])
        self.nu = np.array([0.1, 0.2])

        # Define initial state and simulation duration
        self.x0 = np.array([0.1, -0.1, 0.0, 0.0])
        self.tf = 10.0

        # Extract some information about the model
        self.m = np.stack([self.robot.pinocchio_model.inertias[i].mass
                           for i in [1, 2]], axis=0)
        self.I = 1 / self.m[1] + 1 / self.m[0]

        # Compute the matrix representing this linear system for
        # analytical computation. Dynamics is linear: dX = A X
        self.A = np.array([
            [                   0.0,                   0.0,                     1.0,                   0.0],
            [                   0.0,                   0.0,                     0.0,                   1.0],
            [-self.k[0] / self.m[0], self.k[1] / self.m[0], -self.nu[0] / self.m[0], self.nu[1] / self.m[0]],
            [ self.k[0] / self.m[0],   -self.k[1] * self.I,  self.nu[0] / self.m[0],   -self.nu[1] * self.I]])

        # Initialize numpy random seed
        np.random.seed(42)

    def _spring_force(self, t, q, v, sensor_measurements, out):
        """Update the force generated by the spring.
        """
        out[-2:] = - self.k * q[-2:] - self.nu * v[-2:]

    def _get_simulated_and_analytical_solutions(self, engine, tf, xInit):
        """Simulate the system dynamics and compute the corresponding
        analytical solution at the same timesteps.
        """
        # Run simulation and extract some information from log data
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, xInit, split=False)

        # Compute analytical solution
        x_analytical = np.stack([
            scipy.linalg.expm(self.A * t).dot(self.x0)
            for t in time], axis=0)

        return time, x_jiminy, x_analytical

    def test_continuous_simulation(self):
        """Test simulation of this system using a continuous time controller.
        """
        # Create and initialize the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, compute_command=self._spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 0.0
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine.set_options(engine_options)

        # Compare the numerical and analytical solutions
        _, x_jiminy, x_analytical = \
            self._get_simulated_and_analytical_solutions(
                engine, self.tf, self.x0)
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_discrete_simulation(self):
        """Test simulation of this system using internal dynamics using a
        discrete time controller.
        """
        # Create and initialize the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, internal_dynamics=self._spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1.0e-3
        engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine.set_options(engine_options)

        # Compare the numerical and analytical solutions
        _, x_jiminy, x_analytical = \
            self._get_simulated_and_analytical_solutions(
                engine, self.tf, self.x0)
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_profile_force(self):
        """Test adding an external force profile function to the system.
        """
        # Create and initialize the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, internal_dynamics=self._spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine.set_options(engine_options)

        # Define and register the external force:
        # a spring linking the second mass to the origin.
        k_ext = 50.0
        def external_force(t, q, v, f):
            nonlocal k_ext
            f[0] = - k_ext * (q[0] + q[1])

        engine.register_profile_force("", "SecondMass", external_force)

        # Add the extra external force to second mass
        self.A[3, :] += np.array([
            -k_ext / self.m[1], -k_ext / self.m[1], 0.0, 0.0])

        # Compare the numerical and analytical solutions
        _, x_jiminy, x_analytical = \
            self._get_simulated_and_analytical_solutions(
                engine, self.tf, self.x0)
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

        # =====================================================================
        # Now, apply a force / torque to a non-trivial rotation to verify
        # internal projection of the force onto the joints.

        # Rebuild the model with a freeflyer
        robot = load_urdf_default(
            self.urdf_name, self.motor_names, has_freeflyer=True)

        # Initialize with a random freeflyer configuration and zero velocity
        q_init = np.zeros(9)
        q_init[:7] = np.random.rand(7)
        q_init[3:7] /= np.linalg.norm(q_init[3:7])
        v_init = np.zeros(8)
        q_init[-2:], v_init[-2:] = np.split(self.x0, 2)

        # Define the external wrench to apply on the system
        f_local = np.array([1.0, 1.0, 0.0, 0.0, 0.5, 0.5])
        joint_index = robot.pinocchio_model.getJointId("FirstJoint")

        # Create the controller
        def internal_dynamics(t, q, v, sensor_measurements, u_custom):
            # Apply torque on freeflyer to make it spin
            self.assertTrue(np.allclose(
                np.linalg.norm(q[3:7]), 1.0, atol=TOLERANCE))
            u_custom[3:6] = 1.0

        def compute_command(t, q, v, sensor_measurements, command):
            # Check if local external force is properly computed
            nonlocal f_local
            if engine.is_simulation_running:
                f_ext = engine.robot_states[0].f_external[joint_index].vector
                self.assertTrue(np.allclose(f_ext, f_local, atol=TOLERANCE))

        # Create and initialize the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, robot, compute_command, internal_dynamics)

        # Define and register the external force:
        # a wrench in the local frame.
        def external_force(t, q, v, f):
            nonlocal f_local
            # Rotate the wrench to project it to the world frame
            R = robot.pinocchio_data.oMi[joint_index].rotation
            f[:3] = R @ f_local[:3]
            f[3:] = R @ f_local[3:]

        engine.register_profile_force("", "FirstJoint", external_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["sensorsUpdatePeriod"] = 1e-3
        engine_options["stepper"]["controllerUpdatePeriod"] = 1e-3
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine.set_options(engine_options)

        # Run simulation: Check is done directly by control law
        engine.simulate(self.tf, q_init, v_init)

    def test_fixed_body_constraint(self):
        """Test kinematic constraint: fixed second mass with a constraint.
        """
        # Create and initialize the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, self.robot, internal_dynamics=self._spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine_options["constraints"]["regularization"] = 0.0
        engine.set_options(engine_options)

        # Add a kinematic constraint on the second mass
        constraint = jiminy.FrameConstraint("SecondMass")
        self.robot.add_constraint("fixMass", constraint)

        # The dynamics of the first mass is not changed, the acceleration of
        # the second mass is the opposite of that of the first mass to provide
        # a constant output position.
        self.A[3, :] = -self.A[2, :]

        # Compare the numerical and analytical solutions
        _, x_jiminy, x_analytical = \
            self._get_simulated_and_analytical_solutions(
                engine, self.tf, self.x0)
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

    def test_freeflyer_multiple_constraints(self):
        """Test having several constraints at once.

        This test features:
          - a freeflyer with a fixed body constraint on the
            freeflyer. This gives a non-trivial constraint to solve
            to effectively cancel the freeflyer.
          - a fixed body constraint on the output mass.
        """
        # Rebuild the model with a freeflyer
        robot = load_urdf_default(
            self.urdf_name, self.motor_names, has_freeflyer=True)

        # Create and initialize the engine
        engine = jiminy.Engine()
        setup_controller_and_engine(
            engine, robot, internal_dynamics=self._spring_force)

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["world"]["gravity"] = np.zeros(6)  # Turn off gravity
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine_options["constraints"]["regularization"] = 0.0
        engine.set_options(engine_options)

        # Add a kinematic constraints.
        freeflyer_constraint = jiminy.FrameConstraint("world")
        robot.add_constraint("world", freeflyer_constraint)
        fix_mass_constraint = jiminy.FrameConstraint("SecondMass")
        robot.add_constraint("fixMass", fix_mass_constraint)

        # Initialize with a random freeflyer configuration and zero velocity
        x_init = np.zeros(17)
        x_init[:7] = np.random.rand(7)
        x_init[3:7] /= np.linalg.norm(x_init[3:7])
        x_init[7:9], x_init[-2:] = np.split(self.x0, 2)

        # The acceleration of the second mass should be the opposite of that of
        # the first
        self.A[3, :] = -self.A[2, :]

        # Compare the numerical and analytical solutions
        _, x_jiminy, x_analytical = \
            self._get_simulated_and_analytical_solutions(
                engine, self.tf, x_init)
        self.assertTrue(np.allclose(
            x_jiminy[:, [7, 8, 15, 16]], x_analytical, atol=TOLERANCE))

        # Verify in addition that freeflyer has not moved
        self.assertTrue(np.allclose(x_jiminy[:, 9:15], 0, atol=TOLERANCE))
        self.assertTrue(np.allclose(
            x_jiminy[:, :7], x_jiminy[0, :7], atol=TOLERANCE))

    def test_constraint_external_force(self):
        r"""Test support of external force applied with constraints.

        To provide a non-trivial test case with an external force non-colinear
        to the constraints, simulate two masses oscillating, one along the x
        axis and the other along the y axis, with a spring between them (in
        diagonal). The system may be represented as such ((f) indicates fixed
        bodies):

          [M_22 (f)]
              ^
              ^
            [M_21]\<>\
              ^     \<>\
              ^       \<>\
              ^         \<>\
            [O (f)] <><> [M_11] <><> [M_12 (f)]
        """
        # Build two robots with freeflyer, with a freeflyer and a fixed second
        # body constraint.

        # Instantiate the engine
        engine = jiminy.Engine()

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["stepper"]["odeSolver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-1
        engine_options["constraints"]["regularization"] = 0.0
        engine.set_options(engine_options)

        # Define some internal parameters
        k = np.array([[100, 50], [80, 120]])
        nu = np.array([[0.2, 0.01], [0.05, 0.1]])
        k_cross = 100

        # Initialize and configure the engine
        robot_names = ('FirstSystem', 'SecondSystem')
        for i, robot_name in enumerate(robot_names):
            # Load robot
            robot = load_urdf_default(
                self.urdf_name, self.motor_names, True, robot_name)

            # Apply constraints
            freeflyer_constraint = jiminy.FrameConstraint("world")
            robot.add_constraint("world", freeflyer_constraint)
            fix_mass_constraint = jiminy.FrameConstraint("SecondMass")
            robot.add_constraint("fixMass", fix_mass_constraint)

            # Create controller
            class Controller(jiminy.BaseController):
                def __init__(self, k, nu):
                    super().__init__()
                    self.k = k
                    self.nu = nu

                def internal_dynamics(self, t, q, v, u_custom):
                    u_custom[6:] = - self.k * q[7:] - self.nu * v[6:]

            robot.controller = Controller(k[i, :], nu[i, :])

            # Add robot to engine
            engine.add_robot(robot)

        # Add coupling force
        def force(t, q1, v1, q2, v2, f):
            # Putting a linear spring between both systems would actually
            # decouple them (the force applied on each system would be a
            # function of this system state only). So we use a nonlinear
            # stiffness, proportional to the square of the distance of both
            # systems to the origin.
            d2 = q1[7] ** 2 + q2[7] ** 2
            f[0] = - k_cross * (1 + d2) * q1[7]
            f[1] = + k_cross * (1 + d2) * q2[7]

        engine.register_coupling_force(
            *robot_names, "FirstMass", "FirstMass", force)

        # Initialize the whole system.
        x_init = {}

        ## First system: freeflyer at identity
        x_init['FirstSystem'] = np.zeros(17)
        x_init['FirstSystem'][7:9] = self.x0[:2]
        x_init['FirstSystem'][-2:] = self.x0[2:]
        x_init['FirstSystem'][6] = 1.0

        ## Second system: rotation by pi/2 around Z to bring X axis to Y axis
        x_init['SecondSystem'] = x_init['FirstSystem'].copy()
        x_init['SecondSystem'][5:7] = np.sqrt(2) / 2.0

        # Run simulation and extract some information from log data
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, self.tf, x_init, split=False)

        # Verify that both freeflyers didn't moved
        for x in x_jiminy:
            self.assertTrue(np.allclose(x[:, 9:15], 0.0, atol=TOLERANCE))
            self.assertTrue(np.allclose(x[:, :7], x[0, :7], atol=TOLERANCE))

        # Extract coordinates in a minimum state vector
        x_jiminy_extract = np.hstack([x[:, [7, 8, 15, 16]] for x in x_jiminy])

        # Define dynamics of this system
        def dynamics(t, x):
            # Velocity to position
            dx = np.zeros(8)
            dx[:2] = x[2:4]
            dx[4:6] = x[6:8]

            # Compute acceleration linked to the spring
            for i in range(2):
                dx[2 + 4 * i] = (
                    - k[i, 0] * x[4 * i] - nu[i, 0] * x[2 + 4 * i]
                    + k[i, 1] * x[1 + 4 * i] + nu[i, 1] * x[3 + 4 * i])

            # Coupling force between both system
            dsquared = x[0] ** 2 + x[4] ** 2
            dx[2] += - k_cross * (1 + dsquared) * x[0]
            dx[6] += - k_cross * (1 + dsquared) * x[4]

            for i in [0, 1]:
                # Divide forces by mass
                dx[2 + 4 * i] /= self.m[0]

                # Second mass accelration should be opposite of the first
                dx[3 + 4 * i] = -dx[2 + 4 * i]

            return dx

        x0 = np.hstack([x_init[key][[7, 8, 15, 16]] for key in x_init])
        x_python = integrate_dynamics(time, x0, dynamics)
        np.testing.assert_allclose(x_jiminy_extract, x_python, atol=TOLERANCE)


if __name__ == '__main__':
    unittest.main()
