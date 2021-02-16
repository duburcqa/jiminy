"""This file aims at verifying the sanity of the physics and the integration
method of jiminy on simple mass.
"""
import unittest
import numpy as np
from enum import Enum
from scipy.signal import savgol_filter

import jiminy_py.core as jiminy

from pinocchio import Force

from utilities import (
    load_urdf_default,
    setup_controller_and_engine,
    neutral_state,
    simulate_and_get_state_evolution)

# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class ShapeType(Enum):
    POINT = 0
    BOX = 1
    SPHERE = 2


class SimulateSimpleMass(unittest.TestCase):
    def setUp(self):
        # Model specification
        self.body_name = 'MassBody'

        # Define the parameters of the contact dynamics
        self.k_contact = 1.0e6
        self.nu_contact = 2.0e3
        self.v_stiction = 5.0e-2
        self.r_stiction = 0.5
        self.dry_friction = 5.5
        self.visc_friction = 2.0
        self.dtMax = 1.0e-5

    def _setup(self, shape):
        # Load the right URDF, and create robot.
        if shape == ShapeType.POINT:
            urdf_name = "point_mass.urdf"
        elif shape == ShapeType.BOX:
            urdf_name = "box_collision_mesh.urdf"
        elif shape == ShapeType.SPHERE:
            urdf_name = "sphere_primitive.urdf"

        # Create the jiminy robot and controller
        robot = load_urdf_default(urdf_name, has_freeflyer=True)

        # Add contact point or collision body, along with related sensor,
        # depending on shape type.
        if shape != ShapeType.POINT:
            # Add collision body
            robot.add_collision_bodies([self.body_name])

            # Add a force sensor
            force_sensor = jiminy.ForceSensor(self.body_name)
            robot.attach_sensor(force_sensor)
            force_sensor.initialize(self.body_name)
        else:
            # Add contact point
            robot.add_contact_points([self.body_name])

            # Add a contact sensor
            contact_sensor = jiminy.ContactSensor(self.body_name)
            robot.attach_sensor(contact_sensor)
            contact_sensor.initialize(self.body_name)

        # Extract some information about the engine and the robot
        m = robot.pinocchio_model.inertias[-1].mass
        g = robot.pinocchio_model.gravity.linear[2]
        weight = m * np.linalg.norm(g)
        if shape == ShapeType.POINT:
            height = 0.0
        elif shape == ShapeType.BOX:
            geom = robot.collision_model.geometryObjects[0]
            height = geom.geometry.points(0)[2]  # It is a mesh, not an actual box primitive
        elif shape == ShapeType.SPHERE:
            geom = robot.collision_model.geometryObjects[0]
            height = geom.geometry.radius
        frame_idx = robot.pinocchio_model.getFrameId(self.body_name)
        joint_idx = robot.pinocchio_model.frames[frame_idx].parent
        frame_pose = robot.pinocchio_model.frames[frame_idx].placement

        return robot, weight, height, joint_idx, frame_pose

    def _setup_controller_and_engine(self,
                                     engine, robot,
                                     compute_command=None,
                                     internal_dynamics=None):
        # Initialize the engine
        setup_controller_and_engine(
            engine, robot, compute_command, internal_dynamics)

        # configure the engine
        engine_options = engine.get_options()
        engine_options['contacts']['stiffness'] = self.k_contact
        engine_options['contacts']['damping'] = self.nu_contact
        engine.set_options(engine_options)

        return engine

    def _test_collision_and_contact_dynamics(self, shape):
        """
        @brief    Validate the collision body and contact point dynamics.

        @details  The energy is expected to decrease slowly when penetrating
                  into the ground, but should stay constant otherwise. Then,
                  the equilibrium point must also match the physics. Note that
                  the friction model is not assessed here.
        """
        # Create the robot
        robot, weight, height, joint_idx, _ = self._setup(shape)

        # Create, initialize, and configure the engine
        engine = jiminy.Engine()
        self._setup_controller_and_engine(engine, robot)

        # Set some extra options of the engine, to avoid assertion failure
        # because of problem regularization and outliers
        engine_options = engine.get_options()
        engine_options['contacts']['transitionEps'] = 1.0e-6
        engine_options["stepper"]["controllerUpdatePeriod"] = self.dtMax
        engine.set_options(engine_options)

        # Run simulation and extract some information from log data
        x0 = neutral_state(robot, split=False)
        x0[2] = 1.0
        tf = 1.5
        _, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)
        q_z_jiminy = x_jiminy[:, 2]
        v_z_jiminy = x_jiminy[:, 9]
        penetration_depth = np.minimum(q_z_jiminy - height, 0.0)

        # Total energy and derivative
        log_data, _ = engine.get_log()
        E_robot = log_data['HighLevelController.energy']
        E_contact = 1 / 2 * self.k_contact * penetration_depth ** 2
        E_tot = E_robot + E_contact
        E_diff_robot = np.concatenate((
            np.diff(E_robot) / self.dtMax,
            np.zeros((1,), dtype=E_robot.dtype)))
        E_diff_tot = savgol_filter(E_tot, 201, 2, deriv=1, delta=self.dtMax)

        # Check that the total energy never increases.
        # One must use a specific, less restrictive, tolerance, because of
        # numerical differentiation and filtering error.
        self.assertTrue(np.all(E_diff_tot < 1.0e-3))

        # Check that the energy of robot only increases when the robot is
        # moving upward while still in the ground. This is done by check
        # that there is not two consecutive samples violating this law.
        # Note that the energy must be non-zero to this test to make
        # sense, otherwise the integration error and log accuracy makes
        # the test fail.
        tolerance_depth = 1e-9
        self.assertTrue(np.all(np.diff(np.where(
            (abs(E_diff_robot) > tolerance_depth) & ((E_diff_robot > 0.0) != \
                ((v_z_jiminy > 0.0) & (penetration_depth < 0.0))))[0]) > 1))

        # Compare the numerical and analytical equilibrium state.
        f_ext_z = engine.system_state.f_external[joint_idx].linear[2]
        self.assertTrue(np.allclose(f_ext_z, weight, atol=TOLERANCE))
        self.assertTrue(np.allclose(
            -penetration_depth[-1], weight / self.k_contact, atol=TOLERANCE))

    def test_collision_and_contact_dynamics(self):
        for shape in [ShapeType.POINT, ShapeType.SPHERE]:  # TODO: Implement time of collision + persistent contact points to support BOX shape
            self._test_collision_and_contact_dynamics(shape)

    def test_contact_sensor(self):
        """
        @brief    Validate output of contact sensor.

        @details  The energy is expected to decrease slowly when penetrating
                  into the ground, but should stay constant otherwise. Then,
                  the equilibrium point must also match the physics. Note that
                  the friction model is not assessed here.
        """
        # Create the robot
        robot, _, _, joint_idx, frame_pose = self._setup(ShapeType.POINT)

        # Create the engine
        engine = jiminy.Engine()

        # No control law, only check sensor data
        def check_sensor_data(t, q, v, sensors_data, u):
            nonlocal engine, frame_pose

            # Verify sensor data, if the engine has been initialized
            if engine.is_initialized:
                contact_data = sensors_data[
                    jiminy.ContactSensor.type, self.body_name]
                f = Force(contact_data, np.zeros(3))
                f_joint_sensor = frame_pose * f
                f_jiminy = engine.system_state.f_external[joint_idx]
                self.assertTrue(np.allclose(
                    f_joint_sensor.vector, f_jiminy.vector, atol=TOLERANCE))

            # Set the command torque to zero
            u.fill(0.0)

        # Internal dynamics: make the mass spin to generate nontrivial
        # rotations.
        def spinning_force(t, q, v, sensors_data, u):
            u[3:6] = 1.0

        # Initialize and configure the engine
        self._setup_controller_and_engine(engine, robot,
            compute_command=check_sensor_data,
            internal_dynamics=spinning_force)

        # Run simulation
        q0, v0 = neutral_state(robot, split=True)
        tf = 1.5
        engine.simulate(tf, q0, v0)

    def _test_friction_model(self, shape):
        """
        @brief    Validate the friction model.

        @details  The transition between dry, dry-viscous, and viscous friction
                  is assessed. The energy variation and the steady state are
                  also compared to the theoretical model.
        """
        # Create the robot and engine
        robot, weight, height, _, _ = self._setup(shape)

        # Create, initialize, and configure the engine
        engine = jiminy.Engine()
        self._setup_controller_and_engine(engine, robot)

        # Set some extra options of the engine
        engine_options = engine.get_options()
        engine_options['contacts']['transitionEps'] = 1.0e-6
        engine_options['contacts']['frictionDry'] = self.dry_friction
        engine_options['contacts']['frictionViscous'] = self.visc_friction
        engine_options['contacts']['frictionStictionVel'] = self.v_stiction
        engine_options['contacts']['frictionStictionRatio'] = self.r_stiction
        engine_options["stepper"]["controllerUpdatePeriod"] = self.dtMax
        engine.set_options(engine_options)

        # Register an impulse of force
        t0, dt, Fx = 0.05, 0.8, 5.0
        F = np.array([Fx, 0.0, 0.0, 0.0, 0.0, 0.0])
        engine.register_force_impulse(self.body_name, t0, dt, F)

        # Run simulation
        x0 = neutral_state(robot, split=False)
        x0[2] = height
        tf = 1.5
        time, _, v_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=True)
        v_x_jiminy = v_jiminy[:, 0]

        # Validate the stiction model: check the transition between dry and
        # viscous friction because of stiction phenomena.
        log_data, _ = engine.get_log()
        acceleration = log_data[
            'HighLevelController.currentFreeflyerAccelerationLinX']
        jerk = np.diff(acceleration) / np.diff(time)
        snap =  np.diff(jerk) / np.diff(time[1:])
        snap_rel = np.abs(snap / np.max(snap))
        snap_disc = time[1:-1][snap_rel > 2.0e-4]

        snap_disc_analytical_dry = time[(
            (v_x_jiminy > (self.v_stiction - self.dtMax)) &
            (v_x_jiminy < (self.v_stiction + self.dtMax)))]
        snap_disc_analytical_viscous = time[(
            (v_x_jiminy > ((1.0 + self.r_stiction) *
                self.v_stiction - self.dtMax)) &
            (v_x_jiminy < ((1.0 + self.r_stiction) *
                self.v_stiction + self.dtMax)))]
        snap_disc_analytical = np.sort(np.concatenate(
            (snap_disc_analytical_dry,
             snap_disc_analytical_viscous,
             np.array([t0, t0 + self.dtMax, t0 + dt, t0 + dt + self.dtMax]))))

        self.assertTrue(len(snap_disc) == len(snap_disc_analytical))
        self.assertTrue(np.allclose(
            snap_disc, snap_disc_analytical, atol=TOLERANCE))

        # Check that the energy increases only when the force is applied
        tolerance_E = 1e-9

        E_robot = log_data['HighLevelController.energy']
        E_diff_robot = np.concatenate((
            np.diff(E_robot) / np.diff(time),
            np.zeros((1,), dtype=E_robot.dtype)))
        E_inc_range = time[np.where(E_diff_robot > tolerance_E)[0][[0, -1]]]
        E_inc_range_analytical = np.array([t0, t0 + dt - self.dtMax])

        self.assertTrue(np.allclose(
            E_inc_range, E_inc_range_analytical, atol=tolerance_E))

        # Check that the steady state matches the theory.
        # Note that a specific tolerance is used for the acceleration since the
        # steady state is not perfectly reached.
        tolerance_acc = 1e-6

        v_steady = v_x_jiminy[time == t0 + dt]
        v_steady_analytical = - Fx / (self.visc_friction * weight)
        a_steady = acceleration[
            (time > t0 + dt - self.dtMax) & (time < t0 + dt + self.dtMax)]

        self.assertTrue(len(a_steady) == 1)
        self.assertTrue(a_steady < tolerance_acc)
        self.assertTrue(np.allclose(
            v_steady, v_steady_analytical, atol=TOLERANCE))

    def test_friction_model(self):
        for shape in [ShapeType.POINT]:  # TODO: Implement time of collision + persistent contact points to support shapes
            self._test_friction_model(shape)

if __name__ == '__main__':
    unittest.main()
