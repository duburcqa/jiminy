# This file aims at verifying the sanity of the physics and the integration
# method of jiminy on simple models.
import unittest
import numpy as np
from pinocchio import Force

from jiminy_py import core as jiminy


# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class SimulatePointMass(unittest.TestCase):
    def setUp(self):
        # Load URDF, create robot.
        urdf_path = "data/point_mass.urdf"

        # Define the parameters of the contact dynamics
        self.k_contact = 1.0e6
        self.nu_contact = 2.0e3
        self.v_stiction = 5e-2
        self.r_stiction = 0.5
        self.dry_friction = 5.5
        self.visc_friction = 2.0
        self.dtMax = 1.0e-5

        # Create the jiminy robot and controller
        self.robot = jiminy.Robot()
        self.robot.initialize(urdf_path, has_freeflyer=True)
        self.robot.add_contact_points(['MassBody'])
        force_sensor = jiminy.ContactSensor('MassBody')
        self.robot.attach_sensor(force_sensor)
        force_sensor.initialize('MassBody')

    def test_contact_point_dynamics(self):
        """
        @brief Validate the contact dynamics.

        @details The energy is expected to decrease slowly when penetrating into the ground,
                 but should stay constant otherwise. Then, the equilibrium point must also
                 match the physics. Note that the friction model is not assessed here.
        """
        # Create the engine
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        engine_options = engine.get_options()
        engine_options['contacts']['stiffness'] = self.k_contact
        engine_options['contacts']['damping'] = self.nu_contact
        engine_options['contacts']['transitionEps'] = 1.0 / self.k_contact # To avoid assertion failure because of problem regularization
        engine_options["stepper"]["dtMax"] = self.dtMax
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine.set_options(engine_options)

        # Extract some information about the engine and the robot
        mass = self.robot.pinocchio_model.inertias[-1].mass
        gravity = engine.get_options()['world']['gravity'][2]

        # Run simulation
        x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]) # [TX,TY,TZ],[QX,QY,QZ,QW]
        tf = 1.5

        engine.simulate(tf, x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                                for s in self.robot.logfile_position_headers + \
                                         self.robot.logfile_velocity_headers], axis=-1)

        # Total energy and derivative
        E_contact = 1/2 * self.k_contact * np.minimum(x_jiminy[:, 2], 0.0) ** 2
        E_robot = log_data['HighLevelController.energy']
        E_tot = E_robot + E_contact
        E_diff_robot = np.concatenate((np.diff(E_robot) / np.diff(time), np.array([0.0], dtype=E_robot.dtype)))
        E_diff_tot = np.concatenate((np.diff(E_tot) / np.diff(time), np.array([0.0], dtype=E_robot.dtype)))

        # Check that the total energy never increases
        # One must use a specific, less restrictive, tolerance, because of numerical differentiation error of float32.
        TOLERANCE_diff = 5e-2
        self.assertTrue(np.all(E_diff_tot < TOLERANCE_diff))

        # Check that the energy of robot only increases when the robot is moving upward while still in the ground.
        # This is done by check that there is not two consecutive samples violating this law.
        self.assertTrue(np.all(np.diff(np.where((E_diff_robot > 0.0) != \
                               np.logical_and(x_jiminy[:, 9] > 0.0, x_jiminy[:, 2] < 0.0))) > 1))

        # Compare the numerical and analytical equilibrium state
        idx = self.robot.pinocchio_model.frames[self.robot.pinocchio_model.getFrameId("MassBody")].parent
        self.assertTrue(np.allclose(-engine.system_state.f_external[idx].linear[2], mass * gravity, atol=TOLERANCE))
        self.assertTrue(np.allclose(self.k_contact * x_jiminy[-1, 2], mass * gravity, atol=TOLERANCE))


    def test_force_sensor(self):
        """
        @brief Validate output of force sensor.

        @details The energy is expected to decrease slowly when penetrating into the ground,
                 but should stay constant otherwise. Then, the equilibrium point must also
                 match the physics. Note that the friction model is not assessed here.
        """
        # Create the engine
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        engine_options = engine.get_options()
        engine_options['contacts']['stiffness'] = self.k_contact
        engine_options['contacts']['damping'] = self.nu_contact
        engine_options['contacts']['transitionEps'] = 1.0 / self.k_contact # To avoid assertion failure because of problem regularization
        engine_options["stepper"]["dtMax"] = self.dtMax
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine.set_options(engine_options)


        idx = self.robot.pinocchio_model.getFrameId("MassBody")
        def computeCommand(t, q, v, sensors_data, u):
            # Verify sensor data.
            f = Force(sensors_data[jiminy.ContactSensor.type, "MassBody"], np.zeros(3))
            f_joint_sensor = self.robot.pinocchio_model.frames[idx].placement * f
            f_jiminy = engine.system_state.f_external[self.robot.pinocchio_model.frames[idx].parent]
            self.assertTrue(np.allclose(f_joint_sensor.vector, f_jiminy.vector, atol=TOLERANCE))
            u[:] = 0.0

        # Internal dynamics: make the mass spin to generate nontrivial rotations.
        def internalDynamics(t, q, v, sensors_data, u):
            u[3:6] = 1.0

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        # Run simulation
        x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]) # [TX,TY,TZ],[QX,QY,QZ,QW]
        tf = 1.5

        engine.simulate(tf, x0)

    def test_friction_model(self):
        """
        @brief Validate the friction model.

        @details The transition between dry, dry-viscous, and viscous friction is assessed.
                 The energy variation and the steady state are also compared to the theoretical model.
        """

        # Create the engine
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        engine_options = engine.get_options()
        engine_options['contacts']['stiffness'] = self.k_contact
        engine_options['contacts']['damping'] = self.nu_contact
        engine_options['contacts']['frictionDry'] = self.dry_friction
        engine_options['contacts']['frictionViscous'] = self.visc_friction
        engine_options['contacts']['frictionStictionVel'] = self.v_stiction
        engine_options['contacts']['frictionStictionRatio'] = self.r_stiction
        engine_options['contacts']['transitionEps'] = 1.0 / self.k_contact # To avoid assertion failure because of problem regularization
        engine_options["stepper"]["dtMax"] = self.dtMax
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine.set_options(engine_options)

        # Extract some information about the engine and the robot
        mass = self.robot.pinocchio_model.inertias[-1].mass
        gravity = engine.get_options()['world']['gravity'][2]

        # Register a  impulse of force
        t0 = 0.05
        dt = 0.8
        F = 5.0
        engine.register_force_impulse("MassBody", t0, dt, np.array([F, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # Run simulation
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]) # [TX,TY,TZ],[QX,QY,QZ,QW]
        tf = 1.5

        engine.simulate(tf, x0)

        log_data, _ = engine.get_log()
        time = log_data['Global.Time']
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                            for s in self.robot.logfile_position_headers + \
                                     self.robot.logfile_velocity_headers], axis=-1)

        # Validate the stiction model:
        # check the transition between dry and viscous friction because of stiction phenomena
        acceleration = log_data['HighLevelController.currentFreeflyerAccelerationLinX']
        jerk = np.diff(acceleration) / np.diff(time)
        snap =  np.diff(jerk) / np.diff(time[1:])
        snap_rel = np.abs(snap / np.max(snap))
        snap_disc = time[1:-1][snap_rel > 2.0e-4]

        snap_disc_analytical_dry = log_data['Global.Time'][
            np.logical_and(x_jiminy[:, 7] > self.v_stiction - self.dtMax,
                           x_jiminy[:, 7] < self.v_stiction + self.dtMax)]
        snap_disc_analytical_viscous = log_data['Global.Time'][np.logical_and(
            x_jiminy[:, 7] > (1.0 + self.r_stiction) * self.v_stiction - self.dtMax,
            x_jiminy[:, 7] < (1.0 + self.r_stiction) * self.v_stiction + self.dtMax)]
        snap_disc_analytical = np.sort(np.concatenate(
            (snap_disc_analytical_dry,
             snap_disc_analytical_viscous,
             np.array([t0, t0 + self.dtMax, t0 + dt, t0 + dt + self.dtMax]))))

        self.assertTrue(len(snap_disc) == len(snap_disc_analytical))
        self.assertTrue(np.allclose(snap_disc, snap_disc_analytical, atol=1e-12))

        # Check that the energy increases only when the force is applied
        E_robot = log_data['HighLevelController.energy']
        E_diff_robot = np.concatenate((np.diff(E_robot) / np.diff(time), np.array([0.0], dtype=E_robot.dtype)))
        E_inc_range = log_data['Global.Time'][np.where(E_diff_robot > 1e-5)[0][[0, -1]]]
        E_inc_range_analytical = np.array([t0, t0 + dt - self.dtMax])

        self.assertTrue(np.allclose(E_inc_range, E_inc_range_analytical, atol=5e-3))

        # Check that the steady state matches the theory
        # Note that a specific tolerance is used for the acceleration since the steady state is not perfectly reached
        TOLERANCE_acc = 1e-5

        v_steady = x_jiminy[log_data['Global.Time'] == t0 + dt, 7]
        v_steady_analytical = - F / (self.visc_friction * mass * gravity)
        a_steady = acceleration[np.logical_and(log_data['Global.Time'] > t0 + dt - self.dtMax,
                                               log_data['Global.Time'] < t0 + dt + self.dtMax)]

        self.assertTrue(len(a_steady) == 1)
        self.assertTrue(a_steady < TOLERANCE_acc)
        self.assertTrue(np.allclose(v_steady, v_steady_analytical, atol=TOLERANCE))

if __name__ == '__main__':
    unittest.main()
