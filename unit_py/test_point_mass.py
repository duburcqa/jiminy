# This file aims at verifying the sanity of the physics and the integration
# method of jiminy on simple models.
import unittest
import numpy as np

from jiminy_py import core as jiminy


# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class SimulatePointMass(unittest.TestCase):
    def setUp(self):
        '''
        @brief Validate the contact dynamics.

        @details The energy is expected to decrease slowly when penetrating into the ground,
                 but should stay constant otherwise. Then, the equilibrium point must also
                 match the physics. Note that the friction model is not assessed here.
        '''
        # Load URDF, create robot.
        urdf_path = "data/point_mass.urdf"

        # Define the parameters of the contact dynamics
        self.k_contact = 1.0e6
        self.nu_contact = 2.0e3

        # Create the jiminy robot and controller
        self.robot = jiminy.Robot()
        self.robot.initialize(urdf_path, has_freeflyer=True)
        self.robot.add_contact_points(['MassBody'])
        force_sensor = jiminy.ForceSensor('MassBody')
        self.robot.attach_sensor(force_sensor)
        force_sensor.initialize('MassBody')

    def test_contact_point_dynamics(self):
        # Create the engine
        engine = jiminy.Engine()
        engine.initialize(self.robot)

        engine_options = engine.get_options()
        engine_options['contacts']['stiffness'] = self.k_contact
        engine_options['contacts']['damping'] = self.nu_contact
        engine_options['contacts']['transitionEps'] = 1/self.k_contact # To avoid assertion failure because of problem regularization
        # engine_options['contacts']['frictionDry'] = 5.0
        # engine_options['contacts']['frictionViscous'] = 5.0
        # engine_options['contacts']['dryFrictionVelEps'] = 1.0e-2
        engine_options["stepper"]["dtMax"] = 1.0e-5
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
        x_jiminy = np.stack([log_data['HighLevelController.' + s]
                                for s in self.robot.logfile_position_headers + \
                                         self.robot.logfile_velocity_headers], axis=-1)

        # Total energy and derivative
        E_contact = 1/2 * self.k_contact * np.minimum(x_jiminy[:, 2], 0.0) ** 2
        E_robot = log_data['HighLevelController.energy']
        E_tot = E_robot + E_contact
        E_diff_robot = np.concatenate((np.diff(E_robot), np.array([0.0], dtype=E_robot.dtype)))
        E_diff_tot = np.concatenate((np.diff(E_tot), np.array([0.0], dtype=E_robot.dtype)))

        # Check that the total energy never increases
        # One must use a specific, less restrictive, tolerance, because of numerical differentiation error of float32.
        TOLERANCE_diff = 1e-6
        self.assertTrue(np.all(E_diff_tot < TOLERANCE_diff))

        # Check that the energy of robot only increases when the robot is moving upward while still in the ground.
        # This is done by check that there is not two consecutive samples violating this law.
        self.assertTrue(np.all(np.diff(np.where((E_diff_robot > 0.0) != \
                               np.logical_and(x_jiminy[:, 9] > 0.0, x_jiminy[:, 2] < 0.0))) > 1))

        # Compare the numerical and analytical equilibrium state
        self.assertTrue(np.allclose(-log_data['MassBody.FZ'][-1], mass * gravity, atol=TOLERANCE))
        self.assertTrue(np.allclose(self.k_contact * x_jiminy[-1, 2], mass * gravity, atol=TOLERANCE))

if __name__ == '__main__':
    unittest.main()
