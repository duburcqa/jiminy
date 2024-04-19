"""
@brief This file aims at verifying the sanity of the physics and the
       integration method of jiminy on simple models.
"""
import os
import unittest

import numpy as np


from jiminy_py.core import (  # pylint: disable=no-name-in-module
    ForceSensor as force, ImuSensor as imu, ContactSensor as contact)
from jiminy_py.simulator import Simulator


# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1.0e-5


class SimulateFootedPendulum(unittest.TestCase):
    """Simulate the motion of a pendulum with a square foot having contact
    points at each corner.
    """
    def test_init_and_consistency(self):
        """Verify that the pendulum holds straight without falling when
        perfectly initialized at the unstable equilibrium point.

        This test also checks that there is no discontinuity at initialization
        and that the various sensors are working properly.
        """
        # Create the jiminy robot
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_root_dir = os.path.join(current_dir, "data")
        urdf_path = os.path.join(data_root_dir, "foot_pendulum.urdf")
        hardware_path = os.path.join(data_root_dir, "foot_pendulum.toml")
        simulator = Simulator.build(
            urdf_path, hardware_path=hardware_path, has_freeflyer=True)

        # Set options
        engine_options = simulator.get_options()
        engine_options["telemetry"]["enableConfiguration"] = True
        engine_options["stepper"]["odeSolver"] = "runge_kutta_4"
        engine_options["stepper"]["dtMax"] = 1.0e-5
        engine_options["stepper"]["sensorsUpdatePeriod"] = 0.0
        engine_options["stepper"]["controllerUpdatePeriod"] = 1.0e-3
        engine_options["stepper"]["logInternalStepperSteps"] = True
        engine_options['contacts']['model'] = "constraint"
        engine_options['contacts']['stabilizationFreq'] = 0.0
        engine_options['constraints']['regularization'] = 1e-9
        simulator.set_options(engine_options)

        # Initialize the simulation
        q0 = np.array([0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 1.0, 0.0])
        v0 = np.zeros((simulator.robot.pinocchio_model.nv,))
        mass = simulator.robot.pinocchio_data.mass[0]
        gravity = engine_options["world"]["gravity"]
        simulator.start(q0, v0)

        self.assertTrue(np.all(np.abs(simulator.stepper_state.a) < TOLERANCE))
        imu_data = simulator.robot.sensor_measurements[imu.type, 'Foot']
        gyro_data, accel_data = np.split(imu_data, 2)
        self.assertTrue(np.allclose(gyro_data, 0.0, atol=TOLERANCE))
        self.assertTrue(np.allclose(accel_data, -gravity[:3], atol=TOLERANCE))
        self.assertTrue(np.allclose(
            simulator.robot.sensor_measurements[force.type, 'Foot'],
            -gravity * mass,
            atol=TOLERANCE))
        for i in range(3):
            self.assertTrue(np.allclose(
                simulator.robot.sensor_measurements[
                    contact.type, f'Foot_CollisionBox_0_{2 * i}'],
                simulator.robot.sensor_measurements[
                    contact.type, f'Foot_CollisionBox_0_{2 * (i + 1)}'],
                atol=TOLERANCE))
        with self.assertRaises(KeyError):
            simulator.robot.sensor_measurements[contact.type, 'NA']
        with self.assertRaises(KeyError):
            simulator.robot.sensor_measurements['NA', 'Foot_CollisionBox_0_1']

        # Simulate for a few seconds
        simulator.step(1.0)
        simulator.stop()

        self.assertTrue(np.all(np.abs(simulator.stepper_state.v) < TOLERANCE))
        self.assertTrue(np.all(np.abs(simulator.stepper_state.a) < TOLERANCE))


if __name__ == '__main__':
    unittest.main()
