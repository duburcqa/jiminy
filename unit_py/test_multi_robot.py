# This file aims at verifying the sanity of the physics and the integration
# method of jiminy on simple models.
import unittest
import numpy as np
from scipy.linalg import expm

import jiminy_py.core as jiminy

from utilities import load_urdf_default, simulate_and_get_state_evolution

# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class SimulateMultiRobot(unittest.TestCase):
    def test_multi_robot(self):
        """
        @brief   Simulate interaction between two robots.

        @details The system simulated can be represented as such: each system
                 is a single mass linked to a spring, and a spring k_12 links
                 both systems.

                    k1     M1   k(1->2)  M2
                //| <><>  |__| <><><><> |  |
                             k2         |  |
                //| <><> <><> <><> <><> |__|
        """
        # Specify model
        urdf_name = "linear_single_mass.urdf"
        motors_names = ["Joint"]

        # Specify spring stiffness and damping for this simulation
        # First two parameters are the stiffness of each system,
        # third is the stiffness of the coupling spring
        k = np.array([100, 20, 50])
        nu = np.array([0.1, 0.2, 0.2])

        # Create controllers
        class Controllers:
            def __init__(self, k, nu):
                self.k = k
                self.nu = nu

            def compute_command(self, t, q, v, sensors_data, command):
                pass

            def internal_dynamics(self, t, q, v, sensors_data, u_custom):
                u_custom[:] = - self.k * q - self.nu * v

        # Create two identical robots
        engine = jiminy.EngineMultiRobot()

        # Configure the engine
        engine_options = engine.get_options()
        engine_options["stepper"]["solver"] = "runge_kutta_dopri5"
        engine_options["stepper"]["tolAbs"] = TOLERANCE * 1e-1
        engine_options["stepper"]["tolRel"] = TOLERANCE * 1e-2
        engine.set_options(engine_options)

        system_names = ['FirstSystem', 'SecondSystem']
        robots = []
        for i in range(2):
            robots.append(load_urdf_default(urdf_name, motors_names))

            # Create controller
            controller = Controllers(k[i], nu[i])

            controller = jiminy.ControllerFunctor(
                controller.compute_command, controller.internal_dynamics)
            controller.initialize(robots[i])

            # Add system to engine.
            engine.add_system(system_names[i], robots[i], controller)

        # Add coupling force between both systems: a spring between both masses
        def force(t, q1, v1, q2, v2, f):
            f[0] = k[2] * (q2[0] - q1[0]) + nu[2] * (v2[0] - v1[0])

        engine.register_force_coupling(
            system_names[0], system_names[1], "Mass", "Mass", force)

        # Run simulation and extract some information from log data
        x0 = {'FirstSystem': np.array([0.1, 0.0]),
              'SecondSystem': np.array([-0.1, 0.0])}
        tf = 10.0
        time, x_jiminy = simulate_and_get_state_evolution(
            engine, tf, x0, split=False)
        x_jiminy = np.concatenate(x_jiminy, axis=-1)

        # Define analytical system dynamics: two masses linked by three springs
        m = [r.pinocchio_model_th.inertias[1].mass for r in robots]
        k_eq = [x + k[2] for x in k]
        nu_eq = [x + nu[2] for x in nu]
        A = np.array([[            0.0,              1.0,             0.0,              0.0],
                      [-k_eq[0] / m[0], -nu_eq[0] / m[0],     k[2] / m[0],      nu[2] / m[0]],
                      [            0.0,              0.0,             0.0,              1.0],
                      [    k[2] / m[1],     nu[2] / m[1], -k_eq [1]/ m[1], -nu_eq[1] / m[1]]])

        # Compute analytical solution
        x_analytical = np.stack([
            expm(A * t).dot(x_jiminy[0, :]) for t in time], axis=0)

        # Compare the numerical and analytical solutions
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
