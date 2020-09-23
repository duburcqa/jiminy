# This file aims at verifying the sanity of the physics and the integration
# method of jiminy on simple models.
import unittest
import numpy as np
from pinocchio import Force, Quaternion
from pinocchio.rpy import rpyToMatrix

from jiminy_py import core as jiminy


# Small tolerance for numerical equality.
# The integration error is supposed to be bounded.
TOLERANCE = 1e-7


class SimulateWheel(unittest.TestCase):
    def setUp(self):
        # Load URDF, create robot.
        urdf_path = "data/wheel.urdf"
        from wdc_dynamicsteller import DynamicsTeller, rootjoint
        self.dyn = DynamicsTeller.make(urdf_path, rootjoint.FREEFLYER)

        # Create the jiminy robot
        self.robot = jiminy.Robot()
        self.robot.initialize(urdf_path, has_freeflyer=True)

        # Store wheel radius for further reference.
        idx = self.robot.pinocchio_model.getFrameId("wheel")
        self.wheel_radius = 0.5
        # self.wheel_radius = self.robot.pinocchio_model.frames[idx].placement.translation[2]

        self.ground_normal = np.array([0., 0., 1.])
        self.wheel_axis = np.array([0., 1., 0.])

        # Create wheel constraint
        wheel_constraint = jiminy.WheelConstraint("wheel",
                                                  self.wheel_radius,
                                                  self.ground_normal,
                                                  self.wheel_axis)
        self.robot.add_constraint("wheel", wheel_constraint)


    def test_rolling_motion(self):
        """
        @brief Validate the rolling motion of the wheel when applying an input torque.
        """
        # Extract wheel inertia about its axis.
        I = self.robot.pinocchio_model.inertias[-1].inertia[1, 1]
        m = self.robot.pinocchio_model.inertias[-1].mass
        I += m * self.wheel_radius**2

        # No controller
        def computeCommand(t, q, v, sensors_data, u):
            u[:] = 0.0

        # Internal dynamics: apply constant torque onto wheel axis.
        u_wheel = 0.1
        # u_wheel = 1.0
        def internalDynamics(t, q, v, sensors_data, u):
            u[4] = u_wheel

        controller = jiminy.ControllerFunctor(computeCommand, internalDynamics)
        controller.initialize(self.robot)
        # Create the engine
        engine = jiminy.Engine()
        engine.initialize(self.robot, controller)

        # Run simulation
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]) # [TX,TY,TZ],[QX,QY,QZ,QW]
        # tf = 0.5
        tf = 2.0

        dt_list = np.logspace(-2, -5, 5)
        # dt_list = np.logspace(-2, -6, 10)
        # dt_list = np.logspace(-1, -5, 10)
        error = []

        for dt in dt_list:
            print(dt)
            options = engine.get_options()
            options["stepper"]["dtMax"] = dt
            options["stepper"]["odeSolver"] = "euler_explicit"
            engine.set_options(options)
            engine.simulate(tf, x0)



            log_data, _ = engine.get_log()
            time = log_data['Global.Time']
            x_jiminy = np.stack([log_data['HighLevelController.' + s]
                                    for s in self.robot.logfile_position_headers + \
                                            self.robot.logfile_velocity_headers], axis=-1)
            error.append(np.max(np.abs(x_jiminy[-1000:, 2])))

        import matplotlib.pyplot as plt
        plt.scatter(dt_list, error)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        qt = [x[:7] for x in x_jiminy]
        vt = [x[7:] for x in x_jiminy]

        # Expected motion: the wheel undergoes a constant acceleration u_wheel / I
        w_wheel = u_wheel / I * time
        v_wheel = self.wheel_radius * w_wheel
        angle_wheel = w_wheel * time / 2.0
        quat_wheel = [Quaternion(rpyToMatrix(np.array([[0, y, 0]]).T)) for y in angle_wheel]
        pos_wheel = v_wheel * time / 2.0

        x_analytical = np.zeros_like(x_jiminy)
        x_analytical[:, 0] = pos_wheel

        x_analytical[:, 3:7] =  np.array([q.coeffs() for q in quat_wheel])
        x_analytical[:, 7:10] = np.array([np.array([v, 0, 0]) for q, v in zip(quat_wheel, v_wheel)])

        quat_wheel = [Quaternion((v[3:7].astype(float))) for v in x_jiminy]
        quat_wheel = quat_wheel[1:] + [quat_wheel[0]]

        x_jiminy[:, 7:10] = np.array([q * v for q, v in zip(quat_wheel, x_jiminy[:, 7:10])])
        x_analytical[:, 11] = w_wheel

        import matplotlib.pyplot as plt

        plt.subplot(321)
        plt.plot(x_jiminy[:, 0])
        plt.plot(x_analytical[:, 0])
        plt.title('px')


        plt.subplot(323)
        plt.plot(x_jiminy[:, 1])
        plt.plot(x_analytical[:, 1])
        plt.title('py')

        plt.subplot(325)
        plt.plot(x_jiminy[:, 2])
        plt.plot(x_analytical[:, 2])
        plt.title('pz')


        plt.subplot(322)
        plt.plot(x_jiminy[:, 7])
        plt.plot(x_analytical[:, 7])
        plt.title('vx')


        plt.subplot(324)
        plt.plot(x_jiminy[:, 8])
        plt.plot(x_analytical[:, 8])
        plt.title('vy')

        plt.subplot(326)
        plt.plot(x_jiminy[:, 9])
        plt.plot(x_analytical[:, 9])
        plt.title('vz')

        plt.show()

        # plt.subplot(321)
        # plt.plot(x_jiminy[:, 7])
        # plt.plot(x_analytical[:, 7])


        # plt.subplot(322)
        # plt.plot(x_jiminy[:, 8])
        # plt.plot(x_analytical[:, 8])

        # plt.subplot(323)
        # plt.plot(x_jiminy[:, 9])
        # plt.plot(x_analytical[:, 9])


        # plt.subplot(324)
        # plt.plot(x_jiminy[:, 10])
        # plt.plot(x_analytical[:, 10])


        # plt.subplot(325)
        # plt.plot(x_jiminy[:, 11])
        # plt.plot(x_analytical[:, 11])

        # plt.subplot(326)
        # plt.plot(x_jiminy[:, 12])
        # plt.plot(x_analytical[:, 12])

        plt.show()
        self.assertTrue(np.allclose(x_jiminy, x_analytical, atol=TOLERANCE))

if __name__ == '__main__':
    unittest.main()
