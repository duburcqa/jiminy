"""
@brief This file aims at verifying the sanity of the physics and the
       integration method of jiminy on simple models.
"""
import unittest

import numpy as np

import jiminy_py.core as jiminy
from jiminy_py.simulator import Simulator
from jiminy_py.log import extract_variables_from_log
from jiminy_py.dynamics import update_quantities

from utilities import load_urdf_default


class SimulateDensePole(unittest.TestCase):
    """Simulate the motion of a "dense" pole, comparing against python
    integration.
    """
    def setUp(self):
        # Create the jiminy robot and controller
        self.robot = load_urdf_default("dense_pole.urdf", has_freeflyer=False)

        # Backup original pinocchio model
        self.pinocchio_model_orig = self.robot.pinocchio_model_th.copy()

        # Create a simulator using this robot
        self.simulator = Simulator(self.robot)

        # Define some constant parameters
        self.flex_joint_name = "link1_to_link2"
        self.flex_stiffness = 100.0
        self.flex_damping = 5.0
        self.flex_inertia = 0.5
        self.joint_limit = 0.002
        self.transition_eps = 1e-4

        # Configure joint bounds
        model_options = self.robot.get_model_options()
        model_options['joints']['positionLimitLower'] = [-self.joint_limit]
        model_options['joints']['positionLimitUpper'] = [self.joint_limit]
        model_options['joints']['positionLimitFromUrdf'] = False
        self.robot.set_model_options(model_options)

        # Configure the motors
        for motor in self.robot.motors:
            motor_options = motor.get_options()
            motor_options["enableVelocityLimit"] = False
            motor.set_options(motor_options)

        # Configure the integrator
        engine_options = self.simulator.get_options()
        engine_options['stepper']['tolAbs'] = 1e-9
        engine_options['stepper']['tolRel'] = 1e-8
        engine_options['constraints']['regularization'] = 0.0
        engine_options['contacts']['transitionEps'] = self.transition_eps
        self.simulator.set_options(engine_options)

    def test_flex_model(self):
        """Test if the result is the same with true and virtual inertia in
        the simple case where the flexibility is the only moving joint. Then,
        check that it matches the theoretical dynamical model.
        """
        # Define some constants
        t_end, step_dt = 2.0, 2e-4

        # Add fixed joint constraint
        const = jiminy.JointConstraint("base_to_link1")
        const.baumgarte_freq = 50.0
        self.robot.add_constraint("fixed_joint", const)

        # Configure the engine
        engine_options = self.simulator.get_options()
        engine_options['stepper']['sensorsUpdatePeriod'] = step_dt
        self.simulator.set_options(engine_options)

        # Extract some proxies for convenience
        pinocchio_model_th = self.robot.pinocchio_model_th

        # We set inertia along non-moving axis to 1.0 for numerical stability
        twist_flex_all = []
        q_init, v_init = np.array((0.0,)), np.array((0.0,))
        for is_flex_inertia_virtual in (True, False):
            # Alter flex inertia if non-virtual
            frame_id = pinocchio_model_th.getFrameId(self.flex_joint_name)
            frame = pinocchio_model_th.frames[frame_id]
            joint_id = frame.parent
            pinocchio_model_th.inertias[joint_id] = (
                self.pinocchio_model_orig.inertias[joint_id])
            if not is_flex_inertia_virtual:
                pinocchio_model_th.inertias[joint_id].inertia += np.diag(
                    np.full((3,), self.flex_inertia))
            pinocchio_model_th.frames[frame_id].inertia = (
                self.pinocchio_model_orig.frames[frame_id].inertia)
            if not is_flex_inertia_virtual:
                pinocchio_model_th.frames[frame_id].inertia.inertia += np.diag(
                    np.full((3,), self.flex_inertia))

            # Specify flexibility options
            model_options = self.robot.get_model_options()
            model_options['dynamics']['enableFlexibility'] = True
            flex_options = [{
                'frameName': self.flex_joint_name,
                'stiffness': np.full((3,), self.flex_stiffness),
                'damping': np.full((3,), self.flex_damping),
                'inertia': np.array([1e-3, 0.0, 0.0]),
            }]
            if is_flex_inertia_virtual:
                flex_options[0]['inertia'] = np.full((3,), self.flex_inertia)
                flex_options.append({
                    'frameName': "link2_to_link3",
                    'stiffness': np.zeros((3,)),
                    'damping': np.zeros((3,)),
                    'inertia': np.full((3,), 1e6),
                })
            model_options['dynamics']['flexibilityConfig'] = flex_options
            self.robot.set_model_options(model_options)

            # Launch the simulation
            self.simulator.simulate(
                t_end, q_init, v_init, show_progress_bar=False,
                is_state_theoretical=True)

            # Extract flex twist angle over time
            log_vars = self.simulator.log_data["variables"]
            q_flex = np.stack(extract_variables_from_log(log_vars, [
                f"currentPosition{self.flex_joint_name}Quat{e}"
                for e in ('X', 'Y', 'Z', 'W')
                ]), axis=0)
            twist_flex = 2 * np.arctan2(q_flex[2], q_flex[3])
            twist_flex_all.append(twist_flex)

        assert np.allclose(*twist_flex_all, atol=1e-7)

        # Extract parameters of rigid-body dynamics equation
        q_flex_init = self.robot.get_extended_position_from_theoretical(q_init)
        v_flex_init = self.robot.get_extended_velocity_from_theoretical(v_init)
        update_quantities(
            self.robot, q_flex_init, v_flex_init, use_theoretical_model=False)
        inertia = self.robot.pinocchio_data.Ycrb[2]
        m = inertia.mass
        g = - self.robot.pinocchio_model.gravity.linear[2]
        l = inertia.lever[0]
        I_equiv = inertia.inertia[2, 2] + m * l ** 2

        # Integrate rigid-body model
        theta_all, dtheta = [0.0,], 0.0
        for _ in range(int(np.round(t_end / step_dt))):
            theta = theta_all[-1]
            ddtheta = (
                - m * g * l * np.cos(theta)
                - self.flex_stiffness * theta
                - self.flex_damping * dtheta) / I_equiv
            dtheta += ddtheta * step_dt
            theta_all.append(theta + dtheta * step_dt)

        assert np.allclose(twist_flex_all[0], theta_all, atol=1e-4)

    def test_joint_position_limits(self):
        """Test that the joint position limits are active when they are
        supposed to be.
        """
        # Define some constants
        t_end, step_dt = 0.05, 1e-5

        # Configure the engine
        engine_options = self.simulator.get_options()
        engine_options['stepper']['odeSolver'] = 'euler_explicit'
        engine_options['stepper']['dtMax'] = step_dt
        self.simulator.set_options(engine_options)

        # Start the simulation
        self.simulator.start(np.array((0.0,)), np.array((1.0,)))

        # Get joint bounds constraint
        const = next(iter(self.robot.constraints.bounds_joints.values()))

        # Simulate for a while
        branches = set()
        is_enabled = const.is_enabled
        for _ in range(int(np.round(t_end / step_dt))):
            self.simulator.step(step_dt)
            theta = self.simulator.robot_state.q[0]
            if self.joint_limit - np.abs(theta) <= 0.0:
                assert const.is_enabled
                branches.add(0 if is_enabled else 1)
            elif self.joint_limit - np.abs(theta) < self.transition_eps:
                if is_enabled:
                    assert const.is_enabled
                    branches.add(2)
                else:
                    assert not const.is_enabled
                    branches.add(3)
            else:
                assert not const.is_enabled
                branches.add(4)
            is_enabled = const.is_enabled
        self.simulator.stop()
        assert branches == set((0, 1, 2, 3, 4))

if __name__ == '__main__':
    unittest.main()
