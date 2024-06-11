""" TODO: Write documentation
"""
import gc
import os
import weakref
import unittest

import numpy as np
import gymnasium as gym

from jiminy_py.robot import _gcd
from jiminy_py.log import extract_trajectory_from_log
from gym_jiminy.common.utils import (
    save_trajectory_to_hdf5, build_pipeline, load_pipeline)
from gym_jiminy.common.bases import InterfaceJiminyEnv
from gym_jiminy.common.blocks import PDController


TOLERANCE = 1.0e-6


class PipelineDesign(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        """ TODO: Write documentation
        """
        self.step_dt = 0.04
        self.pid_kp = np.full((12,), fill_value=1500)
        self.pid_kd = np.full((12,), fill_value=0.01)
        self.num_stack = 3
        self.skip_frames_ratio = 2

        self.ANYmalPipelineEnv = build_pipeline(
            env_config=dict(
                cls='gym_jiminy.envs.ANYmalJiminyEnv',
                kwargs=dict(
                    step_dt=self.step_dt,
                    debug=False
                )
            ),
            layers_config=[
                dict(
                    block=dict(
                        cls='gym_jiminy.common.blocks.PDController',
                        kwargs=dict(
                            update_ratio=2,
                            kp=self.pid_kp,
                            kd=self.pid_kd,
                            joint_position_margin=0.0,
                            joint_velocity_limit=float("inf"),
                            joint_acceleration_limit=float("inf")
                        )
                    ),
                    wrapper=dict(
                        kwargs=dict(
                            augment_observation=True
                        )
                    )
                ), dict(
                    block=dict(
                        cls='gym_jiminy.common.blocks.PDAdapter',
                        kwargs=dict(
                            update_ratio=-1,
                            order=1,
                        )
                    ),
                    wrapper=dict(
                        kwargs=dict(
                            augment_observation=False
                        )
                    )
                ), dict(
                    block=dict(
                        cls='gym_jiminy.common.blocks.MahonyFilter',
                        kwargs=dict(
                            update_ratio=1,
                            exact_init=True,
                            kp=1.0,
                            ki=0.1
                        )
                    )
                ), dict(
                    wrapper=dict(
                        cls='gym_jiminy.common.wrappers.StackObservation',
                        kwargs=dict(
                            nested_filter_keys=[
                                ('t',),
                                ('measurements', 'ImuSensor'),
                                ('actions',)
                            ],
                            num_stack=self.num_stack,
                            skip_frames_ratio=self.skip_frames_ratio
                        )
                    )
                )
            ]
        )

    def test_load_files(self):
        """ TODO: Write documentation
        """
        # Get data path
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        # Generate machine-dependent reference trajectory
        env = self.ANYmalPipelineEnv()
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action)
        env.stop()
        trajectory = extract_trajectory_from_log(env.log_data)
        save_trajectory_to_hdf5(
            trajectory, os.path.join(data_dir, "anymal_trajectory.hdf5"))

        # Load TOML pipeline description, create env and perform a step
        toml_file = os.path.join(data_dir, "anymal_pipeline.toml")
        ANYmalPipelineEnv = load_pipeline(toml_file)
        env = ANYmalPipelineEnv()
        env.reset(seed=0)
        env.step(env.action)

        # Load JSON pipeline description, create env and perform a step
        json_file = os.path.join(data_dir, "anymal_pipeline.json")
        ANYmalPipelineEnv = load_pipeline(json_file)
        env = ANYmalPipelineEnv()
        env.reset(seed=0)
        env.step(env.action)

    def test_override_default(self):
        """ TODO: Write documentation
        """
        # Override default environment arguments
        step_dt_2 = 2 * self.step_dt
        env = self.ANYmalPipelineEnv(step_dt=step_dt_2)
        self.assertEqual(env.unwrapped.step_dt, step_dt_2)

        # It does not override the default persistently
        env = self.ANYmalPipelineEnv()
        self.assertEqual(env.unwrapped.step_dt, self.step_dt)

    def test_memory_leak(self):
        """Check that memory is freed when environment goes out of scope.

        This test aims to detect circular references between Python and C++
        objects that cannot be tracked by Python, which would make it
        impossible for the garbage collector to release memory.
        """
        env = self.ANYmalPipelineEnv()
        env.reset(seed=0)
        proxy = weakref.proxy(env)
        env = None
        gc.collect()
        self.assertRaises(ReferenceError, lambda: proxy.action)

    def test_initial_state(self):
        """ TODO: Write documentation
        """
        # Get initial observation
        env = self.ANYmalPipelineEnv()
        obs, _ = env.reset(seed=0)

        # Controller target is observed, and has right name
        self.assertTrue('actions' in obs and 'pd_controller' in obs['actions'])

        # Target, time, and Imu data are stacked
        self.assertEqual(obs['t'].ndim, 1)
        self.assertEqual(len(obs['t']), self.num_stack)
        self.assertEqual(obs['measurements']['ImuSensor'].ndim, 3)
        self.assertEqual(len(obs['measurements']['ImuSensor']), self.num_stack)
        controller_target_obs = obs['actions']['pd_controller']
        self.assertEqual(len(controller_target_obs), self.num_stack)
        self.assertEqual(obs['measurements']['EffortSensor'].ndim, 2)

        # Stacked obs are zeroed
        self.assertTrue(np.all(obs['t'][:-1] == 0.0))
        self.assertTrue(np.all(obs['measurements']['ImuSensor'][:-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs[:-1] == 0.0))

        # Action must be zero
        self.assertTrue(np.all(controller_target_obs[-1] == 0.0))

        # Observation is consistent with internal simulator state
        imu_data_ref = env.simulator.robot.sensor_measurements['ImuSensor']
        imu_data_obs = obs['measurements']['ImuSensor'][-1]
        self.assertTrue(np.all(imu_data_ref == imu_data_obs))
        state_ref = {'q': env.robot_state.q, 'v': env.robot_state.v}
        state_obs = obs['states']['agent']
        self.assertTrue(np.all(state_ref['q'] == state_obs['q']))
        self.assertTrue(np.all(state_ref['v'] == state_obs['v']))

    def test_stacked_obs(self):
        """ TODO: Write documentation
        """
        # Perform a single step
        env = self.ANYmalPipelineEnv()
        env.reset(seed=0)
        action = env.action + 1.0e-3
        obs, *_ = env.step(action)

        # Extract PD controller wrapper env
        env_ctrl = env.env.env.env
        assert isinstance(env_ctrl.controller, PDController)

        # Observation stacking is skipping the required number of frames
        stack_dt = (self.skip_frames_ratio + 1) * env.observe_dt
        t_obs_last = env.step_dt - env.step_dt % stack_dt
        self.assertTrue(np.isclose(
            obs['t'][-1], env.stepper_state.t, TOLERANCE))
        for i in range(1, self.num_stack):
            self.assertTrue(np.isclose(
                obs['t'][::-1][i], t_obs_last - (i - 1) * stack_dt, TOLERANCE))

        # Initial observation is consistent with internal simulator state
        controller_target_obs = obs['actions']['pd_controller']
        self.assertTrue(np.all(controller_target_obs[-1] == env_ctrl.action))
        imu_data_ref = env.simulator.robot.sensor_measurements['ImuSensor']
        imu_data_obs = obs['measurements']['ImuSensor'][-1]
        self.assertTrue(np.all(imu_data_ref == imu_data_obs))
        state_ref = {'q': env.robot_state.q, 'v': env.robot_state.v}
        state_obs = obs['states']['agent']
        self.assertTrue(np.all(state_ref['q'] == state_obs['q']))
        self.assertTrue(np.all(state_ref['v'] == state_obs['v']))

        # Step until to reach the next stacking breakpoint
        n_steps_breakpoint = int(stack_dt // _gcd(env.step_dt, stack_dt))
        for _ in range(1, n_steps_breakpoint):
            obs, *_ = env.step(action)
        for i, t in enumerate(np.flip(obs['t'])):
            self.assertTrue(np.isclose(
                t, n_steps_breakpoint * env.step_dt - i * stack_dt, TOLERANCE))
        imu_data_ref = env.simulator.robot.sensor_measurements['ImuSensor']
        imu_data_obs = obs['measurements']['ImuSensor'][-1]
        self.assertTrue(np.all(imu_data_ref == imu_data_obs))

    def test_update_periods(self):
        # Perform a single step and get log data
        env = self.ANYmalPipelineEnv()

        def configure_telemetry() -> InterfaceJiminyEnv:
            engine_options = env.simulator.get_options()
            engine_options['telemetry']['enableCommand'] = True
            engine_options['stepper']['logInternalStepperSteps'] = False
            env.simulator.set_options(engine_options)
            return env

        env.reset(seed=0, options=dict(reset_hook=configure_telemetry))
        env.step(env.action)
        env.stop()

        controller = env.env.env.env.controller
        assert isinstance(controller, PDController)

        # Check that the PD command is updated 1/2 low-level controller update
        log_vars = env.log_data['variables']
        u_log = log_vars['currentCommandLF_HAA']
        self.assertEqual(controller.control_dt, 2 * env.unwrapped.control_dt)
        self.assertTrue(np.all(u_log[:2] == 0.0))
        self.assertNotEqual(u_log[1], u_log[2])
        self.assertEqual(u_log[2], u_log[3])
