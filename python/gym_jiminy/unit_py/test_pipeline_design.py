""" TODO: Write documentation
"""
import os
import unittest

import numpy as np

from jiminy_py.robot import _gcd
from gym_jiminy.common.pipeline import build_pipeline, load_pipeline


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

        self.ANYmalPipelineEnv = build_pipeline(**{
            'env_config': {
                'env_class': 'gym_jiminy.envs.ANYmalJiminyEnv',
                'env_kwargs': {
                    'step_dt': self.step_dt
                }
            },
            'blocks_config': [{
                'block_class': 'gym_jiminy.common.controllers.PDController',
                'block_kwargs': {
                    'update_ratio': 2,
                    'order': 1,
                    'pid_kp': self.pid_kp,
                    'pid_kd': self.pid_kd,
                    'soft_bounds_margin': 0.0
                },
                'wrapper_kwargs': {
                    'augment_observation': True
                }},
                {
                'wrapper_class': 'gym_jiminy.common.wrappers.StackedJiminyEnv',
                'wrapper_kwargs': {
                    'nested_filter_keys': [
                        ('t',),
                        ('sensors', 'ImuSensor'),
                        ('targets',)
                    ],
                    'num_stack': self.num_stack,
                    'skip_frames_ratio': self.skip_frames_ratio
                }}
            ]
        })

    def test_load_files(self):
        """ TODO: Write documentation
        """
        # Get data path
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        # Load TOML pipeline description, create env and perform a step
        toml_file = os.path.join(data_dir, "anymal_pipeline.toml")
        ANYmalPipelineEnv = load_pipeline(toml_file)
        env = ANYmalPipelineEnv()
        env.reset()
        env.step()

        # Load JSON pipeline description, create env and perform a step
        json_file = os.path.join(data_dir, "anymal_pipeline.json")
        ANYmalPipelineEnv = load_pipeline(json_file)
        env = ANYmalPipelineEnv()
        env.reset()
        env.step()

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

        # Override default 'StackedJiminyEnv' arguments
        num_stack_2 = 2 * self.num_stack
        env = self.ANYmalPipelineEnv(num_stack=num_stack_2)
        self.assertEqual(env.wrapper.num_stack, num_stack_2)

        # Override default 'PDController' arguments
        pid_kp_2 = 2 * self.pid_kp
        env = self.ANYmalPipelineEnv(pid_kp=pid_kp_2)
        self.assertTrue(np.all(env.env.controller.pid_kp == pid_kp_2))

    def test_initial_state(self):
        """ TODO: Write documentation
        """
        # Get initial observation
        env = self.ANYmalPipelineEnv()
        obs = env.reset()

        # Controller target is observed, and has right name
        self.assertTrue('targets' in obs and 'controller_0' in obs['targets'])

        # Target, time, and Imu data are stacked
        self.assertEqual(obs['t'].ndim, 2)
        self.assertEqual(len(obs['t']), self.num_stack)
        self.assertEqual(obs['sensors']['ImuSensor'].ndim, 3)
        self.assertEqual(len(obs['sensors']['ImuSensor']), self.num_stack)
        controller_target_obs = obs['targets']['controller_0']
        self.assertEqual(len(controller_target_obs['q']), self.num_stack)
        self.assertEqual(len(controller_target_obs['v']), self.num_stack)
        self.assertEqual(obs['sensors']['EffortSensor'].ndim, 2)

        # Stacked obs are zeroed
        self.assertTrue(np.all(obs['t'][:-1] == 0.0))
        self.assertTrue(np.all(obs['sensors']['ImuSensor'][:-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs['q'][:-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs['v'][:-1] == 0.0))

        # Action must be zero
        self.assertTrue(np.all(controller_target_obs['q'][-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs['v'][-1] == 0.0))

        # Observation is consistent with internal simulator state
        imu_data_ref = env.simulator.robot.sensors_data['ImuSensor']
        imu_data_obs = obs['sensors']['ImuSensor'][-1]
        self.assertTrue(np.all(imu_data_ref == imu_data_obs))
        state_ref = {'q': env.system_state.q, 'v': env.system_state.v}
        state_obs = obs['state']
        self.assertTrue(np.all(state_ref['q'] == state_obs['q']))
        self.assertTrue(np.all(state_ref['v'] == state_obs['v']))

    def test_step_state(self):
        """ TODO: Write documentation
        """
        # Perform a single step
        env = self.ANYmalPipelineEnv()
        env.reset()
        action = env.env.get_observation()['targets']['controller_0']
        action['q'] += 1.0e-3
        obs, *_ = env.step(action)

        # Observation stacking is skipping the required number of frames
        stack_dt = (self.skip_frames_ratio + 1) * env.observe_dt
        self.assertEqual(obs['t'][-1], stack_dt)

        # Initial observation is consistent with internal simulator state
        controller_target_obs = obs['targets']['controller_0']
        self.assertTrue(np.all(controller_target_obs['q'][-1] == action['v']))
        imu_data_ref = env.simulator.robot.sensors_data['ImuSensor']
        imu_data_obs = obs['sensors']['ImuSensor'][-1]
        self.assertFalse(np.all(imu_data_ref == imu_data_obs))
        state_ref = {'q': env.system_state.q, 'v': env.system_state.v}
        state_obs = obs['state']
        self.assertTrue(np.all(state_ref['q'] == state_obs['q']))
        self.assertTrue(np.all(state_ref['v'] == state_obs['v']))

        # Step until to reach the next stacking breakpoint
        n_steps_breakpoint = int(stack_dt // _gcd(env.step_dt, stack_dt))
        for _ in range(1, n_steps_breakpoint):
            obs, *_ = env.step(action)
        for i, t in enumerate(np.flip(obs['t'])):
            self.assertTrue(np.isclose(
                t, n_steps_breakpoint * env.step_dt - i * stack_dt, 1.0e-6))
        imu_data_ref = env.simulator.robot.sensors_data['ImuSensor']
        imu_data_obs = obs['sensors']['ImuSensor'][-1]
        self.assertTrue(np.all(imu_data_ref == imu_data_obs))

    def test_update_periods(self):
        # Perform a single step and get log data
        env = self.ANYmalPipelineEnv()

        def configure_telemetry() -> None:
            engine_options = env.simulator.engine.get_options()
            engine_options['telemetry']['enableCommand'] = True
            env.simulator.engine.set_options(engine_options)

        env.reset(controller_hook=configure_telemetry)
        env.step()

        # Check that the command is updated 1/2 low-level controller update
        log_vars = env.log_data["variables"]
        u_log = log_vars['HighLevelController.currentCommandLF_HAA']
        self.assertEqual(env.control_dt, 2 * env.unwrapped.control_dt)
        self.assertTrue(np.all(u_log[:2] == 0.0))
        self.assertNotEqual(u_log[1], u_log[2])
        self.assertEqual(u_log[2], u_log[3])
