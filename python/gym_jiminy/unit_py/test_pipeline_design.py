""" TODO: Write documentation
"""
import unittest

import numpy as np

from gym_jiminy.common.control_impl import PDController
from gym_jiminy.common.wrappers import StackedJiminyEnv
from gym_jiminy.common.pipeline_bases import build_pipeline
from gym_jiminy.envs import ANYmalJiminyEnv


class PipelineDesign(unittest.TestCase):
    """ TODO: Write documentation
    """
    def setUp(self):
        """ TODO: Write documentation
        """
        self.step_dt = 5.0e-3
        self.pid_kp = np.full((12,), fill_value=1.5e3)
        self.pid_kd = np.full((12,), fill_value=3.0e-3)
        self.num_stack = 3
        self.skip_frames_ratio = 2

        self.ANYmalPipelineEnv = build_pipeline(**{
            'env_config': {
                'env_class': ANYmalJiminyEnv,
                'env_kwargs': {
                    'step_dt': self.step_dt
                }
            },
            'blocks_config': [{
                'block_class': PDController,
                'block_kwargs': {
                    'update_ratio': 2,
                    'pid_kp': self.pid_kp,
                    'pid_kd': self.pid_kd
                },
                'wrapper_kwargs': {
                    'augment_observation': True
                }},
                {
                'wrapper_class': StackedJiminyEnv,
                'wrapper_kwargs': {
                    'nested_fields_list': [
                        ('t',),
                        ('sensors', 'ImuSensor'),
                        ('targets',)
                    ],
                    'num_stack': self.num_stack,
                    'skip_frames_ratio': self.skip_frames_ratio
                }}
            ]
        })

    def test_override_default(self):
        """ TODO: Write documentation
        """
        # Override default environment arguments
        step_dt_2 = 2 * self.step_dt
        env = self.ANYmalPipelineEnv(step_dt=step_dt_2)
        self.assertTrue(env.unwrapped.step_dt == step_dt_2)

        # It does not override the default persistently
        env = self.ANYmalPipelineEnv()
        self.assertTrue(env.unwrapped.step_dt == self.step_dt)

        # Override default 'StackedJiminyEnv' arguments
        num_stack_2 = 2 * self.num_stack
        env = self.ANYmalPipelineEnv(num_stack=num_stack_2)
        self.assertTrue(env.wrapper.num_stack == num_stack_2)

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
        self.assertTrue(obs['t'].ndim == 2)
        self.assertTrue(len(obs['t']) == self.num_stack)
        self.assertTrue(obs['sensors']['ImuSensor'].ndim == 3)
        self.assertTrue(len(obs['sensors']['ImuSensor']) == self.num_stack)
        controller_target_obs = obs['targets']['controller_0']
        self.assertTrue(len(controller_target_obs['Q']) == self.num_stack)
        self.assertTrue(len(controller_target_obs['V']) == self.num_stack)
        self.assertTrue(obs['sensors']['EffortSensor'].ndim == 2)

        # Stacked obs are zeroed
        self.assertTrue(np.all(obs['t'][:-1] == 0.0))
        self.assertTrue(np.all(obs['sensors']['ImuSensor'][:-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs['Q'][:-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs['V'][:-1] == 0.0))

        # Action must be zero
        self.assertTrue(np.all(controller_target_obs['Q'][-1] == 0.0))
        self.assertTrue(np.all(controller_target_obs['V'][-1] == 0.0))

        # Observation is consistent with internal simulator state
        imu_data_ref = env.simulator.robot.sensors_data['ImuSensor']
        imu_data_obs = obs['sensors']['ImuSensor'][-1]
        self.assertTrue(np.all(imu_data_ref == imu_data_obs))
        state_ref = np.concatenate((env.simulator.engine.system_state.q,
                                    env.simulator.engine.system_state.v))
        state_obs = obs['state']
        self.assertTrue(np.all(state_ref == state_obs))

    def test_step_state(self):
        """ TODO: Write documentation
        """
        # Perform a single step
        env = self.ANYmalPipelineEnv()
        env.reset()
        action = env.env.get_observation()['targets']['controller_0']
        action['Q'] += 1.0e-3
        obs, _, _, _ = env.step(action)

        # Observation stacking is skipping the required number of frames
        stack_dt = (self.skip_frames_ratio + 1) * env.observe_dt
        self.assertTrue(obs['t'][-1] == stack_dt)

        # Initial observation is consistent with internal simulator state
        controller_target_obs = obs['targets']['controller_0']
        self.assertTrue(np.all(controller_target_obs['Q'][-1] == action['Q']))
        imu_data_ref = env.simulator.robot.sensors_data['ImuSensor']
        imu_data_obs = obs['sensors']['ImuSensor'][-1]
        self.assertFalse(np.all(imu_data_ref == imu_data_obs))
        state_ref = np.concatenate((env.simulator.engine.system_state.q,
                                    env.simulator.engine.system_state.v))
        state_obs = obs['state']
        self.assertTrue(np.all(state_ref == state_obs))

        # Step manually to reach the next stacking breakpoint
        env.simulator.step(stack_dt - env.step_dt % stack_dt)
        obs = env.get_observation()
        self.assertTrue(obs['t'][-3] == 0.0)
        self.assertTrue(obs['t'][-2] == stack_dt)
        self.assertTrue(obs['t'][-1] == 2 * stack_dt)
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
        log_data, _ = env.get_log()

        # Check that the command is updated 1/2 low-level controller update
        self.assertTrue(env.control_dt == 2 * env.unwrapped.control_dt)
        u_log = log_data['HighLevelController.currentCommandLF_HAA']
        self.assertTrue(np.all(u_log[:2] == 0.0))
        self.assertTrue(u_log[1] != u_log[2] and u_log[2] == u_log[3])
