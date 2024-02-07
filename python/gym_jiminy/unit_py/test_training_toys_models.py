"""Integration tests to check that everything is working fine, from the
low-level Jiminy engine, to the Gym environment integration. However, it does
not assessed that the viewer is working properly.
"""
import unittest
from typing import Optional, Dict, Any

from torch import nn

from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from utilities import train


SEED = 0
N_THREADS = 5


class ToysModelsStableBaselinesPPO(unittest.TestCase):
    """Solve several official Open AI Gym toys models problems simulated in
    Jiminy using PPO algorithm of Stable Baselines 3 RL framework.

    They are solved consistently in less than 150000 timesteps, and in about
    30000 at best.

    .. warning::
        It requires pytorch>=1.4 and stable-baselines3[extra]>=0.10.0
    """
    @staticmethod
    def _get_default_config_stable_baselines():
        """Return default configuration for stables baselines that should work
        for every toys models.
        """
        # Agent algorithm config
        config = {}
        config['n_steps'] = 4000
        config['batch_size'] = 250
        config['learning_rate'] = 5.0e-4
        config['n_epochs'] = 20
        config['gamma'] = 0.98
        config['gae_lambda'] = 0.94
        config['target_kl'] = 0.1
        config['ent_coef'] = 0.01
        config['vf_coef'] = 0.04
        config['clip_range'] = 0.3
        config['clip_range_vf'] = None
        config['max_grad_norm'] = 1.0
        config['seed'] = SEED

        # Policy model config
        config['policy_kwargs'] = {
            'net_arch': dict(pi=[64, 64], vf=[64, 64]),
            'activation_fn': nn.Tanh,
            'ortho_init': True,
            'log_std_init': 1.0,
            'optimizer_kwargs': {
                'weight_decay': 1e-4,
                'betas': (0.9, 0.999),
                'eps': 1e-6,
            }
        }

        return config

    @classmethod
    def _ppo_training(cls,
                      env_name: str,
                      env_kwargs: Optional[Dict[str, Any]] = None,
                      agent_kwargs: Optional[Dict[str, Any]] = None) -> bool:
        """ Run PPO algorithm on a given algorithm and check if the reward
        threshold has been exceeded.
        """
        # Create a multiprocess environment
        train_env = make_vec_env(
            env_id=env_name,  env_kwargs=env_kwargs or {},
            n_envs=max(0, N_THREADS - 1), vec_env_cls=SubprocVecEnv, seed=SEED)
        test_env = make_vec_env(
            env_id=env_name, env_kwargs=env_kwargs or {},
            n_envs=1, vec_env_cls=DummyVecEnv, seed=SEED)

        # Create the learning agent according to the chosen algorithm
        config = cls._get_default_config_stable_baselines()
        config.update(agent_kwargs or {})
        train_agent = PPO(
            'MlpPolicy', train_env, **config, device='cpu', verbose=False)

        # Run the learning process
        return train(train_agent, test_env, max_timesteps=200000)

    def test_acrobot_stable_baselines(self):
        """Solve acrobot for both continuous and discrete action spaces.
        """
        self.assertTrue(self._ppo_training(
            "gym_jiminy.envs:acrobot", {'continuous': True}))
        self.assertTrue(self._ppo_training(
            "gym_jiminy.envs:acrobot", {'continuous': False}))

    def test_cartpole_stable_baselines(self):
        """Solve cartpole for both continuous and discrete action spaces.
        """
        self.assertTrue(self._ppo_training(
            "gym_jiminy.envs:cartpole", {'continuous': True}))
        self.assertTrue(self._ppo_training(
            "gym_jiminy.envs:cartpole", {'continuous': False}))
