"""Integration tests to check that everything is working fine, from the
low-level Jiminy engine, to the Gym environment integration. However, it does
not assessed that the viewer is working properly.
"""
import warnings
import unittest
from typing import Dict, Any

import numpy as np
from torch import nn

from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from utilities import train


SEED = 0
N_THREADS = 8


class ToysModelsStableBaselinesPPO(unittest.TestCase):
    """Solve several official Open AI Gym toys models problems simulated in
    Jiminy using PPO algorithm of Stable Baselines 3 RL framework.

    They are solved consistently in less than 100000 timesteps, and in about
    30000 at best.

    .. warning::
        It requires pytorch>=1.4 and stable-baselines3[extra]>=0.10.0
    """
    def setUp(self):
        """Disable all warnings to avoid flooding.
        """
        warnings.filterwarnings("ignore")
        np.seterr(all="ignore")

    @staticmethod
    def _get_default_config_stable_baselines():
        """Return default configuration for stables baselines that should work
        for every toys models.
        """
        # Agent algorithm config
        config = {}
        config['n_steps'] = 128
        config['batch_size'] = 128
        config['learning_rate'] = 1.0e-3
        config['n_epochs'] = 8
        config['gamma'] = 0.99
        config['gae_lambda'] = 0.95
        config['target_kl'] = None
        config['ent_coef'] = 0.01
        config['vf_coef'] = 0.5
        config['clip_range'] = 0.2
        config['clip_range_vf'] = float('inf')
        config['max_grad_norm'] = float('inf')
        config['seed'] = SEED

        # Policy model config
        config['policy_kwargs'] = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': nn.Tanh
        }

        return config

    @classmethod
    def _is_success_ppo_training(cls,
                                 env_name: str,
                                 env_kwargs: Dict[str, Any]) -> bool:
        """ Run PPO algorithm on a given algorithm and check if the reward
        threshold has been exceeded.
        """

        # Create a multiprocess environment
        train_env = make_vec_env(
            env_id=env_name, env_kwargs=env_kwargs, n_envs=int(N_THREADS//2),
            vec_env_cls=SubprocVecEnv, seed=0)
        test_env = make_vec_env(
            env_id=env_name, env_kwargs=env_kwargs, n_envs=1,
            vec_env_cls=DummyVecEnv, seed=0)

        # Create the learning agent according to the chosen algorithm
        config = cls._get_default_config_stable_baselines()
        train_agent = PPO('MlpPolicy', train_env, **config, verbose=False)
        train_agent.eval_env = test_env

        # Run the learning process
        return train(train_agent, max_timesteps=100000)

    def test_cartpole_stable_baselines(self):
        """Solve the Cartpole problem for continuous action space.
        """
        is_success = self._is_success_ppo_training(
            "gym_jiminy.envs:jiminy-cartpole-v0", {'continuous': True})
        self.assertTrue(is_success)
