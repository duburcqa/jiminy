"""Solve the official Open AI Gym Ant-v2 problem simulated in Jiminy using
TD3 algorithm of Ray RLlib reinforcement learning framework.

It solves it consistently in less than 500000 timesteps in average, and in
about 300000 at best.

.. warning::
    This script requires pytorch>=1.4 and ray[rllib]==1.2.
"""
# ======================== User parameters =========================

GYM_ENV_NAME = "gym_jiminy.envs:ant-v0"
DEBUG = False
SEED = 0
N_THREADS = 12
N_GPU = 1

# =================== Configure Python workspace ===================

# GPU device selection must be done at system level to be taken into account
__import__("os").environ["CUDA_VISIBLE_DEVICES"] = \
    ",".join(map(str, range(N_GPU)))

import logging
import gym
import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ddpg.td3 as td3

from gym_jiminy.toolbox.rllib.utilities import initialize, train, test

# Register learning environment
register_env("env", lambda env_config: gym.make(GYM_ENV_NAME, **env_config))

# ============= Initialize Ray and Tensorboard daemons =============

# Initialize Ray backend
logger_creator = initialize(
    num_cpus=N_THREADS, num_gpus=N_GPU, debug=DEBUG)

# ================== Configure learning algorithm ==================

# General options
config = td3.TD3_DEFAULT_CONFIG.copy()
config["framework"] = "tf"
config["eager_tracing"] = True
config["log_level"] = logging.DEBUG if DEBUG else logging.ERROR
config["seed"] = SEED

# Environment options
config["horizon"] = 1000

# TD3-specific options
config["learning_starts"] = 10000
config["exploration_config"]["random_timesteps"] = 20000

# ====================== Run the optimization ======================

train_agent = td3.TD3Trainer(config, "env", logger_creator)
checkpoint_path = train(train_agent, max_timesteps=1000000)

# ===================== Enjoy a trained agent ======================

test_agent = td3.TD3Trainer(config, "env")
test_agent.restore(checkpoint_path)
test(test_agent, explore=False)

# =================== Terminate the Ray backend ====================

train_agent.stop()
test_agent.stop()
ray.shutdown()
