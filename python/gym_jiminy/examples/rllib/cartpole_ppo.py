"""Solve the official Open AI Gym Cartpole problem simulated in Jiminy using
PPO algorithm of Ray RLlib reinforcement learning framework.

It solves it consistently in less than 100000 timesteps in average, and in
about 80000 at best.

.. warning::
    This script requires pytorch>=1.4 and ray[rllib]==1.2.
"""
# flake8: noqa

# ======================== User parameters =========================

GYM_ENV_NAME = "gym_jiminy.envs:cartpole-v0"
GYM_ENV_KWARGS = {
    'continuous': True
}
DEBUG = False
SEED = 0
N_THREADS = 8
N_GPU = 1

# =================== Configure Python workspace ===================

# GPU device selection must be done at system level to be taken into account
__import__("os").environ["CUDA_VISIBLE_DEVICES"] = \
    ",".join(map(str, range(N_GPU)))

import logging
import gym
import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.agents.ppo import (PPOTrainer as Trainer,
                                  DEFAULT_CONFIG as AGENT_DEFAULT_CONFIG)

from gym_jiminy.toolbox.rllib.utilities import initialize, train, test

from tools.fcnet import FullyConnectedNetwork

# Register learning environment
register_env("env", lambda env_config: gym.make(GYM_ENV_NAME, **env_config))

# Register the custom model architecture (it implements 'vf_share_layers')
ModelCatalog.register_custom_model("fully_connected_network", FullyConnectedNetwork)

# ============= Initialize Ray and Tensorboard daemons =============

logger_creator = initialize(
    num_cpus=N_THREADS, num_gpus=N_GPU, debug=DEBUG)

# ======================== Configure model =========================

# Copy the default model configuration
mdl_cfg = MODEL_DEFAULTS.copy()

# Fully-connected network settings
mdl_cfg["fcnet_activation"] = "tanh"            # Nonlinearity for built-in fully connected net ["tanh", "relu", "linear"]
mdl_cfg["fcnet_hiddens"] = [64, 64]             # Number of hidden layers for fully connected net
mdl_cfg["no_final_linear"] = False              # Whether to skip the final linear layer used to resize the outputs to `num_outputs`
mdl_cfg["vf_share_layers"] = False              # Whether layers should be shared for the value function.

# Custom model settings
mdl_cfg["custom_model"] = "fully_connected_network"  # Name of a custom model to use. (None to disable)
mdl_cfg["custom_model_config"] = {}                  # Dict of extra options to pass to the custom models, in addition to the usual ones (defined above)

# ========================= Configure RLlib ========================

# Copy the default rllib configuration
rllib_cfg = COMMON_CONFIG.copy()

# Ressources settings
rllib_cfg["framework"] = "torch"                    # Use PyTorch ("torch") instead of Tensorflow ("tf" or "tfe" (eager mode))
rllib_cfg["log_level"] = logging.DEBUG if DEBUG else logging.ERROR
rllib_cfg["num_gpus"] = N_GPU                       # Number of GPUs to reserve for the trainer process
rllib_cfg["num_workers"] = int(N_THREADS//2)        # Number of rollout worker processes for parallel sampling
rllib_cfg["num_envs_per_worker"] = 1                # Number of environments per worker processes
rllib_cfg["remote_worker_envs"] = False             # Whether to create the envs per worker in individual remote processes. Note it adds overheads
rllib_cfg["num_cpus_per_worker"] = 1                # Number of CPUs to reserve per worker processes
rllib_cfg["num_cpus_for_driver"] = 0                # Number of CPUs to allocate for trainer process
rllib_cfg["remote_env_batch_wait_ms"] = 0           # Duration worker processes are waiting when polling environments. (0 to disable: as soon as one env is ready)
rllib_cfg["extra_python_environs_for_driver"] = {}  # Extra python env vars to set for trainer process
rllib_cfg["extra_python_environs_for_worker"] = {   # Extra python env vars to set for worker processes
    "OMP_NUM_THREADS": "1"                          # Disable multi-threaded linear algebra (numpy, torch...etc), ensuring one thread per worker
}

# Environment settings
rllib_cfg["horizon"] = None               # Number of steps after which the episode is forced to terminate
rllib_cfg["soft_horizon"] = False         # Calculate rewards but don't reset the environment when the horizon is hit
rllib_cfg["no_done_at_end"] = False       # Don't set 'done' at the end of the episode
rllib_cfg["normalize_actions"] = False    # Normalize actions to the upper and lower bounds of the action space
rllib_cfg["clip_actions"] = True          # Whether to clip actions to the upper and lower bounds of the action space
rllib_cfg["clip_rewards"] = False         # Whether to clip rewards prior to experience postprocessing
rllib_cfg["env_config"] = GYM_ENV_KWARGS  # Arguments to pass to the env creator

# Model settings
# rllib_cfg["replay_sequence_length"] = 1  # The number of contiguous environment steps to replay at once
rllib_cfg["model"] = mdl_cfg             # Policy model configuration

# Rollout settings
rllib_cfg["rollout_fragment_length"] = 128     # Sample batches of this size (mult. by `num_envs_per_worker`) are collected from each worker process
rllib_cfg["train_batch_size"] = 512            # Sample batches are concatenated together into batches of this size if sample_async disable, per worker process otherwise
rllib_cfg["batch_mode"] = "truncate_episodes"  # Whether to rollout "complete_episodes" or "truncate_episodes" to `rollout_fragment_length`
rllib_cfg["sample_async"] = False              # Use a background thread for sampling (slightly off-policy)
rllib_cfg["observation_filter"] = "NoFilter"   # Element-wise observation filter ["NoFilter", "MeanStdFilter"]
rllib_cfg["compress_observations"] = False     # Whether to LZ4 compress individual observations
rllib_cfg["min_iter_time_s"] = 0               # Minimum time per training iteration. To make sure timing differences do not affect training.
rllib_cfg["seed"] = SEED                       # sets the random seed of each worker processes (in conjunction with worker_index)

# Learning settings
rllib_cfg["lr"] = 1.0e-3              # Default learning rate (Not use for actor-critic algorithms such as TD3)
rllib_cfg["gamma"] = 0.99             # Discount factor of the MDP (0.992: <4% after one walk stride)
rllib_cfg["explore"] = True           # Disable exploration completely
rllib_cfg["exploration_config"] = {
    "type": "StochasticSampling",     # The Exploration class to use ["EpsilonGreedy", "Random", ...]
}
rllib_cfg["shuffle_buffer_size"] = 0  # Shuffle input batches via a sliding window buffer of this size (0 = disable)

# Evaluation settings
rllib_cfg["evaluation_interval"] = None    # Evaluate every `evaluation_interval` training iterations (None to disable)
rllib_cfg["evaluation_num_episodes"] = 50  # Number of episodes to run per evaluation
rllib_cfg["evaluation_num_workers"] = 0    # Number of dedicated parallel workers to use for evaluation (0 to disable)

# Debugging and monitoring settings
rllib_cfg["monitor"] = False                   # Whether to save episode stats and videos
rllib_cfg["ignore_worker_failures"] = False    # Whether to save episode stats and videos
rllib_cfg["log_level"] = "INFO"                # Set the ray.rllib.* log level for the agent process and its workers [DEBUG, INFO, WARN, or ERROR]
rllib_cfg["log_sys_usage"] = True              # Monitor system resource metrics (requires `psutil` and `gputil`)
rllib_cfg["metrics_smoothing_episodes"] = 100  # Smooth metrics over this many episodes
rllib_cfg["collect_metrics_timeout"] = 180     # Wait for metric batches for this duration. If not in time, collect in the next train iteration
rllib_cfg["timesteps_per_iteration"] = 0       # Minimum env steps to optimize for per train call. It does not affect learning, only monitoring

# ================== Configure learning algorithm ==================

# Copy the default learning algorithm configuration, including PPO-specific parameters,
# then overwrite the common parameters that has been updated ONLY.
agent_cfg = AGENT_DEFAULT_CONFIG.copy()
for key, value in rllib_cfg.items():
    if COMMON_CONFIG[key] != value:
        agent_cfg[key] = value

# Estimators settings
agent_cfg["use_gae"] = True     # Use the Generalized Advantage Estimator (GAE) with a value function (https://arxiv.org/pdf/1506.02438.pdf)
agent_cfg["use_critic"] = True  # Use a critic as a value baseline (otherwise don't use any; required for using GAE).
agent_cfg["lambda"] = 0.95      # The GAE(lambda) parameter.

# Learning settings
agent_cfg["kl_coeff"] = 0.0                 # Initial coefficient for KL divergence. (0.0 for L^CLIP)
agent_cfg["kl_target"] = 0.01               # Target value for KL divergence
agent_cfg["vf_share_layers"] = False        # Share layers for value function. If you set this to True, it's important to tune vf_loss_coeff
agent_cfg["vf_loss_coeff"] = 0.5            # Coefficient of the value function loss
agent_cfg["entropy_coeff"] = 0.01           # Coefficient of the entropy regularizer
agent_cfg["entropy_coeff_schedule"] = None  # Decay schedule for the entropy regularizer
agent_cfg["clip_param"] = 0.2               # PPO clip parameter
agent_cfg["vf_clip_param"] = float("inf")   # Clip param for the value function. Note that this is sensitive to the scale of the rewards (-1 to disable)

# Optimization settings
agent_cfg["lr_schedule"] = None        # Learning rate schedule
agent_cfg["sgd_minibatch_size"] = 128  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
agent_cfg["num_sgd_iter"] = 8          # Number of SGD epochs to execute per train batch
agent_cfg["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training
agent_cfg["grad_clip"] = None          # Clamp the norm of the gradient during optimization (None to disable)

# ====================== Run the optimization ======================

agent_cfg["lr"] = 1.0e-3
agent_cfg["lr_schedule"] = None

train_agent = Trainer(agent_cfg, "env", logger_creator)
checkpoint_path = train(train_agent, max_timesteps=100000)

# ===================== Enjoy a trained agent ======================

test_agent = Trainer(agent_cfg, "env", logger_creator)
test_agent.restore(checkpoint_path)
test(test_agent, explore=False)

# =================== Terminate the Ray backend ====================

train_agent.stop()
test_agent.stop()
ray.shutdown()
