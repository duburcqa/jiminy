"""
@brief TODO

@remark This script requires pytorch>=1.4 and and ray[rllib]==1.0.
"""
import os
import copy
import time
import pathlib
import socket
from datetime import datetime

# ======================== User parameters =========================

GYM_ENV_NAME = "gym_jiminy:jiminy-cartpole-v0"
GYM_ENV_KWARGS = {
    'continuous': True
}
AGENT_ALGORITHM = "PPO"
SEED = 0
N_THREADS = 8
N_GPU = 0

# =================== Configure Python workspace ===================

# GPU device selection must be done at system level to be taken into account
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str,range(N_GPU)))

import gym
from tensorboard.program import TensorBoard

import ray
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

from jiminy_py.viewer import sleep

from fcnet import FullyConnectedNetwork

# ============= Initialize Ray and Tensorboard daemons =============
#
# It will be used later for almost everything from dashboard,
# remote/client management, to multithreaded environment.
#

# Initialize Ray server, if not already running
if not ray.is_initialized():
    ray.init(
        address=None,             # The address of the Ray cluster to connect to, if any.
        num_cpus=N_THREADS,       # Number of CPUs assigned to each raylet (None to disable limit)
        num_gpus=N_GPU,           # Number of GPUs assigned to each raylet (None to disable limit)
        _lru_evict=False,         # Enable object eviction in LRU order if under memory pressure (not recommended)
        local_mode=False,         # If true, the code will be executed serially (for debugging purpose)
        logging_level=20,         # Logging level
        log_to_driver=False,      # Whether to redirect the output from every worker to the driver
        include_dashboard=True,   # Whether to start the Ray dashboard, which displays cluster's status
        dashboard_host="0.0.0.0"  # The host to bind the dashboard server to.
    )

# Configure Tensorboard
log_root_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "log_traj-policy")
if not 'tb' in locals().keys():
    tb = TensorBoard()
    tb.configure(host="0.0.0.0", logdir=log_root_path)
    url = tb.launch()
    print(f"Started Tensorboard {url}. Root directory: {log_root_path}")

# Create log directory
log_name = "_".join((datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                     socket.gethostname().replace('-', '_')))
log_path = os.path.join(log_root_path, log_name)
pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
print(f"Tensorboard logfiles directory: {log_path}")

# Define Ray logger
def logger_creator(config):
    return UnifiedLogger(config, log_path, loggers=None)

# ======================== Configure model =========================

# Register the custom model architecture (it implements 'vf_share_layers')
ModelCatalog.register_custom_model("fully_connected_network", FullyConnectedNetwork)

# Copy the default model configuration
mdl_cfg = copy.deepcopy(MODEL_DEFAULTS)

# Custom model settings
mdl_cfg["custom_model"] = "fully_connected_network"  # Name of a custom model to use. (None to disable)
mdl_cfg["custom_model_config"] = {}                  # Dict of extra options to pass to the custom models

# ========================= Configure RLlib ========================

# Register the environment with custom default constructor arguments
env_creator = lambda env_config: \
    gym.make(GYM_ENV_NAME, **GYM_ENV_KWARGS, **env_config)
register_env("cartpole_env", env_creator)

# Copy the default rllib configuration
rllib_cfg = copy.deepcopy(COMMON_CONFIG)

# Ressources settings
rllib_cfg["framework"] = "torch"                    # Use PyTorch instead of Tensorflow
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
rllib_cfg["horizon"] = 5000             # Number of steps after which the episode is forced to terminate
rllib_cfg["soft_horizon"] = False       # Calculate rewards but don't reset the environment when the horizon is hit
rllib_cfg["no_done_at_end"] = False     # Don't set 'done' at the end of the episode
rllib_cfg["normalize_actions"] = False  # Normalize actions to the upper and lower bounds of the action space
rllib_cfg["clip_actions"] = True        # Whether to clip actions to the upper and lower bounds of the action space
rllib_cfg["clip_rewards"] = False       # Whether to clip rewards prior to experience postprocessing
rllib_cfg["env_config"] = {}            # Arguments to pass to the env creator

# Model settings
rllib_cfg["replay_sequence_length"] = 1  # The number of contiguous environment steps to replay at once
rllib_cfg["model"] = mdl_cfg             # Policy model configuration

# Rollout settings
rllib_cfg["rollout_fragment_length"] = 128     # Sample batches of this size (mult. by `num_envs_per_worker`) are collected from each worker process
rllib_cfg["train_batch_size"] = 512            # Sample batches are concatenated together into batches of this size if sample_async disable, per worker process otherwise
rllib_cfg["batch_mode"] = "truncate_episodes"  # Whether to rollout "complete_episodes" or "truncate_episodes" to `rollout_fragment_length`
rllib_cfg["sample_async"] = False              # Use a background thread for sampling (slightly off-policy)
#rllib_cfg["_use_trajectory_view_api"] = True  # Process trajectories as a whole instead of individual timesteps (experimental feature in 1.0)
rllib_cfg["observation_filter"] = "NoFilter"   # Element-wise observation filter ["NoFilter", "MeanStdFilter"]
rllib_cfg["synchronize_filters"] = False       # Whether to synchronize the statistics of remote filters
rllib_cfg["compress_observations"] = False     # Whether to LZ4 compress individual observations
rllib_cfg["min_iter_time_s"] = 0               # Minimum time per training iteration. To make sure timing differences do not affect training.
rllib_cfg["seed"] = SEED                       # sets the random seed of each worker processes (in conjunction with worker_index)

# Learning settings
rllib_cfg["lr"] = 1.0e-3              # Default learning rate (Not use for actor-critic algorithms such as TD3)
rllib_cfg["gamma"] = 0.99             # Discount factor of the MDP (0.992: <4% after one walk stride)
rllib_cfg["explore"] = True           # Disable exploration completely
rllib_cfg["exploration_config"] = {
    "type": "Random",                 # The Exploration class to use ["EpsilonGreedy", "Random", ...]
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

# Select the right algorithm and import the required dependencies
if AGENT_ALGORITHM == "PPO":
    from ray.rllib.agents.ppo import (PPOTrainer as Trainer,
                                      DEFAULT_CONFIG as AGENT_DEFAULT_CONFIG)
elif AGENT_ALGORITHM == "APPO":
    from ray.rllib.agents.ppo.appo import (APPOTrainer as Trainer,
                                           DEFAULT_CONFIG as AGENT_DEFAULT_CONFIG)
elif AGENT_ALGORITHM == "SAC":
    from ray.rllib.agents.sac import (SACTrainer as Trainer,
                                      DEFAULT_CONFIG as AGENT_DEFAULT_CONFIG)

# Copy the default learning algorithm configuration, including PPO-specific parameters,
# then overwrite the common parameters that has been updated ONLY.
agent_cfg = copy.deepcopy(AGENT_DEFAULT_CONFIG)
for key, value in rllib_cfg.items():
    if COMMON_CONFIG[key] != value:
        agent_cfg[key] = value

if AGENT_ALGORITHM == "PPO":
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
    agent_cfg["lr_schedule"] = None  # Learning rate schedule
    agent_cfg["sgd_minibatch_size"] = 128  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
    agent_cfg["num_sgd_iter"] = 8          # Number of SGD epochs to execute per train batch
    agent_cfg["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training
    agent_cfg["grad_clip"] = 0.5     # Clamp the norm of the gradient during optimization (None to disable)

if AGENT_ALGORITHM == "APPO":
    # Estimators settings
    agent_cfg["vtrace"] = False  # Use V-trace weighted advantages. PPO GAE advantages otherwise.

    # Learning settings
    agent_cfg["use_kl_loss"] = False
    agent_cfg["replay_proportion"] = 0.0 # set >0 to enable experience replay
    agent_cfg["replay_buffer_num_slots"] = 100 # replay_buffer_num_slots * rollout_fragment_length samples to store for replay

    # Optimization settings
    agent_cfg["grad_clip"] = 1.0
    agent_cfg["train_batch_size"] = agent_cfg["sgd_minibatch_size"]
    agent_cfg["minibatch_buffer_size"] = 1
    agent_cfg["decay"] = 0.99
    agent_cfg["momentum"] = 0.0
    agent_cfg["epsilon"] = 0.1

    # Miscellaneous settings
    agent_cfg["num_data_loader_buffers"] = 1 # set >1 to load data into GPUs in parallel.  Increases GPU memory usage proportionally
    agent_cfg["broadcast_interval"] = 1 # max number of workers to broadcast one set of weights to
    agent_cfg["learner_queue_size"] = 16 # max queue size for train batches feeding into the learner
    agent_cfg["learner_queue_timeout"] = 300 # wait for train batches to be available in minibatch buffer queue
    agent_cfg["max_sample_requests_in_flight_per_worker"] = 2 # level of queuing for sampling.

    del agent_cfg["sgd_minibatch_size"], agent_cfg["shuffle_sequences"], \
        agent_cfg["vf_share_layers"], agent_cfg["vf_clip_param"]

if AGENT_ALGORITHM == "SAC":
    # Model settings
    agent_cfg["twin_q"] = True
    agent_cfg["use_state_preprocessor"] = False
    agent_cfg["model"]["fcnet_hiddens"] = []
    agent_cfg["Q_model"] = {
        "fcnet_activation": "tanh",
        "fcnet_hiddens": [64, 64],
    }
    agent_cfg["policy_model"] = {
        "fcnet_activation": "tanh",
        "fcnet_hiddens": [64, 64],
    }

    # Learning settings
    agent_cfg["tau"] = 5e-3                     # Update the target by \tau * policy + (1-\tau) * target_policy.
    agent_cfg["initial_alpha"] = 1.0            # Initial value to use for the entropy weight alpha
    agent_cfg["target_entropy"] = "auto"        # Target entropy lower bound. Inverse of reward scale. If "auto", will be set to -|A|
    agent_cfg["n_step"] = 1                     # N-step target updates

    # Replay buffer settings
    agent_cfg["buffer_size"] = 1000000
    agent_cfg["prioritized_replay"] = False    # If True prioritized replay buffer will be used.
    agent_cfg["prioritized_replay_alpha"] = 0.6
    agent_cfg["prioritized_replay_beta"] = 0.4
    agent_cfg["prioritized_replay_eps"] = 1e-6
    agent_cfg["prioritized_replay_beta_annealing_timesteps"] = 20000
    agent_cfg["final_prioritized_replay_beta"] = 0.4

    # Optimization settings
    agent_cfg["optimization"] = {
        "actor_learning_rate": 1.0e-4,
        "critic_learning_rate": 1.0e-4,
        "entropy_learning_rate": 1.0e-4,
    }
    agent_cfg["learning_starts"] = 10000          # How many steps of the model to sample before learning starts
    agent_cfg["target_network_update_freq"] = 10  # Update the target network every `target_network_update_freq` steps
    agent_cfg["timesteps_per_iteration"] = 300    # Number of env steps to optimize for before returning.
    agent_cfg["grad_clip"] = None                 # If not None, clip gradients during optimization at this value



# ====================== Overwrite some options ====================

# agent_cfg = copy.deepcopy(DEFAULT_CONFIG)
# agent_cfg["lr"] = 5.0e-6
# agent_cfg["lr_schedule"] = [
#     [      0, 5.0e-6],
#     [ 100000, 1.0e-6],
#     [ 400000, 1.0e-6],
#     [ 800000, 1.0e-7],
#     [1000000, 1.0e-7],
#     [1200000, 1.0e-9],
# ]
train_agent = Trainer(agent_cfg, env="cartpole_env")

# ====================== Run the optimization ======================

timesteps_total = 400000
results_fields_filter = [
    "training_iteration",
    "time_total_s",
    "timesteps_total",
    "episodes_total",
    "episode_reward_max",
    "episode_reward_mean",
    "episode_len_mean"
]

try:
    train_agent = Trainer(agent_cfg, "cartpole_env", logger_creator)

    result = {"timesteps_total": 0}
    while result["timesteps_total"] < timesteps_total:
        # Perform one iteration of training the policy
        result = train_agent.train()

        # Print the training status
        msg_data = []
        for field in results_fields_filter:
            if field in result.keys():
                msg_data.append(f"{field}: {result[field]:.5g}")
        print(" - ".join(msg_data))
except KeyboardInterrupt:
    print("Interrupting training...")
finally:
    checkpoint_path = train_agent.save()

# ===================== Enjoy a trained agent ======================

t_end = 10.0  # Testing duration

try:
    env = env_creator(rllib_cfg["env_config"])
    test_agent = Trainer(agent_cfg, "cartpole_env", logger_creator)
    test_agent.restore(checkpoint_path)
    t_init = time.time()
    t_prev = t_init
    while t_prev - t_init < t_end:
        cum_step, cum_reward = 0, 0
        done = False
        obs = env.reset()
        while not done:
            if not (t_prev - t_init < t_end):
                break
            action = test_agent.compute_action(obs, explore=False)
            obs, reward, done, _ = env.step(action)
            cum_step += 1
            cum_reward += reward
            env.render()
            sleep(env.dt - (time.time() - t_prev))
            t_prev = time.time()
        print(f"Episode length: {cum_step} - Cumulative reward: {cum_reward}")
except KeyboardInterrupt:
    print("Interrupting testing...")

# =================== Terminate the Ray backend ====================

train_agent.stop()
test_agent.stop()
ray.shutdown()
