import time

import gym
import ray
from ray import rllib
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from jiminy_py.viewer import sleep


GYM_ENV_NAME = "gym_jiminy:jiminy-cartpole-v0"


# ================= Initialize the Ray backend =================
#
# It will be used later for almost everything from dashboard,
# remote/client management, to multithreaded environment.
#

ray.init(
    address=None,         # The address of the Ray cluster to connect to, if any.
    num_cpus=8,           # Number of CPUs assigned to each raylet (None = no limit)
    num_gpus=1,           # Number of GPUs assigned to each raylet (None = no limit)
    webui_host="0.0.0.0", # The host to bind the web UI server to.
    local_mode=False,     # If true, the code will be executed serially (for debugging purpose)
    logging_level=20      # Logging level.
)

# ================= Configure the model =================

# Copy the default model configuration
mdl_cfg = MODEL_DEFAULTS.copy()

# Built-in model settings
mdl_cfg["conv_filters"] = None                  # Filter config. List of [out_channels, kernel, stride] for each filter
mdl_cfg["conv_activation"] = "relu"             # Nonlinearity for built-in convnet
mdl_cfg["fcnet_activation"] = "relu"            # Nonlinearity for built-in fully connected net (tanh, relu)
mdl_cfg["fcnet_hiddens"] = [64, 64]             # Number of hidden layers for fully connected net
mdl_cfg["no_final_linear"] = False              # Whether to skip the final linear layer used to resize the outputs to `num_outputs`
mdl_cfg["vf_share_layers"] = True               # Whether layers should be shared for the value function.
mdl_cfg["use_lstm"] = False                     # Whether to wrap the model with a LSTM
mdl_cfg["max_seq_len"] = 20                     # Max seq len for training the LSTM
mdl_cfg["lstm_cell_size"] = 256                 # Size of the LSTM cell
mdl_cfg["lstm_use_prev_action_reward"] = False  # Whether to feed a_{t-1}, r_{t-1} to LSTM

# Custom models settings
mdl_cfg["custom_model"] = None # Name of a custom model to use
mdl_cfg["custom_options"] = {} # Dict of extra options to pass to the custom models

# ================= Configure rllib =================

# Copy the default rllib configuration
rllib_cfg = COMMON_CONFIG.copy()

# Ressources settings
rllib_cfg["use_pytorch"] = True        # Use PyTorch instead of Tensorflow
rllib_cfg["num_gpus"] = 1              # Number of GPUs to reserve for the trainer process
rllib_cfg["num_workers"] = 8           # Number of rollout worker actors for parallel sampling
rllib_cfg["num_envs_per_worker"] = 16  # Number of environments per worker
rllib_cfg["num_cpus_per_worker"] = 1   # Number of CPUs to reserve per worker
rllib_cfg["num_cpus_for_driver"] = 0   # Number of CPUs to allocate for the trainer

# Rollout settings
rllib_cfg["rollout_fragment_length"] = 128     # Sample batches of this size (mult. by `num_envs_per_worker`) are collected from rollout workers
rllib_cfg["train_batch_size"] = 1024            # Sample batches are concatenated together into batches of this size
rllib_cfg["batch_mode"] = "complete_episodes"  # Whether to rollout "complete_episodes" or "truncate_episodes" to `rollout_fragment_length`
rllib_cfg["sample_async"] = False              # Use a background thread for sampling (slightly off-policy)
rllib_cfg["observation_filter"] = "NoFilter"   # Element-wise observation filter ["NoFilter", "MeanStdFilter"]
rllib_cfg["metrics_smoothing_episodes"] = 100  # Smooth metrics over this many episodes
rllib_cfg["seed"] = None                       # sets the random seed of each worker (in conjunction with worker_index)

# Environment settings
rllib_cfg["horizon"] = None             # Number of steps after which the episode is forced to terminate
rllib_cfg["soft_horizon"] = True        # Calculate rewards but don't reset the environment when the horizon is hit
rllib_cfg["no_done_at_end"] = False     # Don't set 'done' at the end of the episode
rllib_cfg["env_config"] = {}               # Arguments to pass to the env creator
rllib_cfg["normalize_actions"] = False  # Normalize actions to the upper and lower bounds of the action space
rllib_cfg["clip_actions"] = True        # Whether to clip actions to the upper and lower bounds of the action space

# Learning settings
rllib_cfg["gamma"] = 0.99             # Discount factor of the MDP
rllib_cfg["lr"] = 1.0e-3              # Learning rate
rllib_cfg["shuffle_buffer_size"] = 0  # Shuffle input batches via a sliding window buffer of this size (0 = disable)
rllib_cfg["log_level"] = "WARN"       # Set the ray.rllib.* log level for the agent process and its workers [DEBUG, INFO, WARN, or ERROR]
rllib_cfg["model"] = mdl_cfg          # Policy model configuration

# ================= Configure the learning algorithm =================

# Copy the default learning algorithm configuration, including PPO-specific parameters,
# then overwrite the common parameters that has been updated ONLY.
agent_cfg = DEFAULT_CONFIG
for key, value in rllib_cfg.items():
    if COMMON_CONFIG[key] != value:
        agent_cfg[key] = value

# Optimizer settings
agent_cfg["sgd_minibatch_size"] = 512  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
agent_cfg["num_sgd_iter"] = 10         # Number of SGD epochs to execute per train batch
agent_cfg["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training

# Estimators settings
agent_cfg["use_gae"] = True     # Use the Generalized Advantage Estimator (GAE) with a value function (https://arxiv.org/pdf/1506.02438.pdf)
agent_cfg["use_critic"] = True  # Use a critic as a value baseline (otherwise don't use any; required for using GAE).
agent_cfg["lambda"] = 0.95       # The GAE(lambda) parameter.

# PPO settings
agent_cfg["lr_schedule"] = None             # Learning rate schedule
agent_cfg["kl_coeff"] = 0.2                 # Initial coefficient for KL divergence
agent_cfg["kl_target"] = 0.01               # Target value for KL divergence
agent_cfg["vf_share_layers"] = True         # Share layers for value function. If you set this to True, it's important to tune vf_loss_coeff
agent_cfg["vf_loss_coeff"] = 0.5            # Coefficient of the value function loss
agent_cfg["entropy_coeff"] = 0.01           # Coefficient of the entropy regularizer
agent_cfg["entropy_coeff_schedule"] = None  # Decay schedule for the entropy regularizer
agent_cfg["clip_param"] = 0.2               # PPO clip parameter
agent_cfg["vf_clip_param"] = float("inf")   # Clip param for the value function. Note that this is sensitive to the scale of the rewards (-1 to disable)
agent_cfg["grad_clip"] = None               # Clip the global norm of gradients by this amount (None = disable) (No working with PyTorch ML backend)

# ================= Configure the learning algorithm =================

train_agent = PPOTrainer(agent_cfg, GYM_ENV_NAME)

# ================= Run the optimization =================

total_timesteps = 2000000
results_fields_filter = ["time_total_s", "training_iteration", "timesteps_total", "episode_reward_max", "episode_reward_mean",
                         ["info", ["sample_time_ms", "grad_time_ms", "opt_peak_throughput", "sample_peak_throughput"]]]

result = {"info": {"num_steps_trained": 0}}
while result["info"]["num_steps_trained"] < total_timesteps:
    # Perform one iteration of training the policy
    result = train_agent.train()

    # Print the training status
    for field in results_fields_filter:
        if not isinstance(field, list):
            print(f"{field}: {result[field]}")
        else:
            for subfield in field[1]:
                print(f"{subfield} : {result[field[0]][subfield]}")
    print("============================")

checkpoint_path = train_agent.save()

# ================= Enjoy a trained agent =================

t_end = 10.0 # Total duration of the simulation(s) in seconds

env = gym.make(GYM_ENV_NAME)
test_agent = PPOTrainer(agent_cfg, GYM_ENV_NAME)
test_agent.restore(checkpoint_path)
t_init = time.time()
t_prev = t_init
while t_prev - t_init < 20.0:
    observ = env.reset()
    done = False
    cumulative_reward = 0
    while (not done) and (t_prev - t_init < 20.0):
        action = test_agent.compute_action(observ)
        observ, reward, done, _ = env.step(action)
        cumulative_reward += reward
        env.render()
        sleep(env.dt - (time.time() - t_prev))
        t_prev = time.time()
    print(cumulative_reward)
