import os
import time
import pathlib

import gym
import ray
from ray import rllib
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.agents.trainer import COMMON_CONFIG
# from ray.tune.registry import register_env
from tensorboard.program import TensorBoard

from jiminy_py.viewer import sleep


GYM_ENV_NAME = "gym_jiminy:jiminy-cartpole-v0"
algorithm = "ppo"

# ================= Initialize the Ray backend and Tensorboard =================
#
# It will be used later for almost everything from dashboard,
# remote/client management, to multithreaded environment.
#

ray.init(
    address=None,         # The address of the Ray cluster to connect to, if any.
    num_cpus=8,           # Number of CPUs assigned to each raylet (None = no limit)
    num_gpus=0,           # Number of GPUs assigned to each raylet (None = no limit)
    webui_host="0.0.0.0", # The host to bind the web UI server to.
    local_mode=True,     # If true, the code will be executed serially by Tune (for debugging purpose)
    logging_level=20      # Logging level.
)

if not 'tb' in locals():
    tb = TensorBoard()
    tb.configure(host="0.0.0.0",
                logdir=os.path.join(pathlib.Path.home(), 'ray_results'))
    url = tb.launch()
    print(f"Starting Tensorboard {url} ...")

# ================= Configure the model =================

# Copy the default model configuration
mdl_cfg = MODEL_DEFAULTS.copy()

# Convolution network settings
mdl_cfg["conv_filters"] = None                  # Filter config. List of [out_channels, kernel, stride] for each filter
mdl_cfg["conv_activation"] = "relu"             # Nonlinearity for built-in convnet

# Fully-connected network settings
mdl_cfg["fcnet_activation"] = "tanh"            # Nonlinearity for built-in fully connected net (tanh, relu)
mdl_cfg["fcnet_hiddens"] = [64, 64]             # Number of hidden layers for fully connected net
mdl_cfg["no_final_linear"] = False              # Whether to skip the final linear layer used to resize the outputs to `num_outputs`
mdl_cfg["free_log_std"] = False                 # The last half of the output layer does not dependent on the input
mdl_cfg["vf_share_layers"] = False              # Whether layers should be shared for the value function.

# LTSM network settings
mdl_cfg["use_lstm"] = False                     # Whether to wrap the model with a LSTM
mdl_cfg["max_seq_len"] = 20                     # Max seq len for training the LSTM
mdl_cfg["lstm_cell_size"] = 256                 # Size of the LSTM cell
mdl_cfg["lstm_use_prev_action_reward"] = False  # Whether to feed a_{t-1}, r_{t-1} to LSTM

# Custom model settings
mdl_cfg["custom_model"] = None # Name of a custom model to use
mdl_cfg["custom_options"] = {} # Dict of extra options to pass to the custom models

# ================= Configure rllib =================

# Copy the default rllib configuration
rllib_cfg = COMMON_CONFIG.copy()

# Ressources settings
rllib_cfg["use_pytorch"] = True        # Use PyTorch instead of Tensorflow
rllib_cfg["num_gpus"] = 0              # Number of GPUs to reserve for the trainer process
rllib_cfg["num_workers"] = 8           # Number of rollout worker actors for parallel sampling
rllib_cfg["num_envs_per_worker"] = 16  # Number of environments per worker
rllib_cfg["num_cpus_per_worker"] = 1   # Number of CPUs to reserve per worker
rllib_cfg["num_cpus_for_driver"] = 0   # Number of CPUs to allocate for the trainer

# Rollout settings
rllib_cfg["rollout_fragment_length"] = 64      # Sample batches of this size (mult. by `num_envs_per_worker`) are collected from rollout workers
rllib_cfg["train_batch_size"] = 8096            # Sample batches are concatenated together into batches of this size
rllib_cfg["batch_mode"] = "complete_episodes"  # Whether to rollout "complete_episodes" or "truncate_episodes" to `rollout_fragment_length`
rllib_cfg["sample_async"] = False              # Use a background thread for sampling (slightly off-policy)
rllib_cfg["observation_filter"] = "NoFilter"   # Element-wise observation filter ["NoFilter", "MeanStdFilter"]
rllib_cfg["metrics_smoothing_episodes"] = 100  # Smooth metrics over this many episodes
rllib_cfg["seed"] = 0                          # sets the random seed of each worker (in conjunction with worker_index)

# Environment settings
rllib_cfg["horizon"] = 1000             # Number of steps after which the episode is forced to terminate
rllib_cfg["soft_horizon"] = False        # Calculate rewards but don't reset the environment when the horizon is hit
rllib_cfg["no_done_at_end"] = False      # Don't set 'done' at the end of the episode
rllib_cfg["normalize_actions"] = False  # Normalize actions to the upper and lower bounds of the action space
rllib_cfg["clip_actions"] = False       # Whether to clip actions to the upper and lower bounds of the action space
rllib_cfg["env_config"] = {}            # Arguments to pass to the env creator

# Learning settings
rllib_cfg["lr"] = 1.0e-4              # Learning rate
rllib_cfg["gamma"] = 0.99             # Discount factor of the MDP
rllib_cfg["exploration_config"] = {
        "type": "EpsilonGreedy", # The Exploration class to use ["StochasticSampling", "EpsilonGreedy", ...]
    }
rllib_cfg["shuffle_buffer_size"] = 0  # Shuffle input batches via a sliding window buffer of this size (0 = disable)
rllib_cfg["log_level"] = "INFO"       # Set the ray.rllib.* log level for the agent process and its workers [DEBUG, INFO, WARN, or ERROR]
rllib_cfg["model"] = mdl_cfg          # Policy model configuration

# ================= Configure the learning algorithm =================

# Select the right algorithm and import the required dependencies
if algorithm == "ppo":
    from ray.rllib.agents.ppo import PPOTrainer as Trainer, DEFAULT_CONFIG
elif algorithm == "sac":
    from ray.rllib.agents.sac import SACTrainer as Trainer, DEFAULT_CONFIG
elif algorithm == "apex-dqn":
    from ray.rllib.agents.dqn.apex import ApexTrainer as Trainer, APEX_DEFAULT_CONFIG as DEFAULT_CONFIG

# Copy the default learning algorithm configuration, including PPO-specific parameters,
# then overwrite the common parameters that has been updated ONLY.
agent_cfg = DEFAULT_CONFIG.copy()
for key, value in rllib_cfg.items():
    if COMMON_CONFIG[key] != value:
        agent_cfg[key] = value

if algorithm == "ppo":
    # Optimizer settings
    agent_cfg["sgd_minibatch_size"] = 128  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
    agent_cfg["num_sgd_iter"] = 8          # Number of SGD epochs to execute per train batch
    agent_cfg["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training

    # Estimators settings
    agent_cfg["use_gae"] = True      # Use the Generalized Advantage Estimator (GAE) with a value function (https://arxiv.org/pdf/1506.02438.pdf)
    agent_cfg["use_critic"] = True  # Use a critic as a value baseline (otherwise don't use any; required for using GAE).
    agent_cfg["lambda"] = 0.95       # The GAE(lambda) parameter.

    # Learning and optimization settings
    agent_cfg["lr_schedule"] = None             # Learning rate schedule
    agent_cfg["kl_coeff"] = 0.2                 # Initial coefficient for KL divergence
    agent_cfg["kl_target"] = 0.01               # Target value for KL divergence
    agent_cfg["vf_share_layers"] = False        # Share layers for value function. If you set this to True, it's important to tune vf_loss_coeff
    agent_cfg["vf_loss_coeff"] = 0.5            # Coefficient of the value function loss
    agent_cfg["entropy_coeff"] = 0.01           # Coefficient of the entropy regularizer
    agent_cfg["entropy_coeff_schedule"] = None  # Decay schedule for the entropy regularizer
    agent_cfg["clip_param"] = 0.2               # PPO clip parameter
    agent_cfg["vf_clip_param"] = float("inf")   # Clip param for the value function. Note that this is sensitive to the scale of the rewards (-1 to disable)
    agent_cfg["grad_clip"] = None #0.5          # Clip the global norm of gradients by this amount (None = disable) (No working with PyTorch ML backend)
elif algorithm == "sac":
    # Model
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

    # Learning
    agent_cfg["tau"] = 5e-3                     # Update the target by \tau * policy + (1-\tau) * target_policy.
    agent_cfg["initial_alpha"] = 1.0            # Initial value to use for the entropy weight alpha
    agent_cfg["target_entropy"] = "auto"        # Target entropy lower bound. Inverse of reward scale. If "auto", will be set to -|A|
    agent_cfg["n_step"] = 1                     # N-step target updates
    agent_cfg["timesteps_per_iteration"] = 300  # Number of env steps to optimize for before returning.

    # Replay buffer
    agent_cfg["buffer_size"] = 1000000
    agent_cfg["prioritized_replay"] = False    # If True prioritized replay buffer will be used.
    agent_cfg["prioritized_replay_alpha"] = 0.6
    agent_cfg["prioritized_replay_beta"] = 0.4
    agent_cfg["prioritized_replay_eps"] = 1e-6
    agent_cfg["prioritized_replay_beta_annealing_timesteps"] = 20000
    agent_cfg["final_prioritized_replay_beta"] = 0.4

    # Optimization
    agent_cfg["optimization"] = {
        "actor_learning_rate": 1.0e-4,
        "critic_learning_rate": 1.0e-4,
        "entropy_learning_rate": 1.0e-4,
    }
    agent_cfg["learning_starts"] = 10000          # How many steps of the model to sample before learning starts
    agent_cfg["target_network_update_freq"] = 10  # Update the target network every `target_network_update_freq` steps
    agent_cfg["grad_clip"] = None                 # If not None, clip gradients during optimization at this value
else:
    pass

# ================= Configure the learning algorithm =================

# env_creator = lambda env_config: gym.make(GYM_ENV_NAME, **env_config)
# register_env("my_custom_env", env_creator)
agent_cfg = DEFAULT_CONFIG.copy()
agent_cfg["use_pytorch"] = True
# agent_cfg["batch_mode"] = "complete_episodes"
# agent_cfg["horizon"] = 1000
# agent_cfg["soft_horizon"] = False
# agent_cfg["no_done_at_end"] = False
agent_cfg["lr"] = 2.0e-7              # Learning rate
train_agent = Trainer(agent_cfg, GYM_ENV_NAME) #env="my_custom_env")

# ================= Run the optimization =================

timesteps_total = 40000000
results_fields_filter = ["training_iteration", "time_total_s", "timesteps_total", "episode_reward_max", "episode_reward_mean",
                         ["info", ["sample_time_ms", "grad_time_ms", "opt_peak_throughput", "sample_peak_throughput"]]]

try:
    result = {"timesteps_total": 0}
    while result["timesteps_total"] < timesteps_total:
        # Perform one iteration of training the policy
        result = train_agent.train()

        # Print the training status
        for field in results_fields_filter:
            if not isinstance(field, list):
                if field in result.keys():
                    print(f"{field}: {result[field]}")
            else:
                for subfield in field[1]:
                    if subfield in result[field[0]].keys():
                        print(f"{subfield} : {result[field[0]][subfield]}")
        print("============================")
except KeyboardInterrupt:
    print("Interrupting training...")
finally:
    checkpoint_path = train_agent.save()

# ================= Enjoy a trained agent =================

t_end = 10.0 # Total duration of the simulation(s) in seconds

try:
    env = gym.make(GYM_ENV_NAME) #env_creator(rllib_cfg["env_config"])
    test_agent = Trainer(agent_cfg, GYM_ENV_NAME)
    test_agent.restore(checkpoint_path)
    t_init = time.time()
    t_prev = t_init
    while t_prev - t_init < 20.0:
        observ = env.reset()
        done = False
        cumulative_reward = 0
        while (not done) and (t_prev - t_init < 20.0):
            action = test_agent.compute_action(observ, explore=False)
            observ, reward, done, _ = env.step(action)
            cumulative_reward += reward
            env.render()
            sleep(env.dt - (time.time() - t_prev))
            t_prev = time.time()
        print(cumulative_reward)
except KeyboardInterrupt:
    print("Interrupting testing...")
finally:
    checkpoint_path = train_agent.save()

# ================= Terminate the Ray backend =================

ray.shutdown()
