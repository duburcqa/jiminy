import os
import copy
import time
import pathlib

import gym
import ray
from ray import rllib
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.agents.ppo import PPOTrainer as Trainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from tensorboard.program import TensorBoard

from jiminy_py.viewer import sleep

from fcnet import FullyConnectedNetwork


# ================= User parameters =================

GYM_ENV_NAME = "gym_jiminy:jiminy-acrobot-v0"
GYM_ENV_KWARGS = {'continuous': True}

# ================= Initialize the Ray backend and Tensorboard =================
#
# It will be used later for almost everything from dashboard,
# remote/client management, to multithreaded environment.
#

ray.init(
    address=None,          # The address of the Ray cluster to connect to, if any.
    num_cpus=8,            # Number of CPUs assigned to each raylet (None = no limit)
    num_gpus=0,            # Number of GPUs assigned to each raylet (None = no limit)
    webui_host="0.0.0.0",  # The host to bind the web UI server to.
    local_mode=False,      # If true, the code will be executed serially (for debugging purpose)
    logging_level=20       # Logging level.
)

# # Create tensorboard Jupyter cell
# %load_ext tensorboard
# %tensorboard --logdir logs
if not 'tb' in locals().keys():
    tb = TensorBoard()
    tb.configure(host="0.0.0.0",
                 logdir=os.path.join(pathlib.Path.home(), 'ray_results'))
    url = tb.launch()
    print(f"Starting Tensorboard {url} ...")

# ================= Configure the model =================

# Register the custom model architecture (it implements 'vf_share_layers')
ModelCatalog.register_custom_model("my_model", FullyConnectedNetwork)

# Copy the default model configuration
mdl_cfg = copy.deepcopy(MODEL_DEFAULTS)

# Fully-connected network settings
mdl_cfg["fcnet_activation"] = "tanh"            # Nonlinearity for built-in fully connected net ["tanh", "relu", "linear"]
mdl_cfg["fcnet_hiddens"] = [64, 64]             # Number of hidden layers for fully connected net
mdl_cfg["no_final_linear"] = False              # Whether to skip the final linear layer used to resize the outputs to `num_outputs`
mdl_cfg["free_log_std"] = False                 # The last half of the output layer does not dependent on the input
mdl_cfg["vf_share_layers"] = False              # Whether layers should be shared for the value function.

# Custom model settings
mdl_cfg["custom_model"] = "my_model" # Name of a custom model to use
mdl_cfg["custom_options"] = {} # Dict of extra options to pass to the custom models

# ================= Configure rllib =================

# Register the environment with custom default constructor arguments
env_creator = lambda env_config: gym.make(GYM_ENV_NAME, **GYM_ENV_KWARGS)
register_env("my_custom_env", env_creator)

# Copy the default rllib configuration
rllib_cfg = copy.deepcopy(COMMON_CONFIG)

# Ressources settings
rllib_cfg["use_pytorch"] = True        # Use PyTorch instead of Tensorflow
rllib_cfg["num_gpus"] = 0              # Number of GPUs to reserve for the trainer process
rllib_cfg["num_workers"] = 4           # Number of rollout worker actors for parallel sampling
rllib_cfg["num_envs_per_worker"] = 1   # Number of environments per worker
rllib_cfg["num_cpus_per_worker"] = 1   # Number of CPUs to reserve per worker
rllib_cfg["num_cpus_for_driver"] = 0   # Number of CPUs to allocate for the trainer

# Rollout settings
rllib_cfg["rollout_fragment_length"] = 128     # Sample batches of this size (mult. by `num_envs_per_worker`) are collected from rollout workers
rllib_cfg["train_batch_size"] = 512            # Sample batches are concatenated together into batches of this size
rllib_cfg["batch_mode"] = "truncate_episodes"  # Whether to rollout "complete_episodes" or "truncate_episodes" to `rollout_fragment_length`
rllib_cfg["sample_async"] = False              # Use a background thread for sampling (slightly off-policy)
rllib_cfg["observation_filter"] = "NoFilter"   # Element-wise observation filter ["NoFilter", "MeanStdFilter"]
rllib_cfg["metrics_smoothing_episodes"] = 100  # Smooth metrics over this many episodes
rllib_cfg["seed"] = 0                          # sets the random seed of each worker (in conjunction with worker_index)

# Environment settings
rllib_cfg["horizon"] = None             # Number of steps after which the episode is forced to terminate
rllib_cfg["soft_horizon"] = False       # Calculate rewards but don't reset the environment when the horizon is hit
rllib_cfg["no_done_at_end"] = False     # Don't set 'done' at the end of the episode
rllib_cfg["normalize_actions"] = False  # Normalize actions to the upper and lower bounds of the action space
rllib_cfg["clip_actions"] = True        # Whether to clip actions to the upper and lower bounds of the action space
rllib_cfg["env_config"] = {}            # Arguments to pass to the env creator

# Learning settings
rllib_cfg["lr"] = 1.0e-3              # Learning rate
rllib_cfg["gamma"] = 0.99             # Discount factor of the MDP
rllib_cfg["explore"] = True
rllib_cfg["exploration_config"] = {
        "type": "Random"              # The Exploration class to use ["StochasticSampling", "EpsilonGreedy", ...]
    }
rllib_cfg["shuffle_buffer_size"] = 0  # Shuffle input batches via a sliding window buffer of this size (0 = disable)
rllib_cfg["log_level"] = "INFO"       # Set the ray.rllib.* log level for the agent process and its workers [DEBUG, INFO, WARN, or ERROR]
rllib_cfg["model"] = mdl_cfg          # Policy model configuration

# ================= Configure the learning algorithm =================

# Copy the default learning algorithm configuration, including PPO-specific parameters,
# then overwrite the common parameters that has been updated ONLY.
agent_cfg = copy.deepcopy(DEFAULT_CONFIG)
for key, value in rllib_cfg.items():
    if COMMON_CONFIG[key] != value:
        agent_cfg[key] = value

# Optimizer settings
agent_cfg["sgd_minibatch_size"] = 128  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
agent_cfg["num_sgd_iter"] = 8          # Number of SGD epochs to execute per train batch
agent_cfg["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training

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
agent_cfg["lr_schedule"] = None             # Learning rate schedule
agent_cfg["grad_clip"] = 0.5  #float("inf")       # Clip the global norm of gradients by this amount (None = disable)

# ================= Configure the learning algorithm =================

# agent_cfg = copy.deepcopy(DEFAULT_CONFIG)
agent_cfg["use_pytorch"] = False
agent_cfg["model"]["custom_model"] = None
agent_cfg["exploration_config"]["type"] = "StochasticSampling"
agent_cfg["lr"] = 1.0e-4
agent_cfg["lr_schedule"] = [
    [      0, 1.0e-4],
    [1200000, 1.0e-6],
]
train_agent = Trainer(agent_cfg, env="my_custom_env")

# ================= Run the optimization =================

timesteps_total = 1200000
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

t_end = 20.0 # Total duration of the simulation(s) in seconds

try:
    env = env_creator(rllib_cfg["env_config"])
    test_agent = Trainer(agent_cfg, env="my_custom_env")
    test_agent.restore(checkpoint_path)
    t_init = time.time()
    t_prev = t_init
    while t_prev - t_init < t_end:
        observ = env.reset()
        done = False
        cumulative_reward = 0
        while not done:
            if not (t_prev - t_init < t_end):
                break
            action = test_agent.compute_action(observ, explore=False)
            observ, reward, done, _ = env.step(action)
            cumulative_reward += reward
            env.render()
            sleep(env.dt - (time.time() - t_prev))
            t_prev = time.time()
        print(cumulative_reward)
except KeyboardInterrupt:
    print("Interrupting testing...")

# ================= Terminate the Ray backend =================

ray.shutdown()
