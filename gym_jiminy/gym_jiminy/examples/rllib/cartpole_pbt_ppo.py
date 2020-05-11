import os
import pathlib
import random

import gym
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.registry import register_env
from tensorboard.program import TensorBoard

from fcnet import FullyConnectedNetwork


# ================= User parameters =================

GYM_ENV_NAME = "gym_jiminy:jiminy-cartpole-v0"
GYM_ENV_KWARGS = {'continuous': False}
AGENT_ALGORITHM = "PPO"

# ================= Initialize the Ray backend and Tensorboard =================
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

# # Create tensorboard Jupyter cell
# %load_ext tensorboard
# %tensorboard --logdir logs
if not 'tb' in locals().keys():
    tb = TensorBoard()
    tb.configure(host="0.0.0.0",
                 logdir=os.path.join(pathlib.Path.home(), 'ray_results'))
    url = tb.launch()
    print(f"Starting Tensorboard {url} ...")

# ================= Run hyperparameter search =================

# Register the custom model architecture (it implements 'vf_share_layers')
ModelCatalog.register_custom_model("my_model", FullyConnectedNetwork)

# Register the environment with custom default constructor arguments
env_creator = lambda env_config: gym.make(GYM_ENV_NAME, **GYM_ENV_KWARGS)
register_env("my_custom_env", env_creator)

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    # ensure that the size of the train batches exactly batches the sum of length of rollout fragments
    config["rollout_fragment_length"] = config["train_batch_size"] \
        / (config["num_workers"] * config["num_envs_per_worker"])
    return config

pbt = PopulationBasedTraining(
    time_attr="timesteps_total", #time_total_s
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=400000,
    resample_probability=0.25,
    quantile_fraction=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5],
        "num_sgd_iter": lambda: random.randint(1, 32),
        "sgd_minibatch_size": lambda: random.randint(64, 1024),
        "train_batch_size": lambda: random.randint(256, 4096),
    },
    custom_explore_fn=explore
)

tune.run(
    AGENT_ALGORITHM,
    name="_".join([GYM_ENV_NAME, "PBT", AGENT_ALGORITHM]),
    scheduler=pbt,
    num_samples=8,
    reuse_actors=False,
    config={
        # Resource config
        "num_workers": 1,
        "num_gpus": 0.125,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 4,
        "num_cpus_for_driver": 0,
        # Problem setup
        "env": "my_custom_env",
        "use_pytorch": True,
        # Fixed params
        "batch_mode": "truncate_episodes",
        "horizon": 5000,
        "kl_coeff": 0.0,
        "entropy_coeff": 0.01,
        "vf_clip_param": float("inf"),
        "grad_clip": 0.5,
        "exploration_config": {
            "type": "Random"
        },
        "model": {
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [64, 64],
            "free_log_std": False,
            "vf_share_layers": False,
            "custom_model": "my_model"
        },
        "vf_share_layers": False,
        "seed": 0,
        # These params are tuned from a fixed starting value
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1.0e-4,
        # These params start off randomly drawn from a set
        "num_sgd_iter": tune.sample_from(
            lambda spec: random.choice([4, 8, 16, 32])),
        "sgd_minibatch_size": tune.sample_from(
            lambda spec: random.choice([64, 128, 512])),
        "train_batch_size": tune.sample_from(
            lambda spec: random.choice([512, 1024, 2048, 4096]))
    }
)
