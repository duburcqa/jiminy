import os
import pathlib
import random

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from tensorboard.program import TensorBoard

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

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=50,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7],
        "num_sgd_iter": lambda: random.randint(1, 30),
        "sgd_minibatch_size": lambda: random.randint(128, 8192),
        "train_batch_size": lambda: random.randint(2000, 80000),
    },
    custom_explore_fn=explore
)

tune.run(
    "PPO",
    name="cartpole_pbt_ppo",
    scheduler=pbt,
    num_samples=8,
    reuse_actors=True,
    config={
        # Resource config
        "num_workers": 8,
        "num_gpus": 0.125,
        "num_cpus_per_worker": 0.125,
        "num_envs_per_worker": 4,
        "num_cpus_for_driver": 0,
        # Problem setup
        "env": "gym_jiminy:jiminy-cartpole-v0",
        "use_pytorch": True,
        # Fixed params
        "model": {
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [64, 64],
            "free_log_std": True
        },
        "kl_coeff": 1.0,
        # These params are tuned from a fixed starting value
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1.0e-6,
        # These params start off randomly drawn from a set
        "num_sgd_iter": tune.sample_from(
            lambda spec: random.choice([10, 20, 30])),
        "sgd_minibatch_size": tune.sample_from(
            lambda spec: random.choice([128, 512, 2048])),
        "train_batch_size": tune.sample_from(
            lambda spec: random.choice([4096, 8192, 16384, 32768]))
    }
)
