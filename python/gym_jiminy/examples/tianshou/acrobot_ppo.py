"""Solve the official Open AI Gym Acrobot problem simulated in Jiminy using PPO
algorithm of Tianshou reinforcement learning framework.

It solves it consistently in less than 100000 timesteps in average, and in
about 50000 at best.

.. warning::
    This script requires pytorch>=1.4 and tianshou==0.3.0.
"""
# flake8: noqa

# ======================== User parameters =========================

GYM_ENV_NAME = "gym_jiminy.envs:acrobot-v0"
GYM_ENV_KWARGS = {
    "continuous": True
}
SEED = 0
N_THREADS = 8
N_GPU = 1

# =================== Configure Python workspace ===================

# GPU device selection must be done at system level to be taken into account
__import__("os").environ["CUDA_VISIBLE_DEVICES"] = \
    ",".join(map(str, range(N_GPU)))

import numpy as np
import gym
import torch

from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy

from tools.fcnet import build_actor_critic
from tools.utilities import initialize, train, test

# ================== Initialize Tensorboard daemon =================

writer = initialize()

# ================== Configure learning algorithm ==================

### On-policy trainer parameters
trainer_config = {}
trainer_config["collect_per_step"] = 512    # Number of frames the collector would collect in total before the network update
trainer_config["batch_size"] = 128          # Minibatch size of sample data used for policy learning
trainer_config["repeat_per_collect"] = 8    # Number of repeat time for policy learning for each batch (after splitting the batch)
trainer_config["frame_per_epoch"] = 100000  # Number of sample frames in one epoch (testing performance computed after each epoch)
trainer_config["max_epoch"] = 1             # Maximum of epochs for training
trainer_config["episode_per_test"] = 100    # Number of episodes for one policy evaluation

### PPO algorithm parameters
ppo_config = {}
ppo_config["max_batchsize"] = 512
ppo_config["discount_factor"] = 0.99
ppo_config["gae_lambda"] = 0.95
ppo_config["vf_coef"] = 0.5
ppo_config["ent_coef"] = 0.01
ppo_config["eps_clip"] = 0.2
ppo_config["dual_clip"] = None
ppo_config["value_clip"] = False
ppo_config["reward_normalization"] = False
ppo_config["max_grad_norm"] = 0.5

# ====================== Run the optimization ======================

# Create a multiprocess environment
env_creator = lambda *args, **kwargs: gym.make(GYM_ENV_NAME, **GYM_ENV_KWARGS)

# Create training and testing environments
train_envs = SubprocVectorEnv([
    env_creator for _ in range(int(N_THREADS//2))])
test_envs = SubprocVectorEnv([
    env_creator for _ in range(int(N_THREADS//2))])

# Set the seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
train_envs.seed(SEED)
test_envs.seed(SEED)

# Create actor and critic
actor, critic, dist_fn = build_actor_critic(
    env_creator, vf_share_layers=False, free_log_std=True)
actor = actor.to("cuda")
critic = critic.to("cuda")

# Set the action range in continuous mode
env = env_creator()
if isinstance(env.action_space, gym.spaces.Box):
    ppo_config["action_range"] = [
        float(env.action_space.low), float(env.action_space.high)]

# Optimizer parameters
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=1e-5)

# Create the training agent
train_agent = PPOPolicy(actor, critic, optimizer, dist_fn, **ppo_config)

# Run the learning process
checkpoint_path = train(
    train_agent, train_envs, test_envs, writer, trainer_config)

# ===================== Enjoy the trained agent ======================

test_agent = PPOPolicy(actor, critic, optimizer, dist_fn, **ppo_config)
test_agent.load_state_dict(torch.load(checkpoint_path))
test(test_agent, env_creator, num_episodes=1)
