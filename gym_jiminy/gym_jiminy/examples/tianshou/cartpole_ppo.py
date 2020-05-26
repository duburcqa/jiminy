import os
import time
import tqdm
import numpy as np

import gym
from gym.wrappers import TimeLimit
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.program import TensorBoard
from typing import Dict, List, Union, Callable, Optional

from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.policy.utils import DiagGaussian
from tianshou.trainer import test_episode, gather_info
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils import tqdm_config, MovAvg

#  User parameters
GYM_ENV_NAME = "gym_jiminy:jiminy-cartpole-v0"
GYM_ENV_KWARGS = {'continuous': False}
TARGET_EPISODE_STEPS = 5000
MAX_EPISODE_STEPS = 10000
SEED = 0
N_THREADS = 8
VF_SHARE_LAYERS = False

# Define some hyper-parameters:
env_creator = lambda *args, **kwargs: TimeLimit(gym.make(GYM_ENV_NAME, **GYM_ENV_KWARGS), MAX_EPISODE_STEPS)

### On-policy trainer parameters
trainer_config = dict(
    collect_per_step = 128 * N_THREADS,  # Number of frames the collector would collect in total before the network update
    batch_size = 128,                    # Minibatch size of sample data used for policy learning
    repeat_per_collect = 8,              # Number of repeat time for policy learning for each batch (after splitting the batch)
    frame_per_epoch = 100000,            # Number of sample frames in one epoch (testing performance computed after each epoch)
    max_epoch = 10,                      # Maximum of epochs for training
    episode_per_test = N_THREADS,        # Number of episodes for one policy evaluation
)

### PPO algorithm parameters
ppo_config = dict(
    discount_factor = 0.99,
    eps_clip= 0.2,
    vf_coef = 0.5,
    ent_coef = 0.01,
    gae_lambda = 0.95,
    dual_clip = None,
    value_clip = False,
    reward_normalization = False,
    max_grad_norm = 0.5
)

### Optimizer parameters
lr = 1.0e-3
buffer_size = 128 * N_THREADS

# Instantiate the gym environment
train_envs = VectorEnv([lambda: env_creator() for _ in range(N_THREADS)])
test_envs = VectorEnv([lambda: env_creator() for _ in range(N_THREADS)])

# Set the seed
np.random.seed(SEED)
torch.manual_seed(SEED)
train_envs.seed(SEED)
test_envs.seed(SEED)

# Crate the models

### Define the models
class Net(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        n_input = np.prod(state_space.shape)
        self.model = nn.Sequential(*[
            nn.Linear(n_input, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        ])

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            device = next(self.model.parameters()).device
            obs = torch.tensor(obs, device=device, dtype=torch.float32)
        logits = self.model(obs)
        return logits, state

class Actor(nn.Module):
    def __init__(self, preprocess_net, action_space, free_log_std=True):
        super().__init__()
        self.preprocess = preprocess_net
        self.is_action_space_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.free_log_std = free_log_std
        if self.is_action_space_discrete:
            n_output = np.prod(action_space.n)
            self.onehot = nn.Linear(64, n_output)
        else:
            n_output = np.prod(action_space.shape)
            self.mu = nn.Linear(64, n_output)
            if self.free_log_std:
                self.sigma = nn.Parameter(torch.zeros(n_output, 1))
            else:
                self.sigma = nn.Linear(64, n_output)

    def forward(self, obs, state=None, info={}):
        logits, h = self.preprocess(obs, state)
        if self.is_action_space_discrete:
            logits = nn.functional.softmax(self.onehot(logits), dim=-1)
            return logits, h
        else:
            mu = self.mu(logits)
            if self.free_log_std:
                sigma = self.sigma.expand(mu.shape).exp()
            else:
                sigma = torch.exp(self.sigma(logits))
            return (mu, sigma), None

class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(64, 1)

    def forward(self, obs, **kwargs):
        logits, _ = self.preprocess(obs, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits

### Instantiate the models
env = env_creator()
if VF_SHARE_LAYERS:
    net = Net(env.observation_space)
    actor = Actor(net, env.action_space).to('cuda')
    critic = Critic(net).to('cuda')
else:
    actor = Actor(Net(env.observation_space), env.action_space).to('cuda')
    critic = Critic(Net(env.observation_space)).to('cuda')
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

### orthogonal initialization
for m in list(actor.modules()) + list(critic.modules()):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)

### Set the probability distribution
if isinstance(env.action_space, gym.spaces.Discrete):
    dist_fn = torch.distributions.Categorical
else:
    dist_fn = DiagGaussian
    ppo_config["action_range"] = [float(env.action_space.low),
                                  float(env.action_space.high)]

# Setup policy and collectors:
policy = PPOPolicy(actor, critic, optim, dist_fn, **ppo_config)
train_collector = Collector(policy, train_envs, ReplayBuffer(buffer_size))
test_collector = Collector(policy, test_envs)

# Setup the trainer and run the learning process

### Define a custom on-policy trainer
def onpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        frame_per_epoch: int,
        collect_per_step: int,
        repeat_per_collect: int,
        episode_per_test: Union[int, List[int]],
        batch_size: int,
        train_fn: Optional[Callable[[int], None]] = None,
        test_fn: Optional[Callable[[int], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        log_fn: Optional[Callable[[dict], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1,
        verbose: bool = True,
        **kwargs
) -> Dict[str, Union[float, str]]:
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    test_in_train = train_collector.policy == policy
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=frame_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:
                result = train_collector.collect(n_step=collect_per_step,
                                                 log_fn=log_fn)
                data = {}
                if test_in_train and stop_fn and stop_fn(result['rew']):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test)
                    if stop_fn and stop_fn(test_result['rew']):
                        if save_fn:
                            save_fn(policy)
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result['rew'])
                    else:
                        policy.train()
                        if train_fn:
                            train_fn(epoch)
                losses = policy.learn(
                    train_collector.sample(0), batch_size, repeat_per_collect)
                train_collector.reset_buffer()
                global_step += collect_per_step
                for k in result.keys():
                    data[k] = f'{result[k]:.2f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, result[k], global_step=global_step)
                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f'{stat[k].get():.6f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=global_step)
                t.update(collect_per_step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward)

### Configure Tensorboard
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
if not 'tb' in locals().keys():
    tb = TensorBoard()
    tb.configure(host="0.0.0.0", logdir=data_path)
    url = tb.launch()
    print(f"Started Tensorboard {url} at {data_path}...")
writer = SummaryWriter(data_path)

### Configure export
def save_fn(policy):
    torch.save(policy.state_dict(), os.path.join(data_path, 'policy.pth'))

### Configure early stopping of training
def stop_fn(x):
    return x >= TARGET_EPISODE_STEPS

### Run the learning process
result = onpolicy_trainer(
    policy, train_collector, test_collector,
    **trainer_config, stop_fn=stop_fn, save_fn=save_fn,
    writer=writer, verbose=True)
print(f'Finished training! Use {result["duration"]}')

### Stop the data collectors
train_collector.close()
test_collector.close()

# Enjoy a trained agent !
env = env_creator()
collector = Collector(policy, env)
result = collector.collect(n_episode=1, render=env.dt)
print(f'Final reward: {result["rew"]}, length: {result["len"]}')
collector.close()
