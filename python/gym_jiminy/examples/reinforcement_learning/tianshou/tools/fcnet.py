import numpy as np
from typing import Optional, Tuple, Sequence, Callable, Union, Dict, Any

import gym
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal


class Net(nn.Module):
    def __init__(self, obs_space: gym.spaces.Space):
        super().__init__()
        n_input = np.prod(obs_space.shape)
        self.model = nn.Sequential(*[
            nn.Linear(n_input, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        ])

    def forward(self,
                obs: np.ndarray,
                state: Optional[Any] = None,
                info: Optional[Dict[str, Any]] = None
                ) -> Tuple[Union[np.ndarray, Sequence[np.ndarray]], Any]:
        if not isinstance(obs, torch.Tensor):
            device = next(self.model.parameters()).device
            obs = torch.tensor(obs, device=device, dtype=torch.float32)
        logits = self.model(obs)
        return logits, state


class Actor(nn.Module):
    def __init__(self,
                 preprocess_net: Net,
                 action_space: gym.spaces.Space,
                 free_log_std: bool = True):
        super().__init__()
        self.preprocess = preprocess_net
        self.is_action_space_discrete = \
            isinstance(action_space, gym.spaces.Discrete)
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

    def forward(self,
                obs: np.ndarray,
                state: Optional[Any] = None,
                info: Optional[Dict[str, Any]] = None
                ) -> Tuple[Union[np.ndarray, Sequence[np.ndarray]], Any]:
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
    def __init__(self, preprocess_net: Net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(64, 1)

    def forward(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        logits, _ = self.preprocess(obs, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits


def build_actor_critic(
        env_creator: Callable[[], gym.Env],
        vf_share_layers: bool,
        free_log_std: bool) -> Tuple[Actor, Critic, Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor]]:
    # Instantiate actor and critic
    env = env_creator()
    net = Net(env.observation_space)
    actor = Actor(net, env.action_space, free_log_std)
    if vf_share_layers:
        critic = Critic(net)
    else:
        critic = Critic(Net(env.observation_space))

    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # Define the probability distribution
    if isinstance(env.action_space, gym.spaces.Discrete):
        dist_fn = Categorical
    else:
        def dist_fn(*args):
            return Independent(Normal(*args), 1)  # Diagonal normal

    return actor, critic, dist_fn
