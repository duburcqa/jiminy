import os
import math
import socket
import pathlib
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import gym
from gym.wrappers import FlattenObservation
import tensorflow as tf
from tensorboard.program import TensorBoard

import ray
from ray.tune.logger import UnifiedLogger
from ray.rllib.utils.filter import NoFilter
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy import TFPolicy

from gym_jiminy.common.utils import clip


PRINT_RESULT_FIELDS_FILTER = [
    "training_iteration",
    "time_total_s",
    "timesteps_total",
    "episodes_total",
    "episode_reward_max",
    "episode_reward_mean",
    "episode_len_mean"
]


def initialize(num_cpus: int = 0,
               num_gpus: int = 0,
               log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               debug: bool = False,
               verbose: bool = True) -> Callable[[], UnifiedLogger]:
    """Initialize Ray and Tensorboard daemons.

    It will be used later for almost everything from dashboard, remote/client
    management, to multithreaded environment.

    :param log_root_path: Fullpath of root log directory.
                          Optional: location of this file / log by default.
    :param log_name: Name of the subdirectory where to save data.
                     Optional: full date _ hostname by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: lambda function to pass Ray Trainers to monitor the learning
              progress in tensorboard.
    """
    # Initialize Ray server, if not already running
    if not ray.is_initialized():
        ray.init(
            # Address of Ray cluster to connect to, if any
            address=None,
            # Number of CPUs assigned to each raylet (None to disable limit)
            num_cpus=num_cpus,
            # Number of GPUs assigned to each raylet (None to disable limit)
            num_gpus=num_gpus,
            # Enable object eviction in LRU order under memory pressure
            _lru_evict=False,
            # Whether or not to execute the code serially (for debugging)
            local_mode=False,
            # Logging level
            logging_level=logging.DEBUG if debug else logging.ERROR,
            # Whether to redirect the output from every worker to the driver
            log_to_driver=debug,
            # Whether to start Ray dashboard, which displays cluster's status
            include_dashboard=True,
            # The host to bind the dashboard server to
            dashboard_host="0.0.0.0")

    # Configure Tensorboard
    if log_root_path is None:
        log_root_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "log")
    if 'tb' not in locals().keys():
        tb = TensorBoard()
        tb.configure(host="0.0.0.0", logdir=log_root_path)
        url = tb.launch()
        if verbose:
            print(f"Started Tensorboard {url}. "
                  f"Root directory: {log_root_path}")

    # Create log directory
    if log_name is None:
        log_name = "_".join((datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                            socket.gethostname().replace('-', '_')))
    log_path = os.path.join(log_root_path, log_name)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Tensorboard logfiles directory: {log_path}")

    # Define Ray logger
    def logger_creator(config):
        return UnifiedLogger(config, log_path, loggers=None)

    return logger_creator


def compute_action(policy: TFPolicy,
                   dist_class: ActionDistribution,
                   input_dict: Dict[str, np.ndarray],
                   explore: bool) -> np.ndarray:
    """TODO Write documentation.
    """
    if policy.framework == 'torch':
        with torch.no_grad():
            input_dict = policy._lazy_tensor_dict(input_dict)
            action_logits, _ = policy.model(input_dict)
            action_dist = dist_class(action_logits, policy.model)
            if explore:
                action_torch = action_dist.sample()
            else:
                action_torch = action_dist.deterministic_sample()
            action = action_torch.cpu().numpy()
    elif tf.compat.v1.executing_eagerly():
        action_logits, _ = policy.model(input_dict)
        action_dist = dist_class(action_logits, policy.model)
        if explore:
            action_tf = action_dist.sample()
        else:
            action_tf = action_dist.deterministic_sample()
        action = action_tf.numpy()
    else:
        # This obscure piece of code takes advantage of already existing
        # placeholders to avoid creating new nodes to evalute computation
        # graph. It is several order of magnitude more efficient than calling
        # `action_logits, _ = model(input_dict).eval(session=policy._sess)[0]`
        # directly, but also significantly trickier.
        feed_dict = {policy._input_dict[key]: value
                     for key, value in input_dict.items()
                     if key in policy._input_dict.keys()}
        feed_dict[policy._is_exploring] = explore
        action = policy._sess.run(
            policy._sampled_action, feed_dict=feed_dict)
    return action


def evaluate(env_creator: Callable[..., gym.Env],
             policy: TFPolicy,
             dist_class: ActionDistribution,
             obs_filter_fn: Optional[
                 Callable[[np.ndarray], np.ndarray]] = None,
             n_frames_stack: int = 1,
             horizon: Optional[int] = None,
             clip_action: bool = False,
             explore: bool = False,
             enable_stats: bool = True,
             enable_replay: bool = True,
             viewer_kwargs: Optional[Dict[str, Any]] = None) -> gym.Env:
    """TODO Write documentation.
    """
    # Handling of default arguments
    if viewer_kwargs is None:
        viewer_kwargs = {}

    # Instantiate the environment
    env = FlattenObservation(env_creator(debug=True))
    observation_space, action_space = env.observation_space, env.action_space

    # Initialize frame stack
    input_dict = {
        "obs": np.zeros([1, *observation_space.shape]),
        "prev_n_obs": np.zeros([1, n_frames_stack, *observation_space.shape]),
        "prev_n_act": np.zeros([1, n_frames_stack, *action_space.shape]),
        "prev_n_rew": np.zeros([1, n_frames_stack])
    }

    # Initialize the simulation
    obs = env.reset()

    # Run the simulation
    try:
        info_episode = []
        tot_reward = 0.0
        done = False
        while not done:
            if obs_filter_fn is not None:
                obs = obs_filter_fn(obs)
            input_dict["obs"][0] = obs
            action = compute_action(policy, dist_class, input_dict, explore)
            if clip_action:
                action = clip(action_space, action)
            input_dict["prev_n_obs"][0, -1] = input_dict["obs"][0]
            obs, reward, done, info = env.step(action)
            input_dict["prev_n_act"][0, -1] = action
            input_dict["prev_n_rew"][0, -1] = reward
            info_episode.append(info)
            tot_reward += reward
            if done or (horizon is not None and env.num_steps > horizon):
                break
            for field in input_dict.values():
                field[:] = np.roll(field, shift=-1, axis=1)
    except KeyboardInterrupt:
        pass

    # Display some statistic if requested
    if enable_stats:
        print("env.num_steps:", env.num_steps)
        print("cumulative reward:", tot_reward)

    # Replay the result if requested
    if enable_replay:
        env.replay(**{'speed_ratio': 1.0, **viewer_kwargs})

    return env, info_episode


def test(test_agent: Trainer,
         explore: bool = True,
         n_frames_stack: int = 1,
         enable_stats: bool = True,
         enable_replay: bool = True,
         viewer_kwargs: Optional[Dict[str, Any]] = None,
         **kwargs: Any) -> gym.Env:
    """Test a model on a specific environment using a given agent. It will
    render the result in the default viewer.

    .. note::
        This function can be terminated early using CTRL+C.
    """
    # Define environment creator
    def env_creator(**kwargs: Any):
        nonlocal test_agent
        return test_agent.env_creator(
            {**test_agent.config["env_config"], **kwargs})

    # Get policy model
    policy = test_agent.get_policy()
    dist_class = policy.dist_class
    obs_filter = test_agent.workers.local_worker().filters["default_policy"]
    if isinstance(obs_filter, NoFilter):
        obs_filter_fn = None
    else:
        obs_mean, obs_std = obs_filter.rs.mean, obs_filter.rs.std
        obs_filter_fn = \
            lambda obs: (obs - obs_mean) / (obs_std + 1.0e-8)  # noqa: E731

    if viewer_kwargs is not None:
        kwargs.update(viewer_kwargs)

    return evaluate(env_creator,
                    policy,
                    dist_class,
                    obs_filter_fn,
                    n_frames_stack=n_frames_stack,
                    horizon=test_agent.config["horizon"],
                    clip_action=test_agent.config["clip_actions"],
                    explore=explore,
                    enable_stats=enable_stats,
                    enable_replay=enable_replay,
                    viewer_kwargs=kwargs)


def train(train_agent: Trainer,
          max_timesteps: int,
          evaluation_period: int = 0,
          verbose: bool = True) -> str:
    """Train a model on a specific environment using a given agent.

    Note that the agent is associated with a given reinforcement learning
    algorithm, and instanciated for a specific environment and neural network
    model. Thus, it already wraps all the required information to actually
    perform training.

    .. note::
        This function can be terminated early using CTRL+C.

    :param train_agent: Training agent.
    :param max_timesteps: Number of maximum training timesteps.
    :param evaluation_period: Run one simulation without exploration every
                              given number of training steps, and save a video
                              of the esult in log folder. 0 to disable.
                              Optional: Disable by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: Fullpath of agent's final state dump. Note that it also contains
              the trained neural network model.
    """
    env_spec = [spec for ev in train_agent.workers.foreach_worker(
        lambda ev: ev.foreach_env(lambda env: env.spec)) for spec in ev][0]
    if env_spec is None or env_spec.reward_threshold is None:
        reward_threshold = math.inf
    else:
        reward_threshold = env_spec.reward_threshold

    try:
        while True:
            # Perform one iteration of training the policy
            result = train_agent.train()

            # Print current training result
            msg_data = []
            for field in PRINT_RESULT_FIELDS_FILTER:
                if field in result.keys():
                    msg_data.append(f"{field}: {result[field]:.5g}")
            print(" - ".join(msg_data))

            # Record video of the result if requested
            iter = result["training_iteration"]
            if evaluation_period > 0 and iter % evaluation_period == 0:
                record_video_path = f"{train_agent.logdir}/iter_{iter}.mp4"
                test(train_agent, explore=False, viewer_kwargs={
                    "record_video_path": record_video_path,
                    "scene_name": f"iter_{iter}"})

            # Check terminal conditions
            if result["timesteps_total"] > max_timesteps:
                break
            if result["episode_reward_mean"] > reward_threshold:
                if verbose:
                    print("Problem solved successfully!")
                break
    except KeyboardInterrupt:
        if verbose:
            print("Interrupting training...")

    return train_agent.save()
