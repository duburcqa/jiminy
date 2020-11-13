import os
import time
import math
import socket
import pathlib
from datetime import datetime
from typing import Optional, Callable

import gym
from gym.envs.registration import spec
from tensorboard.program import TensorBoard

import ray
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents.trainer import Trainer

from jiminy_py.viewer import sleep


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
            logging_level=20,
            # Whether to redirect the output from every worker to the driver
            log_to_driver=False,
            # Whether to start Ray dashboard, which displays cluster's status
            include_dashboard=True,
            # The host to bind the dashboard server to
            dashboard_host="0.0.0.0"
        )

    # Configure Tensorboard
    if log_root_path is None:
        log_root_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "log")
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


def train(train_agent: Trainer,
          max_timesteps: int,
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
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: Fullpath of agent's final state dump. Note that it also contains
              the trained neural network model.
    """
    env_spec = spec(train_agent._env_id)
    if env_spec.reward_threshold is None:
        env_spec.reward_threshold = math.inf

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

            # Check terminal conditions
            if result["timesteps_total"] > max_timesteps:
                break
            if result["episode_reward_mean"] > env_spec.reward_threshold:
                if verbose:
                    print("Problem solved successfully!")
                break
    except KeyboardInterrupt:
        if verbose:
            print("Interrupting training...")

    return train_agent.save()


def test(test_agent: Trainer,
         max_episodes: int = math.inf,
         max_duration: int = math.inf,
         verbose: bool = True) -> None:
    """Test a model on a specific environment using a given agent. It will
    render the result in the default viewer.

    .. note::
        This function can be terminated early using CTRL+C.

    :param max_episodes: Max number of episodes to run. Can be infinite.
                         Optional: infinite by default.
    :param max_duration: Max total duration of the episodes. Can be infinite.
                         Optional: infinite by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.
    """
    # Check user arguments
    if (math.isinf(max_episodes) and math.isinf(max_duration)):
        raise ValueError(
            "Either 'max_episodes' or 'max_duration' must be finite.")

    # Create environment
    env = gym.make(test_agent._env_id, **test_agent.config["env_config"])

    try:
        t_init, t_cur = time.time(), time.time()
        num_episodes = 0
        while (num_episodes < max_episodes) and \
                (t_cur - t_init < max_duration):
            obs = env.reset()
            cum_step, cum_reward = 0, 0.0
            done = False
            while not done:
                # Update state
                action = test_agent.compute_action(obs, explore=False)
                obs, reward, done, _ = env.step(action)
                cum_step += 1
                cum_reward += reward

                # Render the current state in default viewer
                env.render()
                sleep(env.step_dt - (time.time() - t_cur))
                t_cur = time.time()

                # Break the simulation if max duration reached
                if t_cur - t_init > max_duration:
                    break
            num_episodes += 1

            # Print the simulation final state
            if done and verbose:
                print(f"Episode length: {cum_step} - Cumulative reward: "
                      f"{cum_reward}")
    except KeyboardInterrupt:
        if verbose:
            print("Interrupting testing...")
