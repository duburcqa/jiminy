import os
import tqdm
import time
import socket
import pathlib
import tempfile
from datetime import datetime
from typing import Optional, Sequence, Union, Callable, Dict, Any

import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.program import TensorBoard

from tianshou.utils import tqdm_config, MovAvg
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer import test_episode, gather_info


def onpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        frame_per_epoch: int,
        collect_per_step: int,
        repeat_per_collect: int,
        episode_per_test: Union[int, Sequence[int]],
        batch_size: int,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1,
        verbose: bool = True,
        test_in_train: bool = True,
        **kwargs) -> Dict[str, Union[float, str]]:
    """Slightly modified Tianshou `onpolicy_trainer` original method to enable
    to define the maximum number of training steps instead of number of
    episodes, for consistency with other learning frameworks.
    """
    global_step = 0
    best_epoch, best_reward = -1, -1.0
    stat: Dict[str, MovAvg] = {}
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=frame_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, global_step)
                result = train_collector.collect(n_step=collect_per_step)
                data = {}
                if test_in_train and stop_fn and stop_fn(result["rew"]):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test, writer, global_step)
                    if stop_fn(test_result["rew"]):
                        if save_fn:
                            save_fn(policy)
                        for k in result.keys():
                            data[k] = f"{result[k]:.2f}"
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result["rew"])
                    else:
                        policy.train()
                losses = policy.update(
                    0, train_collector.buffer,
                    batch_size=batch_size, repeat=repeat_per_collect)
                train_collector.reset_buffer()
                step = 1
                for v in losses.values():
                    if isinstance(v, (list, tuple)):
                        step = max(step, len(v))
                global_step += step * collect_per_step
                for k in result.keys():
                    data[k] = f"{result[k]:.2f}"
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            "train/" + k, result[k], global_step=global_step)
                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f"{stat[k].get():.6f}"
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=global_step)
                t.update(collect_per_step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(policy, test_collector, test_fn, epoch,
                              episode_per_test, writer, global_step)
        if best_epoch == -1 or best_reward < result["rew"]:
            best_reward = result["rew"]
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f"Epoch #{epoch}: test_reward: {result['rew']:.6f}, "
                  f"best_reward: {best_reward:.6f} in #{best_epoch}")
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward)


def initialize(log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               verbose: bool = True) -> SummaryWriter:
    """Initialize Tensorboard daemon.

    .. note::
        It will be used later for monitoring the learning progress.

    :param log_root_path: Fullpath of root log directory.
                          Optional: location of this file / log by default.
    :param log_name: Name of the subdirectory where to save data.
                     Optional: full date _ hostname by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: SummaryWriter to pass to the training agent to monitor the
              training progress.
    """
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

    return SummaryWriter(log_path)


def train(train_agent: BasePolicy,
          train_envs: SubprocVectorEnv,
          test_envs: SubprocVectorEnv,
          writer: SummaryWriter,
          config: Dict[str, Any],
          verbose: bool = True) -> str:
    """Train a model on a specific environment using a given agent.

    Note that the agent is associated with a given reinforcement learning
    algorithm.

    .. note::
        This function can be terminated early using CTRL+C.

    :param train_agent: Training agent.
    :param train_envs: Training environment vector.
    :param test_envs: Testing environment vector.
    :param writer: SummaryWriter used to monitor training progress.
    :param config: Configuration dictionary to pass on-policy trainer.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: Fullpath of agent's final state dump. Note that it also contains
              the trained neural network model.
    """
    # Create the replay buffer
    buffer_size = config["collect_per_step"]
    replay_buffer = ReplayBuffer(buffer_size)

    # Get testing environment spec
    spec = train_envs.spec[0]

    # Create callback to stop learning early if reward threshold is exceeded
    if spec.reward_threshold is not None:
        def stop_fn(mean_rewards):
            return mean_rewards >= spec.reward_threshold
    else:
        stop_fn = None

    # Create collectors
    train_collector = Collector(train_agent, train_envs, replay_buffer)
    test_collector = Collector(train_agent, test_envs)

    # Configure export
    fd, checkpoint_path = tempfile.mkstemp(
        dir=writer.log_dir, prefix=spec.id, suffix='.zip')
    os.close(fd)

    def save_fn(train_agent):
        torch.save(train_agent.state_dict(), checkpoint_path)

    # Run the learning process
    try:
        result = onpolicy_trainer(
            train_agent, train_collector, test_collector,
            **config, stop_fn=stop_fn, save_fn=save_fn,
            writer=writer, verbose=True)
        max_timesteps = config["frame_per_epoch"] * config["max_epoch"]
        if verbose and (result["train_step"] < max_timesteps):
            print("Problem solved successfully!")
    except KeyboardInterrupt:
        if verbose:
            print("Interrupting training...")

    return checkpoint_path


def test(test_agent: BasePolicy,
         env_creator: Callable[[], gym.Env],
         num_episodes: int,
         verbose: bool = True) -> None:
    """Test a model on a specific environment using a given agent. It will
    render the result in the default viewer.

    .. note::
        This function can be terminated early using CTRL+C.

    :param env_creator: Lambda function without argument used to create a
                        learning environment.
    :param num_episodes: Max number of episodes to run.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.
    """
    env = env_creator()
    collector = Collector(test_agent, env)
    result = collector.collect(n_episode=1, render=env.step_dt)
    if verbose:
        print(f"Final reward: {result['rew']}, length: {result['len']}")
