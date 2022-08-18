import os
import time
import math
import socket
import pathlib
import tempfile
from datetime import datetime
from typing import Optional

from tensorboard.program import TensorBoard

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold as StopOnReward)

from jiminy_py.viewer import sleep


def initialize(log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               verbose: bool = True) -> str:
    """Initialize Tensorboard daemon.

    It will be used later for monitoring the learning progress.

    :param log_root_path: Fullpath of root log directory.
                          Optional: location of this file / log by default.
    :param log_name: Name of the subdirectory where to save data.
                     Optional: full date _ hostname by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: Tensorboard data path.
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

    return log_path


def train(train_agent: BaseAlgorithm,
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
    # Get testing environment spec
    spec = train_agent.eval_env.envs[0].spec

    # Create callback to stop learning early if reward threshold is exceeded
    if spec.reward_threshold is not None:
        callback_reward = StopOnReward(
            reward_threshold=spec.reward_threshold)
        eval_callback = EvalCallback(
            train_agent.eval_env, callback_on_new_best=callback_reward,
            eval_freq=5000, n_eval_episodes=100)
    else:
        eval_callback = None

    try:
        # Run the learning process
        train_agent.learn(total_timesteps=max_timesteps,
                          log_interval=5,
                          reset_num_timesteps=False,
                          callback=eval_callback)
        if train_agent.num_timesteps < max_timesteps:
            print("Problem solved successfully!")
    except KeyboardInterrupt:
        if verbose:
            print("Interrupting training...")

    fd, checkpoint_path = tempfile.mkstemp(
        dir=train_agent.tensorboard_log, prefix=spec.id, suffix='.zip')
    os.close(fd)
    train_agent.save(checkpoint_path)

    return checkpoint_path


def test(test_agent: BaseAlgorithm,
         max_episodes: int = math.inf,
         max_duration: int = math.inf,
         verbose: bool = True) -> None:
    """Test a model on a specific environment using a given agent. It will
    render the result in the default viewer.

    .. note::
        This function can be terminated early using CTRL+C.

    :param train_agent: Testing agent.
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

    # Get environment timestep
    step_dt = test_agent.eval_env.envs[0].step_dt

    try:
        t_init, t_cur = time.time(), time.time()
        num_episodes = 0
        while (num_episodes < max_episodes) and \
                (t_cur - t_init < max_duration):
            obs = test_agent.eval_env.reset()
            cum_step, cum_reward = 0, 0.0
            done = False
            while not done:
                # Update state
                action = test_agent.predict(obs)
                obs, reward, done, _ = test_agent.eval_env.step(action)
                cum_step += 1
                cum_reward += reward[0]

                # Render the current state in default viewer
                test_agent.eval_env.render()
                sleep(step_dt - (time.time() - t_cur))
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
