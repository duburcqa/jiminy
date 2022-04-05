""" TODO: Write documentation.
"""
import os
import re
import math
import json
import time
import shutil
import socket
import pathlib
import logging
import inspect
import tracemalloc
from tempfile import mkstemp
from datetime import datetime
from collections import defaultdict
from typing import Optional, Callable, Dict, Any, Tuple, Union, List

import gym
import numpy as np
import plotext as plt
from tensorboard.program import TensorBoard

import ray
import ray.cloudpickle as pickle
from ray import ray_constants
from ray.state import GlobalState
from ray._private import services
from ray._private.gcs_utils import AvailableResources
from ray._private.test_utils import monitor_memory_usage
from ray._raylet import GcsClientOptions
from ray.exceptions import RayTaskError
from ray.tune.logger import Logger, TBXLogger
from ray.tune.utils.util import SafeFallbackEncoder
from ray.rllib.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.filter import NoFilter, MeanStdFilter
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet

from jiminy_py.viewer import play_logs_files
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import clip, DataNested

try:
    import torch
except ModuleNotFoundError:
    pass
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


logger = logging.getLogger(__name__)


HISTOGRAM_BINS = 20

PRINT_RESULT_FIELDS_FILTER = [
    "training_iteration",
    "time_total_s",
    "timesteps_total",
    "episodes_total",
    "episode_reward_max",
    "episode_reward_mean",
    "episode_len_mean"
]


def initialize(num_cpus: int,
               num_gpus: int,
               log_root_path: str,
               log_name: Optional[str] = None,
               logger_cls: type = TBXLogger,
               launch_tensorboard: bool = True,
               debug: bool = False,
               verbose: bool = True,
               **ray_init_kwargs: Any) -> Callable[[Dict[str, Any]], Logger]:
    """Initialize Ray and Tensorboard daemons.

    It will be used later for almost everything from dashboard, remote/client
    management, to multithreaded environment.

    .. note:
        The default Tensorboard port will be used, namely 6006 if available,
        using 0.0.0.0 (binding to all IPv4 addresses on local machine).
        Similarly, Ray dashboard port is 8265 if available. In both cases, the
        port will be increased interatively until to find one available.

    :param num_cpus: Maximum number of CPU threads that can be executed in
                     parallel. Note that it does not actually reserve part of
                     the CPU, so that several processes can reserve the number
                     of threads available on the system at the same time.
    :param num_gpu: Maximum number of GPU unit that can be used, which can be
                    fractional to only allocate part of the resource. Note that
                    contrary to CPU resource, the memory is likely to actually
                    be reserve and allocated by the process, in particular
                    using Tensorflow backend.
    :param log_root_path: Fullpath of root log directory.
    :param log_name: Name of the subdirectory where to save data. `None` to
                     use default name, empty string '' to set it interactively
                     in command prompt. It must be a valid Python identifier.
                     Optional: full date _ hostname by default.
    :param logger_cls: Custom logger class type deriving from `TBXLogger`.
                       Optional: `TBXLogger` by default.
    :param launch_tensorboard: Whether or not to launch tensorboard
                               automatically.
                               Optional: Enabled by default.
    :param debug: Whether or not to display debugging trace.
                  Optional: Disabled by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: lambda function to pass a `ray.Trainer` to monitor learning
              progress in Tensorboard.
    """
    # Make sure provided logger class derives from ray.tune.logger.Logger
    assert issubclass(logger_cls, Logger), (
        "Logger class must derive from `ray.tune.logger.Logger`")

    # Check if cluster servers are already running, and if requested resources
    # are available.
    is_cluster_running = False
    redis_addresses = services.find_redis_address()
    if redis_addresses:
        for redis_address in redis_addresses:
            # Connect to redis global state accessor
            state = GlobalState()
            options = GcsClientOptions.from_redis_address(
                redis_address, ray_constants.REDIS_DEFAULT_PASSWORD)
            state._initialize_global_state(options)
            state._really_init_global_state()
            global_state_accessor = state.global_state_accessor
            assert global_state_accessor is not None

            # Get available resources
            resources: Dict[str, int] = defaultdict(int)
            for info in global_state_accessor.get_all_available_resources():
                # pylint: disable=no-member
                message = AvailableResources.FromString(info)
                for field, capacity in message.resources_available.items():
                    resources[field] += capacity

            # Disconnect global state accessor
            time.sleep(0.1)
            state.disconnect()

            # Check if enough computation resources are available
            is_cluster_running = (resources["CPU"] >= num_cpus and
                                  resources["GPU"] >= num_gpus)

            # Stop looking as soon as a cluster with enough resources is found
            if is_cluster_running:
                break

    # Connect to Ray server if necessary, starting one if not already running
    if not ray.is_initialized():
        if not is_cluster_running:
            # Start new Ray server, if not already running
            ray.init(
                # Number of CPUs assigned to each raylet
                num_cpus=num_cpus,
                # Number of GPUs assigned to each raylet
                num_gpus=num_gpus,
                # Logging level
                logging_level=logging.DEBUG if debug else logging.ERROR,
                # Whether to redirect outputs from every worker to the driver
                log_to_driver=debug, **{**dict(
                    # Whether to start Ray dashboard to monitor cluster status
                    include_dashboard=False,
                    # The host to bind the dashboard server to
                    dashboard_host="0.0.0.0"
                ), **ray_init_kwargs})
        else:
            # Connect to existing Ray cluster
            ray.init(
                # Address of Ray cluster to connect to
                address=redis_addresses,
                _node_ip_address=next(iter(redis_addresses)).split(":", 1)[0])

    # Configure Tensorboard
    if launch_tensorboard:
        tb = TensorBoard()
        tb.configure(host="0.0.0.0", logdir=os.path.abspath(log_root_path))
        url = tb.launch()
        if verbose:
            print(f"Started Tensorboard {url}.",
                  f"Root directory: {log_root_path}")

    # Monitor memory usage in debug
    if debug:
        monitor_memory_usage(print_interval_s=60)

    # Define log filename interactively if requested
    if log_name == "":
        while True:
            log_name = input(
                "Enter desired log subdirectory name (empty for default)...")
            if not log_name or re.match(r'^[A-Za-z0-9_]+$', log_name):
                break
            print("Unvalid name. Only Python identifiers are supported.")

    # Handling of default log name and sanity checks
    if not log_name:
        log_name = "_".join((
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            re.sub(r'[^A-Za-z0-9_]', "_", socket.gethostname())))
    else:
        assert re.match(r'^[A-Za-z0-9_]+$', log_name), (
            "Log name must be a valid Python identifier.")

    # Create log directory
    log_path = os.path.join(log_root_path, log_name)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Tensorboard logfiles directory: {log_path}")

    # Define Ray logger
    def logger_creator(config: Dict[str, Any]) -> Logger:
        return logger_cls(config, log_path)

    return logger_creator


def compute_action(policy: Policy,
                   input_dict: Dict[str, np.ndarray],
                   explore: bool) -> Tuple[Any, Any]:
    """Compute predicted action by the policy.

    .. note::
        It supports both Pytorch and Tensorflow backends (both eager and
        compiled graph modes).

    :param policy: `rllib.policy.Policy` to use to predict the action, which is
                   a thin wrapper around the actual policy model.
    :param input_dict: Input dictionary for forward as input of the policy.
    :param explore: Whether or not to enable exploration during sampling of the
                    action.
    """
    if policy.framework == 'torch':
        assert isinstance(policy, TorchPolicy)
        input_dict = policy._lazy_tensor_dict(input_dict)
        with torch.no_grad():
            policy.model.eval()
            if policy.action_distribution_fn is not None:
                action_logits, dist_class, state = \
                    policy.action_distribution_fn(
                        policy=policy,
                        model=policy.model,
                        obs_batch=input_dict["obs"],
                        explore=explore,
                        is_training=False)
            else:
                action_logits, state = policy.model(input_dict)
                dist_class = policy.dist_class
            action_dist = dist_class(action_logits, policy.model)
            if explore:
                action_torch = action_dist.sample()
            else:
                action_torch = action_dist.deterministic_sample()
            action = action_torch.cpu().numpy()
    elif tf.compat.v1.executing_eagerly():
        assert isinstance(policy, TFPolicy)
        action_logits, state = policy.model(input_dict)
        action_dist = policy.dist_class(action_logits, policy.model)
        if explore:
            action_tf = action_dist.sample()
        else:
            action_tf = action_dist.deterministic_sample()
        action = action_tf.numpy()
    else:
        # This obscure piece of code takes advantage of already existing
        # placeholders to avoid creating new nodes to evalute computation
        # graph. It is several order of magnitude more efficient than calling
        # `action_logits, _ = model(input_dict).eval(session=policy._sess)`.
        assert isinstance(policy, TFPolicy)
        feed_dict = {policy._input_dict[key]: value
                     for key, value in input_dict.items()
                     if key in policy._input_dict.keys()}
        feed_dict[policy._is_exploring] = np.array(True)
        action, *state = policy._sess.run(
            [policy._sampled_action] + policy._state_outputs,
            feed_dict=feed_dict)
    return action, state


def build_policy_wrapper(policy: Policy,
                         obs_filter_fn: Optional[
                             Callable[[np.ndarray], np.ndarray]] = None,
                         n_frames_stack: int = 1,
                         clip_action: bool = False,
                         explore: bool = False) -> Callable[
                             [DataNested, Optional[float]], DataNested]:
    """Wrap a policy into a simple callable.

    .. note::
        The internal state of the policy, if any, is managed internally.

    .. warning:
        One is responsible of instantiating a new wrapper to reset the internal
        state between simulations if necessary, for example for recurrent
        network or for policy depending on several frames.

    :param policy: Policy to evaluate.
    :param obs_filter_fn: Observation filter to apply on (flattened)
                          observation from the environment, usually used
                          from moving average normalization. `None` to
                          disable.
                          Optional: Disabled by default.
    :param n_frames_stack: Number of frames to stack in the input to provide
                           to the policy. Note that previous observation,
                           action, and reward will be stacked.
                           Optional: 1 by default.
    :param clip_action: Whether or not to clip action to make sure the
                        prediction by the policy is not out-of-bounds.
                        Optional: Disabled by default.
    :param explore: Whether or not to enable exploration during sampling of the
                    actions predicted by the policy.
                    Optional: Disabled by default.
    """
    # Extract some proxies for convenience
    observation_space = policy.observation_space
    action_space = policy.action_space

    # Build preprocessor to flatten environment observation
    observation_space_orig = observation_space
    if hasattr(observation_space_orig, "original_space"):
        observation_space_orig = observation_space.original_space
    preprocessor_class = get_preprocessor(observation_space_orig)
    preprocessor = preprocessor_class(observation_space_orig)
    obs_flat = observation_space.sample()

    # Initialize frame stack
    input_dict = {
        "obs": np.zeros([1, *observation_space.shape]),
        "prev_n_obs": np.zeros([1, n_frames_stack, *observation_space.shape]),
        "prev_n_act": np.zeros([1, n_frames_stack, *action_space.shape]),
        "prev_n_rew": np.zeros([1, n_frames_stack])}

    # Run the simulation
    def forward(obs: DataNested,
                reward: Optional[float]) -> DataNested:
        nonlocal policy, obs_flat, input_dict, explore, clip_action

        # Compute flat observation
        preprocessor.write(obs, obs_flat, 0)

        # Filter observation if necessary
        if obs_filter_fn is not None:
            obs_flat = obs_filter_fn(obs_flat)

        # Update current observation and previous reward buffers
        input_dict["obs"][0] = obs_flat
        if reward is not None:
            input_dict["prev_n_rew"][0, -1] = reward

        # Compute action
        action, _ = compute_action(policy, input_dict, explore)
        if clip_action:
            action = clip(action_space, action)

        # Update previous observation and action buffers
        input_dict["prev_n_obs"][0, -1] = input_dict["obs"][0]
        input_dict["prev_n_act"][0, -1] = action

        # Shift input dict history by one
        for field in input_dict.values():
            field[:] = np.roll(field, shift=-1, axis=1)

        return action.squeeze(0)

    return forward


def build_policy_from_checkpoint(
        policy_class: type,
        env_creator: Callable[[Dict[str, Any]], gym.Env],
        checkpoint_path: str,
        config: Dict[str, Any]) -> Policy:
    """ TODO: Write documentation
    """
    # Load checkpoint policy state
    with open(checkpoint_path, "rb") as checkpoint_dump:
        checkpoint_state = pickle.load(checkpoint_dump)
        worker_dump = checkpoint_state['worker']
        worker_state = pickle.loads(worker_dump)
        policy_state = worker_state['state']['default_policy']

    # Initiate temporary environment to get observation and action spaces
    env = env_creator(config.get("env_config", {}))

    # Get preprocessed observation space
    preprocessor_class = get_preprocessor(env.observation_space)
    preprocessor = preprocessor_class(env.observation_space)
    observation_space = preprocessor.observation_space

    # Instantiate policy and load checkpoint state
    policy = policy_class(observation_space, env.action_space, config)
    policy.set_state(policy_state)

    return policy


def train(train_agent: Trainer,
          max_timesteps: int = 0,
          max_iters: int = 0,
          evaluation_num: int = 10,
          evaluation_period: int = 0,
          checkpoint_period: int = 0,
          record_video: bool = True,
          verbose: bool = True,
          debug: bool = False) -> str:
    """Train a model on a specific environment using a given agent.

    Note that the agent is associated with a given reinforcement learning
    algorithm, and instanciated for a specific environment and neural network
    model. Thus, it already wraps all the required information to actually
    perform training.

    .. note::
        This function can be terminated early using CTRL+C.

    :param train_agent: Training agent.
    :param max_timesteps: Maximum number of training timesteps. 0 to disable.
                          Optional: Disabled by default.
    :param max_iters: Maximum number of training iterations. 0 to disable.
                      Optional: Disabled by default.
    :param evaluation_num: How any evaluation to run. The log files of the best
                           and worst performing trials will be exported, and
                           some statistics will be reported if 'verbose' is
                           enabled.
    :param evaluation_period: Run one simulation (with exploration) every given
                              number of training steps, and save the log file
                              and a video of the result in log folder if
                              requested. 0 to disable.
                              Optional: Disabled by default.
    :param checkpoint_period: Backup trainer every given number of training
                              steps in log folder if requested. 0 to disable.
                              Optional: Disabled by default.
    :param record_video: Whether or not to enable video recording during
                         evaluation. Video will be recorded for best and worst
                         trials.
                         Optional: True by default.
    :param debug: Whether or not to monitor memory allocation for debugging
                  memory leaks.
                  Optional: Disabled by default.
    :param verbose: Whether or not to print high-level information after each
                    training iteration.
                    Optional: True by default.

    :returns: Fullpath of agent's final state dump. Note that it also contains
              the trained neural network model.
    """
    # Get environment's reward threshold, if any
    assert isinstance(train_agent.workers, WorkerSet)
    env_spec, *_ = [val for worker in train_agent.workers.foreach_worker(
        lambda worker: worker.foreach_env(lambda env: env.spec))
        for val in worker]
    if env_spec is None or env_spec.reward_threshold is None:
        reward_threshold = math.inf
    else:
        reward_threshold = env_spec.reward_threshold

    # Backup some information
    if not train_agent.iteration:
        # Make sure log dir exists
        os.makedirs(train_agent.logdir, exist_ok=True)

        # Backup environment's source file
        env_type, *_ = [val for worker in train_agent.workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: type(env.unwrapped)))
            for val in worker]
        while True:
            try:
                path = inspect.getfile(env_type)
                shutil.copy2(path, train_agent.logdir, follow_symlinks=True)
            except TypeError:
                pass
            try:
                env_type = env_type.__bases__[0]
            except IndexError:
                break

        # Backup main's source file, if any
        frame = inspect.stack()[1]  # assuming called directly from main script
        main_file = inspect.getfile(frame[0])
        main_backup_name = f"{train_agent.logdir}/main.py"
        if main_file.endswith(".py"):
            shutil.copy2(main_file, main_backup_name, follow_symlinks=True)

        # Backup RLlib config
        with open(f"{train_agent.logdir}/params.json", 'w') as file:
            json.dump(train_agent.config,
                      file,
                      indent=2,
                      sort_keys=True,
                      cls=SafeFallbackEncoder)

    if debug:
        tracemalloc.start(10)
        snapshot = tracemalloc.take_snapshot()

    # Run several training iterations until terminal condition is reached
    try:
        while True:
            # Perform one iteration of training the policy
            result = train_agent.train()
            iter_num = result["training_iteration"]

            # Monitor memory allocation since the beginning and between iters
            if debug:
                snapshot_new = tracemalloc.take_snapshot()
                top_stats = snapshot_new.compare_to(snapshot, 'lineno')
                for stat in top_stats[:10]:
                    print(stat)
                top_trace = snapshot_new.statistics('traceback')
                if top_trace:
                    for line in top_trace[0].traceback.format():
                        print(line)
                snapshot = snapshot_new

            # Print current training result summary
            msg_data = []
            for field in PRINT_RESULT_FIELDS_FILTER:
                if field in result.keys():
                    msg_data.append(f"{field}: {result[field]:.5g}")
            print(" - ".join(msg_data))

            # Record video and log data of the result
            if evaluation_period > 0 and iter_num % evaluation_period == 0:
                duration = []
                total_rewards = []
                log_files_tmp = []
                test_env = train_agent.env_creator(
                    train_agent.config["env_config"])
                seed = train_agent.config["seed"] or 0
                for i in range(evaluation_num):
                    # Evaluate the policy once
                    test(train_agent,
                         explore=True,
                         seed=seed+i,
                         test_env=test_env,
                         enable_stats=False,
                         enable_replay=False)

                    # Export temporary log file
                    fd, log_path = mkstemp(prefix="log_", suffix=".hdf5")
                    os.close(fd)
                    test_env.write_log(log_path)

                    # Monitor some statistics
                    duration.append(test_env.num_steps * test_env.step_dt)
                    total_rewards.append(test_env.total_reward)
                    log_files_tmp.append(log_path)

                # Backup log file of best trial
                trial_best_idx = np.argmax(duration)
                log_path = f"{train_agent.logdir}/iter_{iter_num}.hdf5"
                shutil.move(log_files_tmp[trial_best_idx], log_path)

                # Record video of best trial if requested
                if record_video:
                    video_path = f"{train_agent.logdir}/iter_{iter_num}.mp4"
                    play_logs_files(log_path,
                                    record_video_path=video_path,
                                    scene_name=f"iter_{iter_num}")

                # Ascii histogram if requested
                if verbose:
                    try:
                        plt.clp()
                        plt.subplots(1, 2)
                        for i, (title, data) in enumerate(zip(
                                ("Episode duration", "Total reward"),
                                (duration, total_rewards))):
                            plt.subplot(1, i)
                            plt.hist(data, HISTOGRAM_BINS)
                            plt.plotsize(50, 20)
                            plt.title(title)
                        plt.show()
                    except IndexError as e:
                        logger.warning(f"Rendering statistics failed: {e}")

            # Backup the policy
            if checkpoint_period > 0 and iter_num % checkpoint_period == 0:
                train_agent.save()

            # Check terminal conditions
            if 0 < max_timesteps < result["timesteps_total"]:
                break
            if 0 < max_iters < iter_num:
                break
            if reward_threshold < result["episode_reward_mean"]:
                if verbose:
                    print("Problem solved successfully!")
                break
    except KeyboardInterrupt:
        if verbose:
            print("Interrupting training...")
    except RayTaskError as e:
        logger.warning(str(e))

    # Backup trained agent and return file location
    return train_agent.save()


def test(test_agent: Trainer,
         explore: bool = True,
         seed: Optional[int] = None,
         n_frames_stack: int = 1,
         enable_stats: bool = True,
         enable_replay: bool = True,
         test_env: Optional[BaseJiminyEnv] = None,
         viewer_kwargs: Optional[Dict[str, Any]] = None,
         **kwargs: Any) -> Union[BaseJiminyEnv, List[Dict[str, Any]]]:
    """Test a model on a specific environment using a given agent.

    .. note::
        This function can be terminated early using CTRL+C.

    :param test_agent: Agent to evaluate on a single simulation.
    :param seed: Seed of the environment to be used for the evaluation of the
                 policy. Note that the environment's seed is always reset.
                 Optional: `test_agent.config["seed"] or 0` if not especified.
    :param explore: Whether or not to enable exploration during sampling of the
                    actions predicted by the policy.
                    Optional: Disabled by default.
    :param n_frames_stack: Number of frames to stack in the input to provide
                           to the policy. Note that previous observation,
                           action, and reward will be stacked.
                           Optional: 1 by default.
    :param enable_stats: Whether or not to print high-level statistics after
                         simulation.
                         Optional: Enabled by default.
    :param enable_replay: Whether or not to enable replay of the simulation,
                          and eventually recording through `viewer_kwargs`.
                          Optional: Enabled by default.
    :param test_env: Environment on which to evaluate the policy. It must be
                     already instantiated and ready-to-use.
    :param viewer_kwargs: Extra keyword arguments to forward to the viewer if
                          replay has been requested.
    """
    # Instantiate the environment if not provided
    if test_env is None:
        test_env = test_agent.env_creator(EnvContext(
            **test_agent.config["env_config"], **kwargs))

    # Get policy model
    policy = test_agent.get_policy()

    # Get observation filter if any
    assert isinstance(test_agent.workers, WorkerSet)
    obs_filter = test_agent.workers.local_worker().filters["default_policy"]
    if isinstance(obs_filter, NoFilter):
        obs_filter_fn = None
    elif isinstance(obs_filter, MeanStdFilter):
        obs_mean, obs_std = obs_filter.rs.mean, obs_filter.rs.std
        obs_filter_fn = \
            lambda obs: (obs - obs_mean) / (obs_std + 1.0e-8)  # noqa: E731
    else:
        raise RuntimeError(f"Filter '{obs_filter.__class__}' not supported.")

    # Forward viewer keyword arguments
    if viewer_kwargs is not None:
        kwargs.update(viewer_kwargs)

    # Wrap policy as a callback function
    policy_fn = build_policy_wrapper(policy,
                                     obs_filter_fn,
                                     n_frames_stack,
                                     test_agent.config["clip_actions"],
                                     explore)

    # Evaluate the policy
    info_episode = test_env.evaluate(test_env,
                                     policy_fn,
                                     seed or test_agent.config["seed"] or 0,
                                     horizon=test_agent.config["horizon"],
                                     enable_stats=enable_stats,
                                     enable_replay=enable_replay,
                                     **kwargs)

    return test_env, info_episode


__all__ = [
    "initialize",
    "build_policy_wrapper",
    "build_policy_from_checkpoint",
    "train",
    "test"
]
