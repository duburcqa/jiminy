""" TODO: Write documentation.
"""
import os
import re
import json
import time
import shutil
import socket
import pathlib
import logging
import inspect
import tracemalloc
from itertools import chain
from datetime import datetime
from collections import defaultdict
from tempfile import mkstemp, mkdtemp
from typing import Optional, Callable, Dict, Any, Union
from operator import attrgetter

import gym
import numpy as np
import plotext as plt
import tree

import ray
from ray._private import services
from ray._private.state import GlobalState
from ray._private.gcs_utils import AvailableResources
from ray._private.test_utils import monitor_memory_usage
from ray._raylet import GcsClientOptions
from ray.exceptions import RayTaskError
from ray.tune.logger import Logger, TBXLogger
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.utils.util import SafeFallbackEncoder
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_THIS_ITER
from ray.rllib.utils.typing import SampleBatchType

from jiminy_py.viewer import play_logs_files
from gym_jiminy.common.envs import BaseJiminyEnv


logger = logging.getLogger(__name__)


HISTOGRAM_BINS = 15

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
               log_root_path: Optional[str] = None,
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
                          Default to root temporary directory.
    :param log_name: Name of the subdirectory where to save data. `None` to
                     use default name, empty string '' to set it interactively
                     in command prompt. It must be a valid Python identifier.
                     Optional: full date _ hostname by default.
    :param logger_cls: Custom logger class type deriving from `TBXLogger`.
                       Optional: `TBXLogger` by default.
    :param launch_tensorboard: Whether to launch tensorboard automatically.
                               Optional: Enabled by default.
    :param debug: Whether to display debugging trace.
                  Optional: Disabled by default.
    :param verbose: Whether to print information about what is going on.
                    Optional: True by default.

    :returns: lambda function to pass a `ray.Trainer` to monitor learning
              progress in Tensorboard.
    """
    # Make sure provided logger class derives from ray.tune.logger.Logger
    assert issubclass(logger_cls, Logger), (
        "Logger class must derive from `ray.tune.logger.Logger`")

    # handling of default log directory
    log_root_path = mkdtemp()

    # Check if cluster servers are already running, and if requested resources
    # are available.
    is_cluster_running, ray_address = False, None
    for ray_address in services.find_gcs_addresses():
        # Connect to redis global state accessor
        state = GlobalState()
        options = GcsClientOptions.from_gcs_address(ray_address)
        state._initialize_global_state(options)
        state._really_init_global_state()
        global_state_accessor = state.global_state_accessor
        assert global_state_accessor is not None

        # Get available resources
        resources: Dict[str, Union[int, float]] = defaultdict(int)
        for info in global_state_accessor.get_all_available_resources():
            # pylint: disable=no-member
            message = AvailableResources.FromString(info)
            for field, capacity in message.resources_available.items():
                resources[field] += capacity

        # Disconnect global state accessor
        time.sleep(0.1)
        state.disconnect()

        # Check if enough computation resources are available
        is_cluster_running = (
            resources["CPU"] >= num_cpus and resources["GPU"] >= num_gpus)

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
                address=ray_address,
                # _node_ip_address=next(iter(ray_address)).split(":", 1)[0]
                logging_level=logging.DEBUG if debug else logging.ERROR,
                )

    # Configure Tensorboard
    if launch_tensorboard:
        try:
            # pylint: disable=import-outside-toplevel,import-error
            from tensorboard.program import TensorBoard
            from contextlib import redirect_stdout
            tb = TensorBoard()
            tb.configure(host="0.0.0.0",
                         load_fast=False,
                         logdir=os.path.abspath(log_root_path))
            with open(os.devnull, 'w') as stdout, redirect_stdout(stdout):
                url = tb.launch()
            if verbose:
                print(f"Started Tensorboard {url}.",
                      f"Root directory: {log_root_path}")
        except ImportError:
            logger.warning("Tensorboard not available. Cannot start server.")

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
            print("Invalid name. Only Python identifiers are supported.")

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


def train(algo: Algorithm,
          max_timesteps: int = 0,
          max_iters: int = 0,
          checkpoint_period: int = 0,
          verbose: bool = True,
          debug: bool = False) -> str:
    """Train a model on a specific environment using a given algorithm.

    The algorithm is associated with a given reinforcement learning algorithm,
    instantiated for a specific environment and policy model. Thus, it already
    wraps all the required information to actually perform training.

    .. note::
        This function can be aborted using CTRL+C without raising an exception.

    :param algo: Training algorithm.
    :param max_timesteps: Maximum number of training timesteps. 0 to disable.
                          Optional: Disabled by default.
    :param max_iters: Maximum number of training iterations. 0 to disable.
                      Optional: Disabled by default.
    :param checkpoint_period: Backup trainer every given number of training
                              steps in log folder if requested. 0 to disable.
                              Optional: Disabled by default.
    :param verbose: Whether to print high-level information after each training
                    iteration.
                    Optional: True by default.
    :param debug: Whether to monitor memory allocation to debug memory leaks.
                  Optional: Disabled by default.

    :returns: Fullpath of algorithm's final state dump. This includes the
              trained policy model.
    """
    # Get environment's reward threshold, if any
    assert isinstance(algo.workers, WorkerSet)
    env_spec, *_ = chain(*algo.workers.foreach_env(attrgetter('spec')))
    if env_spec is None or env_spec is None:
        reward_threshold = float('inf')
    else:
        reward_threshold = env_spec.reward_threshold

    # Backup some information
    if algo.iteration == 0:
        # Make sure log dir exists
        os.makedirs(algo.logdir, exist_ok=True)

        # Backup environment's source file
        (env_type,) = set(
            chain(*algo.workers.foreach_env(lambda e: type(e.unwrapped))))
        while True:
            try:
                path = inspect.getfile(env_type)
                shutil.copy2(path, algo.logdir, follow_symlinks=True)
            except TypeError:
                pass
            try:
                env_type, *_ = env_type.__bases__
            except ValueError:
                break

        # Backup main's source file, if any
        frame_info_0, frame_info_1, *_ = inspect.stack()
        root_file = frame_info_0.filename
        if os.path.exists(root_file):
            source_file = frame_info_1.filename
            main_backup_name = f"{algo.logdir}/main.py"
            shutil.copy2(source_file, main_backup_name, follow_symlinks=True)

        # Backup RLlib config
        with open(f"{algo.logdir}/params.json", 'w') as file:
            json.dump(algo.config.to_dict(),
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
            result = algo.train()
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

            # Backup the policy
            if checkpoint_period > 0 and iter_num % checkpoint_period == 0:
                algo.save()

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
    return algo.save()


def evaluate(algo: Algorithm,
             eval_workers: Optional[WorkerSet] = None,
             evaluation_num: Optional[int] = None,
             print_stats: bool = False,
             enable_replay: Optional[bool] = None,
             record_video: bool = False,
             raw_data: bool = False,
             **kwargs: Any) -> Union[dict, SampleBatchType]:
    """Evaluates the current policy under `evaluation_config` configuration,
    then returns some performance metrics.

    .. details::
        This method is specifically tailored for Gym environments inheriting
        from `BaseJiminyEnv`. It can be used to monitor the training progress.

    .. note::
        This function can be aborted using CTRL+C without raising an exception.

    .. warning::
        Remote evaluation workers are not supported for now.

    :param eval_workers: Rollout workers for evaluation.
                         Optional: `algo.evaluation_workers` by default.
    :param evaluation_num: How any evaluation to run. The log files of the best
                           performing trial will be exported.
                           Optional: `algo.config["evaluation_duration"]` by
                           default.
    :param print_stats: Whether to print high-level statistics.
                        Optional: Enabled by default.
    :param enable_replay: Whether to enable replay of the simulation.
                          Optional: The opposite of `record_video` by default.
    :param record_video: Whether to enable video recording during evaluation.
                         The video will feature the best and worst trials.
                         Optional: Disable by default.
    :param raw_data: Whether to return raw collected data or aggregated
                     performance metrics.
    :param kwargs: Extra keyword arguments to forward to the viewer if any.
    """
    # Handling of default arguments
    if eval_workers is None:
        eval_workers = algo.evaluation_workers

    # Extract evaluation config
    eval_cfg = algo.evaluation_config

    # Check some pre-conditions for using this method
    (env_type,) = set(
        chain(*eval_workers.foreach_env(lambda e: type(e.unwrapped))))
    if not issubclass(env_type, BaseJiminyEnv):
        raise RuntimeError("Test env must inherit from `BaseJiminyEnv`.")
    if algo.config["evaluation_duration_unit"] == "auto":
        raise ValueError("evaluation_duration_unit='timesteps' not supported.")
    num_episodes = evaluation_num or algo.config["evaluation_duration"]
    if num_episodes == "auto":
        raise ValueError("evaluation_duration='auto' not supported.")

    # Get the step size of the environment
    (env_dt,) = set(chain(*eval_workers.foreach_env(attrgetter("step_dt"))))

    # Helpers to force writing log at episode termination
    class WriteLogHook:
        """Stateful function used has a temporary wrapper around an original
        `on_episode_end` callback instance method to force writing log right
        after termination and before reset. This is necessary because
        `worker.sample()` always reset the environment at the end by design.
        """
        def __init__(self, on_episode_end: Callable) -> None:
            self.__func__ = on_episode_end

        def __call__(self: DefaultCallbacks, *,
                     worker: RolloutWorker,
                     **kwargs: Any) -> None:
            def write_log(env: gym.Env) -> str:
                fd, log_path = mkstemp(prefix="log_", suffix=".hdf5")
                os.close(fd)
                env.write_log(log_path, format="hdf5")
                return log_path

            worker.callbacks.log_files = worker.foreach_env(write_log)
            self.__func__(worker=worker, **kwargs)

    def toggle_write_log_hook(worker: RolloutWorker) -> None:
        """Add write log callback hook if not already setup, remove otherwise.
        """
        callbacks = worker.callbacks
        if isinstance(callbacks.on_episode_end, WriteLogHook):
            callbacks.on_episode_end = callbacks.on_episode_end.__func__
        else:
            callbacks.on_episode_end = WriteLogHook(callbacks.on_episode_end)

    # Collect samples.
    # `sample` either produces 1 episode or exactly `evaluation_duration` based
    # on `unit` being set to "episodes" or "timesteps" respectively.
    # See https://github.com/ray-project/ray/blob/98b267f390290f2b2e839a9f1f762cf8c67d1a4a/rllib/algorithms/algorithm.py#L937  # noqa: E501  # pylint: disable=line-too-long
    all_batches = []
    all_log_files, all_num_steps, all_total_rewards = [], [], []
    eval_workers.foreach_worker(toggle_write_log_hook)
    for _ in range(num_episodes):
        # Run a complete episode
        batch = local_worker.sample().as_multi_agent()[DEFAULT_POLICY_ID]
        num_steps = batch.env_steps()
        total_reward = np.sum(batch[batch.REWARDS])
        (log_file,) = local_worker.callbacks.log_files

        # Store all batches for later use
        if raw_data or algo.reward_estimators:
            all_batches.append(batch)

        # Keep track of basic info
        all_num_steps.append(num_steps)
        all_total_rewards.append(total_reward)
        all_log_files.append(log_file)
    eval_workers.foreach_worker(toggle_write_log_hook)

    # Backup only the log file corresponding of the best trial
    log_labels = ("best", "worst")[:num_episodes]
    log_paths = []
    for suffix, idx in zip(log_labels, (
            np.argmax(all_total_rewards), np.argmin(all_total_rewards))):
        log_path = f"{algo.logdir}/iter_{algo.iteration}-{suffix}.hdf5"
        shutil.move(all_log_files[idx], log_path)
        log_paths.append(log_path)

    # Compute high-level performance metrics
    metrics = collect_metrics(
        eval_workers,
        keep_custom_metrics=eval_cfg["keep_per_episode_custom_metrics"],
        timeout_seconds=eval_cfg["metrics_episode_collection_timeout_s"])
    metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = sum(all_num_steps)

    # Ascii histogram if requested
    if print_stats:
        if num_episodes >= 10:
            try:
                plt.clear_figure()
                plt.subplots(1, 2)
                plt.theme('pro')  # 'clear' for terminal-like black and white
                for i, (data, title) in enumerate((
                        (env_dt * np.array(all_num_steps), "Episode duration"),
                        (all_total_rewards, "Total reward"))):
                    plt.subplot(1, i + 1)
                    plt.hist(data, HISTOGRAM_BINS)
                    plt.plotsize(50, 20)
                    plt.title(title)
                plt.show()
            except IndexError as e:
                logger.warning("Ascii rendering failure for statistics: %s", e)
        else:
            logger.warning(
                "'evaluation_duration' must be at least than 10 to compute "
                "and print meaningful statistics.")

    # Replay and/or record a video of the best trial if requested.
    # The viewer must be closed after replay if recording is requested,
    # otherwise the graphical window will dramatically slowdown rendering.
    viewer_kwargs, *_ = chain(
        *eval_workers.foreach_env(attrgetter("viewer_kwargs")))
    viewer_kwargs = {**dict(
        viewers=[],
        delete_robot_on_close=True),
        **viewer_kwargs, **dict(
        scene_name=f"iter_{algo.iteration}",
        robots_colors=('green', 'red') if len(log_labels) == 2 else None,
        close_backend=enable_replay and record_video),
        **kwargs, **dict(
        legend=log_labels if len(log_labels) == 2 else None)}
    record_video = record_video or "record_video_path" in viewer_kwargs
    if record_video:
        video_path = viewer_kwargs.pop(
            "record_video_path", f"{algo.logdir}/iter_{algo.iteration}.mp4")
    if enable_replay is None:
        enable_replay = not record_video
    if enable_replay:
        viewer_kwargs["viewers"] = play_logs_files(log_paths, **viewer_kwargs)
    if record_video:
        viewer_kwargs["viewers"] = play_logs_files(
            log_paths, record_video_path=video_path, **viewer_kwargs)
    for viewer in viewer_kwargs["viewers"]:
        viewer.close()

    # Return collected data without computing metrics if requested
    if raw_data:
        return concat_samples(all_batches)

    # Compute off-policy estimates
    estimates = defaultdict(list)
    for name, estimator in algo.reward_estimators.items():
        for batch in all_batches:
            estimate_result = estimator.estimate(
                batch,
                split_batch_by_episode=algo.config[
                    "ope_split_batch_by_episode"])
            estimates[name].append(estimate_result)

    # Accumulate estimates from all batches
    if estimates:
        metrics["off_policy_estimator"] = {}
        for name, estimate_list in estimates.items():
            avg_estimate = tree.map_structure(
                lambda *x: np.mean(x, axis=0), *estimate_list)
            metrics["off_policy_estimator"][name] = avg_estimate

    return metrics


__all__ = [
    "initialize",
    "train",
    "evaluate"
]
