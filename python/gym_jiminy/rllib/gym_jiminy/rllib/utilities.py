"""This module provides helper methods to make it easier to run learning
pipelines using Ray-RLlib framework.

The main focus are:
  * making it easy to build a policy from a checkpoint,
  * enabling evaluating the performance of a given policy without having to
    fire up a dedicated ray server instance, which is normally required.
"""
import os
import re
import json
import shutil
import socket
import ctypes
import pickle
import logging
import inspect
import tracemalloc
from pathlib import Path
from contextlib import redirect_stdout
from functools import partial
from itertools import chain
from datetime import datetime
from collections import defaultdict
from tempfile import mkdtemp
from traceback import TracebackException
from typing import (
    Optional, Any, Union, Sequence, Tuple, List, Literal, Dict, Set, Callable,
    DefaultDict, Collection, Iterable, SupportsFloat, overload, cast)

import tree
import numpy as np
import gymnasium as gym
import plotext as plt

import ray
from ray._private import services
from ray._private.internal_api import get_state_from_address
from ray._private.test_utils import monitor_memory_usage
from ray.exceptions import RayTaskError
from ray.tune.logger import NoopLogger
from ray.tune.result import TRAINING_ITERATION, TIME_TOTAL_S
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import SafeFallbackEncoder
from ray.rllib.core import (
    DEFAULT_MODULE_ID, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER,
    COMPONENT_RL_MODULE)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.connectors.module_to_env import (
    GetActions, ListifyDataForVectorEnv, ModuleToEnvPipeline,
    RemoveSingleTsTimeRankFromBatch, TensorToNumpy,
    UnBatchToIndividualItems)
from ray.rllib.connectors.env_to_module import (
    AddObservationsFromEpisodesToBatch, AddStatesFromEpisodesToBatch,
    BatchIndividualItems, EnvToModulePipeline, NumpyToTensor,
    MeanStdFilter as MeanStdFilterConnector)
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.filter import MeanStdFilter as _MeanStdFilter, RunningStat
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED_LIFETIME, NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES_LIFETIME, EPISODE_RETURN_MEAN, EPISODE_RETURN_MAX,
    EPISODE_LEN_MEAN, EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_EPISODES)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import (
    AgentID, EpisodeID, ResultDict, EpisodeType, StateDict)

from jiminy_py.viewer import async_play_and_record_logs_files
from gym_jiminy.common.bases import Obs, Act
from gym_jiminy.common.envs import PolicyCallbackFun, BaseJiminyEnv


HISTOGRAM_BINS = 15

PRINT_RESULT_FIELDS_FILTER = (
    TRAINING_ITERATION,
    TIME_TOTAL_S,
    "/".join((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME)),
    "/".join((ENV_RUNNER_RESULTS, NUM_EPISODES_LIFETIME)),
    "/".join((ENV_RUNNER_RESULTS, EPISODE_RETURN_MAX)),
    "/".join((ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN)),
    "/".join((ENV_RUNNER_RESULTS, EPISODE_LEN_MEAN)),
)

VALID_SUMMARY_TYPES = (int, float, np.float32, np.float64, np.int32, np.int64)

LOGGER = logging.getLogger(__name__)


class MeanStdFilter(MeanStdFilterConnector):
    """A connector used to mean-std-filter observations.

    This class patches `ray.rllib.connectors.env_to_module.MeanStdFilter` to
    fix statistics accumulation, which is currently broken for ray<=2.40.
    See PR for details: https://github.com/ray-project/ray/pull/49718
    """
    def merge_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Initialize filters if not already done, clear running stats otherwise
        if self._filters is None:
            self._init_new_filters()
        else:
            for _filter in self._filters.values():
                _filter.running_stats = tree.map_structure(
                    RunningStat, _filter.shape)
        assert self._filters is not None

        # Make sure data is uniform across given states.
        ref = next(iter(states[0].values()))

        for state in states:
            for agent_id, agent_state in state.items():
                _filter = _MeanStdFilter(
                    ref["shape"],
                    demean=ref["de_mean_to_zero"],
                    destd=ref["de_std_to_one"],
                    clip=ref["clip_by_value"],
                )
                # Override running stats of the filter with the ones stored in
                # `agent_state`.
                _filter.buffer = tree.unflatten_as(
                    agent_state["shape"],
                    [
                        RunningStat.from_state(stats)
                        # ------------------------ FIX ------------------------
                        for stats in agent_state["buffer"]
                        # -----------------------------------------------------
                    ],
                )

                # Leave the buffers as-is, since they should always only
                # reflect what has happened on the particular env runner.
                self._filters[agent_id].apply_changes(
                    _filter, with_buffer=False)

        # Calling base `_get_state_from_filters` to avoid syncing `buffer` attr
        return MeanStdFilterConnector._get_state_from_filters(
            self._filters)  # type: ignore[arg-type]

    @staticmethod
    def _get_state_from_filters(
            filters: Dict[AgentID, _MeanStdFilter]  # type: ignore[override]
            ) -> Dict[AgentID, Dict[str, Any]]:
        ret = MeanStdFilterConnector._get_state_from_filters(
            filters)  # type: ignore[arg-type]
        # -------------------------------- FIX --------------------------------
        for agent_id, agent_filter in filters.items():
            if "buffer" not in ret[agent_id]:
                ret[agent_id]["buffer"] = [
                    s.to_state() for s in tree.flatten(agent_filter.buffer)]
        # ---------------------------------------------------------------------
        return ret


class MonitorEpisodeCallback(DefaultCallbacks):
    """Extend monitoring of training batches.

    This method extends monitoring in several ways:
      * Log raw dictionary of extra information returned by the environment.
      * Log reduced statistics (mean, max, min) about the episode length
        (#timesteps) and return (undiscounted cumulative reward).
        Note that it is already done natively, but the original implementation
        is buggy. More specially, the same samples are repeated multiple time.
      * Log a histogram of the episode durations (seconds).
    """
    def __init__(self) -> None:
        # Unique ID of the ongoing episode for each environments being
        # managed by the runner associated with this callback instance.
        self._ongoing_episodes: Dict[int, EpisodeID] = {}

        # Episodes that were started by never reached termination before the
        # end of the previous sampling iteration.
        self._partial_episodes: DefaultDict[
            EpisodeID, List[EpisodeType]] = defaultdict(list)

        # Keep track of all manually registered logger metrics
        self._clear_metrics_keys: Set[str] = set()

        # Whether to clear all manually registered logger metrics at the end of
        # the next episode.
        self._must_clear_metrics = False

    def on_episode_start(self,
                         *,
                         episode: EpisodeType,
                         env_runner: EnvRunner,
                         metrics_logger: MetricsLogger,
                         env: gym.Env,
                         env_index: int,
                         rl_module: RLModule,
                         **kwargs: Any) -> None:
        # Drop all partial episodes associated with the environment at hand
        # when starting a fresh new one since it will never be done anyway.
        if env_index in self._ongoing_episodes:
            episode_id_prev = self._ongoing_episodes[env_index]
            self._partial_episodes.pop(episode_id_prev, None)
        self._ongoing_episodes[env_index] = episode.id_

    def on_episode_end(self,
                       *,
                       episode: EpisodeType,
                       env_runner: EnvRunner,
                       metrics_logger: MetricsLogger,
                       env: gym.vector.VectorEnv,
                       env_index: int,
                       rl_module: RLModule,
                       **kwargs: Any) -> None:
        # Force clearing all custom metrics if necessary.
        # If not cleared manually, the internal buffer of all metrics that are
        # not already cleared automatically when reduced would keep filling-up
        # and contain the whole history of data, not just the ones collected at
        # the current sampling iteration.
        # This is problematic, because at the end of each simple iteration, the
        # internal buffer of these metrics will get merged with the global
        # logger at algorithm-level, which already contain the whole history
        # except the current sampling iteration. As a result, the history will
        # be stored twice, recursively.
        # It is not possible to store "lifetime" metrics at runner-level
        # because of this design flaw. From there, the only option that makes
        # sense is clearing all metrics at the beginning of every sampling
        # iteration, so that the global logger never ends up corrupted.
        if self._must_clear_metrics:
            for key in self._clear_metrics_keys:
                metrics_logger.stats.pop(key, None)
            self._must_clear_metrics = False

        # Get all the chunks associated with the episode at hand
        episodes = (*self._partial_episodes.pop(episode.id_, []), episode)

        # Get window size.
        # Note that `log_value` already forces `clear_on_reduce=True` based on
        # window size to avoid memory overflow.
        window = env_runner.config.metrics_num_episodes_for_smoothing
        clear_on_reduce = window in (None, float("inf"))

        # Log raw infos without any scalar reduction (mean, max, min...)
        for info in episode.get_infos():
            metrics_logger.log_dict(info, reduce=None, clear_on_reduce=True)

        # Log the duration of all episodes without any scalar reduction.
        # Note that relying on `num_steps` is not possible as the environment
        # has already been reset at this point.
        assert isinstance(episode, SingleAgentEpisode)
        num_steps = episode.t
        step_dt = env.unwrapped.get_attr('step_dt')[env_index]

        episode_duration = step_dt * num_steps
        metrics_logger.log_value("episode_duration",
                                 episode_duration,
                                 reduce=None,
                                 clear_on_reduce=True)

        # Log reduced (min, max, mean) cumulative reward and episode length.
        # Note that the window size must be set to `float("inf")` instead of
        # `None` to avoid triggering Exponential Moving Average (EMA).
        episode_return = sum(episode.get_return() for episode in episodes)
        # Beware `episode.env_steps()` does NOT correspond to the absolute
        # "index" of the last step of the episode before termination, but
        # rather the total number of steps in this chunk.
        for reduce in ("min", "max", "mean"):
            for field, value in (
                    (f"episode_return_{reduce}", episode_return),
                    (f"episode_len_{reduce}", num_steps),
                    (f"episode_duration_{reduce}", episode_duration)):
                metrics_logger.log_value(field,
                                         value,
                                         reduce=reduce,
                                         window=window,
                                         clear_on_reduce=clear_on_reduce)
                if not clear_on_reduce:
                    self._clear_metrics_keys.add(field)

    def on_sample_end(self,
                      *,
                      env_runner: EnvRunner,
                      metrics_logger: MetricsLogger,
                      samples: List[EpisodeType],
                      **kwargs: Any) -> None:
        # Store all the partial episodes that did not reached done yet.
        # Note that they are already "finalized" at this point.
        for episode in samples:
            if episode.is_done:
                continue
            self._partial_episodes[episode.id_].append(episode)

        # Last change to clear all custom metrics if no episode has been
        # collected with this runner during this sampling iteration.
        if self._must_clear_metrics:
            for key in self._clear_metrics_keys:
                metrics_logger.stats.pop(key, None)

        # Clear all metrics after sampling.
        # Note that this action must be delayed until the next training
        # iteration, as this method is called before merging metrics.
        self._must_clear_metrics = True


def initialize(num_cpus: int,
               num_gpus: int,
               log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               launch_tensorboard: bool = True,
               env_vars: Optional[Dict[str, Any]] = None,
               debug: bool = False,
               verbose: bool = True,
               **ray_init_kwargs: Any) -> str:
    """Initialize Ray and Tensorboard daemons.

    It will be used later for almost everything from dashboard, remote/client
    management, to multithreaded environment.

    .. note:
        The default Tensorboard port will be used, namely 6006 if available,
        using 0.0.0.0 (binding to all IPv4 addresses on local machine).
        Similarly, Ray dashboard port is 8265 if available. In both cases, the
        port will be increased interactively until to find one available.

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
    :param launch_tensorboard: Whether to launch tensorboard automatically.
                               Optional: Enabled by default.
    :param env_vars: Environment variables at cluster-level as a dictionary.
                     Optional: `None` by default.
    :param debug: Whether to display debugging trace.
                  Optional: Disabled by default.
    :param verbose: Whether to print information about what is going on.
                    Optional: True by default.

    :returns: Fully-qualified path of the logging directory.
    """
    # Handling of default log directory
    if log_root_path is None:
        log_root_path = mkdtemp()

    # Check if cluster servers are already running, and if requested resources
    # are available.
    is_cluster_running, ray_address = False, None
    for ray_address in services.find_gcs_addresses():
        # Get available resources
        state = get_state_from_address(ray_address)
        resources = state.available_resources()
        state.disconnect()

        # Check if enough computation resources are available
        is_cluster_running = (
            resources.get("CPU", 0) >= num_cpus and
            resources.get("GPU", 0) >= num_gpus)

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
                # Environment variables
                runtime_env={"env_vars": env_vars or {}},
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
            LOGGER.warning("Tensorboard not available. Cannot start server.")

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
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3],
            re.sub(r'[^A-Za-z0-9_]', "_", socket.gethostname())))
    else:
        assert re.match(r'^[A-Za-z0-9_]+$', log_name), (
            "Log name must be a valid Python identifier.")

    # Create log directory
    log_path = os.path.join(log_root_path, log_name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Experiment logfiles directory: {log_path}")

    return log_path


class MultiCallbacks(DefaultCallbacks, Checkpointable):
    """Wrapper to combine multiple callback classes as one to fit with the
    standard RLlib API.

    .. note::
        Based on `ray.rllib.algorithms.callbacks.make_multi_callbacks`, which
        has been extended to support stateful callbacks and the so-called new
        API.
    """
    def __init__(self,
                 callbacks_list: Tuple[Callable[[], DefaultCallbacks], ...]
                 ) -> None:
        """
        :param callbacks_list: The list of sub-classes of DefaultCallbacks to
                               be baked into the to-be-returned class. All of
                               these sub-classes' implemented methods will be
                               called in the given order.
        """
        self._ctor_kwargs = dict(callbacks_list=callbacks_list)
        self._callbacks_list = tuple(
            callback_class() for callback_class in callbacks_list)

    def on_algorithm_init(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_algorithm_init(**kwargs)

    def on_workers_recreated(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_workers_recreated(**kwargs)

    def on_checkpoint_loaded(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_checkpoint_loaded(**kwargs)

    def on_environment_created(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_environment_created(**kwargs)

    def on_episode_start(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_episode_start(**kwargs)

    def on_episode_step(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_episode_step(**kwargs)

    def on_episode_end(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_episode_end(**kwargs)

    def on_evaluate_start(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_evaluate_start(**kwargs)

    def on_evaluate_end(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_evaluate_end(**kwargs)

    def on_sample_end(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_sample_end(**kwargs)

    def on_train_result(self, **kwargs: Any) -> None:
        for callbacks in self._callbacks_list:
            callbacks.on_train_result(**kwargs)

    def get_state(self,
                  components: Optional[Union[str, Collection[str]]] = None,
                  *,
                  not_components: Optional[Union[str, Collection[str]]] = None,
                  **kwargs: Any) -> StateDict:
        # Sanitize input argument(s)
        if isinstance(components, str):
            components = (components,)
        if isinstance(not_components, str):
            not_components = (not_components,)

        # Aggregate sequentially states of all the wrapped callbacks if any.
        # Note that the wrapper itself is stateless.
        state = {}
        for i, callbacks in enumerate(self._callbacks_list):
            # Skip individual callbacks are not requested
            key = str(i)
            if components is not None and key not in components:
                continue
            if not_components is not None and key in not_components:
                continue

            # Append the state of the individual callback
            if isinstance(callbacks, Checkpointable):
                state[key] = callbacks.get_state()
        return state

    def set_state(self, state: StateDict) -> None:
        for i, callbacks in enumerate(self._callbacks_list):
            key = str(i)
            state_i = state.get(key, None)
            if state_i:
                assert isinstance(callbacks, Checkpointable)
                callbacks.set_state(state_i)

    def get_checkpointable_components(
            self) -> List[Tuple[str, Checkpointable]]:
        return [(str(i), callbacks)
                for i, callbacks in enumerate(self._callbacks_list)
                if isinstance(callbacks, Checkpointable)]

    def get_ctor_args_and_kwargs(self) -> Tuple[Tuple, Dict[str, Any]]:
        return (
            (),  # *args
            self._ctor_kwargs,  # **kwargs
        )


def train(algo_config: AlgorithmConfig,
          logdir: str,
          max_timesteps: int = 0,
          max_iters: int = 0,
          checkpoint_interval: int = 0,
          enable_evaluation_replay: bool = False,
          enable_evaluation_record_video: bool = False,
          verbose: bool = True,
          debug: bool = False) -> Tuple[Dict[str, Any], str]:
    """Train a model on a specific environment using a given algorithm.

    The algorithm is associated with a given reinforcement learning algorithm,
    instantiated for a specific environment and policy model. Thus, it already
    wraps all the required information to actually perform training.

    .. note::
        This function can be aborted using CTRL+C without raising an exception.

    :param algo: Training algorithm.
    :param logdir: Directory where to store the logfiles created by the logger.
    :param max_timesteps: Maximum number of training timesteps. 0 to disable.
                          Optional: Disabled by default.
    :param max_iters: Maximum number of training iterations. 0 to disable.
                      Optional: Disabled by default.
    :param checkpoint_interval: Backup trainer every given number of training
                                steps in log folder if requested. 0 to disable.
                                Optional: Disabled by default.
    :param enable_evaluation_replay:
        Whether to replay a video of the best and worst trials after completing
        evaluation.
        Optional: False by default.
    :param enable_evaluation_record_video:
        Whether to record a video of the best and worst trials after completing
        evaluation. This video would be stored in `algo.logdir` with suffix
        `algo.iteration`.
        Optional: False by default.
    :param verbose: Whether to print high-level information after each training
                    iteration.
                    Optional: True by default.

    :returns: tuple (metrics: Dict[str, Any], checkpoint_path: str) where
    `metrics` gathers some highl-level metrics, and `checkpoint_path` is the
    fullpath of algorithm's final state dump, including the trained module.
    """
    # Copy original configuration before customizing it
    algo_config = algo_config.copy()

    # PyTorch is the only ML framework supported by `gym_jiminy.rllib`
    algo_config.framework(
        framework="torch"
    )

    # Force new API stack as the old one is not supported anymore
    algo_config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    # Disable multi-threading by default at worker level
    algo_config.python_environment(
        # Extra python env vars to set for worker processes
        extra_python_environs_for_worker={
            # Disable multi-threaded linear algebra at worker-level
            **dict(
                OMP_NUM_THREADS=1,
                OPENBLAS_NUM_THREADS=1,
                MKL_NUM_THREADS=1,
                VECLIB_MAXIMUM_THREADS=1,
                NUMEXPR_NUM_THREADS=1,
                NUMBA_NUM_THREADS=1,
            ),
            **algo_config.extra_python_environs_for_worker,
        }
    )

    # Help making training more deterministic if a seed is specified
    seed = cast(Optional[int], algo_config.seed)
    if seed is not None:
        algo_config.reporting(
            min_time_s_per_iteration=None
        )

    # Disable default logger but configure logging directory nonetheless
    algo_config.debugging(
        logger_config=dict(
            type=NoopLogger,
            logdir=logdir
        )
    )

    # Keep custom metrics as-is, without reducing them as (max, min, mean)
    algo_config.reporting(
        keep_per_episode_custom_metrics=True
    )

    # Enable monitoring callbacks
    if algo_config.callbacks_class is DefaultCallbacks:
        algo_config.callbacks(MonitorEpisodeCallback)
    else:
        algo_config.callbacks(partial(MultiCallbacks, (
            algo_config.callbacks_class, MonitorEpisodeCallback)))

    # Configure evaluation
    algo_config.evaluation(
        custom_evaluation_function=partial(
            evaluate_from_algo,
            print_stats=True,
            enable_replay=enable_evaluation_replay,
            record_video=enable_evaluation_record_video,
            _return_metrics=False,
        ),
    )
    evaluation_config = cast(
        Optional[AlgorithmConfig], algo_config.evaluation_config)
    if evaluation_config is None:
        algo_config.evaluation(
            evaluation_config=dict(
                env=algo_config.env,
                env_config={
                    **algo_config.env_config,
                    **dict(
                        debug=debug
                    ),
                },
                render_env=False,
                callbacks_class=DefaultCallbacks,
                explore=False,
            )
        )
    else:
        evaluation_config.environment(
            env_config={
                **evaluation_config.env_config,
                **dict(
                    debug=debug
                ),
            },
            render_env=False,
        )

    # Initialize the learning algorithm
    algo = algo_config.build()

    # Backup some information
    if algo.iteration == 0:
        # Make sure log dir exists
        os.makedirs(logdir, exist_ok=True)

        # Backup all environment's source files
        assert algo.env_runner_group is not None
        (env_type,) = set(chain(*algo.env_runner_group.foreach_worker(
            lambda worker: [
                type(env.unwrapped) for env in worker.env.unwrapped.envs])))
        while True:
            try:
                path = inspect.getfile(env_type)
                shutil.copy2(path, logdir, follow_symlinks=True)
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
            main_backup_name = f"{logdir}/main.py"
            shutil.copy2(source_file, main_backup_name, follow_symlinks=True)

        # Backup RLlib config
        with open(f"{logdir}/params.json", 'w') as file:
            json.dump(algo_config.to_dict(),
                      file,
                      indent=2,
                      sort_keys=True,
                      cls=SafeFallbackEncoder)

    # Instantiate logger that will be used throughout the experiment
    try:
        # pylint: disable=import-outside-toplevel,import-error
        from tensorboardX import SummaryWriter
        file_writer = SummaryWriter(logdir, flush_secs=30)
    except ImportError:
        LOGGER.warning("Tensorboard not available. Cannot start server.")

    # Get the reward threshold of the environment, if any.
    # Note that the original environment is always re-registered as an new gym
    # entry-point "rllib-single-agent-env-v0". Most of the original spec are
    # lost in the process when the environment is a callable that has been
    # registered through `ray.tune.registry.register_env`. The only way to
    # access the original spec is by instantiating the entry-point manually
    # without relying on `gym.make`.
    assert algo.env_runner_group is not None
    env_spec, *_ = chain(*algo.env_runner_group.foreach_worker(
        lambda worker: worker.env.unwrapped.get_attr('spec')))
    assert env_spec is not None
    if env_spec.reward_threshold is None:
        env_spec = env_spec.entry_point().spec
    if env_spec is None or env_spec.reward_threshold is None:
        return_threshold = float('inf')
    else:
        return_threshold = env_spec.reward_threshold

    # Set the seed of the training and evaluation environments
    if seed is not None:
        seed_seq_gen = np.random.SeedSequence(seed)
        seed_seq = seed_seq_gen.generate_state(2)
        for seed_seq_gen_i, workers in zip(
                tuple(map(np.random.SeedSequence, seed_seq)),
                (algo.env_runner_group, algo.eval_env_runner_group)):
            if workers is None:
                continue
            num_envs = workers.foreach_worker(lambda worker: worker.num_envs)
            seeds = [seed_seq_gen_i.generate_state(n).tolist()
                     for n in num_envs]
            workers.foreach_worker_with_id(
                lambda idx, worker: worker.env.reset(
                    seed=seeds[idx]))  # pylint: disable=cell-var-from-loop

    # Restore checkpoint if any
    checkpoints_paths = sorted([
        str(path) for path in Path(logdir).iterdir()
        if path.is_dir() and path.name.startswith("checkpoint_")])
    if checkpoints_paths:
        checkpoint_dir = checkpoints_paths[-1]
        algo.restore(checkpoint_dir)
        if isinstance(algo.callbacks, Checkpointable):
            algo.callbacks.restore_from_path(
                os.path.join(checkpoint_dir, "callbacks"))
            state_callbacks = algo.callbacks.get_state()
            algo.env_runner_group.foreach_worker(
                lambda worker: worker._callbacks.set_state(state_callbacks))

    # Synchronize connectors of training and evaluation remote workers with the
    # local training runner. This is necessary if a checkpoint has just been
    # restored, otherwise that is a no-op that does no harm.
    def sync_connectors(state_connectors: Dict[str, Any],
                        env_runner: EnvRunner) -> None:
        """Internal helper to synchronise all the env-to-module
        connectors of a given runner with a given state.

        :param state_connectors: Expected state of the connectors
                                    after synchronization.
        :param env_runner: Environment runner to consider.
        """
        assert isinstance(env_runner, SingleAgentEnvRunner)
        env_runner._env_to_module.set_state(state_connectors)

    state_connectors = algo.env_runner._env_to_module.get_state()
    algo.env_runner_group.foreach_worker(partial(
        sync_connectors, state_connectors))
    if algo.eval_env_runner_group is not None:
        algo.eval_env_runner_group.foreach_worker(partial(
            sync_connectors, state_connectors))

    # Disable connector update for evaluation runner
    def disable_update_connectors(env_runner: EnvRunner) -> None:
        """Internal helper to disable automatic update of statistics (mean,
        std) when collecting samples used by MeanStdFilter to empirically
        normalized the observation.

        :param env_runner: Environment runner to consider.
        """
        assert isinstance(env_runner, SingleAgentEnvRunner)
        for connector in env_runner._env_to_module:
            if isinstance(connector, MeanStdFilter):
                connector._update_stats = False
                break

    if algo.eval_env_runner_group is not None:
        algo.eval_env_runner_group.foreach_worker(disable_update_connectors)

    # Monitor memory allocations to detect leaks if any
    if debug:
        tracemalloc.start(10)
        snapshot = tracemalloc.take_snapshot()

    # Run several training iterations until terminal condition is reached
    try:
        iter_num = 0
        while True:
            # Perform one iteration of training the policy
            result = algo.train()

            # Synchronize evaluation connectors with training connectors
            if algo.eval_env_runner_group is not None:
                state_connectors = algo.env_runner._env_to_module.get_state()
                algo.eval_env_runner_group.foreach_worker(partial(
                    sync_connectors, state_connectors))

            # Log results
            num_timesteps = result[NUM_ENV_STEPS_SAMPLED_LIFETIME]
            if file_writer is not None:
                # Flatten result dict after excluding irrelevant special keys
                masked_fields = (
                    "config", "pid", "date", "hostname", "node_ip", "trial_id",
                    "timestamp", TIME_TOTAL_S, TRAINING_ITERATION)
                result_filtered = result.copy()
                for k in masked_fields:
                    result_filtered.pop(k, None)
                result_flat = flatten_dict(result_filtered, delimiter="/")

                # Keep track of the tag of all the scalar fields
                scalar_tags: List[str] = []

                # Logger variables in accordance with their respective types
                for attr, value in result_flat.items():
                    # First, try to log the variable as a scalar
                    try:
                        file_writer.add_scalar(
                            attr, value, global_step=num_timesteps)
                        scalar_tags.append(attr)
                        continue
                    except (TypeError, AssertionError, NotImplementedError):
                        pass

                    if isinstance(value, np.ndarray):
                        # Assuming single image
                        if value.ndim == 3:
                            file_writer.add_image(
                                attr, value, global_step=num_timesteps)
                            continue

                        # Assuming batch of images
                        if value.ndim == 4:
                            file_writer.add_images(
                                attr, value, global_step=num_timesteps)
                            continue

                        # Assuming video with arbitrary FPS
                        if value.ndim == 5:
                            file_writer.add_video(
                                attr, value, fps=20, global_step=num_timesteps)
                            continue

                    # In last resort, try to log the variable as an histogram
                    if isinstance(value, list):
                        if not value:
                            continue
                        try:
                            file_writer.add_histogram(
                                attr, value, global_step=num_timesteps)
                            continue
                        except (ValueError, TypeError):
                            pass

                    # Warn and move on if it was impossible to log the variable
                    LOGGER.warning(
                        "You are trying to log an invalid value (%s=%s).",
                        attr, value)

                # Add multiline charts for dictionary of named scalar metrics
                # stored under key `ENV_RUNNER_RESULTS`.
                custom_layout_data: Dict[str, Tuple[str, List[str]]] = {}
                metrics_nested_list: List[Tuple[str, Dict[str, Any]]] = [(
                    "/".join(("ray", "tune", "env_runners")),
                    result[ENV_RUNNER_RESULTS])]
                while metrics_nested_list:
                    # Pop out the first element in queue
                    title, metrics_nested = metrics_nested_list.pop(0)

                    # Check if it corresponds to a dict of named scalar metrics
                    tags = ["/".join(map(str, (title, name)))
                            for name in metrics_nested.keys()]
                    for tag, data in zip(tags, metrics_nested.values()):
                        if tag not in scalar_tags:
                            break
                    else:
                        # If so, add multiline chart with all named scalars
                        custom_layout_data[title] = ('Multiline', tags)
                        continue

                    # If not, add candidate values to the queue
                    for tag, data in zip(tags, metrics_nested.values()):
                        if isinstance(data, dict) and len(data) > 1:
                            metrics_nested_list.append((tag, data))

                # Create all multiline chart at once due to API limitations
                file_writer.add_custom_scalars({"default": custom_layout_data})

                # Flush the log file
                file_writer.flush()

            # Monitor memory allocations since the beginning and between iters
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
            if verbose or debug:
                msg_data = []
                result_flat = flatten_dict(result, delimiter="/")
                for field in PRINT_RESULT_FIELDS_FILTER:
                    if field in result_flat.keys():
                        msg_data.append(
                            f"{field.rsplit('/', 1)[-1]}: "
                            f"{result_flat[field]:.5g}")
                print(" - ".join(msg_data))

            # Backup the policy
            iter_num = result[TRAINING_ITERATION]
            if checkpoint_interval > 0 and iter_num % checkpoint_interval == 0:
                checkpoint_dir = os.path.join(
                    logdir, f"checkpoint_{iter_num:06d}")
                algo.save(checkpoint_dir)
                if isinstance(algo.callbacks, Checkpointable):
                    algo.callbacks.save_to_path(
                        os.path.join(checkpoint_dir, "callbacks"))

            # Check terminal conditions
            num_timesteps = result[NUM_ENV_STEPS_SAMPLED_LIFETIME]
            if 0 < max_timesteps < num_timesteps:
                if verbose or debug:
                    print("Reached maximum number of environment steps.")
                break
            if 0 < max_iters < iter_num:
                if verbose or debug:
                    print("Reached maximum number of iterations.")
                break
            if result[ENV_RUNNER_RESULTS][NUM_EPISODES] > 0:
                return_mean = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
                if return_threshold < return_mean:
                    if verbose or debug:
                        print("Problem solved successfully!")
                    break
    except KeyboardInterrupt:
        if verbose or debug:
            print("Interrupting training...")
    except RayTaskError as e:
        traceback = TracebackException.from_exception(e)
        LOGGER.warning(''.join(traceback.format()))

    # Flush and close logger before returning
    if file_writer is not None:
        file_writer.flush()
        file_writer.close()

    # Backup trained agent and return file location
    checkpoint_result = algo.save(
        os.path.join(logdir, f"checkpoint_{iter_num:06d}"))
    assert checkpoint_result.metrics is not None
    assert checkpoint_result.checkpoint is not None

    # Stop the algorithm
    algo.stop()

    return (checkpoint_result.metrics, checkpoint_result.checkpoint.path)


def partition(length: int, num_chunks: int) -> Iterable[int]:
    """Generate partition with chunks of minimal variation in length.

    :param length: Total number of elements to partition.
    :param num_chunks: Size of each chunks.
    """
    num_chunks = min(length, num_chunks)
    chunk_length_min, stepdown = divmod(length, num_chunks)
    return (chunk_length_min + (i < stepdown) for i in range(num_chunks))


def sample_from_runner(
        env_runner: EnvRunner,
        num_episodes: int
        ) -> Tuple[Sequence[ResultDict], Sequence[EpisodeType], Sequence[str]]:
    """Sample multiple evaluation episodes on any of the environments being
    managed by a given evaluation runner.

    In practice, it stops any ongoing episodes, backup the current mode of each
    environment (training vs. evaluation) before switching to evaluation mode,
    collect the desired number of episodes, stops ongoing episodes if any and
    delete their corresponding log files are now useless, then restore the
    orginal mode of each environment.

    .. warning::
        This method only supports environments deriving from `BaseJiminyEnv`.

    :param env_runner: Environment runner to use for collecting samples.
    :param num_episodes: Minimum number of episodes to collect. There may be
                         more, if several episodes terminated simultaneously.

    :returns: tuple gathering logged high-level metrics, raw episode data, and
    simulation log paths of all the episodes that were collected.
    """
    # Make sure that the input argument(s) are valid
    assert num_episodes > 0

    # Assert(s) for type checker
    assert isinstance(env_runner, SingleAgentEnvRunner)

    # FIXME: When there is a single thread, the same environments are used for
    # sampling during training and evaluation. One is expecting to sample
    # complete episode from reset to termination during evaluation, which means
    # that it is necessary to force reset the environment before sample.
    # Although all the environments were initiallly reset systematically before
    # sampling a given number of episodes, it is no longer the case after
    # applying our on patch that specifically revert this behavior. This patch
    # was necessary as it causes episodes from which the agent is performing
    # poorly to be over-represented in training batches.
    # Passing `force_reset=True` as input argument of `sample` is not suffisant
    # in practice, as this input argument is ignored when specifying the number
    # of episodes. The only option is to override the internal flag
    # `_needs_initial_reset` used internally to automatically trigger reset.
    env_runner._needs_initial_reset = True

    # Stop any simulation that may be running
    env = env_runner.env.unwrapped
    try:
        env.call('stop')
    except AttributeError as e:
        raise RuntimeError(
            "Evaluation env must inherit from `BaseJiminyEnv`.") from e

    # Backup the current mode of the environment then switch to evaluation
    is_training_all = env.get_attr('training')
    env.call('eval')

    # Collect a bunch of episodes
    episodes = env_runner.sample(
        num_episodes=num_episodes, random_actions=False)

    # Extract the log files of interest.
    log_paths = tuple(episode.infos[0]['log_path'] for episode in episodes)

    # Collect metrics
    metrics = env_runner.get_metrics()

    # Remove log paths from metrics to prevent irrelevant tensorboard logging.
    # Note that 'log_path' is not stored as metrics by default with official
    # RLlib algo config. Nevertheless, `gym_jiminy.rllib.utilities.train`
    # forces all infos to be stored as metrics for the evaluation runners via
    # `MonitorEpisodeCallback`, which includes 'log_path'.
    metrics.pop('log_path', None)

    # Enable once-again auto-reset, to avoid starting back training from where
    # evaluation stopped.
    env_runner._needs_initial_reset = True

    # Stop any simulation that may be running.
    # This is necessary to make sure that all log files have been written.
    env.call('stop')

    # Delete useless log files to avoid filling up the hard drive.
    # Note that the log path would be `None` for environments has been reset
    # right before ending sampling and terminating the simulation. This is
    # expected as internally, try to step a vector environment will only call
    # `reset` without performing a single step if the previous episode has just
    # terminated.
    for log_path in env.get_attr('log_path'):
        if log_path is not None and log_path not in log_paths:
            os.remove(log_path)

    # Restore the original training/evaluation mode.
    # FIXME: `set_attr` is buggy on`gymnasium<=1.0` and cannot be used
    # reliability in conjunction with `BasePipelineWrapper`.
    # See PR: https://github.com/Farama-Foundation/Gymnasium/pull/1294
    # env.set_attr('training', is_training_all)
    assert isinstance(env, gym.vector.SyncVectorEnv)
    for env, is_training in zip(env.envs, is_training_all):
        env.get_wrapper_attr("train")(is_training)

    return (metrics,), episodes, log_paths


def sample_from_runner_group(
        workers: EnvRunnerGroup,
        num_episodes: int,
        evaluation_sample_timeout_s: Optional[float] = None
        ) -> Tuple[Sequence[ResultDict], Sequence[EpisodeType], Sequence[str]]:
    """Sample multiple evaluation episodes on any of the environments being
    managed by all the remote and local evaluation runners of a given group.

    Sampling would be distributed across all remote runners if any, otherwise
    it will be sequential on the local runner.

    .. warning::
        This method only supports environments deriving from `BaseJiminyEnv`.

    :param workers: Group of environment runners to use for collecting samples.
    :param num_episodes: Minimum number of episodes to collect. There may be
                         more, if several episodes terminated simultaneously.
    :param evaluation_sample_timeout_s:
        Timeout in seconds for evaluation runners to sample a complete episode
        of remote evluation. After this time, the user receives a warning and
        instructions on how to fix the issue.

    :returns: tuple gathering logged high-level metrics, raw episode data, and
    simulation log paths for all the episode that were collected.
    """
    # This method only supports environments deriving from `BaseJiminyEnv`
    (env_type,) = set(chain(*workers.foreach_worker(
        lambda worker: [
            type(env.unwrapped) for env in worker.env.unwrapped.envs])))
    if not issubclass(env_type, BaseJiminyEnv):
        raise RuntimeError("Test env must inherit from `BaseJiminyEnv`.")

    if workers.num_remote_env_runners() == 0:
        all_metrics, all_episodes, all_log_paths = (
            sample_from_runner(workers.local_env_runner, num_episodes))
    elif (num_workers := workers.num_healthy_remote_workers()) > 0:
        # Split workload across workers as fairly as possibly
        worker_ids = workers.healthy_worker_ids()
        chunks = dict(zip(worker_ids, partition(num_episodes, num_workers)))

        # Run a complete episode per selected worker and collect metrics
        results = workers.foreach_worker_with_id(
            func=lambda ident, worker: sample_from_runner(
                worker, num_episodes=chunks[ident]),
            local_env_runner=False,
            remote_worker_ids=worker_ids,
            timeout_seconds=evaluation_sample_timeout_s)
        all_metrics, all_episodes, all_log_paths = [], [], []
        for worker_ids, (metrics, episodes, log_paths) in zip(
                worker_ids, results):
            if len(episodes) < chunks[worker_ids]:
                LOGGER.warning(
                    "Calling `sample()` on your remote evaluation worker(s) "
                    "resulted in a timeout. Please configure the parameter "
                    "`evaluation_sample_timeout_s` accordingly.")
            all_metrics += metrics
            all_episodes += episodes
            all_log_paths += log_paths
    else:
        raise RuntimeError("No runner available for running the evaluation.")

    return all_metrics, all_episodes, all_log_paths


def _pretty_print_statistics(data: Sequence[Tuple[str, np.ndarray]]) -> None:
    """Render histograms directly from within the terminal without involving
    any graphical server. These figures can be saves in text files and
    copy-pasted easily. Internally, it is simply calling on `plotext.hist`.

    .. note::
        Fallback to basic print if the terminal (stdout) does not support
        unicode or falls to render the graph for any reason.

    :param data: Sequence of pairs containing first the label and second
                 all the samples available as a 1D array.
    """
    try:
        plt.clear_figure()
        fig = plt.subplots(1, len(data))
        plt.theme('clear')  # 'pro' to support color lines
        for i, (title, values) in enumerate(data):
            ax = fig.subplot(1, i + 1)
            ax.hist(values, HISTOGRAM_BINS, marker="sd")
            ax.plotsize(50, 20)
            ax.title(title)
        plt.show()
    except (IndexError, UnicodeEncodeError) as e:
        LOGGER.warning("'plotext' figure rendering failure: %s", e)
        for i, (title, values) in enumerate(data):
            print(
                f"* {title}: {np.mean(values):.2f} +/- {np.std(values):.2f} "
                f"(min = {np.min(values):.2f}, max = {np.max(values):.2f})")


def _pretty_print_episode_metrics(all_episodes: Sequence[EpisodeType],
                                  step_dt: Optional[float]) -> None:
    """Render ASCII histograms horizontally side-by-side of high-level
    performance metrics about the batch of episodes.

    In practice, one histogram on the left corresponds to the episode duration
    (seconds) or length (#timesteps) on the left depending on whether the
    timestep unit has been specified, while the right one corresponds to the
    return (undiscounted cumulative reward).

    :param all_episodes: Batch of episodes from which to extract high-level
                         metrics. Note that the number of episodes must be
                         larger than 10.
    :param step_dt: Unique timestep of the environment for the whole batch of
                    episodes if applicable, `None` otherwise.
    """
    # Keep track of basic info
    all_num_env_steps = np.array([
        episode.env_steps() for episode in all_episodes])
    all_returns = np.array([
        episode.get_return() for episode in all_episodes])

    # Early return if not enough data is available
    if len(all_episodes) < 10:
        LOGGER.warning(
            "The number of episodes must be larger than 10 for printing "
            "meaningful statistics.")
        return

    # Gather meaningful statistics
    episode_stats = []
    if step_dt is not None:
        # Print the episode duration if the environment timestep is identical
        # for all instances, which is extremely likely but not mandorary.
        all_env_durations = step_dt * all_num_env_steps
        episode_stats.append(("Episode duration", all_env_durations))
    else:
        # Print the episode length as a fallback
        episode_stats.append(("Episode length", all_num_env_steps))
    episode_stats.append(("Return", all_returns))

    # Print ASCII histograms as horizontal subplots
    _pretty_print_statistics(episode_stats)


@overload
def evaluate_from_algo(algo: Algorithm,
                       workers: Optional[EnvRunnerGroup] = ...,
                       print_stats: bool = ...,
                       enable_replay: Optional[bool] = ...,
                       record_video: bool = ...,
                       _return_metrics: Literal[True] = ...) -> ResultDict:
    ...


@overload
def evaluate_from_algo(algo: Algorithm,
                       workers: Optional[EnvRunnerGroup] = ...,
                       print_stats: bool = ...,
                       enable_replay: Optional[bool] = ...,
                       record_video: bool = ...,
                       _return_metrics: Literal[False] = ...
                       ) -> Tuple[ResultDict, int, int]:
    ...


def evaluate_from_algo(algo: Algorithm,
                       workers: Optional[EnvRunnerGroup] = None,
                       print_stats: bool = True,
                       enable_replay: Optional[bool] = False,
                       record_video: bool = False,
                       _return_metrics: bool = True
                       ) -> Union[ResultDict, Tuple[ResultDict, int, int]]:
    """Evaluates the current algorithm under `evaluation_config` configuration,
    then returns some performance metrics to monitor the training progress.

    The log files associated with the best and worst trials among the complete
    evaluation batch (respectively the two episodes with highest and lowest
    return) will be stored in `algo.logdir` with suffix `algo.iteration`.

    .. warning::
        This method is specifically tailored for Gym environments inheriting
        from `BaseJiminyEnv`.

    :param workers: Rollout workers for evaluation.
                    Optional: `algo.eval_workers` by default.
    :param print_stats: Whether to print high-level statistics.
                        Optional: True by default.
    :param enable_replay: Whether to replay a video after completing evaluation
                          of the two episodes with highest and lowest return.
                          Optional: False by default.
    :param record_video: Whether to record a video after completing evaluation
                         of the two episodes with highest and lowest return.
                         Optional: False by default.
    :param _return_metrics: Whether to return aggregated metrics. If not, then
                            a tuple `(None, num_env_steps, num_agent_steps)` is
                            returned instead. Must be set to `False` if passed
                            as `custom_evaluation_function` to the algorithm.
    """
    # Handling of default argument(s)
    if workers is None:
        workers = algo.eval_env_runner_group
    assert workers is not None

    # Extract evaluation config
    eval_cfg = algo.evaluation_config
    assert eval_cfg is not None
    if eval_cfg.evaluation_duration_unit == "auto":
        raise ValueError("evaluation_duration_unit='timesteps' not supported.")
    num_episodes: Union[int, str] = eval_cfg.evaluation_duration
    if num_episodes == "auto":
        raise ValueError("evaluation_duration='auto' not supported.")
    assert isinstance(num_episodes, int)

    # Sample episodes
    all_metrics, all_episodes, all_log_paths = sample_from_runner_group(
        workers, num_episodes, eval_cfg.evaluation_sample_timeout_s)

    # Print stats in ASCII histograms if requested
    if print_stats:
        try:
            (step_dt,) = set(chain(*workers.foreach_worker(
                lambda worker: worker.env.unwrapped.get_attr('step_dt'))))
        except ValueError:
            step_dt = None
        _pretty_print_episode_metrics(all_episodes, step_dt)

    # Backup only the log file corresponding to the best and worst trial, while
    # deleting all the others.
    all_returns = [
        episode.get_return() for episode in all_episodes]
    idx_worst, idx_best = np.argsort(all_returns)[[0, -1]]
    log_labels, log_paths = [], []
    for idx, log_path_orig in tuple(enumerate(all_log_paths))[::-1]:
        if idx not in (idx_worst, idx_best):
            os.remove(log_path_orig)
            continue
        ext = Path(log_path_orig).suffix
        label = "best" if idx == idx_best else "worst"
        log_path = f"{algo.logdir}/iter_{algo.iteration}-{label}{ext}"
        shutil.move(log_path_orig, log_path)
        log_paths.append(log_path)
        log_labels.append(label)

    # Replay and/or record a video of the best and worst trials if requested.
    # Async to enable replaying and recording while training keeps going.
    viewer_kwargs, *_ = chain(*workers.foreach_worker(
        lambda worker: worker.env.unwrapped.get_attr('viewer_kwargs')))
    viewer_kwargs.update(
        scene_name=f"iter_{algo.iteration}",
        robots_colors=('green', 'red') if num_episodes > 1 else None,
        legend=log_labels if num_episodes > 1 else None,
        close_backend=True)
    if record_video:
        viewer_kwargs.setdefault(
            "record_video_path", f"{algo.logdir}/iter_{algo.iteration}.mp4")
    async_play_and_record_logs_files(
        log_paths, enable_replay=enable_replay, **viewer_kwargs)

    # Centralized all metrics in algorithm logger
    algo.metrics.merge_and_log_n_dicts(
        list(all_metrics), key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS))

    # Early return if computing reduced metrics is not requested.
    # Note that it expects a non-empty `ResultDict` as first output, even
    # though it is not used anywhere in practice. Using a mock value instead.
    num_env_steps = sum(episode.env_steps() for episode in all_episodes)
    num_agent_steps = sum(episode.agent_steps() for episode in all_episodes)
    if not _return_metrics:
        return {"_": None}, num_env_steps, num_agent_steps

    # Update lifetime statistics in global logger
    algo.metrics.log_dict(
        {
            NUM_ENV_STEPS_SAMPLED_LIFETIME: num_env_steps,
            NUM_AGENT_STEPS_SAMPLED_LIFETIME: num_agent_steps,
            NUM_EPISODES_LIFETIME: algo.metrics.peek(
                (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_EPISODES),
                default=0)
        },
        key=EVALUATION_RESULTS,
        reduce="sum")

    # Compute reduced metrics
    results = algo.metrics.reduce(
        key=EVALUATION_RESULTS, return_stats_obj=False)

    # Compute off-policy estimates
    estimates = defaultdict(list)
    for name, estimator in algo.reward_estimators.items():
        for episode in all_episodes:
            estimate_result = estimator.estimate(
                episode,
                split_batch_by_episode=eval_cfg.ope_split_batch_by_episode)
            estimates[name].append(estimate_result)

    # Accumulate estimates from all episodes
    if estimates:
        results["off_policy_estimator"] = {}
        for name, estimate_list in estimates.items():
            avg_estimate = tree.map_structure(
                lambda *x: np.mean(x, axis=0), *estimate_list)
            results["off_policy_estimator"][name] = avg_estimate

    return results


def evaluate_from_runner(
        env_runner: EnvRunner,
        num_episodes: int = 1,
        print_stats: Optional[bool] = None,
        enable_replay: Optional[bool] = None,
        delete_log_files: bool = True,
        block: bool = True,
        **kwargs: Any
        ) -> Tuple[Sequence[EpisodeType], Optional[Sequence[str]]]:
    """Evaluates the performance of a given local worker.

    This method is specifically tailored for Gym environments inheriting from
    `BaseJiminyEnv`.

    :param env_runner: Local environment runner for evaluation.
    :param num_episodes: Number of episodes to sample. The log files of the
                         best and worst performing trials will be exported.
                         Optional: 1 by default.
    :param print_stats: Whether to print high-level statistics.
                        Optional: `num_episodes >= 10` by default.
    :param enable_replay: Whether to enable replay of the simulation.
                          Optional: True by default if `record_video_path` is
                          not provided and the default/current backend supports
                          it, False otherwise.
    :param delete_log_files: Whether to delete log files instead of returning
                             them. Note that this option is not supported if
                             `enable_replay=True` and `block=False`.
    :param block: Whether calling this method should be blocking.
                  Optional: True by default.
    :param kwargs: Extra keyword arguments to forward to the viewer if any.

    :returns: Sequences of episodes, along with the sequence of corresponding
              log files if `delete_log_files=False`, None otherwise.
    """
    # Assert(s) for type checker
    assert isinstance(env_runner, SingleAgentEnvRunner)

    # Make sure that the input arguments are valid
    if delete_log_files and enable_replay and not block:
        raise ValueError(
            "Specifying `delete_log_files=True` is not available in "
            "conjunction with `enable_replay=True` and `block=True`.")

    # Handling of default argument(s)
    if print_stats is None:
        print_stats = num_episodes >= 10

    # Sample episodes
    all_log_paths: Optional[Sequence[str]]
    _, all_episodes, all_log_paths = (
        sample_from_runner(env_runner, num_episodes))

    # Print stats in ASCII histograms if requested
    env = env_runner.env.unwrapped
    if print_stats:
        try:
            (step_dt,) = set(env.get_attr('step_dt'))
        except ValueError:
            step_dt = None
        _pretty_print_episode_metrics(all_episodes, step_dt)

    # Extract the indices of the best and worst trial
    all_returns = np.array([
        episode.get_return() for episode in all_episodes])
    idx_worst, idx_best = np.argsort(all_returns)[[0, -1]]

    # Replay and/or record a video of the best and worst trials if requested.
    # Async to enable replaying and recording while training keeps going.
    viewer_kwargs, *_ = env.get_attr('viewer_kwargs')
    viewer_kwargs = {
        **viewer_kwargs, **dict(
            robots_colors=('green', 'red') if num_episodes > 1 else None),
        **kwargs, **dict(
            legend=("best", "worst") if num_episodes > 1 else None,
            close_backend=True)}
    thread = async_play_and_record_logs_files(
        list(set(all_log_paths[idx] for idx in (idx_best, idx_worst))),
        enable_replay=enable_replay,
        **viewer_kwargs)
    if block and thread is not None:
        try:
            thread.join()
        except KeyboardInterrupt:
            assert thread.ident is not None
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), ctypes.py_object(SystemExit))

    # Delete log files if requested
    if delete_log_files:
        for log_path in all_log_paths:
            os.remove(log_path)
        all_log_paths = None

    # Return all collected data
    return all_episodes, all_log_paths


def build_runner_from_checkpoint(
        checkpoint_path: str,
        env_config_kwargs: Optional[Dict[str, Any]] = None,
        is_eval_runner: bool = True) -> EnvRunner:
    """Build a local runner from a checkpoint generated by calling
    `algo.save()` during training of the policy.

    This local env runner can then be passed to `evaluate_from_runner` for
    evaluating the performance of a policy without having to initialize Ray
    distributed computing backend.

    .. warning::
        This method is *NOT* standalone in the event where the evaluation
        env has been registered by calling `ray.tune.registry.register_env`.
        In such a case, one must ensure that this registration has been done
        prior to calling this method, otherwise it will raise an exception.

    :param checkpoint_path: Checkpoint directory to be restored.
    :param env_config_kwargs: Keyword-only arguments that will be passed at
                              environment instantiation. This is useful to
                              partially override the original configuration,
                              i.e. to fix absolute path that may have change.
                              Optional: `None` by default.
    :param is_eval_runner: Whether to restore the evaluation runner in place
                           of the training one.
                           Optional: True by default.
    """
    # Instantiate the runner
    env_runner_checkpoint_path = Path(checkpoint_path) / (
        "eval_env_runner" if is_eval_runner else "env_runner")
    class_and_ctor_args_fullpath = (
        env_runner_checkpoint_path / "class_and_ctor_args.pkl")
    with open(class_and_ctor_args_fullpath, "rb") as f:
        ctor_info = pickle.load(f)
    ctor = ctor_info["class"]
    # env_runner = ctor.from_checkpoint(env_runner_checkpoint_path)
    env_runner_class_ctor_checkpoint_path = (
        env_runner_checkpoint_path / ctor.CLASS_AND_CTOR_ARGS_FILE_NAME)
    with open(env_runner_class_ctor_checkpoint_path, "rb") as f:
        ctor_info = pickle.load(f)
    ctor_args, ctor_kwargs = ctor_info["ctor_args_and_kwargs"]
    env_config = ctor_kwargs['config'].env_config
    if env_config_kwargs:
        env_config.update(env_config_kwargs)
    env_runner = ctor(*ctor_args, **ctor_kwargs)

    # Restore the state of the runner
    env_runner.restore_from_path(env_runner_checkpoint_path)

    # Sync the weights from the learner to the runner.
    # Note that it is necessary to load the learner module because weights are
    # not up-to-date at runner-level.
    rl_module = RLModule.from_checkpoint(
        Path(checkpoint_path) / COMPONENT_LEARNER_GROUP /
        COMPONENT_LEARNER / COMPONENT_RL_MODULE / DEFAULT_MODULE_ID)
    env_runner.set_state({COMPONENT_RL_MODULE: rl_module.get_state()})

    return env_runner


def build_module_from_checkpoint(checkpoint_path: str) -> RLModule:
    """Build a single-agent evaluation policy from a checkpoint generated by
    calling `algo.save()` during training of the policy.

    .. warning::
        This method supports single-agent policies, with further restrictions:
          * without custom connectors, i.e.
            `config.module_to_env_connector = None`,
            `config.env_to_module_connector = None`.
          * with default connectors enables, i.e.
            `config.add_default_connectors_to_module_to_env_pipeline = True`,
            `config.add_default_connectors_to_module_to_env_pipeline = True`.
          * without action normalization or clipping at runner-level, i.e.
            `config.normalize_actions = False`,
            `config.clip_actions = False`.

        As an alternative to rllib-specific pre- and post- processors at
        runner-level such as action normalization and clipping, one can
        leverage the environment pipeline design introduced by jiminy, e.g. by
        adding `NormalizeAction` and/or `ClipAction` layers.

    :param checkpoint_path: Checkpoint directory to be restored.
    """
    # Restore a complete runner instead of just the policy, in order to perform
    # checks regarding the pre- and post- processing of the policy.
    env_runner = build_runner_from_checkpoint(checkpoint_path)
    config = env_runner.config

    # Assert(s) for type checker
    assert isinstance(env_runner, SingleAgentEnvRunner)

    # Make sure that the environment is single-agent
    if config.is_multi_agent():
        raise RuntimeError("Multi-agent environments are not supported")

    # Make sure that no custom module from/to env connector has been specified
    if config._module_to_env_connector is not None:
        raise RuntimeError("Custom module to env connectors are not supported")
    if config._env_to_module_connector is not None:
        raise RuntimeError("Custom env to module connectors are not supported")

    # Make sure that default module from/to env connectors are enabled
    if not config.add_default_connectors_to_env_to_module_pipeline:
        raise RuntimeError(
            "Disabling default env to module connectors are not supported")
    if not config.add_default_connectors_to_module_to_env_pipeline:
        raise RuntimeError(
            "Disabling default module to env connectors are not supported")

    # Make sure that action normalization and clipping is disabled
    if config.normalize_actions or config.clip_actions:
        raise RuntimeError(
            "Action normalization and/or clipping is not supported")

    return env_runner.module


def build_module_wrapper(rl_module: RLModule,
                         explore: bool = False) -> PolicyCallbackFun:
    """Wrap a single-agent RL module into a simple callable that can be passed
    to `BaseJiminyEnv.evaluate` for assessing the performance of the underlying
    policy on a given environment.

    .. note::
        Internally, this method leverages connectors to perform observation and
        action pre- and post-processing. This is especially convenient to h
        andle automatically module view requirements, and store the internal
        state of the policy, if any, without having to manage buffer manually.
        In practice, this methods keeps tracks of all the data being collected
        at every timesteps since the beginning of the ongoing episode. This
        information is passed as input of every connectors. This means that
        connectors are now stateless, which is much better for reproducibility
        and observability.

    :param rl_module: Single-agent RL module to evaluate.
    :param explore: Whether to enable exploration during policy inference.
    """
    # Enable evaluation module
    assert isinstance(rl_module, TorchRLModule)
    rl_module.eval()

    # Instantiate default env_to_module and module_to_env pipelines
    env_to_module = EnvToModulePipeline(
        input_observation_space=rl_module.observation_space,
        input_action_space=rl_module.action_space,
        connectors=[
            AddObservationsFromEpisodesToBatch(),
            AddStatesFromEpisodesToBatch(),
            BatchIndividualItems(),
            NumpyToTensor()])
    module_to_env = ModuleToEnvPipeline(
        input_observation_space=rl_module.observation_space,
        input_action_space=rl_module.action_space,
        connectors=[
            GetActions(),
            TensorToNumpy(),
            UnBatchToIndividualItems(),
            RemoveSingleTsTimeRankFromBatch(),
            ListifyDataForVectorEnv()])

    # Pre-allocate nonlocal memory buffers
    episode = SingleAgentEpisode()
    extra_model_outputs_prev: Dict[str, Any] = {}
    shared_data: Dict[str, Any] = {}

    def forward(obs: Obs,
                action_prev: Optional[Act],
                reward: Optional[SupportsFloat],
                terminated: bool,
                truncated: bool,
                info: Dict[str, Any]) -> Act:
        """Pre-process the observation, compute the action and post-process it.
        """
        # Bind local connectors and buffers managing the internal state
        nonlocal rl_module, module_to_env, env_to_module, episode, \
            extra_model_outputs_prev, shared_data, explore

        # Reset the internal buffer at initialization of the episode, namely
        # when `reward` is `None`.
        if reward is None:
            # Instantiate new empty episode
            episode = SingleAgentEpisode(
                observations=[obs],
                observation_space=rl_module.observation_space,
                action_space=rl_module.action_space)

            # Clear shared data buffer
            shared_data.clear()

        # Update observation buffer
        else:
            episode.add_env_step(
                obs,
                action_prev,
                reward,
                infos=info,
                terminated=terminated,
                truncated=truncated,
                extra_model_outputs=extra_model_outputs_prev)

        # Observation pre-processing
        to_module = env_to_module(
            rl_module=rl_module,
            episodes=(episode,),
            explore=explore,
            shared_data=shared_data)

        # RLModule forward pass
        if explore:
            to_env = rl_module.forward_exploration(to_module, t=episode.t)
        else:
            to_env = rl_module.forward_inference(to_module)

        # Action post-processing
        to_env = module_to_env(
            rl_module=rl_module,
            batch=to_env,
            episodes=(episode,),
            explore=explore,
            shared_data=shared_data)

        # Extract the (vectorized) actions to be sent to the env
        actions = to_env.pop(Columns.ACTIONS)
        actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, actions)

        # Backup extra module outputs
        extra_model_outputs_prev = {k: v[0] for k, v in to_env.items()}

        return actions_for_env[0]

    return forward


__all__ = [
    "initialize",
    "train",
    "sample_from_runner",
    "sample_from_runner_group",
    "evaluate_from_algo",
    "evaluate_from_runner",
    "build_runner_from_checkpoint",
    "build_module_from_checkpoint",
    "build_module_wrapper",
]
