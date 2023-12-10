""" TODO: Write documentation.
"""
import os
import re
import json
import time
import shutil
import socket
import ctypes
import pathlib
import logging
import inspect
import tracemalloc
from contextlib import redirect_stdout
from importlib.abc import MetaPathFinder
from itertools import chain
from datetime import datetime
from collections import defaultdict
from tempfile import mkstemp, mkdtemp
from pkgutil import walk_packages
from operator import attrgetter, itemgetter
from typing import (
    Optional, Callable, Dict, Any, Union, List, Sequence, Tuple, Type, cast)

import gymnasium as gym
import numpy as np
import plotext as plt
import tree

import ray
import ray.rllib.connectors
from ray._private import services
from ray._private.state import GlobalState
from ray._private.gcs_utils import AvailableResources
from ray._private.test_utils import monitor_memory_usage
from ray._raylet import GcsClientOptions  # type: ignore[attr-defined]
from ray.exceptions import RayTaskError
from ray.tune.logger import Logger
from ray.tune.result import TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import SafeFallbackEncoder
from ray.rllib.connectors.connector import Connector, ConnectorContext
from ray.rllib.connectors.registry import register_connector
from ray.rllib.connectors.util import (
    get_agent_connectors_from_config, get_action_connectors_from_config)
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_THIS_ITER
from ray.rllib.utils.typing import (EnvCreator,
                                    EnvID,
                                    AgentID,
                                    PolicyID,
                                    ActionConnectorDataType,
                                    AgentConnectorDataType)

from jiminy_py.viewer import async_play_and_record_logs_files
from gym_jiminy.common.envs import BaseJiminyEnv
from gym_jiminy.common.utils import DataNested


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

VALID_SUMMARY_TYPES = (int, float, np.float32, np.float64, np.int32, np.int64)

LOGGER = logging.getLogger(__name__)


class TBXLogger(Logger):
    """TensorBoardX Logger.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
    """
    # pylint: disable=attribute-defined-outside-init

    VALID_HPARAMS = (str, bool, int, float, list, type(None))
    VALID_NP_HPARAMS = (np.bool_, np.float32, np.float64, np.int32, np.int64)

    def _init(self) -> None:
        # pylint: disable=import-outside-toplevel

        from tensorboardX import SummaryWriter
        self._file_writer = SummaryWriter(self.logdir, flush_secs=30)
        self.last_result: Optional[Dict[str, Union[
            int, float, np.floating, np.integer, Sequence[Union[
                int, float, np.floating, np.integer]], np.ndarray]]] = None

    def on_result(self, result: Dict) -> None:
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        # Remove keys that are not supposed to be logged
        tmp = result.copy()
        for k in ("config", "pid", "timestamp", TIME_TOTAL_S,
                  TRAINING_ITERATION):
            tmp.pop(k, None)

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result: Dict[str, Union[
            int, float, np.floating, np.integer, Sequence[Union[
                int, float, np.floating, np.integer]], np.ndarray]] = {}

        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if isinstance(value, VALID_SUMMARY_TYPES) and not np.isnan(value):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(
                    full_attr, value, global_step=step)
            elif (isinstance(value, list) and len(value) > 0) or (
                isinstance(value, np.ndarray) and value.size > 0
            ):
                valid_result[full_attr] = value

                # Must be video
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    self._file_writer.add_video(
                        full_attr, value, global_step=step, fps=20)
                    continue

                try:
                    self._file_writer.add_histogram(
                        full_attr, value, global_step=step)
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    LOGGER.warning(
                        "You are trying to log an invalid value (%s=%s) "
                        "via %s!", full_attr, value, type(self).__name__)

        self.last_result = valid_result
        self._file_writer.flush()

    def flush(self) -> None:
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self) -> None:
        if self._file_writer is not None:
            if self.trial and self.trial.evaluated_params and self.last_result:
                flat_result = flatten_dict(self.last_result, delimiter="/")
                scrubbed_result = {
                    k: value
                    for k, value in flat_result.items()
                    if isinstance(value, tuple(VALID_SUMMARY_TYPES))
                }
                self._try_log_hparams(scrubbed_result)
            self._file_writer.close()

    def _try_log_hparams(self,
                         result: Dict[
                             str, Union[int, float, np.floating, np.integer]]
                         ) -> None:
        # pylint: disable=import-outside-toplevel

        # TBX currently errors if the hparams value is None.
        assert self.trial and self.trial.evaluated_params
        flat_params = flatten_dict(self.trial.evaluated_params)
        scrubbed_params = {
            k: v for k, v in flat_params.items()
            if isinstance(v, self.VALID_HPARAMS)}

        np_params = {
            k: v.tolist() for k, v in flat_params.items()
            if isinstance(v, self.VALID_NP_HPARAMS)}

        scrubbed_params.update(np_params)

        removed = {
            k: v for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)}
        if removed:
            LOGGER.info(
                "Removed the following hyperparameter values when logging to "
                "tensorboard: %s", str(removed))

        from tensorboardX.summary import hparams

        try:
            experiment_tag, session_start_tag, session_end_tag = hparams(
                hparam_dict=scrubbed_params, metric_dict=result)
            self._file_writer.file_writer.add_summary(experiment_tag)
            self._file_writer.file_writer.add_summary(session_start_tag)
            self._file_writer.file_writer.add_summary(session_end_tag)
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.exception(
                "TensorboardX failed to log hparams. This may be due to an "
                "unsupported type in the hyperparameter values.")


def initialize(num_cpus: int,
               num_gpus: int,
               log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               launch_tensorboard: bool = True,
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
    :param debug: Whether to display debugging trace.
                  Optional: Disabled by default.
    :param verbose: Whether to print information about what is going on.
                    Optional: True by default.

    :returns: Fully-qualified path of the logging directory.
    """
    # handling of default log directory
    if log_root_path is None:
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
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            re.sub(r'[^A-Za-z0-9_]', "_", socket.gethostname())))
    else:
        assert re.match(r'^[A-Za-z0-9_]+$', log_name), (
            "Log name must be a valid Python identifier.")

    # Create log directory
    log_path = os.path.join(log_root_path, log_name)
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Experiment logfiles directory: {log_path}")

    return log_path


def train(algo: Algorithm,
          max_timesteps: int = 0,
          max_iters: int = 0,
          logger_cls: type = TBXLogger,
          logdir: Optional[Logger] = None,
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
    :param logger_cls: Logger class deriving from `ray.tune.logger.Logger`.
                       Optional: `ray.tune.logger.TBXLogger` by default.
    :param logdir: Directory where to store the logfiles created by the logger.
                   Optional: None by default.
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
    if env_spec is None or env_spec.reward_threshold is None:
        reward_threshold = float('inf')
    else:
        reward_threshold = env_spec.reward_threshold

    # Instantiate logger that will be used throughout the experiment
    result_logger = None
    if logdir is not None:
        result_logger = logger_cls({}, logdir=logdir)

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
            config = cast(AlgorithmConfig, algo.config)
            json.dump(config.to_dict(),
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

            # Log results
            if result_logger is not None:
                result_logger.on_result(result)

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
        LOGGER.warning("%s", e)

    # Flush and close logger before returning
    if result_logger is not None:
        result_logger.flush()
        result_logger.close()

    # Backup trained agent and return file location
    return algo.save()


def _restore_default_connectors() -> None:
    """Restore default connectors.

    They would no longer exists after calling `ray.shutdown`, otherwise this
    is a no-op. This is necessary to restore policies resorting on agent and
    action connectors to pre- and post-process observations and actions.
    """
    for root_pkg in (ray.rllib.connectors.action, ray.rllib.connectors.agent):
        for loader, module_name, _ in walk_packages(root_pkg.__path__):
            if isinstance(loader, MetaPathFinder):
                module_info = loader.find_module(module_name, path=None)
            else:
                module_info = loader.find_module(module_name)
            assert module_info is not None
            module = module_info.load_module(module_name)
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, Connector):
                    register_connector(name, obj)


def _extract_eval_worker_info_from_checkpoint(
        checkpoint_path: str
        ) -> Tuple[Type[Policy], EnvCreator, Dict, AlgorithmConfig]:
    """Internal helper to extract the information required to build a local
    evaluation worker or a multi-agent policy from a checkpoint and restore
    the weight of the model.

    :param checkpoint_path: Checkpoint directory to be restored.
    """
    # Restore default connectors if needed
    _restore_default_connectors()

    # Load checkpoint data
    checkpoint_info = get_checkpoint_info(checkpoint_path)
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    worker_state = state["worker"]

    # Extract algorithm class and config
    algorithm_class, config = state.get("algorithm_class"), state.get("config")
    config = algorithm_class.get_default_config().update_from_dict(config)

    # Extract original evaluation config and tweak it to be local only
    config.output = None
    config.evaluation_num_workers = 0
    config.num_envs_per_worker = 1
    config.evaluation_parallel_to_training = False
    evaluation_config = config.get_evaluation_config_object()
    if (evaluation_config.off_policy_estimation_methods or
            config.input_ == "dataset"):
        raise ValueError("Offline evaluation is not supported for now.")

    # Extract policy class and env creator
    policy_class = algorithm_class.get_default_policy_class(config)
    _, env_creator = algorithm_class._get_env_id_and_creator(
        evaluation_config.env, evaluation_config)

    return policy_class, env_creator, worker_state, evaluation_config


def build_eval_policy_from_checkpoint(checkpoint_path: str) -> PolicyMap:
    """Build a multi-agent evaluation policy from a checkpoint generated by
    calling `algo.save()` during training of the policy.

    .. note::
        Internally, the method `ray.rllib.policy.policy.Policy.from_checkpoint`
        is not called because the configuration of the policy must be tweaked
        for the purpose of evaluation.

    :param checkpoint_path: Checkpoint directory to be restored.
    """
    # Extract local evaluation worker information
    _, _, worker_state, config = \
        _extract_eval_worker_info_from_checkpoint(checkpoint_path)

    # Create empty preprocessor and filter maps
    preprocessing_enabled = not config._disable_preprocessor_api and (
        config.model.get("custom_preprocessor") or
        config.preprocessor_pref is not None)

    # Create empty policy map
    policy_map = PolicyMap(
        capacity=config.policy_map_capacity,
        policy_states_are_swappable=config.policy_states_are_swappable)

    # Restore all saved policies
    policy_states = worker_state["policy_states"]
    for pid, policy_state in policy_states.items():
        # Extract policy spec
        spec = policy_state["policy_spec"]
        policy_spec = PolicySpec.deserialize(spec) if (
            config.enable_connectors or
            isinstance(spec, dict)) else spec
        if isinstance(policy_spec.config, AlgorithmConfig):
            merged_conf = policy_spec.config
        else:
            merged_conf = config.copy(copy_frozen=False)
            merged_conf.update_from_dict(policy_spec.config or {})
        merged_conf.worker_index = 0

        # Get post-processed observation space
        obs_space = policy_spec.observation_space
        if preprocessing_enabled:
            preprocessor = ModelCatalog.get_preprocessor_for_space(
                obs_space, merged_conf.model)
            if preprocessor is not None:
                obs_space = preprocessor.observation_space

        # Instantiate the policy
        policy_map[pid] = policy_spec.policy_class(
            obs_space,
            policy_spec.action_space,
            merged_conf)

        # Restore its state
        policy_map[pid].set_state(policy_state)

    return policy_map


def build_policy_wrapper(policy_map: PolicyMap,
                         policy_id: PolicyID = DEFAULT_POLICY_ID,
                         env_id: EnvID = 0,
                         agent_id: AgentID = 0,
                         **kwargs: Any) -> Callable[[
                             DataNested, Optional[float], bool, Dict[str, Any]
                             ], DataNested]:
    """Wrap a single policy into a simple callable. This function can then be
    passed to `BaseJiminyEnv.evaluate` to assess its performance on a given
    environment.

    .. note::
        The internal state of the policy, if any, is managed internally. To do
        so, it leverages connectors to perform observation and action pre- and
        post-processing. This is especially convenient to handle automatically
        policy view requirements and store the internal state of the policy if
        any without having to manage buffer manually.

    :param policy: Single policy to evaluate.
    :param kwargs: Extra keyword arguments that must correspond to a valid
                   subset of options from `AlgorithmConfig`. These options take
                   precedence over the original configuration of the policy.
                   For instance, `clip_actions=False` or `explore=False`.
    """
    # Pick the desired policy
    policy = policy_map[policy_id]

    # Extract policy configuration
    if isinstance(policy.config, AlgorithmConfig):
        config = policy.config.copy(copy_frozen=False)
    else:
        config = AlgorithmConfig().update_from_dict(policy.config or {})
    config.update_from_dict(kwargs)
    config.worker_index = 0

    # Create custom connectors, ignoring the existing ones if any, in order to
    # enforce the desired configuration. For reference, see
    # `ray.rllib.connectors.util.create_connectors_for_policy`.
    ctx = ConnectorContext.from_policy(policy)
    agent_connectors = get_agent_connectors_from_config(ctx, config)
    action_connectors = get_action_connectors_from_config(ctx, config)

    # Set inference mode to avoid spending time on training-only connectors
    agent_connectors.in_eval()
    action_connectors.in_eval()

    def forward(obs: DataNested,
                reward: Optional[float],
                done: bool,
                info: Dict[str, Any]) -> DataNested:
        """Pre-process the observation, compute the action and post-process it.

        .. seealso::
            See `ray.rllib.utils.policy.local_policy_inference` for reference.
        """
        # Bind local connectors managing the internal state of the policy
        nonlocal agent_connectors, action_connectors

        # Reset the connectors at initialization rather than termination
        # to ease debug, namely when `reward` is `None`.
        if reward is None:
            agent_connectors.reset(env_id)

        # Store available input data in a dictionary
        input_dict: Dict[str, Any] = {
            SampleBatch.NEXT_OBS: obs,
            SampleBatch.DONES: done,
            SampleBatch.INFOS: info}
        input_dict[SampleBatch.REWARDS] = reward

        # Pre-process the input
        acd_list = [AgentConnectorDataType(
            env_id, agent_id, input_dict)]  # type: ignore[arg-type]
        (ac_output,) = agent_connectors(acd_list)

        # Compute a single action from raw post-processed input dict.
        # Note that input has to be batched because it is assumed as such
        # by AgentCollector. Because of this, the Policy output is also
        # batched, so the single action must be unpacked.
        policy_output = policy.compute_actions_from_input_dict(
            ac_output.data.sample_batch, explore=config.explore)
        policy_output = tree.map_structure(itemgetter(0), policy_output)

        # Post-process the policy output
        action_connector_data = ActionConnectorDataType(
            env_id,  # type: ignore[arg-type]
            agent_id,
            ac_output.data.raw_dict,
            policy_output)
        acd = action_connectors(action_connector_data)
        action, _, _ = acd.output

        # Notify agent connectors with this new policy output.
        # Necessary for state buffering agent connectors, for example.
        agent_connectors.on_policy_output(action_connector_data)

        return action

    return forward


def build_eval_worker_from_checkpoint(
        checkpoint_path: str) -> RolloutWorker:
    """Build an evaluation worker from a checkpoint generated by calling
    `algo.save()` during training of the policy.

    This local worker can then be passed to `worker_evaluation` in order to
    evaluate the performance of a policy without requiring initialization of
    Ray distributed computing backend.

    .. warning::
        This method is *NOT* standalone in the event where a custom connector
        or environment has been registered by calling
        `ray.rllib.connectors.registry.register_connector` or
        `ray.tune.registry.register_env` and then used during training. In such
        a case, it is necessary to ensure this registration has been done prior
        to calling this method otherwise it will raise an exception.

    :param checkpoint_path: Checkpoint directory to be restored.
    """
    # Extract local evaluation worker information
    policy_class, env_creator, worker_state, config = \
        _extract_eval_worker_info_from_checkpoint(checkpoint_path)

    # Build and restore worker state
    worker = RolloutWorker(
        env_creator=env_creator,
        default_policy_class=policy_class,
        config=config,
        num_workers=0)
    worker.set_state(worker_state)

    return worker


class _WriteLogHook:
    """Stateful function used has a temporary wrapper around an original
    `on_episode_end` callback instance method to force writing log right after
    termination and before reset. This is necessary because `worker.sample()`
    always reset the environment at the end by design.
    """
    def __init__(self, on_episode_end: Callable[..., Any]) -> None:
        self.__func__ = on_episode_end

    def __call__(self, *, worker: RolloutWorker, **kwargs: Any) -> None:
        def write_log(env: gym.Env) -> str:
            fd, log_path = mkstemp(prefix="log_", suffix=".hdf5")
            os.close(fd)
            env.write_log(  # type: ignore[attr-defined]
                log_path, format="hdf5")
            return log_path

        worker.callbacks.log_paths = (  # type: ignore[attr-defined]
            worker.foreach_env(write_log))
        self.__func__(worker=worker, **kwargs)


def _toggle_write_log_hook(worker: RolloutWorker) -> None:
    """Add write log callback hook if not already setup, remove otherwise.
    """
    callbacks = worker.callbacks
    if isinstance(callbacks.on_episode_end, _WriteLogHook):
        callbacks.on_episode_end = callbacks.on_episode_end.__func__
    else:
        callbacks.on_episode_end = _WriteLogHook(callbacks.on_episode_end)


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


def evaluate_local_worker(worker: RolloutWorker,
                          evaluation_num: int = 1,
                          print_stats: Optional[bool] = None,
                          enable_replay: Optional[bool] = None,
                          block: bool = True,
                          **kwargs: Any
                          ) -> Tuple[List[SampleBatch], List[str]]:
    """Evaluates the performance of a given local worker.

    This method is specifically tailored for Gym environments inheriting from
    `BaseJiminyEnv`.

    :param worker: Rollout workers for evaluation.
    :param evaluation_num: How any evaluation to run. The log files of the best
                           performing trial will be exported.
                           Optional: 1 by default.
    :param print_stats: Whether to print high-level statistics.
                        Optional: `evaluation_num >= 10` by default.
    :param enable_replay: Whether to enable replay of the simulation.
                          Optional: True by default if `record_video_path` is
                          not provided and the default/current backend supports
                          it, False otherwise.
    :param block: Whether calling this method should be blocking.
                  Optional: True by default.
    :param kwargs: Extra keyword arguments to forward to the viewer if any.

    :returns: One sample batch per evaluation.
    """
    # Handling of default argument(s)
    if print_stats is None:
        print_stats = evaluation_num >= 10

    # Enforce restriction(s) for using this method
    (env_type,) = set(worker.foreach_env(lambda e: type(e.unwrapped)))
    if not issubclass(env_type, BaseJiminyEnv):
        raise RuntimeError("Test env must inherit from `BaseJiminyEnv`.")

    # Collect samples.
    # `sample` either produces 1 episode or exactly `evaluation_duration` based
    # on `unit` being set to "episodes" or "timesteps" respectively.
    # See https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/algorithms/algorithm.py#L937  # noqa: E501  # pylint: disable=line-too-long
    all_batches, all_log_paths = [], []
    all_num_steps, all_total_rewards = [], []
    _toggle_write_log_hook(worker)
    for _ in range(evaluation_num):
        # Run a complete episode
        batch = worker.sample().as_multi_agent()[DEFAULT_POLICY_ID]
        num_steps = batch.env_steps()
        total_reward = np.sum(batch[batch.REWARDS])

        # Backup the log files
        (log_path,) = worker.callbacks.log_paths  # type: ignore[attr-defined]
        all_log_paths.append(log_path)

        # Store all batches for later use
        all_batches.append(batch)

        # Keep track of basic info
        all_num_steps.append(num_steps)
        all_total_rewards.append(total_reward)
    _toggle_write_log_hook(worker)

    # Ascii histogram if requested
    if print_stats:
        # Get the step size of the environment
        (env_dt,) = set(worker.foreach_env(attrgetter("step_dt")))

        # Print statistics if enough data available
        if evaluation_num >= 10:
            _pretty_print_statistics((
                ("Episode duration", env_dt * np.array(all_num_steps)),
                ("Total reward", np.array(all_total_rewards))))
        else:
            LOGGER.warning(
                "'evaluation_duration' must be at least 10 to print "
                "meaningful statistics.")

    # Extract the indices of the best and worst trial
    idx_worst, idx_best = np.argsort(all_total_rewards)[[0, -1]]

    # Replay and/or record a video of the best and worst trials if requested.
    # Async to enable replaying and recording while training keeps going.
    viewer_kwargs, *_ = worker.foreach_env(attrgetter("viewer_kwargs"))
    viewer_kwargs = {
        **viewer_kwargs, **dict(
            robots_colors=('green', 'red') if evaluation_num > 1 else None),
        **kwargs, **dict(
            legend=("best", "worst") if evaluation_num > 1 else None,
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

    # Return all collected data
    return all_batches, all_log_paths


def evaluate_algo(algo: Algorithm,
                  eval_workers: Optional[WorkerSet] = None,
                  print_stats: bool = True,
                  enable_replay: Optional[bool] = None,
                  record_video: bool = False) -> Dict[str, Any]:
    """Evaluates the current algorithm under `evaluation_config` configuration,
    then returns some performance metrics.

    This method is specifically tailored for Gym environments inheriting from
    `BaseJiminyEnv`. It can be used to monitor the training progress.

    :param eval_workers: Rollout workers for evaluation.
                         Optional: `algo.eval_workers` by default.
    :param print_stats: Whether to print high-level statistics.
                        Optional: True by default.
    :param enable_replay: Whether to enable replay of the simulation.
                          Optional: The opposite of `record_video` by default.
    :param record_video: Whether to enable video recording during evaluation.
                         The video will feature the best and worst trials.
                         Optional: False by default.
    """
    # Handling of default argument(s)
    if eval_workers is None:
        eval_workers = algo.evaluation_workers
    assert eval_workers is not None
    if isinstance(eval_workers, RolloutWorker):
        local_worker = eval_workers
    else:
        local_worker = eval_workers.local_worker()

    # Extract evaluation config
    eval_cfg = algo.evaluation_config
    assert isinstance(eval_cfg, AlgorithmConfig)
    algo_cfg = algo.config

    # Enforce restriction(s) for using this method
    (env_type,) = set(
        chain(*eval_workers.foreach_env(lambda e: type(e.unwrapped))))
    if not issubclass(env_type, BaseJiminyEnv):
        raise RuntimeError("Test env must inherit from `BaseJiminyEnv`.")
    if algo_cfg["evaluation_duration_unit"] == "auto":
        raise ValueError("evaluation_duration_unit='timesteps' not supported.")
    num_episodes = algo_cfg["evaluation_duration"]
    if num_episodes == "auto":
        raise ValueError("evaluation_duration='auto' not supported.")

    # Collect samples.
    # `sample` either produces 1 episode or exactly `evaluation_duration` based
    # on `unit` being set to "episodes" or "timesteps" respectively. See:
    # https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/algorithms/algorithm.py#L937  # noqa: E501  # pylint: disable=line-too-long
    eval_workers.foreach_worker(_toggle_write_log_hook)
    if eval_workers.num_remote_workers() == 0:
        # Collect the data
        all_batches, all_log_paths = evaluate_local_worker(
            local_worker, num_episodes, print_stats=False, enable_replay=False)

        # Extract some high-level statistics
        all_num_steps = [batch.env_steps() for batch in all_batches]
        all_total_rewards = [
            np.sum(batch[batch.REWARDS]) for batch in all_batches]
    elif eval_workers.num_healthy_remote_workers() > 0:
        all_batches, all_log_paths = [], []
        all_num_steps, all_total_rewards = [], []
        while (delta_episodes := num_episodes - len(all_log_paths)) > 0:
            if eval_workers.num_healthy_remote_workers() == 0:
                # All of the remote evaluation workers died. Stopping.
                break

            # Select proper number of evaluation workers for this round
            selected_eval_worker_ids = [
                worker_id for i, worker_id in enumerate(
                    eval_workers.healthy_worker_ids()) if i < delta_episodes]

            # Run a complete episode per selected worker
            batches = eval_workers.foreach_worker(
                func=lambda w: w.sample(),
                local_worker=False,
                remote_worker_ids=selected_eval_worker_ids,
                timeout_seconds=algo_cfg["evaluation_sample_timeout_s"])
            if len(batches) != len(selected_eval_worker_ids):
                LOGGER.warning(
                    "Calling `sample()` on your remote evaluation worker(s) "
                    "resulted in a timeout. Please configure the parameter "
                    "`evaluation_sample_timeout_s` accordingly.")
                break

            # Keep track of basic info
            for ma_batch in batches:
                batch = ma_batch.as_multi_agent()[DEFAULT_POLICY_ID]
                all_num_steps.append(batch.env_steps())
                all_total_rewards.append(np.sum(batch[batch.REWARDS]))
            all_log_paths += chain(*eval_workers.foreach_worker(
                lambda w: w.callbacks.log_paths, local_worker=False))

            # Store all batches for later use
            if algo.reward_estimators:
                all_batches.extend(batches)
    else:
        # Can't find a good way to run the evaluation. Wait for next iteration.
        return {}
    eval_workers.foreach_worker(_toggle_write_log_hook)

    # Compute high-level performance metrics
    metrics = collect_metrics(
        eval_workers,
        keep_custom_metrics=eval_cfg["keep_per_episode_custom_metrics"],
        timeout_seconds=eval_cfg["metrics_episode_collection_timeout_s"])
    metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = sum(all_num_steps)

    # Ascii histogram if requested
    if print_stats:
        # Get the step size of the environment
        (env_dt,) = set(chain(
            *eval_workers.foreach_env(attrgetter("step_dt"))))

        # Print statistics if enough data available
        if num_episodes >= 10:
            _pretty_print_statistics((
                ("Episode duration", env_dt * np.array(all_num_steps)),
                ("Total reward", np.array(all_total_rewards))))
        else:
            LOGGER.warning(
                "'evaluation_num' must be at least 10 to print meaningful "
                "statistics.")

    # Backup only the log file corresponding to the best and worst trial
    idx_worst, idx_best = np.argsort(all_total_rewards)[[0, -1]]
    log_labels, log_paths = ("best", "worst")[:num_episodes], []
    for suffix, idx in zip(log_labels, (idx_best, idx_worst)):
        log_path = f"{algo.logdir}/iter_{algo.iteration}-{suffix}.hdf5"
        shutil.move(all_log_paths[idx], log_path)
        log_paths.append(log_path)

    # Replay and/or record a video of the best and worst trials if requested.
    # Async to enable replaying and recording while training keeps going.
    viewer_kwargs, *_ = chain(*eval_workers.foreach_env(
        attrgetter("viewer_kwargs")))
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

    # Compute off-policy estimates
    estimates = defaultdict(list)
    for name, estimator in algo.reward_estimators.items():
        for batch in all_batches:
            estimate_result = estimator.estimate(
                batch,
                split_batch_by_episode=algo_cfg[
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
    "build_eval_policy_from_checkpoint",
    "build_policy_wrapper",
    "build_eval_worker_from_checkpoint",
    "evaluate_local_worker",
    "evaluate_algo"
]
