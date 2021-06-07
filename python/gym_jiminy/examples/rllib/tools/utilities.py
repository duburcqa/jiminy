import os
import re
import math
import json
import shutil
import socket
import pathlib
import logging
import inspect
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Type

import gym
import numpy as np
from tensorboard.program import TensorBoard

import ray
from ray.exceptions import RayTaskError
from ray.tune.logger import Logger, TBXLogger
from ray.tune.utils.util import SafeFallbackEncoder
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.utils.filter import NoFilter
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.preprocessors import get_preprocessor

from gym_jiminy.common.utils import clip, SpaceDictNested

try:
    import torch
except ModuleNotFoundError:
    pass
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


PRINT_RESULT_FIELDS_FILTER = [
    "training_iteration",
    "time_total_s",
    "timesteps_total",
    "episodes_total",
    "episode_reward_max",
    "episode_reward_mean",
    "episode_len_mean"
]


logger = logging.getLogger(__name__)


class MonitorInfoCallback:
    # Base on `rllib/examples/custom_metrics_and_callbacks.py` example.

    def on_episode_step(self,
                        *,
                        episode: MultiAgentEpisode,
                        **kwargs) -> None:
        super().on_episode_step(episode=episode, **kwargs)
        info = episode.last_info_for()
        if info is not None:
            for key, value in info.items():
                episode.hist_data.setdefault(key, []).append(value)

    def on_episode_end(self,
                       *,
                       base_env: BaseEnv,
                       episode: MultiAgentEpisode,
                       **kwargs) -> None:
        super().on_episode_end(base_env=base_env, episode=episode, **kwargs)
        episode.custom_metrics["episode_duration"] = \
            base_env.get_unwrapped()[0].step_dt * episode.length


class CurriculumUpdateCallback:
    def on_train_result(self,
                        *,
                        trainer,
                        result: dict,
                        **kwargs) -> None:
        super().on_train_result(trainer=trainer, result=result, **kwargs)
        trainer.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.update(result)))


def build_callbacks(*callback_mixins: Type) -> DefaultCallbacks:
    """Aggregate several callback mixin together.

    .. note::
        Note that the order is important if several mixin are implementing the
        same method. It follows the same precedence roles than usual multiple
        inheritence, namely ordered from highest to lowest priority.

    :param callback_mixins: Sequence of callback mixin objects.
    """
    # TODO: Remove this method after release of ray 1.4.0 and use instead
    # `ray.rllib.agents.callbacks.MultiCallbacks`.
    return type("UnifiedCallbacks", (*callback_mixins, DefaultCallbacks), {})


def _flatten_dict(dt: Dict[str, Any],
                  delimiter: str = "/",
                  prevent_delimiter: bool = False) -> Dict[str, Any]:
    """Must be patched to use copy instead of deepcopy to prevent memory
    allocation, significantly impeding computational efficiency of `TBXLogger`,
    and slowing down the optimization by about 25%.
    """
    dt = dt.copy()
    if prevent_delimiter and any(delimiter in key for key in dt):
        # Raise if delimiter is any of the keys
        raise ValueError(
            "Found delimiter `{}` in key when trying to flatten array."
            "Please avoid using the delimiter in your specification.")
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise  if delimiter is in any of the subkeys
                        raise ValueError(
                            "Found delimiter `{}` in key when trying to "
                            "flatten array. Please avoid using the delimiter "
                            "in your specification.")
                    add[delimiter.join([key, str(subkey)])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


ray.tune.logger.flatten_dict = _flatten_dict


def initialize(num_cpus: int,
               num_gpus: int,
               log_root_path: Optional[str] = None,
               log_name: Optional[str] = None,
               logger_cls: type = TBXLogger,
               debug: bool = False,
               verbose: bool = True) -> Callable[[], Logger]:
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
                          Optional: location of this file / log by default.
    :param log_name: Name of the subdirectory where to save data. `None` to
                     use default name, empty string '' to set it interactively
                     in command prompt. It must be a valid Python identifier.
                     Optional: full date _ hostname by default.
    :param debug: Whether or not to display debugging trace.
                  Optional: Disable by default.
    :param verbose: Whether or not to print information about what is going on.
                    Optional: True by default.

    :returns: lambda function to pass a `ray.Trainer` to monitor learning
              progress in Tensorboard.
    """
    # Make sure provided logger class derives from ray.tune.logger.Logger
    assert issubclass(logger_cls, Logger), (
        "Logger class must derive from `ray.tune.logger.Logger`")

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
            local_mode=debug,
            # Logging level
            logging_level=logging.DEBUG if debug else logging.ERROR,
            # Whether to redirect the output from every worker to the driver
            log_to_driver=debug,
            # Whether to start Ray dashboard, which displays cluster's status
            include_dashboard=True,
            # The host to bind the dashboard server to
            dashboard_host="0.0.0.0")

    # Handling of default log root directory
    if log_root_path is None:
        log_root_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "log")
    log_root_path = os.path.abspath(log_root_path)

    # Configure Tensorboard
    if 'tb' not in locals().keys():
        tb = TensorBoard()
        tb.configure(host="0.0.0.0", logdir=log_root_path)
        url = tb.launch()
        if verbose:
            print(f"Started Tensorboard {url}. "
                  f"Root directory: {log_root_path}")

    # Define log filename interactively if requested
    if log_name == "":
        while True:
            log_name = input(
                "Enter desired log subdirectory name (empty for default)...")
            if not log_name or re.match(r'^[A-Za-z0-9_]+$', log_name):
                break
            else:
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
    def logger_creator(config):
        return logger_cls(config, log_path)

    return logger_creator


def compute_action(policy: Policy,
                   input_dict: Dict[str, np.ndarray],
                   explore: bool) -> Any:
    """Compute predicted action by the policy.

    .. note::
        It supports both Pytorch and Tensorflow backends (both eager and
        compiled graph modes).

    :param policy: `rllib.poli.Policy` to use to predict the action, which is
                   a thin wrapper around the actual policy model.
    :param input_dict: Input dictionary for forward as input of the policy.
    :param explore: Whether or not to enable exploration during sampling of the
                    action.
    """
    if policy.framework == 'torch':
        with torch.no_grad():
            input_dict = policy._lazy_tensor_dict(input_dict)
            action_logits, _ = policy.model(input_dict)
            action_dist = policy.dist_class(action_logits, policy.model)
            if explore:
                action_torch = action_dist.sample()
            else:
                action_torch = action_dist.deterministic_sample()
            action = action_torch.cpu().numpy()
    elif tf.compat.v1.executing_eagerly():
        action_logits, _ = policy.model(input_dict)
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
        # `action_logits, _ = model(input_dict).eval(session=policy._sess)[0]`
        # directly, but also significantly trickier.
        feed_dict = {policy._input_dict[key]: value
                     for key, value in input_dict.items()
                     if key in policy._input_dict.keys()}
        feed_dict[policy._is_exploring] = explore
        action = policy._sess.run(
            policy._sampled_action, feed_dict=feed_dict)
    return action


def build_policy_wrapper(policy: Policy,
                         obs_filter_fn: Optional[
                             Callable[[np.ndarray], np.ndarray]] = None,
                         explore: bool = True,
                         n_frames_stack: int = 1,
                         clip_action: bool = False) -> Callable[
                             [np.ndarray, Optional[float]], SpaceDictNested]:
    """ TODO: Write documentation.
    """
    # Extract some proxies for convenience
    observation_space = policy.observation_space
    action_space = policy.action_space

    # Build preprocessor to flatten environment observation
    preprocessor_class = get_preprocessor(observation_space.original_space)
    preprocessor = preprocessor_class(observation_space.original_space)
    obs_flat = preprocessor.observation_space.sample()

    # Initialize frame stack
    input_dict = {
        "obs": np.zeros([1, *observation_space.shape]),
        "prev_n_obs": np.zeros([1, n_frames_stack, *observation_space.shape]),
        "prev_n_act": np.zeros([1, n_frames_stack, *action_space.shape]),
        "prev_n_rew": np.zeros([1, n_frames_stack])}

    # Run the simulation
    def forward(obs: SpaceDictNested,
                reward: Optional[float]) -> SpaceDictNested:
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
        action = compute_action(policy, input_dict, explore)
        if clip_action:
            action = clip(action_space, action)

        # Update previous observation and action buffers
        input_dict["prev_n_obs"][0, -1] = input_dict["obs"][0]
        input_dict["prev_n_act"][0, -1] = action

        # Shift input dict history by one
        for field in input_dict.values():
            field[:] = np.roll(field, shift=-1, axis=1)

        return action[0]

    return forward


def train(train_agent: Trainer,
          max_timesteps: int = 0,
          max_iters: int = 0,
          evaluation_period: int = 0,
          checkpoint_period: int = 0,
          record_video: bool = True,
          verbose: bool = True) -> str:
    """Train a model on a specific environment using a given agent.

    Note that the agent is associated with a given reinforcement learning
    algorithm, and instanciated for a specific environment and neural network
    model. Thus, it already wraps all the required information to actually
    perform training.

    .. note::
        This function can be terminated early using CTRL+C.

    :param train_agent: Training agent.
    :param max_timesteps: Maximum number of training timesteps. 0 to disable.
                          Optional: Disable by default.
    :param max_iters: Maximum number of training iterations. 0 to disable.
                      Optional: Disable by default.
    :param evaluation_period: Run one simulation (with exploration) every given
                              number of training steps, and save the log file
                              and a video of the result in log folder if
                              requested. 0 to disable.
                              Optional: Disable by default.
    :param checkpoint_period: Backup trainer every given number of training
                              steps in log folder if requested. 0 to disable.
                              Optional: Disable by default.
    :param record_video: Whether or not to enable video recording during
                         evaluation.
                         Optional: True by default.
    :param verbose: Whether or not to print high-level information after each
                    training iteration.
                    Optional: True by default.

    :returns: Fullpath of agent's final state dump. Note that it also contains
              the trained neural network model.
    """
    # Get environment's reward threshold, if any
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
        env_file = inspect.getfile(env_type)
        shutil.copy2(env_file, train_agent.logdir, follow_symlinks=True)

        # Backup main's source file, if any
        frame = inspect.stack()[1]  # assuming called directly from main script
        main_file = inspect.getfile(frame[0])
        main_backup_name = f"{train_agent.logdir}/main.py"
        if main_file.endswith(".py"):
            shutil.copy2(main_file, main_backup_name, follow_symlinks=True)

        # Backup RLlib config
        with open(f"{train_agent.logdir}/params.json", 'w') as f:
            json.dump(train_agent.config,
                      f,
                      indent=2,
                      sort_keys=True,
                      cls=SafeFallbackEncoder)

    # Run several training iterations until terminal condition is reached
    try:
        while True:
            # Perform one iteration of training the policy
            result = train_agent.train()
            iter = result["training_iteration"]

            # Print current training result summary
            msg_data = []
            for field in PRINT_RESULT_FIELDS_FILTER:
                if field in result.keys():
                    msg_data.append(f"{field}: {result[field]:.5g}")
            print(" - ".join(msg_data))

            # Record video and log data of the result
            if evaluation_period > 0 and iter % evaluation_period == 0:
                record_video_path = f"{train_agent.logdir}/iter_{iter}.mp4"
                env, _ = test(train_agent,
                              explore=True,
                              enable_replay=record_video,
                              viewer_kwargs={
                                  "record_video_path": record_video_path,
                                  "scene_name": f"iter_{iter}"
                              })
                env.write_log(f"{train_agent.logdir}/iter_{iter}.hdf5")

            # Backup the policy
            if checkpoint_period > 0 and iter % checkpoint_period == 0:
                train_agent.save()

            # Check terminal conditions
            if max_timesteps > 0 and result["timesteps_total"] > max_timesteps:
                break
            if max_iters > 0 and iter > max_iters:
                break
            if result["episode_reward_mean"] > reward_threshold:
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


def evaluate(env: gym.Env,
             policy: Policy,
             obs_filter_fn: Optional[
                 Callable[[np.ndarray], np.ndarray]] = None,
             n_frames_stack: int = 1,
             horizon: Optional[int] = None,
             clip_action: bool = False,
             explore: bool = False,
             enable_stats: bool = True,
             enable_replay: bool = True,
             viewer_kwargs: Optional[Dict[str, Any]] = None) -> gym.Env:
    """Evaluate a policy on a given environment over a complete episode.

    :param env: Environment on which to evaluate the policy. Note that the
                environment must be already instantiated and ready-to-use.
    :param policy: Policy to evaluate.
    :param obs_filter_fn: Observation filter to apply on (flattened)
                          observation from the environment, usually used
                          from moving average normalization. `None` to
                          disable.
                          Optional: Disable by default.
    :param n_frames_stack: Number of frames to stack in the input to provide
                           to the policy. Note that previous observation,
                           action, and reward will be stacked.
                           Optional: 1 by default.
    :param horizon: Horizon of the simulation, namely maximum number of steps
                    before termination. `None` to disable.
                    Optional: Disable by default.
    :param clip_action: Whether or not to clip action to make sure the
                        prediction by the policy is not out-of-bounds.
                        Optional: Disable by default.
    :param explore: Whether or not to enable exploration during sampling of the
                    actions predicted by the policy.
                    Optional: Disable by default.
    :param enable_stats: Whether or not to print high-level statistics after
                         simulation.
                         Optional: Enable by default.
    :param enable_replay: Whether or not to enable replay of the simulation,
                          and eventually recording through `viewer_kwargs`.
                          Optional: Enable by default.
    :param viewer_kwargs: Extra keyword arguments to forward to the viewer if
                          replay has been requested.
    """
    # Handling of default arguments
    if viewer_kwargs is None:
        viewer_kwargs = {}

    # Initialize frame stack
    policy_forward = build_policy_wrapper(
        policy, obs_filter_fn, explore, n_frames_stack, clip_action)

    # Initialize the simulation
    obs = env.reset()
    reward = None

    # Run the simulation
    try:
        info_episode = []
        done = False
        while not done:
            action = policy_forward(obs=obs, reward=reward)
            obs, reward, done, info = env.step(action)
            info_episode.append(info)
            if done or (horizon is not None and env.num_steps > horizon):
                break
    except KeyboardInterrupt:
        pass

    # Display some statistic if requested
    if enable_stats:
        print("env.num_steps:", env.num_steps)
        print("cumulative reward:", env.total_reward)

    # Replay the result if requested
    if enable_replay:
        try:
            env.replay(**{'speed_ratio': 1.0, **viewer_kwargs})
        except Exception as e:
            # Do not fail because of replay/recording exception
            logger.warning(str(e))

    return env, info_episode


def test(test_agent: Trainer,
         explore: bool = True,
         n_frames_stack: int = 1,
         enable_stats: bool = True,
         enable_replay: bool = True,
         test_env: Optional[gym.Env] = None,
         viewer_kwargs: Optional[Dict[str, Any]] = None,
         **kwargs: Any) -> gym.Env:
    """Test a model on a specific environment using a given agent.

    .. note::
        This function can be terminated early using CTRL+C.

    :param test_agent: Agent to evaluate on a single simulation.
    :param explore: Whether or not to enable exploration during sampling of the
                    actions predicted by the policy.
                    Optional: Disable by default.
    :param n_frames_stack: Number of frames to stack in the input to provide
                           to the policy. Note that previous observation,
                           action, and reward will be stacked.
                           Optional: 1 by default.
    :param enable_stats: Whether or not to print high-level statistics after
                         simulation.
                         Optional: Enable by default.
    :param enable_replay: Whether or not to enable replay of the simulation,
                          and eventually recording through `viewer_kwargs`.
                          Optional: Enable by default.
    :param test_env: Environment on which to evaluate the policy. It must be
                     already instantiated and ready-to-use.
    :param viewer_kwargs: Extra keyword arguments to forward to the viewer if
                          replay has been requested.
    """
    # Instantiate the environment if not provided
    if test_env is None:
        test_env = test_agent.env_creator(
            {**test_agent.config["env_config"], **kwargs})

    # Get policy model
    policy = test_agent.get_policy()

    # Get observation filter if any
    obs_filter = test_agent.workers.local_worker().filters["default_policy"]
    if isinstance(obs_filter, NoFilter):
        obs_filter_fn = None
    else:
        obs_mean, obs_std = obs_filter.rs.mean, obs_filter.rs.std
        obs_filter_fn = \
            lambda obs: (obs - obs_mean) / (obs_std + 1.0e-8)  # noqa: E731

    if viewer_kwargs is not None:
        kwargs.update(viewer_kwargs)

    return evaluate(test_env,
                    policy,
                    obs_filter_fn,
                    n_frames_stack=n_frames_stack,
                    horizon=test_agent.config["horizon"],
                    clip_action=test_agent.config["clip_actions"],
                    explore=explore,
                    enable_stats=enable_stats,
                    enable_replay=enable_replay,
                    viewer_kwargs=kwargs)
