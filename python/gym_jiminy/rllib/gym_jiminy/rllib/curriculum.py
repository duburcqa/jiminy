
""" TODO: Write documentation.
"""
import math
from functools import partial
from collections import defaultdict
from typing import (
    List, Any, Dict, Tuple, Optional, Callable, DefaultDict, Union, Collection,
    cast)

import numpy as np
import gymnasium as gym

from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import (
    ResultDict, EpisodeID, EpisodeType, StateDict)

from jiminy_py import tree
from gym_jiminy.common.bases import BasePipelineWrapper
try:
    from gym_jiminy.toolbox.wrappers.meta_envs import (
        ProbaTaskTree, TaskPath, BaseTaskSettableWrapper)
except ImportError as e:
    raise ImportError(
        "Submodule not available. Please install 'gym_jiminy[toolbox]'."
        ) from e


def _update_proba_task_tree_from_runner(
        env_runner: EnvRunner,
        proba_task_tree: ProbaTaskTree) -> None:
    """Update the probability task tree of all the environments being managed
    by a given runner.

    :param env_runner: Environment runner to consider.
    :param proba_task_tree:
        Probability tree consistent with the task space of the underlying
        environment, which must derive from
        `gym_jiminy.toolbox.wrappers.meta_envs.BaseTaskSettableWrapper`.
    """
    # FIXME: `set_attr` is buggy on`gymnasium<=1.0` and cannot be used
    # reliability in conjunction with `BasePipelineWrapper`.
    # See PR: https://github.com/Farama-Foundation/Gymnasium/pull/1294
    assert isinstance(env_runner, SingleAgentEnvRunner)
    env = env_runner.env.unwrapped
    assert isinstance(env, gym.vector.SyncVectorEnv)
    for env in env.unwrapped.envs:
        while not isinstance(env, BaseTaskSettableWrapper):
            assert isinstance(
                env, (gym.Wrapper, BasePipelineWrapper))
            env = env.env
        env.proba_task_tree = proba_task_tree


def _update_proba_task_tree_from_runner_group(
        workers: EnvRunnerGroup,
        proba_task_tree: ProbaTaskTree) -> None:
    """Update the probability tree for a group of environment runners.

    :param workers: Group of environment runners to be updated.
    :param proba_task_tree:
        Probability tree consistent with the task space of the underlying
        environment, which must derive from
        `gym_jiminy.toolbox.wrappers.meta_envs.BaseTaskSettableWrapper`.
    """
    workers.foreach_worker(partial(
        _update_proba_task_tree_from_runner,
        proba_task_tree=proba_task_tree))
    # workers.foreach_worker(
    #     lambda worker: worker.env.unwrapped.set_attr(
    #         'proba_task_tree',
    #         (proba_task_tree,) * worker.num_envs))


class TaskSchedulingSamplingCallback(DefaultCallbacks, Checkpointable):
    r"""Scheduler that automatically adapt the probability distribution of the
    tasks of a task-settable environment in order to maintain the same level of
    performance among all the task, no matter if some of them are much harder
    to solve than others.

    Mathematically, it is equivalent to solving the following maximin
    Reinforcement Learning problem in computationally tractable way:

        theta^* = argmax_{theta}(argmin_{task \in tasks}(
            E_{tau \sim pi_theta}[R_task(tau)]))

    The probability of each node (task group or individual task) is updated
    based on the aggregated mean score of all their respective child nodes. For
    the task tree presented in example of the documentations of
    `TaskSettableEnv` and `TaskSchedulingWrapper`:

        ((E[S0, S1, S2, S3, S4, S5, S6, S7, S8],
            (E[S0, S1, S2],
                (E[S0, S1], E[], E[S2,])),
            E[S3, S4, S5],
            (E[S6, S7, S8],
                (E[S6,], E[S7, S8]))),
        E[S9, S10, S11],
        (E[S12, S13, S14, S15, S16, S17],
            (E[S12, S13],
                (E[S12,], E[S13,])),
            E[S14],
            E[],
            (E[S15, S16, S17],
                (E[S15, S16, S17],
                    (E[S15, S16], E[S17])),
                E[])))

    In practice, the mean score is computed by moving average, which means that
    only a fixed length history of the most recent scores is keep in memory for
    each node. Finally, probability is computed using softmin formula along all
    nodes sharing the same parent:

        (P0, P1, P2) = normalized(exp(- beta * (
            E[S0, S1, S2, S3, S4, S5, S6, S7, S8],
            E[S9, S10, S11],
            E[S12, S13, S14, S15, S16, S17])))

    .. warning:
        To use this callback, the base environment must wrapped with
        `BaseTaskSettableWrapper` (but not necessarily as top-most layer).

    :param history_length: Number of past episodes over which the average score
                           is computed for each node individually. Basically a
                           moving average over a sliding window. This time
                           constant must be at least one order of magnitude
                           slower than the update of the policy for the RL
                           problem to appear stationary from its standpoint.
    :param softmin_beta: Inverse temperature parameter of the softmin formula
                         used to infer sampling probabilities for each task
                         from all their scores. For large beta, very small
                         difference in scores would be enough to make induce
                         large discrependies in probabilities between tasks.
                         However, if the distribution of tasks is very
                         unbalanced, then the policy may forget some skills
                         that were previously already learned.
    :param score_fn:
        Function used to score each episode with signature:

        .. code-block:: python

            score_fn(episode_chunks: Tuple[
                         ray.rllib.utils.typing.EpisodeType, ...],
                     env: gym.vector.VectorEnv,
                     env_index: int) -> float  # score

        `None` to use the standardized return, i.e. the undiscounted cumulative
        reward over complete episodes divided by the maximum number of steps of
        an episode before truncation. This means that its value is ranging from
        0.0 to 1.0 iif the reward is normalized.
        Optional: `None` by default.
    """
    def __init__(self,
                 history_length: int,
                 softmin_beta: float,
                 score_fn: Optional[Callable[[
                    Tuple[EpisodeType, ...], gym.vector.VectorEnv, int
                    ], float]] = None) -> None:
        # Backup user-argument(s)
        self.history_length = history_length
        self.softmin_beta = softmin_beta
        self.score_fn = score_fn

        # Unique ID of the ongoing episode for each environments being
        # managed by the runner associated with this callback instance.
        self._ongoing_episodes: Dict[int, EpisodeID] = {}

        # Episodes that were started by never reached termination before
        # the end of the previous sampling iteration.
        self._partial_episodes: DefaultDict[
            EpisodeID, List[EpisodeType]] = defaultdict(list)

        # Whether to clear all task metrics at the end of the next episode
        self._must_clear_metrics = False

        # Whether the internal state of this callback instance has been
        # initialized.
        # Note that initialization must be postponed because it requires having
        # access to attributes of the environments, but they are not available
        # at this point.
        self._is_initialized = False

        # Whether the internal state of this callback has just been restored
        # but still must to be propagated to all managed environments.
        self._is_restored = False

        # Maximum number of steps of the episodes.
        # This is used to standardize the return because 0.0 and 1.0 (assuming
        # the reward is normalized), or at least to make it independent from
        # the maximum episode duration.
        self._max_num_steps_all: Tuple[int, ...] = ()

        # Arbitrarily nested task space
        self._task_space = gym.spaces.Tuple([])

        # Flatten task space representation for efficiency
        self._task_paths: Tuple[TaskPath, ...] = ()
        self._task_names: Tuple[str, ...] = ()

        # Current probablity task tree
        self.proba_task_tree: ProbaTaskTree = ()

        # Flattened probablity task tree representation for efficiency
        self._proba_task_tree_flat_map: Dict[str, int] = {}

        # Use custom logger and aggregate stats
        self.stats_logger = MetricsLogger()
        self._buffer = MetricsLogger()

    def on_environment_created(self,
                               *,
                               env_runner: EnvRunner,
                               metrics_logger: MetricsLogger,
                               env: gym.vector.VectorEnv,
                               env_context: EnvContext,
                               **kwargs: Any) -> None:
        # Early return if the callback is already initialized
        if self._is_initialized:
            return

        # Add stats proxy at worker-level to ease centralization later on.
        # Note that the proxy must NOT be added if it was already done before.
        # This would be the case when no remote workers are available, causing
        # the local one is used for sample collection as a fallback.
        if not hasattr(env_runner, "_task_stats_logger"):
            env_runner.__dict__["_task_stats_logger"] = self._buffer

        # Backup tree information
        try:
            self._task_space, *_ = env.unwrapped.get_attr("task_space")
        except AttributeError as e:
            raise RuntimeError("Base environment must be wrapped with "
                               "`BaseTaskSettableWrapper`.") from e
        if not self.proba_task_tree:
            self.proba_task_tree, *_ = (
                env.unwrapped.get_attr("proba_task_tree"))

        # Get the maximum episode duration
        self._max_num_steps_all = tuple(
            math.ceil(simulation_duration_max / step_dt)
            for simulation_duration_max, step_dt in zip(
                env.unwrapped.get_attr("simulation_duration_max"),
                env.unwrapped.get_attr("step_dt")))

        # Pre-compute the list of all possible tasks
        self._task_paths = cast(Tuple[TaskPath, ...], tuple(
            (*path, i)
            for path, space in tree.flatten_with_path(self._task_space)
            for i in range(space.n)))
        self._task_names = tuple(
            "/".join(map(str, task)) for task in self._task_paths)

        # Initialize proba task tree flat ordering map
        self._proba_task_tree_flat_map = {
            "/".join(map(str, path[::2])): i for i, (path, _) in enumerate(
                tree.flatten_with_path(self.proba_task_tree))}

        # The callback is now fully initialized
        self._is_initialized = True

    def on_episode_start(self,
                         *,
                         episode: EpisodeType,
                         env_runner: EnvRunner,
                         metrics_logger: MetricsLogger,
                         env: gym.Env,
                         env_index: int,
                         rl_module: RLModule,
                         **kwargs: Any) -> None:
        # Propagate task probability to the environments if not already done.
        # FIXME: At this point, the environment managed by the runner at hand
        # has already been reset. As a result, the tasks associated with the
        # very first episodes to be collected after restoring the state of this
        # callback instance would be sampled according to the old probability
        # task tree. To address this issue, propagation of the probability tree
        # at environment-level after restoring state should be moved in
        # `on_episode_created`. However, this callback method is not available
        # prior to `ray>=2.41`.
        if self._is_restored:
            _update_proba_task_tree_from_runner(
                env_runner, self.proba_task_tree)
            self._is_restored = False

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
        # Clear all custom metrics at the beginning of every sampling iteration
        if self._must_clear_metrics:
            self._buffer.reset()
            self._must_clear_metrics = False

        # Get all the chunks associated with the episode at hand
        episodes = (*self._partial_episodes.pop(episode.id_, []), episode)

        # Compute the score for the episode
        if self.score_fn is None:
            episode_return = sum(
                episode.get_return() for episode in episodes)
            score = episode_return / self._max_num_steps_all[env_index]
        else:
            score = self.score_fn(episodes, env, env_index)

        # Pop task information out from the episode to avoid monitoring it
        task_index = episodes[0].get_infos()[0].pop("task_index")

        # Update score history of all the nodes from root to leave for the
        # task associated with the episode.
        task_path = self._task_paths[task_index]
        for i in range(len(task_path)):
            task_branch = "/".join(map(str, task_path[:(i + 1)]))
            self._buffer.log_value(
                ("score", task_branch),
                score,
                reduce="mean",
                window=self.history_length,
                clear_on_reduce=False)
            self._buffer.log_value(
                ("num", task_branch), 1, reduce="sum")

    def on_sample_end(self,
                      *,
                      env_runner: EnvRunner,
                      metrics_logger: MetricsLogger,
                      samples: List[EpisodeType],
                      **kwargs: Any) -> None:
        # Store all the partial episodes that did not reached done yet
        for episode in samples:
            if episode.is_done:
                continue
            self._partial_episodes[episode.id_].append(episode)

        # Clear all metrics after sampling
        self._must_clear_metrics = True

    def on_train_result(self,
                        *,
                        algorithm: Algorithm,
                        metrics_logger: MetricsLogger,
                        result: ResultDict,
                        **kwargs: Any) -> None:
        # Make sure that the internal state of the callback is initialized
        if not self._is_initialized:
            self.on_environment_created(
                env_runner=algorithm.env_runner,
                metrics_logger=algorithm.metrics,
                env=algorithm.env_runner.env,
                env_context=EnvContext({}, worker_index=0))

        # Centralized reduced task statistics across all remote workers.
        # Note that it is necessary to also include the local worker, as it
        # would be used for sample collection if there is no remote worker.
        workers = algorithm.env_runner_group
        assert workers is not None
        task_stats_all = workers.foreach_worker(
            lambda worker: worker._task_stats_logger.reduce())
        self.stats_logger.merge_and_log_n_dicts(task_stats_all)
        task_stats = self.stats_logger.reduce()

        # Extract task metrics and aggregate them with the results
        task_metrics = {
            key: {
                task_path: stat.peek()
                for task_path, stat in task_stats_group.items()}
            for key, task_stats_group in task_stats.items()}
        result[ENV_RUNNER_RESULTS]["task_metrics"] = task_metrics

        # Extract mean task scores aggregated across runners
        score_task_metrics = task_metrics.get("score", {})

        # Re-order flat task tree and complete missing data with nan
        score_task_tree_flat: List[float] = [
            float('nan') for _ in self._proba_task_tree_flat_map]
        for field, scores in score_task_metrics.items():
            index = self._proba_task_tree_flat_map[field]
            score_task_tree_flat[index] = scores

        # Unflatten mean task score tree
        score_task_tree = tree.unflatten_as(
            self.proba_task_tree, score_task_tree_flat)

        # Compute the probability tree
        proba_task_tree: ProbaTaskTree = []
        score_and_proba_task_branches: List[Tuple[
            ProbaTaskTree, ProbaTaskTree, gym.Space
            ]] = [(score_task_tree, proba_task_tree, self._task_space)]
        while score_and_proba_task_branches:
            # Pop out the first tuple (score, proba, space) in queue
            score_task_branch, proba_task_branch, task_space_branch = (
                score_and_proba_task_branches.pop(0))
            assert isinstance(proba_task_branch, list)

            # Extract the scores from which to compute probabilities
            if isinstance(task_space_branch, gym.spaces.Discrete):
                scores = score_task_branch
            else:
                assert isinstance(task_space_branch, gym.spaces.Tuple)
                assert isinstance(score_task_branch, (tuple, list))
                scores = [score for score, _ in score_task_branch]

            # Compute the probabilities with a fallback for missing data
            probas = np.exp(- self.softmin_beta * np.array(scores))
            probas_undef = np.isnan(probas)
            if probas_undef.all():
                probas = np.ones_like(probas)
            else:
                probas[probas_undef] = np.nanmean(probas)
            probas /= np.sum(probas)

            # Build the probability tree depending on the type of space
            if isinstance(task_space_branch, gym.spaces.Discrete):
                proba_task_branch += probas.tolist()
            else:
                assert isinstance(task_space_branch, gym.spaces.Tuple)
                assert isinstance(score_task_branch, (tuple, list))
                for (_, score_task_branch_), proba, space in zip(
                        score_task_branch, probas, task_space_branch):
                    proba_task_branch_: ProbaTaskTree = []
                    proba_task_branch.append((proba, proba_task_branch_))
                    score_and_proba_task_branches.append((
                        score_task_branch_, proba_task_branch_, space))

        # Update the probability tree at runner-level
        _update_proba_task_tree_from_runner_group(workers, proba_task_tree)
        self.proba_task_tree = proba_task_tree

        # Compute flattened probability tree
        proba_task_tree_flat: List[float] = []
        for (*task_branch, task_leaf) in self._task_paths:
            proba = 1.0
            proba_task_tree_i = proba_task_tree
            for i in task_branch:
                elem_i = proba_task_tree_i[i]
                assert isinstance(elem_i, tuple)
                proba_i, proba_task_tree_i = elem_i
                proba *= proba_i
            elem_i = proba_task_tree_i[task_leaf]
            assert isinstance(elem_i, float)
            proba *= elem_i
            proba_task_tree_flat.append(proba)

        # Monitor flattened probability tree
        proba_task_metrics = task_metrics.setdefault("proba", {})
        for path, proba in zip(self._task_paths, proba_task_tree_flat):
            proba_task_metrics["/".join(map(str, path))] = proba

        # Make sure that no entry is missing for the  number of episodes
        num_task_metrics = task_metrics.setdefault("num", {})
        for path in self._task_paths:
            num_task_metrics.setdefault("/".join(map(str, path)), 0)

        # Filter out all non-leaf task metrics to avoid cluttering plots
        for data in (score_task_metrics, num_task_metrics):
            for key in tuple(data.keys()):
                if key not in self._task_names:
                    del data[key]

    def get_state(self,
                  components: Optional[Union[str, Collection[str]]] = None,
                  *,
                  not_components: Optional[Union[str, Collection[str]]] = None,
                  **kwargs: Any) -> StateDict:
        return dict(
            proba_task_tree=self.proba_task_tree,
            stats_logger=self.stats_logger.get_state())

    def set_state(self, state: StateDict) -> None:
        self.proba_task_tree = state["proba_task_tree"]
        self.stats_logger.set_state(state["stats_logger"])
        self._is_restored = True

    def get_ctor_args_and_kwargs(self) -> Tuple[Tuple, Dict[str, Any]]:
        return (
            (),  # *args
            dict(  # **kwargs
                history_length=self.history_length,
                softmin_beta=self.softmin_beta,
                score_fn=self.score_fn
            )
        )


__all__ = [
    "TaskSchedulingSamplingCallback",
]
