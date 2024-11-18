
""" TODO: Write documentation.
"""
from typing import List, Any, Dict, Tuple, Type

import numpy as np
import gymnasium as gym

from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import ResultDict, EpisodeType

from jiminy_py import tree
from gym_jiminy.common.bases import BasePipelineWrapper
from gym_jiminy.common.envs import BaseJiminyEnv
try:
    from gym_jiminy.toolbox.wrappers.meta_envs import (
        DataTaskTree, Task, TaskSchedulingWrapper)
except ImportError as e:
    raise ImportError(
        "Submodule not available. Please install 'gym_jiminy[toolbox]'."
        ) from e


def build_task_scheduling_callback(history_length: int,
                                   softmin_beta: float
                                   ) -> Type[DefaultCallbacks]:
    """ TODO: Write documentation.

    .. warning:
        To use this callback, the base environment must derive from classes
        `TaskSettableEnv` and `BaseJiminyEnv` (in this exact order), while
        being wrapped with `TaskSchedulingWrapper` (but not necessarily as
        top-most layer).
    """
    class TaskSchedulingSamplingCallback(DefaultCallbacks):
        """ TODO: Write documentation.

        The probability of each node (task group or individual task) is updated
        based on the aggregated mean score of all their respective child nodes.
        For the task tree presented in example of the documentations of
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

        In practice, the mean score is computed by moving average, which means
        that only a fixed length history of the most recent scores is keep in
        memory for each node. Finally, probability is computed using softmin
        formula along all nodes sharing the same parent:

            (P0, P1, P2) = normalized(exp(- beta * (
                E[S0, S1, S2, S3, S4, S5, S6, S7, S8],
                E[S9, S10, S11],
                E[S12, S13, S14, S15, S16, S17])))
        """
        def __init__(self) -> None:
            # Whether to clear all task metrics at the end of the next episode
            self._is_initialized = False
            self._must_clear_metrics = False
            self._task_space = gym.spaces.Tuple([])
            self._task_list: Tuple[Task, ...] = ()
            self._proba_task_tree: DataTaskTree[float] = ()
            self._proba_task_tree_flat_map: Dict[str, int] = {}

        def on_environment_created(self,
                                   *,
                                   env_runner: EnvRunner,
                                   metrics_logger: MetricsLogger,
                                   env: gym.Env,
                                   env_context: EnvContext,
                                   **kwargs: Any) -> None:
            # Early return if the callback is already initialized
            if self._is_initialized:
                return

            # Make sure that the environment is valid
            assert isinstance(env.unwrapped, gym.vector.SyncVectorEnv)
            for env_unwrapped in env.unwrapped.envs:
                assert isinstance(env_unwrapped.unwrapped, BaseJiminyEnv)
                while not isinstance(env_unwrapped, BaseJiminyEnv):
                    if isinstance(env_unwrapped, TaskSchedulingWrapper):
                        break
                    assert isinstance(
                        env_unwrapped, (gym.Wrapper, BasePipelineWrapper))
                    env_unwrapped = env_unwrapped.env
                else:
                    raise RuntimeError(
                        "Base environment must derive from `TaskSettableEnv` "
                        "and `BaseJiminyEnv` in this exact order, while being "
                        "wrapped with `TaskSchedulingWrapper`.")

            # Backup tree information
            assert isinstance(env_unwrapped, TaskSchedulingWrapper)
            self._task_space = env_unwrapped.task_space
            self._task_list = env_unwrapped.task_list
            self._proba_task_tree = env_unwrapped.proba_task_tree

            # Initialize proba task tree flat ordering map
            self._proba_task_tree_flat_map = {
                "/".join(map(str, path[::2])): i for i, (path, _) in enumerate(
                    tree.flatten_with_path(self._proba_task_tree))}

            # The callback is now fully initialized
            self._is_initialized = True

        def on_episode_end(self,
                           *,
                           episode: EpisodeType,
                           env_runner: EnvRunner,
                           metrics_logger: MetricsLogger,
                           env: gym.Env,
                           env_index: int,
                           rl_module: RLModule,
                           **kwargs: Any) -> None:
            # Force clearing all custom metrics at the beginning of every
            # sampling iteration. See `MonitorEpisodeCallback.on_episode_end`.
            if self._must_clear_metrics:
                metrics_logger.stats.pop("task_metrics", None)
                self._must_clear_metrics = False

            # Pop out task information from the episode to avoid monitoring it
            task, score = (), 0.0
            for info in episode.get_infos():
                if "task" in info:  # "task" missing at reset
                    task, score = info.pop("task")

            # Update score history of all the nodes from root to leave for the
            # task associated with the episode.
            for i in range(len(task)):
                field = "/".join(map(str, task[:(i + 1)]))
                metrics_logger.log_value(("task_metrics", "score", field),
                                         score,
                                         reduce="mean",
                                         window=history_length,
                                         clear_on_reduce=False)
                metrics_logger.log_value(
                    ("task_metrics", "num", field), 1, reduce="sum")

        def on_sample_end(self,
                          *,
                          env_runner: EnvRunner,
                          metrics_logger: MetricsLogger,
                          samples: List[EpisodeType],
                          **kwargs: Any) -> None:
            # Clear all metrics after sampling.
            # See `MonitorEpisodeCallback.on_episode_end`.
            self._must_clear_metrics = True

        def on_train_result(self,
                            *,
                            algorithm: Algorithm,
                            metrics_logger: MetricsLogger,
                            result: ResultDict,
                            **kwargs: Any) -> None:
            # Make sure that the internal state of the callback is initialized
            if not self._is_initialized:
                TaskSchedulingSamplingCallback.on_environment_created(
                    self,
                    env_runner=algorithm.env_runner,
                    metrics_logger=algorithm.metrics,
                    env=algorithm.env_runner.env,
                    env_context=EnvContext({}, worker_index=0))

            # Extract from metrics mean task scores aggregated across runners
            metrics = result[ENV_RUNNER_RESULTS]
            task_metrics = metrics.setdefault("task_metrics", {})
            score_task_metrics = task_metrics.get("score", {})

            # Re-order flat task tree and complete missing data with nan
            score_task_tree_flat: List[float] = [
                float('nan') for _ in self._proba_task_tree_flat_map]
            for field, scores in score_task_metrics.items():
                index = self._proba_task_tree_flat_map[field]
                score_task_tree_flat[index] = scores

            # Unflatten mean task score tree
            score_task_tree = tree.unflatten_as(
                self._proba_task_tree, score_task_tree_flat)

            # Compute the probability tree
            proba_task_tree: DataTaskTree[float] = []
            score_and_proba_task_branches: List[Tuple[
                DataTaskTree[float], DataTaskTree[float], gym.Space
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
                probas = np.exp(- softmin_beta * np.array(scores))
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
                        proba_task_branch_: DataTaskTree[float] = []
                        proba_task_branch.append((proba, proba_task_branch_))
                        score_and_proba_task_branches.append((
                            score_task_branch_, proba_task_branch_, space))

            # Update the probability tree at runner-level.
            # FIXME: Rely on `set_attr` when moving to `gymnasium>=1.0`, which
            # fixes attribute lookup in wrapped layers.
            self._proba_task_tree = proba_task_tree
            workers = algorithm.env_runner_group
            assert workers is not None

            def _update_runner_proba_task_tree(env_runner: EnvRunner) -> None:
                """Update the probability task tree of all the environments
                being managed by a given runner.

                :param env_runner: Environment runner to consider.
                """
                nonlocal proba_task_tree
                assert isinstance(env_runner, SingleAgentEnvRunner)
                env = env_runner.env.unwrapped
                assert isinstance(env, gym.vector.SyncVectorEnv)
                for env in env.envs:
                    while not isinstance(env, TaskSchedulingWrapper):
                        assert isinstance(
                            env, (gym.Wrapper, BasePipelineWrapper))
                        env = env.env
                    env.proba_task_tree = proba_task_tree

            workers.foreach_worker(_update_runner_proba_task_tree)
            # workers.foreach_worker(
            #     lambda worker: worker.env.set_attr(
            #         'proba_task_tree', proba_task_tree))

            # Monitor #sampling and probabilities at every task levels
            num_task_metrics = task_metrics.setdefault("num", {})
            proba_task_metrics = task_metrics.setdefault("proba", {})
            for path, proba in tree.flatten_with_path(self._proba_task_tree):
                field = "/".join(map(str, path[::2]))
                proba_task_metrics[field] = proba
                if field not in num_task_metrics:
                    num_task_metrics[field] = 0

            # Filter out all non-leaf metrics to avoid cluttering plots
            field_list = ["/".join(map(str, task)) for task in self._task_list]
            for data in (score_task_metrics,
                         num_task_metrics,
                         proba_task_metrics):
                for key in tuple(data.keys()):
                    if key not in field_list:
                        del data[key]

    return TaskSchedulingSamplingCallback


__all__ = [
    "build_task_scheduling_callback",
]
