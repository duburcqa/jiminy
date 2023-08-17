
""" TODO: Write documentation.
"""
import re
from typing import List, Any, Dict, Union, Optional, Callable, Type

import numpy as np

from ray.tune.result import TRAINING_ITERATION, TIMESTEPS_TOTAL
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID

try:
    from gym_jiminy.toolbox.wrappers.meta_envs import DataTreeT
except ImportError as e:
    raise ImportError(
        "Submodule not available. Please install 'gym_jiminy[toolbox]'."
        ) from e

from .utilities import TBXLogger


def build_task_scheduling_callback(
        history_length: int,
        softmin_beta: float,
        callbacks_class: Type[DefaultCallbacks] = DefaultCallbacks
        ) -> Type[DefaultCallbacks]:
    """ TODO: Write documentation.
    """
    class TaskSchedulingSamplingCallback(
            callbacks_class):  # type: ignore[valid-type,misc]
        """ TODO: Write documentation.

        .. warning::
            Note that the driver and the workers have different instantiation
            of the callbacks. As a result, they do not share the same internal
            state, i.e. `on_episode_end` and `on_train_result` methods cannot
            rely on shared attributes.
        """
        def __init__(
                self,
                legacy_callbacks_dict: Optional[
                    Dict[str, Callable[..., Any]]] = None) -> None:
            """ TODO: Write documentation.
            """
            super().__init__(legacy_callbacks_dict)
            self._task_tree_scores: DataTreeT = {}

        def on_episode_end(self,
                           *,
                           worker: RolloutWorker,
                           base_env: BaseEnv,
                           policies: Dict[PolicyID, Policy],
                           episode: Union[Episode, EpisodeV2, Exception],
                           env_index: Optional[int] = None,
                           **kwargs: Any) -> None:
            """ TODO: Write documentation.
            """
            # Call base implementation
            super().on_episode_end(worker=worker,
                                   base_env=base_env,
                                   policies=policies,
                                   episode=episode,
                                   **kwargs)
            if not isinstance(episode, (Episode, EpisodeV2)):
                return

            # Monitor episode duration for each gait
            for env in base_env.get_sub_environments():
                # Gather the set of tasks.
                # It corresponds to all the leaves of the task decision tree.
                tasks: List[List[Any]] = [[]]
                task_branches = [env.task_tree_probas]
                while task_branches:
                    task_path, task_branch = tasks.pop(0), task_branches.pop(0)
                    for task_node, (_, task_branch_next) in \
                            task_branch.items():
                        tasks.append(task_path + [task_node])
                        if task_branch_next:
                            task_branches.append(task_branch_next)

                # Initialize histogram data for all tasks
                for task in tasks:
                    field = "/".join(map(str, ("task", *task, "score")))
                    episode.hist_data.setdefault(field, [])

                # Update statistics for the task corresponding to the episode
                task = env.get_task()
                if task:
                    field = "/".join(map(str, ("task", *task, "score")))
                    episode.hist_data[field].append(env.get_score())

        def on_train_result(self,
                            *,
                            algorithm: Algorithm,
                            result: Dict[str, Any],
                            **kwargs: Any) -> None:
            """ TODO: Write documentation.
            """
            # Gather task scores
            task_tree_scores_new: DataTreeT = {}
            for field, value in result['hist_stats'].items():
                if field.startswith("task/"):
                    task = map(int, field.split("/")[1:-1])
                    task_branch = task_tree_scores_new
                    for task_node in task:
                        task_scores, task_branch = task_branch.setdefault(
                            task_node, ([], {}))
                        task_scores += value

            # Keep track of scores over training iterations for a moving window
            task_branches = [self._task_tree_scores]
            task_branches_new = [task_tree_scores_new]
            while task_branches_new:
                task_branch = task_branches.pop(0)
                task_branch_new = task_branches_new.pop(0)
                for task_node, (task_scores_new, task_branch_new_next) in \
                        task_branch_new.items():
                    task_scores, task_branch_next = task_branch.setdefault(
                        task_node, ([], {}))
                    task_scores += task_scores_new
                    del task_scores[-(history_length + 1)::-1]
                    if task_branch_new_next:
                        task_branches_new.append(task_branch_new_next)
                        task_branches.append(task_branch_next)

            # Compute tree of mean values
            task_tree_mean: DataTreeT = {}
            task_branches_mean = [task_tree_mean]
            task_branches = [self._task_tree_scores]
            while task_branches:
                task_branch_mean = task_branches_mean.pop(0)
                task_branch = task_branches.pop(0)
                for task_node, (
                        task_scores, task_branch_next) in task_branch.items():
                    if len(task_scores) > 0:
                        task_scores_mean = np.mean(task_scores, dtype=float)
                    else:
                        task_scores_mean = np.nan
                    task_branch_next_mean: DataTreeT = {}
                    task_branch_mean[task_node] = (
                        task_scores_mean, task_branch_next_mean)
                    if task_branch_next:
                        task_branches.append(task_branch_next)
                        task_branches_mean.append(task_branch_next_mean)

            # Compute tree of probabilities
            task_tree_probas: DataTreeT = {}
            task_branches_probas = [task_tree_probas]
            task_branches_mean = [task_tree_mean]
            while task_branches_probas:
                task_branch_probas = task_branches_probas.pop(0)
                task_branch_mean = task_branches_mean.pop(0)

                task_scores_mean = np.array([
                    task_scores_mean
                    for task_scores_mean, _ in task_branch_mean.values()])
                task_probas = np.exp(- softmin_beta * task_scores_mean)
                task_probas_undef = np.isnan(task_probas)
                if task_probas_undef.all():
                    task_probas = np.ones_like(task_probas)
                else:
                    task_probas[task_probas_undef] = np.nanmean(task_probas)
                task_probas /= np.sum(task_probas)

                for (task_node, (_, task_branch_next_mean)), task_proba in \
                        zip(task_branch_mean.items(), task_probas):
                    task_branch_next_probas: DataTreeT = {}
                    task_branch_probas[task_node] = (
                        task_proba, task_branch_next_probas)
                    if task_branch_next_mean:
                        task_branches_probas.append(task_branch_next_probas)
                        task_branches_mean.append(task_branch_next_mean)

            # Compute sampling statistics for currents training iteration
            task_tree_num: DataTreeT = {}
            task_branches_num = [task_tree_num]
            task_branches_new = [task_tree_scores_new]
            while task_branches_new:
                task_branch_num = task_branches_num.pop(0)
                task_branch_new = task_branches_new.pop(0)
                for task_node, (task_scores, task_branch_next) in \
                        task_branch_new.items():
                    task_branch_next_mean = {}
                    task_branch_num[task_node] = (
                        len(task_scores), task_branch_next_mean)
                    if task_branch_next:
                        task_branches_new.append(task_branch_next)
                        task_branches_num.append(task_branch_next_mean)

            # Add statistics regarding task curriculum learning
            result["task_metrics"] = {}

            # Monitor sampling statistics for every task levels
            tasks: List[List[Any]] = [[]]
            task_branches = [task_tree_num]
            while task_branches:
                task_path, task_branch = tasks.pop(0), task_branches.pop(0)
                for task_node, (task_num, task_branch_next) in \
                        task_branch.items():
                    tasks.append(task_path + [task_node])
                    field = "/".join(map(str, ("task", *tasks[-1], "num")))
                    result["task_metrics"][field] = task_num
                    if task_branch_next:
                        task_branches.append(task_branch_next)

            # Monitor probabilities for every task levels
            tasks: List[List[Any]] = [[]]
            task_branches = [task_tree_probas]
            while task_branches:
                task_path, task_branch = tasks.pop(0), task_branches.pop(0)
                for task_node, (task_num, task_branch_next) in \
                        task_branch.items():
                    tasks.append(task_path + [task_node])
                    field = "/".join(map(str, ("task", *tasks[-1], "proba")))
                    result["task_metrics"][field] = task_num
                    if task_branch_next:
                        task_branches.append(task_branch_next)

            # Update envs accordingly
            assert algorithm.workers is not None
            algorithm.workers.foreach_env(
                lambda env: env.task_tree_probas.update(task_tree_probas))

    return TaskSchedulingSamplingCallback


class TBXLoggerLayout(TBXLogger):
    """ TODO: Write documentation.
    """
    def on_result(self, result: dict) -> None:
        """ TODO: Write documentation.
        """
        # Get current training iteration
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        # Add custom multiline charts to compare statistics between gaits
        task_stats_tags: Dict[str, List[str]] = {}
        path = ["ray", "tune", "task_metrics"]
        for field, value in result["hist_stats"].items():
            if re.search("task/", field):
                *_, title = field.split("/")
                task_stats_tag = "/".join(path + [f"{field}_mean"])
                mean = np.mean(value) if len(value) > 0 else 0.0
                self._file_writer.add_scalar(
                    task_stats_tag, mean, global_step=step)
                task_stats_tags.setdefault(title, []).append(task_stats_tag)
        for title, tags in task_stats_tags.items():
            self._file_writer.add_custom_scalars_multilinechart(
                tags=tags, title=title)

        # Add custom multiline charts to monitor task metrics
        task_metrics_tags: Dict[str, List[str]] = {}
        for field in result.get("task_metrics", {}).keys():
            *_, title = field.split("/")
            task_metrics_tag = "/".join(path + [field])
            task_metrics_tags.setdefault(title, []).append(task_metrics_tag)
        for title, tags in task_metrics_tags.items():
            self._file_writer.add_custom_scalars_multilinechart(
                tags=tags, title=title)

        # Call base implementation
        super().on_result(result)


__all__ = [
    "build_task_scheduling_callback",
    "TBXLoggerLayout"
]
