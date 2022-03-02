""" TODO: Write documentation.
"""
from typing import Any, Dict, Optional

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID


class MonitorInfoCallback(DefaultCallbacks):
    """ TODO: Write documentation.
    """
    # Base on `rllib/examples/custom_metrics_and_callbacks.py` example.

    def on_episode_step(self,
                        *,
                        worker: RolloutWorker,
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: Episode,
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        super().on_episode_step(worker=worker,
                                base_env=base_env,
                                policies=policies,
                                episode=episode,
                                **kwargs)
        info = episode.last_info_for()
        if info is not None:
            for key, value in info.items():
                # TODO: This line cause memory to grow unboundely
                episode.hist_data.setdefault(key, []).append(value)

    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: Episode,
                       **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        super().on_episode_end(worker=worker,
                               base_env=base_env,
                               policies=policies,
                               episode=episode,
                               **kwargs)
        episode.custom_metrics["episode_duration"] = \
            base_env.get_sub_environments()[0].step_dt * episode.length


class CurriculumUpdateCallback(DefaultCallbacks):
    """ TODO: Write documentation.
    """
    def on_train_result(self,
                        *,
                        trainer: Trainer,
                        result: dict,
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        super().on_train_result(trainer=trainer, result=result, **kwargs)

        # Assertion(s) for type checker
        workers = trainer.workers
        assert isinstance(workers, WorkerSet)

        workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.update(result)))


__all__ = [
    "MonitorInfoCallback",
    "CurriculumUpdateCallback"
]
