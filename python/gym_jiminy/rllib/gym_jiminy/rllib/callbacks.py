""" TODO: Write documentation.
"""
from operator import methodcaller
from typing import Any, Dict, Optional, Union

from ray.rllib.env import BaseEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID


class MonitorInfoCallback(DefaultCallbacks):
    """ TODO: Write documentation.

    Base on `rllib/examples/custom_metrics_and_callbacks.py` example.
    """
    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: Union[Episode, EpisodeV2],
                        env_index: Optional[int] = None,
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        super().on_episode_step(worker=worker,
                                base_env=base_env,
                                policies=policies,
                                episode=episode,
                                env_index=env_index,
                                **kwargs)
        if isinstance(episode, Episode):
            info = episode.last_info_for()
        else:
            info = episode._last_infos.get(_DUMMY_AGENT_ID)
        if info is not None:
            for key, value in info.items():
                # TODO: This line cause memory to grow unboundedly
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
        (step_dt,) = set(e.step_dt for e in base_env.get_sub_environments())
        episode.custom_metrics["episode_duration"] = step_dt * episode.length


class CurriculumUpdateCallback(DefaultCallbacks):
    """ TODO: Write documentation.
    """
    def on_train_result(self,
                        *,
                        algorithm: Algorithm,
                        result: Dict[str, Any],
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        assert algorithm.workers is not None
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        algorithm.workers.foreach_env(methodcaller('update', result))


__all__ = [
    "MonitorInfoCallback",
    "CurriculumUpdateCallback"
]
