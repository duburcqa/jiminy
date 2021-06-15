""" TODO: Write documentation.
"""
from typing import Type, Any

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks


class MonitorInfoCallback(DefaultCallbacks):
    """ TODO: Write documentation.
    """
    # Base on `rllib/examples/custom_metrics_and_callbacks.py` example.

    def on_episode_step(self,
                        *,
                        episode: MultiAgentEpisode,
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        super().on_episode_step(episode=episode, **kwargs)
        info = episode.last_info_for()
        if info is not None:
            for key, value in info.items():
                episode.hist_data.setdefault(key, []).append(value)

    def on_episode_end(self,
                       *,
                       base_env: BaseEnv,
                       episode: MultiAgentEpisode,
                       **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        super().on_episode_end(base_env=base_env, episode=episode, **kwargs)
        episode.custom_metrics["episode_duration"] = \
            base_env.get_unwrapped()[0].step_dt * episode.length


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
        trainer.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.update(result)))


def build_callbacks(*callbacks: Type) -> DefaultCallbacks:
    """Aggregate several callback mixin together.

    .. note::
        Note that the order is important if several mixin are implementing the
        same method. It follows the same precedence roles than usual multiple
        inheritence, namely ordered from highest to lowest priority.

    :param callbacks: Sequence of callback objects.
    """
    # TODO: Remove this method after release of ray 1.4.0 and use instead
    # `ray.rllib.agents.callbacks.MultiCallbacks`.
    return type("UnifiedCallbacks", callbacks, {})


__all__ = [
    "MonitorInfoCallback",
    "CurriculumUpdateCallback",
    "build_callbacks"
]
