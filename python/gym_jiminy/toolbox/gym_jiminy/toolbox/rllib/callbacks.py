""" TODO: Write documentation.
"""
from typing import Type, Any

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks


class MonitorInfoCallback:
    """ TODO: Write documentation.
    """
    # Base on `rllib/examples/custom_metrics_and_callbacks.py` example.

    def on_episode_step(self,
                        *,
                        episode: MultiAgentEpisode,
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # pylint: disable=no-member
        super().on_episode_step(  # type: ignore[misc]
            episode=episode, **kwargs)
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
        # pylint: disable=no-member
        super().on_episode_end(  # type: ignore[misc]
            base_env=base_env, episode=episode, **kwargs)
        episode.custom_metrics["episode_duration"] = \
            base_env.get_unwrapped()[0].step_dt * episode.length


class CurriculumUpdateCallback:
    """ TODO: Write documentation.
    """
    def on_train_result(self,
                        *,
                        trainer: Trainer,
                        result: dict,
                        **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # pylint: disable=no-member
        super().on_train_result(  # type: ignore[misc]
            trainer=trainer, result=result, **kwargs)
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


__all__ = [
    "MonitorInfoCallback",
    "CurriculumUpdateCallback",
    "build_callbacks"
]
