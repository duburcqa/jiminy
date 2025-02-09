# mypy: disable-error-code="no-untyped-def, var-annotated"
""" TODO: Write documentation.
"""
import os
import sys
import math
import warnings
import unittest
from functools import partial
from typing import Any, Callable, Dict, Tuple, Optional

import numpy as np
import gymnasium as gym

try:
    import ray
    from ray.tune.registry import register_env
    from ray.rllib.algorithms import PPOConfig
    from ray.rllib.utils.typing import EpisodeType
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    IS_RAY_AVAILABLE = True
except ImportError:
    IS_RAY_AVAILABLE = False

from gym_jiminy.common.wrappers import FlattenObservation
from gym_jiminy.envs import AcrobotJiminyEnv
from gym_jiminy.toolbox.wrappers.meta_envs import BaseTaskSettableWrapper
if IS_RAY_AVAILABLE:
    from gym_jiminy.rllib.utilities import initialize, train
    from gym_jiminy.rllib.curriculum import TaskSchedulingSamplingCallback


# Fix the seed for CI stability
SEED = 0

# Dummy task scoring parameters
TASK_SCORE_MIN = 0.15
TASK_SCORE_MAX = 0.85
TASK_SCORE_STD = 0.05

# Task curriculum parameters
SOFTMIN_BETA = 3.0
HISTORY_LENGTH = 40


class DummyTaskWrapper(BaseTaskSettableWrapper):
    def _initialize_task_space(self) -> None:
        self.task_space = gym.spaces.Tuple([
            gym.spaces.Tuple([
                gym.spaces.Discrete(3),
                gym.spaces.Discrete(1),
                gym.spaces.Discrete(2)]),
            gym.spaces.Discrete(1),
            gym.spaces.Tuple([
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(1),
                gym.spaces.Discrete(1),
                gym.spaces.Tuple([
                    gym.spaces.Discrete(2),
                    gym.spaces.Discrete(1)])])])

    def set_task(self, task_index: int) -> None:
        pass


def compute_score(episode_chunks: Tuple["EpisodeType", ...],
                  env: gym.vector.VectorEnv,
                  env_index: int) -> float:
    task_index = env.get_attr("task_index")[env_index]
    num_tasks = env.get_attr("num_tasks")[env_index]
    ratio = task_index / num_tasks
    score = TASK_SCORE_MIN + ratio * (TASK_SCORE_MAX - TASK_SCORE_MIN)
    score += np.random.uniform(low=-1.0, high=1.0) * TASK_SCORE_STD
    return float(score)


@unittest.skipIf(not IS_RAY_AVAILABLE, "Ray is not available.")
class AcrobotTaskCurriculum(unittest.TestCase):
    """ TODO: Write documentation.
    """
    def setUp(self):
        if not sys.warnoptions:
            # Disable Ray "unclosed file" warnings, which are not legitimate
            warnings.simplefilter("ignore", ResourceWarning)

    def test_task_curriculum(self):
        """ TODO: Write documentation.
        """
        # Reset numpy global seed
        np.random.seed(0)

        # Start Ray and Tensorboard background processes.
        # Note that it is necessary to specify Python Path manually, otherwise
        # serialization of `DummyTaskWrapper` will fail miserably if the tests
        # are executed from another directory via:
        # `python -m unittest discover -s "unit_py" -v`
        logdir = initialize(num_cpus=0,
                            num_gpus=0,
                            launch_tensorboard= False,
                            env_vars={"PYTHONPATH": os.path.dirname(__file__)},
                            verbose=False)

        # Register the environment
        env_creator: Callable[[Dict[str, Any]], gym.Env]
        env_creator = lambda kwargs: gym.wrappers.TimeLimit(
            FlattenObservation(DummyTaskWrapper(AcrobotJiminyEnv(**kwargs))),
            max_episode_steps=1)
        register_env("env", env_creator)

        # PPO configuration
        algo_config = PPOConfig()
        algo_config.env_runners(
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_env_runners=0,
            num_envs_per_env_runner=1)
        algo_config.learners(
            num_learners=0,
            num_cpus_per_learner=1,
            num_gpus_per_learner=0)
        algo_config.environment(env="env")
        algo_config.debugging(seed=SEED)
        algo_config.callbacks(partial(
            TaskSchedulingSamplingCallback,
            history_length=HISTORY_LENGTH,
            softmin_beta=SOFTMIN_BETA,
            score_fn=compute_score))
        algo_config.training(
            train_batch_size_per_learner=25,
            minibatch_size=25,
            num_epochs=1)
        algo_config.rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[8, 8],
            ))

        # Initialize the learning algorithm and train the agent
        result, checkpoint_path = train(
            algo_config,
            logdir,
            max_iters=100,
            verbose=False)

        # Terminate Ray backend
        ray.shutdown()

        # Check that the number of episodes are valid
        result_runner = result['env_runners']
        nums = result_runner['task_metrics']['num']
        num_tot = sum(nums.values())
        assert result_runner['num_episodes_lifetime'] == num_tot

        # Check the the number of episodes per task is consistent with probas
        task_paths = sorted(nums.keys())
        probas = result_runner['task_metrics']['proba']
        for task_path_str in task_paths:
            proba_stat = probas[task_path_str]
            proba_est = nums[task_path_str] / num_tot
            assert (proba_stat - proba_est) < 0.1

        # Check if the average scores are valid
        tol_abs = 2 * (TASK_SCORE_STD / HISTORY_LENGTH ** 0.5)
        scores = result_runner['task_metrics']['score']
        for task_index, task_path_str in enumerate(task_paths):
            if task_path_str not in scores.keys():
                continue
            score_stat = scores[task_path_str]
            task_path = tuple(map(int, task_path_str.split('/')))
            ratio = task_index / len(task_paths)
            score_th = TASK_SCORE_MIN + ratio * (
                TASK_SCORE_MAX - TASK_SCORE_MIN)
            assert abs(score_stat - score_th) < tol_abs

        # Check that the probability are valid
        assert abs(1.0 - sum(probas.values())) < 1e-6
        for task_index, task_path_str in enumerate(task_paths):
            proba_stat = probas[task_path_str]
            probas_depth = []
            task_branch_path = list(map(int, task_path_str.split('/')))
            while task_branch_path:
                task_leaf_index = task_branch_path.pop()
                task_branch_root = "/".join(map(str, task_branch_path))
                task_branch_width = 1 + max(
                    int(task_path_str.split('/')[len(task_branch_path)])
                    for task_path_str in task_paths
                    if task_path_str.startswith(task_branch_root))

                scores_branch = np.full((task_branch_width,), float("nan"))
                for i in range(task_branch_width):
                    task_branch = "/".join(
                        filter(None, (task_branch_root, str(i))))
                    num, score = 0, 0.0
                    for task_index, task_path_str in enumerate(task_paths):
                        if task_path_str.startswith(task_branch):
                            ratio = task_index / len(task_paths)
                            score += (
                                nums[task_path_str] * scores[task_path_str])
                            num += nums[task_path_str]
                    if num > 0:
                        score /= num
                        scores_branch[i] = score

                probas_branch = np.exp(
                    - SOFTMIN_BETA * np.array(scores_branch))
                probas_branch /= np.sum(probas_branch)
                probas_depth.append(probas_branch[task_leaf_index])
            proba_th = math.prod(probas_depth)
            assert abs(proba_stat - proba_th) < tol_abs
