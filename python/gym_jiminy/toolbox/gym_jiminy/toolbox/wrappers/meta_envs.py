""" TODO: Write documentation.
"""
from operator import mul
from functools import reduce
from typing import Any, List, Optional, Dict, Tuple

import numpy as np

import gym
from gym import spaces

from gym_jiminy.common.utils.helpers import SpaceDictNested


DataTreeT = Dict[Any, Tuple[Any, "DataTreeT"]]  # type: ignore[misc]


class HierarchicalTaskSettableEnv(gym.Env):
    """Extension of gym.Env to define a task-settable Env.

    .. note::
        This class extends the API of Ray RLlib `TaskSettableEnv`. See:
        https://github.com/ray-project/ray/blob/master/rllib/env/apis/task_settable_env.py
    """
    task_space: Optional[spaces.Tuple] = None

    def get_task(self) -> Tuple[Any, ...]:
        """Gets the task that the agent is performing in the current
        environment.
        """
        raise NotImplementedError

    def set_task(self, task: Tuple[Any, ...]) -> None:
        """Sets the specified task to the current environment.

        :param task: task of the meta-learning environment.
        """
        raise NotImplementedError

    def sample_tasks(self, n_tasks: int) -> List[Tuple[Any, ...]]:
        """Samples task of the meta-environment.

        :param n_tasks: number of different meta-tasks needed.
        """
        raise NotImplementedError

    def get_score(self) -> float:
        """ TODO: Write documentation.
        """
        raise NotImplementedError


class TaskSchedulingWrapper(gym.Wrapper):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 env: HierarchicalTaskSettableEnv,
                 initial_task_tree: Optional[DataTreeT] = None
                 ) -> None:
        """ TODO: Write documentation.
        """
        # Call base implementation
        super().__init__(env)

        # Make sure the task space of the environment is supported
        task_space = env.task_space
        if (task_space is None or
            not isinstance(task_space, spaces.Tuple) or
            any(not isinstance(space, (
                spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary))
                for space in task_space)):
            raise ValueError(
                "The task space must be a Tuple or discrete spaces, i.e. "
                "Discrete, MultiDiscrete or MultiBinary.")

        # Default initial task probabilities
        if initial_task_tree is None:
            initial_task_tree = {}
            task_branches = [initial_task_tree]
            for space in task_space:
                # Get task level size
                if isinstance(space, spaces.Discrete):
                    space_size = space.n
                if isinstance(space, spaces.MultiBinary):
                    space_size = 2 ** space.n
                elif isinstance(space, spaces.MultiDiscrete):
                    space_size = reduce(mul, space.nvec)

                # Initialize task level probas
                task_branches_next: List[DataTreeT] = []
                for task_branch in task_branches:
                    for k in range(space_size):
                        task_branch_next: DataTreeT = {}
                        task_branches_next.append(task_branch_next)
                        task_branch[k] = (1.0 / space_size, task_branch_next)
                task_branches = task_branches_next

        # Define task probabilities
        self.task_tree_probas = initial_task_tree

    def sample_tasks(self, n_tasks: int) -> List[Tuple[Any, ...]]:
        """ TODO: Write documentation.
        """
        tasks = []
        for _ in range(n_tasks):
            task_path, task_branch = [], self.task_tree_probas
            while task_branch:
                task_nodes, task_branches_next = zip(*task_branch.items())
                task_branch_proba = [
                    task_proba for task_proba, _ in task_branches_next]
                task_branch_idx = int(np.where(
                    self.rg.multinomial(1, task_branch_proba))[0])
                task_path.append(task_nodes[task_branch_idx])
                _, task_branch = task_branches_next[task_branch_idx]
            tasks.append(tuple(task_path))
        return tasks

    def reset(self, **kwargs: Any) -> SpaceDictNested:
        """ TODO: Write documentation.
        """
        # Sample task
        task = self.sample_tasks(1)[0]

        # Set task
        self.env.set_task(task)

        # Reset environment
        return self.env.reset(**kwargs)
