""" TODO: Write documentation.
"""
from abc import abstractmethod, ABCMeta
from typing import (
    Any, Optional, Dict, List, Tuple, Sequence, Union, Generic,
    SupportsFloat, TypeVar, cast)

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from jiminy_py import tree
from gym_jiminy.common.bases import Obs, Act, InfoType


ValueT = TypeVar('ValueT')

DataTaskTree = Sequence[Union[ValueT, Tuple[ValueT, "DataTaskTree[ValueT]"]]]
Task = Tuple[int, ...]


class TaskSettableEnv(gym.Env[Obs, Act], Generic[Obs, Act], metaclass=ABCMeta):
    """Extension of gym.Env to define a task-settable environment.

    When defining a custom task-settable environment, the user must inherite
    from this "mixin" class and implement all its abstract methods.

    The task space can be any arbitrarily nested data structure, e.g.:

        Tuple([Tuple([Discrete(3), Discrete(1), Discrete(2)]),
               Discrete(1),
               Tuple([Discrete(2), Discrete(1), Discrete(1),
                      Tuple([Discrete(2), Discrete(1)])])])

                    o------------o----------------o
                    |            |                |
             *-------*-----*    T_7    *-----*----*-----------o
             |       |     |           |     |    |           |
         *---*---*  T_4  *---*       *---*  T_10 T_11     *-------*
         |   |   |       |   |       |   |                |       |
        T_1 T_2 T_3     T_5 T_6     T_8 T_9            *-----*   T_14
                                                       |     |
                                                      T_12  T_13

    .. warning::
        All jiminy-based environments must inherite from `BaseJiminyEnv`, which
        means that one must resort on multiple inheritence to make it
        task-settable on top of that. When doing so, it is absolutely necessary
        that `TaskSettableEnv` appears first in the inheritence list for Python
        Method Resolution Order (MRO) to work properly. Otherwise task
        monitoring will not be added automatically to the extra info `info`.
    """

    task_space: spaces.Tuple
    """Task space of the environment.
    """

    def step(self,
             action: Act
             ) -> Tuple[Obs, SupportsFloat, bool, bool, InfoType]:
        """Monitor how well the agent is performing at the time being for the
        current task.

        It simply stores as extra information under key "task" of `info` a
        tuple `(task, score)` gathering the task being addressed by the agent
        and the associated score of the agent at the end of the current step.

        :param terminated: Whether the episode has reached the terminal state
                           of the MDP at the current step. This flag can be
                           used to compute a specific terminal reward.
        :param info: Dictionary of extra information for monitoring.
        """
        # Call base implementation
        obs, reward, terminated, truncated, info = super().step(action)

        # Add the current score of the agent for the task of the episode
        assert "task" not in info
        info["task"] = (self.get_task(), self.get_score())

        # Return total reward
        return obs, reward, terminated, truncated, info

    # methods to override:
    # ----------------------------

    @abstractmethod
    def set_task(self, task: Task) -> None:
        """Set the task that the agent will have to address for now on.
        """

    @abstractmethod
    def get_task(self) -> Task:
        """Get the task that the agent is currently trying to addressing.
        """

    @abstractmethod
    def get_score(self) -> float:
        """Assess how well the agent is performing at the time being for the
        current task. This score must be standardized between 0.0 and 1.0.
        """


class TaskSchedulingWrapper(gym.Wrapper[Obs, Act, Obs, Act],
                            Generic[Obs, Act]):
    """Randomly sample a new task at the beginning of the every episode based
    on a user-specified probability tree.

    The probability tree associated with the task tree presented in example of
    the documentation of `TaskSettableEnv` should look like this:

        ((P0,
            (P00,
                (P000, P001, P002)),
             P01,
            (P02,
                (P020, P021))),
          P1,
         (P2,
            (P20,
                (P200, P201)),
             P21,
             P22,
            (P23,
                (P230,
                    (P2300, P2301)),
                 P231)))

    .. note::
        The underlying probability tree can be updated dynamically by the user
        at any point in time without restriction. It will be used for drawing
        the task of the next epsiode.
    """
    task_space: spaces.Tuple
    """Task space of the wrapped environment.
    """

    task_list: Tuple[Task, ...]
    """(Static) list of all possible tasks.
    """

    def __init__(self,
                 env: gym.Env[Obs, Act],
                 initial_proba_task_tree: Optional[DataTaskTree[float]] = None
                 ) -> None:
        """
        :param env: Environment deriving from `TaskSettableEnv`.
        :param initial_proba_task_tree: Initial probability tree associated
                                        with the task tree of the environment.
        """
        # Make sure that the based environment is compatible
        env_unwrapped = env.unwrapped
        if not isinstance(env_unwrapped, TaskSettableEnv):
            raise RuntimeError(
                "The base environment `env.unwrapped` must derive from "
                "`gym_jiminy.toolbox.wrappers.TaskSettableEnv`.")

        # Make sure the task space of the environment is supported
        task_space_branches: List[gym.Space] = [env_unwrapped.task_space]
        while task_space_branches:
            space = task_space_branches.pop(0)
            if isinstance(space, spaces.Tuple):
                task_space_branches += space
            elif not isinstance(space, spaces.Discrete) or space.start:
                raise ValueError(
                    "The task space must be a arbitrarily nested structure "
                    "whose branches are `gym.spaces.Tuple` spaces and leaves "
                    "are `gym.spaces.Discrete` spaces starting at 0.")

        # Initialize task space attributes
        self.task_space = env_unwrapped.task_space
        self.task_list = cast(Tuple[Task, ...], tuple(
            (*path, i)
            for path, space in tree.flatten_with_path(self.task_space)
            for i in range(space.n)))

        # Underlying original and flattened probability tree
        self._proba_task_tree: DataTaskTree[float] = []
        self._proba_task_tree_flat: Tuple[float, ...] = ()

        # Call base implementation
        super().__init__(env)

        # Evenly distributed initial task probabilities if not specified
        if initial_proba_task_tree is None:
            initial_proba_task_tree = []
            proba_task_branches: List[
                Tuple[DataTaskTree[float], gym.Space]
                ] = [(initial_proba_task_tree, self.task_space)]
            while proba_task_branches:
                probas, space = proba_task_branches.pop(0)
                assert isinstance(probas, list)
                if isinstance(space, spaces.Discrete):
                    size = int(space.n)
                    probas += (1.0 / size for _ in range(space.n))
                else:
                    assert isinstance(space, spaces.Tuple)
                    size = len(space)
                    for space in space:
                        proba_task_branch: DataTaskTree[float] = []
                        probas.append((1.0 / size, proba_task_branch))
                        proba_task_branches.append((proba_task_branch, space))

        # Define task probabilities
        self.proba_task_tree = initial_proba_task_tree

    @property
    def proba_task_tree(self) -> DataTaskTree[float]:
        """Get the probability tree of the environment.
        """
        return self._proba_task_tree

    @proba_task_tree.setter
    def proba_task_tree(self, proba_task_tree: DataTaskTree[float]) -> None:
        """Update the probability tree of the environment.
        """
        # Make sure that the probability tree is consistent with the task space
        proba_task_branches: List[
            Tuple[DataTaskTree[float], gym.Space]
            ] = [(proba_task_tree, self.task_space)]
        while proba_task_branches:
            proba_task_branch, space = proba_task_branches.pop(0)
            if isinstance(space, spaces.Discrete):
                assert isinstance(proba_task_branch, (tuple, list))
                assert len(proba_task_branch) == space.n
            else:
                assert isinstance(space, spaces.Tuple)
                assert isinstance(proba_task_branch, (tuple, list))
                assert len(proba_task_branch) == len(space)
                for elem_i, space_i in zip(proba_task_branch, space):
                    assert isinstance(elem_i, tuple) and len(elem_i) == 2
                    proba_i, proba_task_branch_i = elem_i
                    assert isinstance(proba_i, float)
                    proba_task_branches.append((proba_task_branch_i, space_i))

        # Compute flattened probability tree
        proba_task_tree_flat: List[float] = []
        for (*task_branch, task_leaf) in self.task_list:
            proba = 1.0
            proba_task_tree_i = proba_task_tree
            for i in task_branch:
                elem_i = proba_task_tree_i[i]
                assert isinstance(elem_i, tuple) and len(elem_i) == 2
                proba_i, proba_task_tree_i = elem_i
                proba *= proba_i
            elem_i = proba_task_tree_i[task_leaf]
            assert isinstance(elem_i, float)
            proba *= elem_i
            proba_task_tree_flat.append(proba)

        # Make sure that it sums to 1.0.
        assert np.abs(1.0 - sum(proba_task_tree_flat)) < 1e-3

        # Update internal buffers
        self._proba_task_tree = proba_task_tree
        self._proba_task_tree_flat = tuple(proba_task_tree_flat)

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Obs, InfoType]:
        # Sample new task
        task_index = np.random.choice(
            len(self.task_list), p=self._proba_task_tree_flat)

        # Set current task
        env_unwrapped = self.unwrapped
        assert isinstance(env_unwrapped, TaskSettableEnv)
        env_unwrapped.set_task(self.task_list[task_index])

        # Reset the environment as usual
        return self.env.reset(seed=seed, options=options)
