""" TODO: Write documentation.
"""
from abc import abstractmethod
from typing import (
    Any, Optional, List, Tuple, Sequence, Union, Generic, SupportsFloat,
    TypeVar, cast)

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from jiminy_py import tree
from gym_jiminy.common.bases import (
    BaseObs, Obs, Act, InfoType, EngineObsType, InterfaceJiminyEnv,
    BasePipelineWrapper)
from gym_jiminy.common.bases.pipeline import _merge_base_env_with_wrapper
from gym_jiminy.common.utils import DataNested


_Task = Union[np.int64, Tuple["_Task", ...]]
Task = TypeVar('Task', bound=_Task)
ProbaTaskTree = Sequence[Union[float, Tuple[float, "ProbaTaskTree"]]]
TaskIndex = int
TaskPath = Tuple[int, ...]


class BaseTaskSettableWrapper(BasePipelineWrapper[Obs, Act, BaseObs, Act],
                              Generic[Obs, Act, Task, BaseObs]):
    """Wrapper extending a base jiminy environment to make it task-settable.

    A new task will be randomly sampled at the beginning of the every episode,
    based on a user-specified probability tree `proba_task_tree`. Although one
    can manually call `set_task` to forceable set the current task at any point
    in time, this approach is not recommended unless you know exactly what you
    are doing since a new task will be selected at reset anyway.

    The task space must be some arbitrarily nested data structure whose
    branches are `gym.spaces.Tuple` spaces and leaves are `gym.spaces.Discrete`
    spaces starting at 0, e.g.:

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

    While its corresponding probability tree should look like this:

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

    Where the probability of each task is the product of all the intermediary
    probability, e.g.:

        P(T_1) = P0 * P00 * P000
        ...
        P(T_7) = P1
        ...
        P(T_13) = P2 * P23 * P230 * P2301

    Task are fully specified either by its flattened index or its path, e.g.:

        T_1  := (index =  0, path = (0, 0, 0))
        ...
        T_7  := (index =  7, path = (1,))
        ...
        T_13 := (index = 13, path = (2, 3, 0, 1))

    .. note::
        The underlying probability tree can be updated dynamically by the user
        at any point in time without restriction. It will have no effect until
        the beginning of the next episode.
    """

    task_space: gym.Space[Task]
    """Original task space of the environment.
    """

    def __init__(self,
                 env: InterfaceJiminyEnv[BaseObs, Act],
                 *,
                 initial_proba_task_tree: Optional[ProbaTaskTree] = None,
                 augment_observation: bool = True
                 ) -> None:
        """
        :param env: Environment to extend, eventually already wrapped.
        :param initial_proba_task_tree: Initial probability tree associated
                                        with the task space of the environment.
        :param augment_observation: Whether to add the flattened index of the
                                    current task to the observation of the
                                    environment.
                                    Optional: `True` by default.
        """
        # Backup user argument(s)
        self.augment_observation = augment_observation

        # Initialize base class
        super().__init__(env)

        # Pre-compute the list of all possible tasks as characterized by their
        # respective path (ordered by flattened index) for efficiency.
        self._task_paths = cast(Tuple[TaskPath, ...], tuple(
            (*path, i)
            for path, space in tree.flatten_with_path(self.task_space)
            for i in range(space.n)))
        self.num_tasks = len(self._task_paths)

        # Underlying original and flattened probability tree
        self._proba_task_tree: ProbaTaskTree = []
        self._proba_task_tree_flat: Tuple[float, ...] = ()

        # Evenly distributed initial task probabilities if not specified
        if initial_proba_task_tree is None:
            initial_proba_task_tree = []
            proba_task_branches: List[
                Tuple[ProbaTaskTree, gym.Space]
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
                        proba_task_branch: ProbaTaskTree = []
                        probas.append((1.0 / size, proba_task_branch))
                        proba_task_branches.append((proba_task_branch, space))

        # Initialize task probabilities
        self.proba_task_tree = initial_proba_task_tree

        # Bind action of the base environment
        assert self.action_space.contains(env.action)
        self.action = env.action

        # Allocate memory for the task index
        self.task_index = np.array(-1, dtype=np.int64)

        # Initialize the observation
        self.observation = cast(Obs, _merge_base_env_with_wrapper(
            "task",
            self.env.observation,
            self.task_index if self.augment_observation else None,
            None,
            None))

        # Enable direct forwarding by default for efficiency
        methods_names = ["compute_command"]
        if self.augment_observation and self.num_tasks:
            methods_names.append("refresh_observation")
        for method_name in methods_names:
            method_orig = getattr(BaseTaskSettableWrapper, method_name)
            method = getattr(type(self), method_name)
            if method_orig is method:
                self.__dict__[method_name] = getattr(self.env, method_name)

    @property
    def proba_task_tree(self) -> ProbaTaskTree:
        """Get the task probability tree of the environment.
        """
        return self._proba_task_tree

    @proba_task_tree.setter
    def proba_task_tree(self, proba_task_tree: ProbaTaskTree) -> None:
        """Update the task probability tree of the environment.
        """
        # Make sure that the probability tree is consistent with the task space
        proba_task_branches: List[
            Tuple[ProbaTaskTree, gym.Space]
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
                    assert isinstance(elem_i, tuple)
                    proba_i, proba_task_branch_i = elem_i
                    assert isinstance(proba_i, float)
                    proba_task_branches.append((proba_task_branch_i, space_i))

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

        # Make sure that it sums to 1.0.
        assert np.abs(1.0 - sum(proba_task_tree_flat)) < 1e-3

        # Update internal buffers
        self._proba_task_tree = proba_task_tree
        self._proba_task_tree_flat = tuple(proba_task_tree_flat)

    def _initialize_action_space(self) -> None:
        """Configure the action space.

        It simply copies the action space of the wrapped environment.
        """
        self.action_space = self.env.action_space

    def _initialize_observation_space(self) -> None:
        """Configure the observation space.
        """
        # Initialize the task space.
        # Note that this cannot be done in `__init__` because it would be
        # either too late or too early in the construction of the object.
        # This is not a big deal for the end-user since this method is not
        # supposed to be overloaded anyway.
        self._initialize_task_space()

        # Make sure the task space is valid
        num_tasks = 0
        task_space_branches: List[gym.Space] = [self.task_space]
        while task_space_branches:
            space = task_space_branches.pop(0)
            if isinstance(space, spaces.Tuple):
                task_space_branches += space
            elif not isinstance(space, spaces.Discrete) or space.start:
                raise ValueError(
                    "The task space must be a arbitrarily nested structure "
                    "whose branches are `gym.spaces.Tuple` spaces and leaves "
                    "are `gym.spaces.Discrete` spaces starting at 0.")
            else:
                num_tasks += int(space.n)

        # Get the base observation space from the wrapped environment
        observation_space: gym.Space[Any] = self.env.observation_space

        # Aggregate the task space with the base observation if requested
        if self.augment_observation and num_tasks:
            task_index_space = spaces.Discrete(num_tasks)
            observation_space = _merge_base_env_with_wrapper(
                "task",
                observation_space,
                task_index_space,  # type: ignore[arg-type]
                None,
                None)

        self.observation_space = cast(gym.Space[Obs], observation_space)

    def _setup(self) -> None:
        """Configure the wrapper.

        In addition to calling the base implementation, it sets the observe
        and control update period.
        """
        # Call base implementation
        super()._setup()

        # Copy observe and control update periods from wrapped environment
        self.observe_dt = self.env.observe_dt
        self.control_dt = self.env.control_dt

        # Sample a new task if necessary
        if self.num_tasks:
            self.task_index[()] = self.np_random.choice(
                self.num_tasks, p=self._proba_task_tree_flat)
            self.set_task(int(self.task_index))

    def refresh_observation(self, measurement: EngineObsType) -> None:
        """Compute high-level features based on the current wrapped
        environment's observation.

        It simply forwards the observation computed by the wrapped environment
        without any processing.

        :param measurement: Low-level measure from the environment to process
                            to get higher-level observation.
        """
        # Refresh environment observation
        self.env.refresh_observation(measurement)

    def compute_command(self, action: Act, command: np.ndarray) -> None:
        """Compute the motors efforts to apply on the robot.

        It simply forwards the command computed by the wrapped environment
        without any processing.

        :param action: High-level target to achieve by means of the command.
        :param command: Lower-level command to updated in-place.
        """
        self.env.compute_command(action, command)

    def step(self,  # type: ignore[override]
             action: Act
             ) -> Tuple[DataNested, SupportsFloat, bool, bool, InfoType]:
        """Run a simulation step for a given action.

        This method monitors the performance of the agent for the task at hand
        at the end of the episode. This information is stored under key "task"
        of `info` a tuple `(task, score)`.

        :param terminated: Whether the episode has reached the terminal state
                           of the MDP at the current step. This flag can be
                           used to compute a specific terminal reward.
        :param info: Dictionary of extra information for monitoring.
        """
        # Call base implementation
        obs, reward, terminated, truncated, info = super().step(action)

        # FIXME: Add the score of the agent at the end of the episode for the
        # task at hand if any, otherwise just move on.
        # Note that, at this point, the flags `terminated` and `truncated` have
        # been evaluated for the top-most layer of the pipeline. As a result,
        # the score is guarantee to be added at the end of the episode as long
        # as no additional termination condition is triggered by some higher-
        # level classical wrappers deriving from `gym.Wrapper`. Unfortunately,
        # scenario is quite common, especially via `gym.wrappers.TimeLimit`.
        # Because of this limitation, it is preferrable to store the score at
        # every step.
        if self.num_tasks:  # and (terminated or truncated):
            assert "task" not in info
            score = float(self.get_score())
            assert 0.0 <= score <= 1.0
            info["task"] = (int(self.task_index), score)

        # Return total reward
        return obs, reward, terminated, truncated, info

    # methods to override:
    # ----------------------------

    @abstractmethod
    def _initialize_task_space(self) -> None:
        """Configure the original hierarchical task space.
        """

    @abstractmethod
    def set_task(self, task_index: int) -> None:
        """Set the task that the agent will have to address from now on.
        """

    @abstractmethod
    def get_score(self) -> float:
        """Assess how well the agent is performing so far for the current task.

        .. warning::
            This score must be standardized between 0.0 and 1.0.
        """
