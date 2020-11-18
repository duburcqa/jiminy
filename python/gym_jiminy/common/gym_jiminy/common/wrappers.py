from copy import deepcopy
from functools import reduce
from collections import deque
from typing import Tuple, Type, Dict, Sequence, List, Optional, Any, Iterator

import numpy as np

import gym

from .utils import SpaceDictRecursive


class PartialFrameStack(gym.Wrapper):
    """Observation wrapper that stacks parts of the observations in a rolling
    manner.

    .. note::
        The observation space must be `gym.spaces.Dict`, while, ultimately,
        stacked leaf fields must be `gym.spaces.Box`.
    """
    def __init__(self,
                 env: gym.Env,
                 nested_fields_list: Sequence[Sequence[str]],
                 num_stack: int):
        """
        :param env: Environment to wrap.
        :param nested_fields_list: List of nested observation fields to stack.
                                   Those fields does not have to be leaves. If
                                   not, then every leaves fields from this root
                                   will be stacked.
        :param num_stack: Number of observation frames to partially stack.
        """
        # Backup user arguments
        self.nested_fields_list: List[List[str]] = list(
            map(list, nested_fields_list))  # type: ignore[arg-type]
        self.leaf_fields_list: List[List[str]] = []
        self.num_stack = num_stack

        # Define some internal buffers
        self._observation: Optional[SpaceDictRecursive] = None

        # Initialize base wrapper
        super().__init__(env)

        # Create internal buffers
        self._frames: List[deque] = []

    def get_observation(self) -> SpaceDictRecursive:
        assert (self._observation is not None and
                all(len(frames) == self.num_stack for frames in self._frames))

        # Replace nested fields of original observation by the stacked ones
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            root_obs = reduce(lambda d, key: d[key], fields[:-1],
                              self._observation)
            root_obs[fields[-1]] = np.stack(frames)

        return self._observation

    def step(self,
             action: SpaceDictRecursive
             ) -> Tuple[SpaceDictRecursive, float, bool, Dict[str, Any]]:
        self._observation, reward, done, info = self.env.step(action)

        # Backup the nested observation fields to stack
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(lambda d, key: d[key], fields, self._observation)
            frames.append(leaf_obs)

        return self.get_observation(), reward, done, info

    def reset(self, **kwargs: Any) -> SpaceDictRecursive:
        def _get_branches(root: Any) -> Iterator[List[str]]:
            if isinstance(root, dict):
                for field, node in root.items():
                    if isinstance(node, dict):
                        for path in _get_branches(node):
                            yield [field] + path
                    else:
                        yield [field]

        self._observation = self.env.reset(**kwargs)

        # Determine leaf fields to stack
        self.leaf_fields_list = []
        for fields in self.nested_fields_list:
            root_field = reduce(
                lambda d, key: d[key], fields, self._observation)
            if isinstance(root_field, dict):
                leaf_paths = _get_branches(root_field)
                self.leaf_fields_list += [fields + path for path in leaf_paths]
            else:
                self.leaf_fields_list.append(fields)

        # Compute stacked observation space
        self.observation_space = deepcopy(self.env.observation_space)
        for fields in self.leaf_fields_list:
            root_space = reduce(lambda d, key: d[key], fields[:-1],
                                self.observation_space)
            space = root_space[fields[-1]]
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(
                    "Stacked leaf fields must be associated with "
                    "`gym.spaces.Box` space")
            low = np.repeat(
                space.low[np.newaxis, ...], self.num_stack, axis=0)
            high = np.repeat(
                space.high[np.newaxis, ...], self.num_stack, axis=0)
            root_space.spaces[fields[-1]] = gym.spaces.Box(
                low=low, high=high, dtype=space.dtype)

        # Allocate the frames buffers
        self._frames = [deque(maxlen=self.num_stack)
                        for _ in range(len(self.leaf_fields_list))]

        # Initialize the frames by duplicating the original one
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(lambda d, key: d[key], fields, self._observation)
            for _ in range(self.num_stack):
                frames.append(leaf_obs)

        return self.get_observation()


def build_wrapper(env_config: Tuple[
                      Type[gym.Env],
                      Dict[str, Any]],
                  wrapper_config: Tuple[
                      Type[gym.Wrapper],
                      Dict[str, Any]]) -> Type[gym.Env]:
    """Generate a class inheriting from `gym.Wrapper` wrapping a given type of
    environment.

    .. warning::
        The generated class takes no input argument. Therefore it will not be
        possible to set the arguments of the constructor of the environment and
        wrapper after generation.

    :param env_config:
        Configuration of the environment, as a tuple:

          - [0] Environment class type.
          - [1] Keyword arguments to forward to the constructor of the wrapped
                environment.

    :param wrapper_config:
        Configuration of the wrapper, as a tuple:

          - [0] Wrapper class type to apply on the environment.
          - [1] Keyword arguments to forward to the constructor of the wrapper,
                'env' itself excluded.
    """
    # pylint: disable-all

    # Extract user arguments
    env_class, env_kwargs = env_config
    wrapper_class, wrapper_kwargs = wrapper_config

    wrapped_env_class = type(
        f"{wrapper_class.__name__}Env",
        (wrapper_class,),
        {})

    def __init__(self: wrapped_env_class) -> None:  # type: ignore[valid-type]
        env = env_class(**env_kwargs)
        super(wrapped_env_class, self).__init__(  # type: ignore[arg-type]
            env, **wrapper_kwargs)

    wrapped_env_class.__init__ = __init__  # type: ignore[misc]

    return wrapped_env_class
