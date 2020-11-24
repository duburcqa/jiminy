""" TODO: Write documentation.
"""
from copy import deepcopy
from functools import reduce
from collections import deque
from typing import Tuple, Type, Dict, Sequence, List, Any, Iterator

import numpy as np

import gym

from .utils import _is_breakpoint, zeros, SpaceDictNested
from .pipeline_bases import BasePipelineWrapper


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
                 num_stack: int,
                 **kwargs: Any):
        """
        :param env: Environment to wrap.
        :param nested_fields_list: List of nested observation fields to stack.
                                   Those fields does not have to be leaves. If
                                   not, then every leaves fields from this root
                                   will be stacked.
        :param num_stack: Number of observation frames to partially stack.
        :param kwargs: Extra keyword arguments to allow automatic pipeline
                       wrapper generation.
        """
        # Define helper that will be used to determine the leaf fields to stack
        def _get_branches(root: Any) -> Iterator[List[str]]:
            if isinstance(root, gym.spaces.Dict):
                for field, node in root.spaces.items():
                    if isinstance(node, gym.spaces.Dict):
                        for path in _get_branches(node):
                            yield [field] + path
                    else:
                        yield [field]

        # Backup user arguments
        self.nested_fields_list: List[List[str]] = list(
            map(list, nested_fields_list))  # type: ignore[arg-type]
        self.num_stack = num_stack

        # Initialize base wrapper
        super().__init__(env)  # Do not forward extra arguments, if any

        # Get the leaf fields to stack
        self.leaf_fields_list: List[List[str]] = []
        for fields in self.nested_fields_list:
            root_field = reduce(
                lambda d, key: d[key], fields, self.env.observation_space)
            if isinstance(root_field, gym.spaces.Dict):
                leaf_paths = _get_branches(root_field)
                self.leaf_fields_list += [fields + path for path in leaf_paths]
            else:
                self.leaf_fields_list.append(fields)

        # Compute stacked observation space
        self.observation_space = deepcopy(self.env.observation_space)
        for fields in self.leaf_fields_list:
            root_space = reduce(
                lambda d, key: d[key], fields[:-1], self.observation_space)
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

        # Allocate internal frames buffers
        self._frames: List[deque] = [
            deque(maxlen=self.num_stack)
            for _ in range(len(self.leaf_fields_list))]

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        # Initialize the frames by duplicating the original one
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_space = reduce(
                lambda d, key: d[key], fields, self.env.observation_space)
            for _ in range(self.num_stack):
                frames.append(zeros(leaf_space))

    def observation(self, observation: SpaceDictNested) -> SpaceDictNested:
        """ TODO: Write documentation.
        """
        # Replace nested fields of original observation by the stacked ones
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            root_obs = reduce(lambda d, key: d[key], fields[:-1],
                              observation)
            root_obs[fields[-1]] = np.stack(frames)

        # Return the stacked observation
        return observation

    def compute_observation(self, measure: SpaceDictNested) -> SpaceDictNested:
        """ TODO: Write documentation.
        """
        # Backup the nested observation fields to stack
        for fields, frames in zip(self.leaf_fields_list, self._frames):
            leaf_obs = reduce(lambda d, key: d[key], fields, measure)
            frames.append(leaf_obs.copy())  # Copy to make sure not altered

        # Return the stacked observation
        return self.observation(measure)

    def step(self,
             action: SpaceDictNested
             ) -> Tuple[SpaceDictNested, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        return self.compute_observation(observation), reward, done, info

    def reset(self, **kwargs: Any) -> SpaceDictNested:
        observation = self.env.reset(**kwargs)
        self._setup()
        return self.compute_observation(observation)


class StackedJiminyEnv(BasePipelineWrapper):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 env: gym.Env,
                 skip_frames_ratio: int = 0,
                 **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # Backup some user argument(s)
        self.skip_frames_ratio = skip_frames_ratio

        # Initialize base classes
        super().__init__(env, **kwargs)

        # Instantiate wrapper
        self.wrapper = PartialFrameStack(env, **kwargs)

        # Assertion(s) for type checker
        assert self.env.action_space is not None

        # Define the observation and action spaces
        self.action_space = self.env.action_space
        self.observation_space = self.wrapper.observation_space

        # Initialize some internal buffers
        self.__n_last_stack = 0
        self._action = zeros(self.action_space)
        self._observation = zeros(self.observation_space)

    def _setup(self) -> None:
        # Call base implementation
        super()._setup()

        # Setup wrapper
        self.wrapper._setup()

        # Re-initialize some internal buffer(s)
        # Note that the initial observation is always stored.
        self.__n_last_stack = self.skip_frames_ratio - 1

        # Compute the observe and control update periods
        self.control_dt = self.env.control_dt
        self.observe_dt = self.env.observe_dt

        # Make sure observe update is discrete-time
        if (self.observe_dt <= 0.0):
            raise ValueError(
                "`StackedJiminyEnv` does not support time-continuous update.")

    def compute_observation(self) -> SpaceDictNested:  # type: ignore[override]
        # Get environment observation
        obs = super().compute_observation()

        # Update observed features if necessary
        t = self.simulator.stepper_state.t
        if self.simulator.is_simulation_running and \
                _is_breakpoint(t, self.observe_dt, self._dt_eps):
            self.__n_last_stack += 1
        if self.__n_last_stack == self.skip_frames_ratio:
            self.__n_last_stack = -1
            return self.wrapper.compute_observation(obs)
        else:
            return self.wrapper.observation(obs)


def build_outer_wrapper(env_config: Tuple[
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

    def __init__(self: gym.Wrapper) -> None:
        env = env_class(**env_kwargs)
        super(wrapped_env_class, self).__init__(  # type: ignore[arg-type]
            env, **wrapper_kwargs)

    def __dir__(self: gym.Wrapper) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super(  # type: ignore[arg-type]
            wrapped_env_class, self).__dir__() + self.env.__dir__()

    wrapped_env_class.__init__ = __init__  # type: ignore[misc]
    wrapped_env_class.__dir__ = __dir__  # type: ignore[assignment]

    return wrapped_env_class
