import numpy as np
from gym import ObservationWrapper, spaces

from functools import reduce
from operator import __mul__


def flatten(space, x=None):
    def _flatten_bounds(space, bounds_type):
        if isinstance(space, spaces.Box):
            if bounds_type == 'high':
                return np.asarray(space.high, dtype=np.float32).flatten()
            else:
                return np.asarray(space.low, dtype=np.float32).flatten()
        elif isinstance(space, spaces.Discrete):
            if bounds_type == 'high':
                return np.one(space.n, dtype=np.float32)
            else:
                return np.zeros(space.n, dtype=np.float32)
        elif isinstance(space, spaces.Tuple):
            return np.concatenate([_flatten_bounds(s, bounds_type)
                                   for s in space.spaces])
        elif isinstance(space, spaces.Dict):
            return np.concatenate([_flatten_bounds(s, bounds_type)
                                   for s in space.spaces.values()])
        elif isinstance(space, spaces.MultiBinary):
            if bounds_type == 'high':
                return np.one(space.n, dtype=np.float32)
            else:
                return np.zeros(space.n, dtype=np.float32)
        elif isinstance(space, spaces.MultiDiscrete):
            if bounds_type == 'high':
                return np.one(reduce(__mul__, space.nvec), dtype=np.float32)
            else:
                return np.zeros(reduce(__mul__, space.nvec), dtype=np.float32)
        else:
            raise NotImplementedError
    if x is None:
        return spaces.Box(low=_flatten_bounds(space, 'low'),
                        high=_flatten_bounds(space, 'high'),
                        dtype=np.float32)
    else:
        return spaces.flatten(space, x)


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten(self.env.observation_space) # Note that it does not preserve dtype

    def observation(self, observation):
        return flatten(self.env.observation_space, observation)
