""" TODO: Write documentation
"""
import unittest
from functools import reduce

from gym_jiminy.envs import AtlasPDControlJiminyEnv
from gym_jiminy.common.wrappers import (
    FilterObservation,
    NormalizeAction,
    NormalizeObservation,
    FlattenAction,
    FlattenObservation,
)


class Wrappers(unittest.TestCase):
    """ TODO: Write documentation
    """
    def test_filter_normalize_flatten_wrappers(self):
        env = reduce(
            lambda env, wrapper: wrapper(env),
            (
                NormalizeObservation,
                NormalizeAction,
                FlattenObservation,
                FlattenAction
            ),
            FilterObservation(
                AtlasPDControlJiminyEnv(debug=False),
                nested_filter_keys=(
                    ("states", "pd_controller"),
                    ("measurements", "EncoderSensor"),
                    ("features", "mahony_filter"),
                ),
            ),
        )
        env.reset()
        env.step(env.action)
