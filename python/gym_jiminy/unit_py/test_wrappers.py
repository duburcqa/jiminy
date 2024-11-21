# mypy: disable-error-code="no-untyped-def, var-annotated"
""" TODO: Write documentation
"""
import unittest
from functools import reduce, partial

import numpy as np

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
                partial(NormalizeObservation, ignore_unbounded=True),
                NormalizeAction,
                partial(FlattenObservation, dtype=np.float32),
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
