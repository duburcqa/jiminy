import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import time
from functools import reduce, partial

from gym_jiminy.envs import AtlasPDControlJiminyEnv
from gym_jiminy.common.wrappers import (
    FilterObservation,
    NormalizeAction,
    NormalizeObservation,
    FlattenAction,
    FlattenObservation)

env = reduce(
    lambda env, wrapper: wrapper(env), (
        FlattenObservation,
        FlattenAction,
        partial(NormalizeObservation, ignore_unbounded=True),
        NormalizeAction
    ), FilterObservation(
        AtlasPDControlJiminyEnv(
            # std_ratio={'disturbance': 4.0},
            debug=False
        ),
        nested_filter_keys=(
            # 't',
            # ('states', 'agent', 'q'),
            # ('states', 'agent', 'v'),
            ("states", "pd_controller"),
            # ('states', 'mahony_filter'),
            # ('measurements', 'ImuSensor'),
            # ('measurements', 'ForceSensor'),
            ("measurements", "EncoderSensor"),
            ("features", "mahony_filter"),
        ),
    ),
)

# Run in 30.7s on jiminy==1.8.11 (29.7s with PGO, 28.4s with eigen-dev)
env.reset()
action = env.action
time_start = time.time()
for _ in range(100000):
    env.step(action)
print("time elapsed:", time.time() - time_start)
