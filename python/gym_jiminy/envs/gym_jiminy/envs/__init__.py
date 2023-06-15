from gymnasium.envs.registration import register

from .cartpole import CartPoleJiminyEnv
from .acrobot import AcrobotJiminyEnv
from .ant import AntEnv
from .cassie import CassieJiminyEnv, CassiePDControlJiminyEnv
from .anymal import ANYmalJiminyEnv, ANYmalPDControlJiminyEnv
from .atlas import (AtlasJiminyEnv,
                    AtlasReducedJiminyEnv,
                    AtlasPDControlJiminyEnv,
                    AtlasReducedPDControlJiminyEnv)


__all__ = [
    'CartPoleJiminyEnv',
    'AcrobotJiminyEnv',
    'AntEnv',
    'CassieJiminyEnv',
    'CassiePDControlJiminyEnv',
    'ANYmalJiminyEnv',
    'ANYmalPDControlJiminyEnv',
    'AtlasJiminyEnv',
    'AtlasReducedJiminyEnv',
    'AtlasPDControlJiminyEnv',
    'AtlasReducedPDControlJiminyEnv'
]

register(
    id='cartpole',
    entry_point='gym_jiminy.envs:CartPoleJiminyEnv',
    reward_threshold=475.0,
    max_episode_steps=500
)
register(
    id='acrobot',
    entry_point='gym_jiminy.envs:AcrobotJiminyEnv',
    reward_threshold=-100.0,
    max_episode_steps=500
)
register(
    id='ant',
    entry_point='gym_jiminy.envs:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)
register(
    id='cassie',
    entry_point='gym_jiminy.envs:CassieJiminyEnv'
)
register(
    id='cassie-pid',
    entry_point='gym_jiminy.envs:CassiePDControlJiminyEnv'
)
register(
    id='anymal',
    entry_point='gym_jiminy.envs:ANYmalJiminyEnv'
)
register(
    id='anymal-pid',
    entry_point='gym_jiminy.envs:ANYmalPDControlJiminyEnv'
)
register(
    id='atlas',
    entry_point='gym_jiminy.envs:AtlasJiminyEnv'
)
register(
    id='atlas-reduced',
    entry_point='gym_jiminy.envs:AtlasReducedJiminyEnv'
)
register(
    id='atlas-pid',
    entry_point='gym_jiminy.envs:AtlasPDControlJiminyEnv'
)
register(
    id='atlas-reduced-pid',
    entry_point='gym_jiminy.envs:AtlasReducedPDControlJiminyEnv'
)
