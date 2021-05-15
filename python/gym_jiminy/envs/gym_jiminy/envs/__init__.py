from gym.envs.registration import register

from .cartpole import CartPoleJiminyEnv
from .acrobot import AcrobotJiminyEnv, AcrobotJiminyGoalEnv
from .ant import AntEnv
from .spotmicro import SpotmicroJiminyEnv
from .cassie import CassieJiminyEnv, CassiePDControlJiminyEnv
from .anymal import ANYmalJiminyEnv, ANYmalPDControlJiminyEnv
from .atlas import AtlasJiminyEnv, AtlasPDControlJiminyEnv


__all__ = [
    'CartPoleJiminyEnv',
    'AcrobotJiminyEnv',
    'AcrobotJiminyGoalEnv',
    'AntEnv',
    'SpotmicroJiminyEnv',
    'CassieJiminyEnv',
    'CassiePDControlJiminyEnv',
    'ANYmalJiminyEnv',
    'ANYmalPDControlJiminyEnv',
    'AtlasJiminyEnv',
    'AtlasPDControlJiminyEnv'
]

register(
    id='cartpole-v0',
    entry_point='gym_jiminy.envs:CartPoleJiminyEnv',
    reward_threshold=475.0,
    max_episode_steps=500
)
register(
    id='acrobot-v0',
    entry_point='gym_jiminy.envs:AcrobotJiminyEnv',
    reward_threshold=-100.0,
    max_episode_steps=500
)
register(
    id='ant-v0',
    entry_point='gym_jiminy.envs:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0
)
register(
    id='spotmicro-v0',
    entry_point='gym_jiminy.envs:SpotmicroJiminyEnv'
)
register(
    id='cassie-v0',
    entry_point='gym_jiminy.envs:CassieJiminyEnv'
)
register(
    id='cassie-pid-v0',
    entry_point='gym_jiminy.envs:CassiePDControlJiminyEnv'
)
register(
    id='anymal-v0',
    entry_point='gym_jiminy.envs:ANYmalJiminyEnv'
)
register(
    id='anymal-pid-v0',
    entry_point='gym_jiminy.envs:ANYmalPDControlJiminyEnv'
)
register(
    id='atlas-v0',
    entry_point='gym_jiminy.envs:AtlasJiminyEnv'
)
register(
    id='atlas-pid-v0',
    entry_point='gym_jiminy.envs:AtlasPDControlJiminyEnv'
)
