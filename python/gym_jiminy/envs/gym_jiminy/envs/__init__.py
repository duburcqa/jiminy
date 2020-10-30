from gym.envs.registration import register

from .cartpole import CartPoleJiminyEnv
from .acrobot import AcrobotJiminyEnv, AcrobotJiminyGoalEnv
from .anymal import ANYmalJiminyEnv, ANYmalPDControlJiminyEnv
from .atlas import AtlasJiminyEnv, AtlasPDControlJiminyEnv
from .spotmicro import SpotmicroJiminyEnv


__all__ = [
    'CartPoleJiminyEnv',
    'AcrobotJiminyEnv',
    'AcrobotJiminyGoalEnv',
    'ANYmalJiminyEnv',
    'ANYmalPDControlJiminyEnv',
    'AtlasJiminyEnv',
    'AtlasPDControlJiminyEnv',
    'SpotmicroJiminyEnv'
]

register(
    id='jiminy-cartpole-v0',
    entry_point='gym_jiminy.envs:CartPoleJiminyEnv',
    reward_threshold=475.0,
    max_episode_steps=500
)
register(
    id='jiminy-acrobot-v0',
    entry_point='gym_jiminy.envs:AcrobotJiminyEnv',
    reward_threshold=-100.0,
    max_episode_steps=500
)
register(
    id='jiminy-anymal-v0',
    entry_point='gym_jiminy.envs:ANYmalJiminyEnv'
)
register(
    id='jiminy-anymal-pid-v0',
    entry_point='gym_jiminy.envs:ANYmalPDControlJiminyEnv'
)
register(
    id='jiminy-atlas-v0',
    entry_point='gym_jiminy.envs:AtlasJiminyEnv'
)
register(
    id='jiminy-atlas-pid-v0',
    entry_point='gym_jiminy.envs:AtlasPDControlJiminyEnv'
)
register(
    id='jiminy-spotmicro-v0',
    entry_point='gym_jiminy.envs:SpotmicroJiminyEnv'
)
