from gymnasium.envs.registration import register

from .cartpole import CartPoleJiminyEnv
from .acrobot import AcrobotJiminyEnv
from .ant import AntEnv
from .cassie import CassieJiminyEnv, CassiePDControlJiminyEnv
from .digit import DigitJiminyEnv, DigitPDControlJiminyEnv
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
    'DigitJiminyEnv',
    'DigitPDControlJiminyEnv',
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
    max_episode_steps=500,
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='acrobot',
    entry_point='gym_jiminy.envs:AcrobotJiminyEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='ant',
    entry_point='gym_jiminy.envs:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='cassie',
    entry_point='gym_jiminy.envs:CassieJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='cassie-pid',
    entry_point='gym_jiminy.envs:CassiePDControlJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='digit',
    entry_point='gym_jiminy.envs:DigitJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='digit-pid',
    entry_point='gym_jiminy.envs:DigitPDControlJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='anymal',
    entry_point='gym_jiminy.envs:ANYmalJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='anymal-pid',
    entry_point='gym_jiminy.envs:ANYmalPDControlJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='atlas',
    entry_point='gym_jiminy.envs:AtlasJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='atlas-reduced',
    entry_point='gym_jiminy.envs:AtlasReducedJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='atlas-pid',
    entry_point='gym_jiminy.envs:AtlasPDControlJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
register(
    id='atlas-reduced-pid',
    entry_point='gym_jiminy.envs:AtlasReducedPDControlJiminyEnv',
    order_enforce=False,
    disable_env_checker=True
)
