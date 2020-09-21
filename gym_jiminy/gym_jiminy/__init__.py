import jiminy_py # Preload jiminy
from gym.envs.registration import register


register(
    id='jiminy-cartpole-v0',
    entry_point='gym_jiminy.envs:CartPoleJiminyEnv',
    reward_threshold=10000.0
)
register(
    id='jiminy-acrobot-v0',
    entry_point='gym_jiminy.envs:AcrobotJiminyEnv',
    max_episode_steps=12000,
    reward_threshold=-3000.0
)

