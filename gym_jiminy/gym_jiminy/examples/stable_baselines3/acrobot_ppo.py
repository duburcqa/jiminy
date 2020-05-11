import os
import time

import gym
from torch import nn
from tensorboard.program import TensorBoard

from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import PPOPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv


### Create a multiprocess environment
n_thread = 8
env = SubprocVecEnv(
    [lambda: gym.make("gym_jiminy:jiminy-acrobot-v0", continuous=True)
     for _ in range(n_thread)],
    start_method='fork'
)

### Configure Tensorboard
tensorboard_data_path = os.path.dirname(os.path.realpath(__file__))
if not 'tb' in locals().keys():
    tb = TensorBoard()
    tb.configure(host="0.0.0.0", logdir=tensorboard_data_path)
    url = tb.launch()
    print(f"Started Tensorboard {url} at {tensorboard_data_path}...")

### Create the agent or load one

# Define a custom MLP policy with two hidden layers of size 64
class CustomPolicy(PPOPolicy):
    # Necessary to avoid having to specify the policy when loading a model
    __module__ = None

    def __init__(self, *args, **_kwargs):
        super().__init__(*args, **_kwargs,
                         net_arch=[dict(pi=[64, 64],
                                        vf=[64, 64])],
                         activation_fn=nn.Tanh)

# Define a custom linear scheduler for the learning rate
class LinearSchedule(object):
    def __init__(self, initial_p, final_p):
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, fraction):
        return self.final_p - fraction * (self.final_p - self.initial_p)

learning_rate_scheduler = LinearSchedule(1.0e-3, 1.0e-5)

# PPO config
config = dict(
    n_steps=128,
    batch_size=128,
    learning_rate=learning_rate_scheduler.value,
    n_epochs=8,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=float('inf'),
    max_grad_norm=0.5 #float('inf')
)

# Create the 'agent' according to the chosen algorithm
agent = PPO(
    CustomPolicy, env, **config,
    tensorboard_log=tensorboard_data_path,
    verbose=True
)

# Load a agent if desired
# agent = PPO2.load("acrobot_ppo2_baseline.pkl")

# Run the learning process
agent.learn(
    total_timesteps=1200000,
    log_interval=5,
    reset_num_timesteps=False
)

# Save the agent if desired
# agent.save("acrobot_ppo2_baseline.pkl")

### Enjoy a trained agent

# duration of the simulations in seconds
t_end = 20.0

# Get the time step of Jiminy
env.remotes[0].send(('get_attr','dt'))
dt = env.remotes[0].recv()

# Run the simulation in real-time
obs = env.reset()
for _ in range(int(t_end/dt)):
    action, _states = agent.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.remotes[0].send(('render',((), {'mode': 'rgb_array'})))
    env.remotes[0].recv()
    time.sleep(dt)
