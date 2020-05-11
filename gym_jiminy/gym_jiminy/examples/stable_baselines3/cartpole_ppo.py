import os
import time

import gym
from tensorboard.program import TensorBoard

from stable_baselines3.ppo import MlpPolicy, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


### Create a multiprocess environment
n_thread = 4
env = SubprocVecEnv(
    [lambda: gym.make("gym_jiminy:jiminy-cartpole-v0", continuous=False)
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

# PPO config
config = dict(
    n_steps=128,
    batch_size=128,
    learning_rate=0.001,
    n_epochs=8,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=float('inf'),
    max_grad_norm=0.5 #float('inf')
)

# Create the learning agent according to the chosen algorithm
agent = PPO(
    MlpPolicy, env, **config,
    tensorboard_log=tensorboard_data_path,
    verbose=True
)

# Load an agent if desired
# agent = PPO2.load("cartpole_ppo2_baseline.pkl")

# Run the learning process
agent.learn(
    total_timesteps=400000,
    log_interval=5,
    reset_num_timesteps=False
)

# Save the agent if desired
# agent.save("cartpole_ppo2_baseline.pkl")

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
