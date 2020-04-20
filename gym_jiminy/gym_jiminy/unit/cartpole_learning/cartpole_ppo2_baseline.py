import os
import time

import gym
from gym_jiminy.common import SubprocVecEnvLock

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


### Create a multiprocess environment
n_thread = 4
env = SubprocVecEnvLock([lambda: gym.make("gym_jiminy:jiminy-cartpole-v0") for _ in range(n_thread)])

### Create the model or load one

# Set the Tensorboard path
tensorboard_data_path = os.path.dirname(os.path.realpath(__file__))

# Create the 'model' according to the chosen algorithm
model = PPO2(MlpPolicy, env,
             noptepochs=8, learning_rate=0.001,
             tensorboard_log=tensorboard_data_path)

# Load a model if desired
# model = PPO2.load("cartpole_ppo2_baseline.pkl")

# Run the learning process
model.learn(total_timesteps=400000,
            log_interval=5,
            reset_num_timesteps=False)

# Save the model if desired
# model.save("cartpole_ppo2_baseline.pkl")

### Enjoy a trained agent

# duration of the simulations in seconds
t_end = 20

# Get the time step of Jiminy
env.remotes[0].send(('get_attr','dt'))
dt = env.remotes[0].recv()

# Run the simulation in real-time
obs = env.reset()
for _ in range(int(t_end/dt)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.remotes[0].send(('render',((), {'mode': 'rgb_array'})))
    env.remotes[0].recv()
    time.sleep(dt)
