import os
import time

import gym

from gym.wrappers import FlattenDictWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

import jiminy_py
from gym_jiminy.common import SubprocVecEnvLock

### Create a multiprocess environment
nb_cpu = 4
env = SubprocVecEnvLock([lambda: gym.make("gym_jiminy:jiminy-acrobot-v0") for _ in range(nb_cpu)])

### Create the model or load one

# Set the Tensorboard path
tensorboard_data_path = os.path.dirname(os.path.realpath(__file__))

# Define a custom linear scheduler for the learning rate
class LinearSchedule(object):
    def __init__(self, initial_p, final_p):
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, fraction):
        return self.final_p - fraction * (self.final_p - self.initial_p)

learning_rate_scheduler = LinearSchedule(1e-3, 1e-5)

# Create the 'model' according to the chosen algorithm
model = PPO2(MlpPolicy, env,
             noptepochs=8, learning_rate=learning_rate_scheduler.value,
             gamma=0.994,
             tensorboard_log=tensorboard_data_path, verbose=1)

# Load a model if desired
# model = PPO2.load("acrobot_ppo2_baseline.pkl")

# Run the learning process
model.learn(total_timesteps=1200000,
            log_interval=5,
            reset_num_timesteps=False)

# Save the model if desired
# model.save("acrobot_ppo2_baseline.pkl")

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
