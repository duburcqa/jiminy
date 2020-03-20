import os
import time

import gym

from stable_baselines.her import GoalSelectionStrategy
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import HER, DQN, SAC, TD3, #DDPG

import jiminy_py

# Select the model class
model_class = SAC

# Create a single-process environment
env = gym.make("gym_jiminy:jiminy-acrobot-v0",
               continuous=model_class in [SAC, TD3],#, DDPG],
               enableGoalEnv=True)

### Create the model or load one

# Define a custom MLP policy with two hidden layers of size 64
class CustomPolicy(FeedForwardPolicy):
    # Necessary to avoid having to specify the policy when loading a model
    __module__ = None

    def __init__(self, *args, **_kwargs):
        super(CustomPolicy, self).__init__(*args, **_kwargs,
                                           layers=[64, 64],
                                           feature_extraction="mlp")

# Define a custom linear scheduler for the learning rate
class LinearSchedule(object):
    def __init__(self, initial_p, final_p):
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, fraction):
        return self.final_p - fraction * (self.final_p - self.initial_p)

learning_rate_scheduler = LinearSchedule(1e-3, 1e-5)

# Define the number of artificial transitions generated per real transition
# according to the chosen goal selection strategy
n_sampled_goal = 4

# Set the Tensorboard path
tensorboard_data_path = os.path.dirname(os.path.realpath(__file__))

# Create the 'model' according to the chosen algorithm
model = HER(CustomPolicy, env, model_class,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE, buffer_size=1000000,
            learning_rate=learning_rate_scheduler.value, target_update_interval=256,
            batch_size=64, train_freq=1, random_exploration=0.05, gradient_steps=1,
            learning_starts=4096, gamma=0.994,
            tensorboard_log=tensorboard_data_path, verbose=1)

# Load a model if desired
# model = HER.load("acrobot_her_baseline.pkl", env=env, policy=CustomPolicy)

### Run the learning process
model.learn(total_timesteps=400000,
            log_interval=1,
            reset_num_timesteps=False)

# Save the model if desired
# model.save("acrobot_her_baseline.pkl")

### Enjoy a trained agent

# duration of the simulations in seconds
t_end = 20

# Desired goal
desired_goal = 0.95 * env.env._tipPosZMax # As difficult as possible

# Run the simulation in real-time
env.reset()
env.env.goal[0] = desired_goal
obs = env.env._get_obs()
episode_reward = 0
for _ in range(int(t_end/env.dt)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(env.dt)

    episode_reward += reward
    if done or info.get('is_success', False):
        print("Reward:", episode_reward,
              "Success:", info.get('is_success', False))
        break
