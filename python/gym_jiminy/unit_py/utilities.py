from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold as StopOnReward)


def train(train_agent: BaseAlgorithm,
          max_timesteps: int) -> str:
    """Train a model on a specific environment using a given agent.

    :param train_agent: Training agent.
    :param max_timesteps: Number of maximum training timesteps.

    :returns: Whether or not the threshold reward has been exceeded in average
              over 10 episodes.
    """
    # Get testing environment spec
    spec = train_agent.eval_env.envs[0].spec

    # Create callback to stop learning early if reward threshold is exceeded
    if spec.reward_threshold is not None:
        callback_reward = StopOnReward(
            reward_threshold=spec.reward_threshold)
        eval_callback = EvalCallback(
            train_agent.eval_env, callback_on_new_best=callback_reward,
            eval_freq=10000 // train_agent.n_envs, n_eval_episodes=10,
            verbose=False)
    else:
        eval_callback = None

    # Run the learning process
    train_agent.learn(total_timesteps=max_timesteps, callback=eval_callback)

    return train_agent.num_timesteps < max_timesteps
