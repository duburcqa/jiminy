import math
import logging
from functools import partial
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from gym import ActionWrapper, spaces
from gym.wrappers import (
    FilterObservation, FlattenObservation, TransformObservation)

from gym_jiminy.common.utils import clip
from gym_jiminy.common.controllers import GenericOrderHoldController
from gym_jiminy.common.pipeline import build_pipeline
from gym_jiminy.envs import DoubleTableCartJiminyMetaEnv
from gym_jiminy.envs.double_table_cart import (
    L_X_F_MIN_RANGE, L_X_F_MAX_RANGE, L_Y_F_RANGE)

import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.deprecation import DEPRECATED_VALUE

from helpers import initialize
from fcnet import FrameStackingModel


# Constants of the universe
N_THREADS = 12
N_GPU = 1

DEBUG = False
SEED = 0

HAS_PATIENT = True
HAS_FLEXIBILITY = True
TASK_FEATURES = [
    "robot", "flexibility", "patient", "coupling", "behavior", "trajectory"]
INTERP_ORDER = None

HORIZON = 1000
N_FRAMES_STACK = 24


# Rescale action wrapper
class RescaleAction(ActionWrapper):
    def __init__(self, env, a, b):
        super(RescaleAction, self).__init__(env)
        action_shape = env.action_space.shape
        action_dtype = env.action_space.dtype
        self.a = np.full(action_shape, fill_value=a, dtype=action_dtype)
        self.b = np.full(action_shape, fill_value=b, dtype=action_dtype)
        self.action_space = spaces.Box(low=self.a, high=self.b)

    def action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (high - low) * (action - self.a) / (self.b - self.a)


# Flatten action wrapper
class FlattenAction(ActionWrapper):
    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, action):
        return spaces.unflatten(self.env.action_space, action)


# Register learning environment
def env_creator_base(env_config=None, **kwargs):
    if INTERP_ORDER is None:
        DoubleTableCartEnv = DoubleTableCartJiminyMetaEnv
    else:
        DoubleTableCartEnv = build_pipeline(**{
            'env_config': {
                'env_class': DoubleTableCartJiminyMetaEnv
            },
            'blocks_config': [{
                'block_class': GenericOrderHoldController,
                'block_kwargs': {
                    'order': INTERP_ORDER
                }}
            ]
        })

    return DoubleTableCartEnv(
        has_patient=HAS_PATIENT,
        has_flexiblity=HAS_FLEXIBILITY,
        active_task_features=TASK_FEATURES,
        auto_sampling=True, **{
            "viewer_backend": "panda3d",
            "debug": False,
            **kwargs
        })

def env_creator(env_config=None, **kwargs):
    env = env_creator_base(env_config, **kwargs)
    env = FilterObservation(
        env, filter_keys=("sensors", "trajectory"))
    clip_fn = partial(clip, env.observation_space)
    env = TransformObservation(env, clip_fn)
    env = RescaleAction(env, -50.0, 50.0)
    if INTERP_ORDER is not None:
        env = FlattenAction(env)
    return env

register_env("double_table_cart", env_creator)

# Register model
ModelCatalog.register_custom_model("frame_stack_network", FrameStackingModel)

# Initialize Ray backend
log_creator = initialize(
    num_cpus=N_THREADS, num_gpus=N_GPU, debug=DEBUG)

# PPO configuration
config = ppo.DEFAULT_CONFIG.copy()
config["framework"] = "tf"  # Enable "eager" mode if possible
config["eager_tracing"] = True
config["log_level"] = logging.DEBUG if DEBUG else logging.ERROR
config["num_workers"] = 10
config["num_envs_per_worker"] = 4
config["num_cpus_per_worker"] = 1
config["seed"] = SEED

config["observation_filter"] = "MeanStdFilter"
config["normalize_actions"] = False
config["clip_actions"] = False
config["explore"] = True

config["horizon"] = HORIZON
config["batch_mode"] = "truncate_episodes"
config["shuffle_sequences"] = True

config["train_batch_size"] = 4000
config["rollout_fragment_length"] = 100
config["sgd_minibatch_size"] = 128
config["num_sgd_iter"] = 30
config["entropy_coeff"] = 0.0

config["model"]["custom_model"] = "frame_stack_network"
config["model"]["custom_model_config"] = {
    "num_frames": N_FRAMES_STACK
}
config["model"]["fcnet_activation"] = "tanh"
config["model"]["fcnet_hiddens"] = [64, 64]
config["model"]["vf_share_layers"] = False
config["model"]["no_final_linear"] = False

# Handling of deprecations
config["vf_share_layers"] = config["model"]["vf_share_layers"]
config["model"]["framestack"] = DEPRECATED_VALUE

# Learning rate scheduling
config["lr"] = 5.0e-5
config["lr_schedule"] = [[      0, 5.0e-5],
                         [ 200000, 5.0e-5],
                         [ 500000, 1.0e-5],
                         [ 500001, 1.0e-6],
                         [ 600000, 1.0e-6],
                         [ 600001, 1.0e-7],
                         [1200000, 1.0e-7]]

# Instantiate trainer
trainer = ppo.PPOTrainer(
    config=config, env="double_table_cart", logger_creator=log_creator)

# Perform training
n_iter = 300
try:
    for _ in range(n_iter):
        result = trainer.train()
        print(" - ".join([
            f"{field}: {result[field]:.5g}" for field in (
                "training_iteration", "time_total_s", "timesteps_total",
                "episodes_total", "episode_reward_max", "episode_reward_mean",
                "episode_len_mean"
        )]))
except KeyboardInterrupt:
    pass

# Get policy model
policy = trainer.get_policy()
model = policy.model
dist_class = policy.dist_class
obs_filter = trainer.workers.local_worker().filters["default_policy"]
obs_mean, obs_std = obs_filter.rs.mean, obs_filter.rs.std
obs_filter_fn = lambda obs: (obs - obs_mean) / (obs_std + 1.0e-8)

def compute_action(input_dict: Dict[str, np.ndarray], explore: bool):
    import tensorflow as tf
    if tf.compat.v1.executing_eagerly():
        action_logits, _ = model(input_dict)
        action_dist = dist_class(action_logits, model)
        if explore:
            action_tf = action_dist.sample()
        else:
            action_tf = action_dist.deterministic_sample()
        action = action_tf.numpy()
    else:
        # This obscure piece of code takes advantage of already existing
        # placeholders to avoid creating new nodes to evalute computation
        # graph. It is several order of magnitude more efficient than calling
        # `action_logits, _ = model(input_dict).eval(session=policy._sess)[0]`
        # directly, but also significantly trickier.
        from ray.rllib.utils.tf_run_builder import run_timeline
        feed_dict = {policy._input_dict[key]: value
                     for key, value in input_dict.items()
                     if key in policy._input_dict.keys()}
        feed_dict[policy._is_exploring] = explore
        action = policy._sess.run(
            policy._sampled_action, feed_dict=feed_dict)
    return action

# Instantiate testing environment
env = FlattenObservation(env_creator(debug=True))

def evaluate(n_steps: int = HORIZON,
             explore: bool = False,
             enable_stats: bool = True,
             enable_plots: bool = True,
             enable_replay: bool = False) -> None:
    # Initialize the simulation
    obs = env.reset()

    # Initialize frame stack
    input_dict = {
        "obs": np.zeros([1, *model.obs_space.shape]),
        "n_obs": np.zeros([1, N_FRAMES_STACK, *model.obs_space.shape]),
        "prev_n_act": np.zeros([1, N_FRAMES_STACK, *env.action_space.shape]),
        "prev_n_rew": np.zeros([1, N_FRAMES_STACK])
    }
    input_dict["obs"][0] = input_dict["n_obs"][0, -1] = obs_filter_fn(obs)

    # Run the simulation
    tot_reward = 0.0
    for _ in range(n_steps):
        action = compute_action(input_dict, explore=explore)
        if config["clip_actions"]:
            action = clip(env.action_space, action)
        obs, reward, done, _ = env.step(action)
        tot_reward += reward
        if done:
            break
        for v in input_dict.values():
            v[:] = np.roll(v, shift=-1, axis=1)
        input_dict["obs"][0] = input_dict["n_obs"][0, -1] = obs_filter_fn(obs)
        input_dict["prev_n_act"][0, -1] = action
        input_dict["prev_n_rew"][0, -1] = reward

    # Compute reference and actual COP trajectory, then tracking error
    if enable_stats or enable_plots:
        log_data, _ = env.get_log()
        task_dict = env.get_task()
        L_x_F_min = env.unwrapped._L_x_F_min
        L_x_F_max = env.unwrapped._L_x_F_max
        L_y_F = env.unwrapped._L_y_F
        L_x_T = task_dict["trajectory"]["L_x_T"]
        L_y_T = task_dict["trajectory"]["L_y_T"]
        beta = task_dict["trajectory"]["beta"]
        x_T = task_dict["trajectory"]["x_T"]
        y_T = task_dict["trajectory"]["y_T"]
        v_T = task_dict["trajectory"]["v_T"]

        R = np.array([[math.cos(beta), -math.sin(beta)],
                    [math.sin(beta),  math.cos(beta)]])
        t = log_data["Global.Time"]
        x_com, y_com = [log_data['.'.join(('HighLevelController', field))]
                        for field in env.robot.logfile_position_headers]
        x_cop = -log_data["Foot.MY"] / log_data["Foot.FZ"]
        y_cop = log_data["Foot.MX"] / log_data["Foot.FZ"]
        x_ref, y_ref = np.array([x_T, y_T]) + R @ np.stack([
            L_x_T * np.cos(v_T * t), L_y_T * np.sin(v_T * t)], axis=0)
        err_rel = np.linalg.norm([
            (x_cop - x_ref) / (L_X_F_MIN_RANGE[1] + L_X_F_MAX_RANGE[1]),
            (y_cop - y_ref) / (2.0 * L_Y_F_RANGE[1])], axis=0)
        tot_rew_mes = (env.num_steps / len(err_rel)) * np.sum(
            np.clip(1.0 - err_rel[t >= 0.2] ** 2, 1.0e-2, 1.0))

    # Display some statistic if requested
    if enable_stats:
        print("env.num_steps:", env.num_steps)
        print("cumulative reward:", tot_reward)
        print("estimated reward:", tot_rew_mes)
        print(env.simulator.stepper_state)

    # Replay the result if requested
    if enable_replay:
        env.simulator.replay(speed_ratio=1.0)

    # Display trajectory tracking analysis if requested
    if enable_plots:
        plt.figure()
        plt.plot(x_ref, y_ref, ".-b")
        plt.plot(x_cop[t <= 0.2], y_cop[t <= 0.2], ".-g")
        plt.plot(x_cop[t >= 0.2], y_cop[t >= 0.2], ".-r")
        plt.plot(x_com, y_com, ".-k")
        plt.plot([L_x_F_max, L_x_F_max, -L_x_F_min, -L_x_F_min, L_x_F_max],
                [L_y_F, -L_y_F, -L_y_F, L_y_F, L_y_F], ":")
        plt.axis("scaled")
        plt.show()

        plt.figure()
        plt.plot(t, err_rel)
        plt.show()

evaluate(n_steps=2*HORIZON,
         explore=False,
         enable_stats=True,
         enable_plots=True,
         enable_replay=False)
