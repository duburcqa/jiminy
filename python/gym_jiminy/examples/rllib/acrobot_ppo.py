"""Solve the official Open AI Gym Acrobot problem simulated in Jiminy using PPO
algorithm of Ray RLlib reinforcement learning framework.

It solves it consistently in less than 100000 timesteps in average.

.. warning::
    This script has been tested for pytorch~=2.3 and ray[rllib]~=2.9.3
"""

# ====================== Configure Python workspace =======================

import os
import logging
from functools import partial

import gymnasium as gym
import ray
from ray.tune.registry import register_env
from ray.rllib.models import MODEL_DEFAULTS
from ray.tune.logger import NoopLogger

from gym_jiminy.toolbox.wrappers import FrameRateLimiter
from gym_jiminy.rllib.ppo import PPOConfig
from gym_jiminy.rllib.utilities import (initialize,
                                        train,
                                        build_eval_policy_from_checkpoint,
                                        build_policy_wrapper,
                                        build_eval_worker_from_checkpoint,
                                        evaluate_local_worker,
                                        evaluate_algo)

# ============================ User parameters ============================

GYM_ENV_NAME = "gym_jiminy.envs:acrobot"
GYM_ENV_KWARGS = {
    'continuous': True
}
ENABLE_RECORDING = True
ENABLE_VIEWER = "JIMINY_VIEWER_DISABLE" not in os.environ
DEBUG = False
SEED = 0
N_THREADS = 9
N_GPU = 0

if __name__ == "__main__":
    # ==================== Initialize Ray and Tensorboard =====================

    # Start Ray and Tensorboard background processes
    logdir = initialize(
        num_cpus=N_THREADS, num_gpus=N_GPU, debug=DEBUG, verbose=True)

    # Register the environment
    register_env("env", lambda env_config: FrameRateLimiter(
        gym.make(GYM_ENV_NAME, **env_config), speed_ratio=1.0))

    # ====================== Configure policy's network =======================

    # Default model configuration
    model_config = MODEL_DEFAULTS.copy()

    # Fully-connected network settings
    model_config.update(dict(
        # Nonlinearity for built-in fully connected net
        fcnet_activation="tanh",
        # Number of hidden layers for fully connected net
        fcnet_hiddens=[64, 64],
        # The last half of the output layer does not dependent on the input
        free_log_std=True,
        # Whether to share layers between the policy and value function
        vf_share_layers=False
    ))

    # ===================== Configure learning algorithm ======================

    # Default PPO configuration
    algo_config = PPOConfig()

    # Resources settings
    algo_config.resources(
        # Number of GPUs to reserve for the trainer process
        num_gpus=N_GPU,
        # Number of CPUs to reserve per worker processes
        num_cpus_per_worker=1,
        # Number of CPUs to allocate for trainer process
        num_gpus_per_worker=0,
    )
    algo_config.rollouts(
        # Number of rollout worker processes for parallel sampling
        num_rollout_workers=N_THREADS-1,
        # Number of environments per worker processes
        num_envs_per_worker=1,
        # Whether to create the envs per worker in individual remote processes
        remote_worker_envs=False,
        # Duration worker processes are waiting when polling environments
        remote_env_batch_wait_ms=0,
    )
    algo_config.python_environment(
        # Extra python env vars to set for trainer process
        extra_python_environs_for_driver = {},
        # Extra python env vars to set for worker processes
        extra_python_environs_for_worker = {
            # Disable multi-threaded linear algebra at worker-level
            "OMP_NUM_THREADS": "1"
        }
    )

    # Debugging and monitoring settings
    algo_config.fault_tolerance(
        # Whether to attempt to continue training if a worker crashes
        recreate_failed_workers=False
    )
    algo_config.debugging(
        # Set the log level for the whole learning process
        log_level=logging.DEBUG if DEBUG else logging.ERROR,
        # Monitor system resource metrics (requires `psutil` and `gputil`)
        log_sys_usage=True,
        # Disable default logger but configure logging directory nonetheless
        logger_config=dict(
            type=NoopLogger,
            logdir=logdir
        )
    )
    algo_config.reporting(
        # Smooth metrics over this many episodes
        metrics_num_episodes_for_smoothing=100,
        # Wait for metric batches for this duration
        metrics_episode_collection_timeout_s=180,
        # Minimum training timesteps to accumulate between each reporting
        min_train_timesteps_per_iteration=0,
        # Whether to store custom metrics without calculating max, min, mean
        keep_per_episode_custom_metrics=True
    )

    # Environment settings
    algo_config.environment(
        # The environment specifier
        env="env",
        # Normalize actions to the bounds of the action space
        normalize_actions=False,
        # Whether to clip actions to the bounds of the action space
        clip_actions=False,
        # Whether to clip rewards during postprocessing by the policy
        clip_rewards=False,
        # Arguments to pass to the env creator
        env_config=GYM_ENV_KWARGS
    )

    # Rollout settings
    algo_config.rollouts(
        # Number of collected samples per environments
        rollout_fragment_length="auto",
        # Whether to rollout "complete_episodes" or "truncate_episodes"
        batch_mode="truncate_episodes",
        # Use a background thread for sampling (slightly off-policy)
        sample_async=False,
        # Element-wise observation filter ["NoFilter", "MeanStdFilter"]
        observation_filter="NoFilter",
        # Whether to LZ4 compress individual observations
        compress_observations=False,
    )
    algo_config.debugging(
        # Set the random seed of each worker processes based on worker_index
        seed=SEED
    )

    # Model settings
    algo_config.training(
        # Policy model configuration
        model=model_config
    )

    # Learning settings
    algo_config.training(
        # Sample batches are concatenated together into batches of this size
        train_batch_size=2000,
        # Learning rate
        lr=5.0e-4,
        # Discount factor of the MDP (0.991: ~1% after 500 step)
        gamma=0.991,
    )
    algo_config.exploration(
        # Whether to disable exploration completely
        explore=True,
        exploration_config=dict(
            # The Exploration class to use ["EpsilonGreedy", "Random", ...]
            type="StochasticSampling",
        )
    )

    # ====================== Configure agent evaluation =======================

    algo_config.evaluation(
        # Evaluate every `evaluation_interval` training iterations
        evaluation_interval=20,
        # Number of evaluation steps based on specified unit
        evaluation_duration=10,
        # The unit with which to count the evaluation duration
        evaluation_duration_unit="episodes",
        # Number of parallel workers to use for evaluation
        evaluation_num_workers=1,
        # Whether to run evaluation in parallel to a Algorithm.train()
        evaluation_parallel_to_training=True,
        # Custom evaluation method
        custom_evaluation_function=partial(
            evaluate_algo,
            print_stats=True,
            enable_replay=ENABLE_VIEWER,
            record_video=ENABLE_RECORDING
        ),
        # Partially override configuration for evaluation
        evaluation_config=dict(
            env="env",
            env_config=dict(
                **GYM_ENV_KWARGS,
                viewer_kwargs=dict(
                    display_com=False,
                    display_dcm=False,
                    display_f_external=False
                )
            ),
            # Real-time rendering during evaluation (no post-processing)
            render_env=False,
            explore=False,
            horizon=None
        )
    )

    # ===================== Configure learning algorithm ======================

    algo_config.framework(
        # PyTorch is the only ML framework supported by `gym_jiminy.rllib`
        framework="torch"
    )

    # Estimators settings
    algo_config.training(
        # Use the Generalized Advantage Estimator (GAE)
        use_gae=True,
        # Whether to user critic as a value baseline (required if using GAE)
        use_critic=True,
        # GAE(lambda) parameter
        lambda_=0.95,
    )

    # Learning settings
    algo_config.training(
        # Initial coefficient for KL divergence. (0.0 for L^CLIP)
        kl_coeff=0.0,
        # Target value for KL divergence
        kl_target=0.1,
        # Coefficient of the value function loss
        vf_loss_coeff=0.5,
        # Coefficient of the entropy regularizer
        entropy_coeff=0.01,
        # Decay schedule for the entropy regularizer
        entropy_coeff_schedule=None,
        # PPO clip parameter
        clip_param=0.2,
        # Clip param for the value function (sensitive to the reward scale)
        vf_clip_param=float("inf"),
    )

    # Regularization settings
    algo_config.training(
        enable_adversarial_noise=True,
        temporal_barrier_threshold=0.5,
        temporal_barrier_reg=1e-1,
        caps_temporal_reg=5e-3,
        caps_spatial_reg=1e-2,
        caps_global_reg=1e-4,
        l2_reg=1e-6,
    )

    # Optimization settings
    algo_config.training(
        # Learning rate schedule
        lr_schedule=None,
        # Minibatch size of each SGD epoch
        sgd_minibatch_size=250,
        # Number of SGD epochs to execute per training iteration
        num_sgd_iter=8,
        # Whether to shuffle sequences in the batch when training
        shuffle_sequences=True,
        # Clamp the norm of the gradient during optimization (None to disable)
        grad_clip=None,
    )

    # ========================= Run the optimization ==========================

    # Initialize the learning algorithm
    algo = algo_config.build()

    # Train the agent
    result = train(algo, max_timesteps=200000, logdir=algo.logdir)
    checkpoint_path = result.checkpoint.path

    # ========================= Terminate Ray backend =========================

    algo.stop()
    ray.shutdown()

    # ===================== Enjoy a trained agent locally =====================

    # Build a standalone local evaluation worker (not requiring ray backend)
    register_env("env", lambda env_config: FrameRateLimiter(
        gym.make(GYM_ENV_NAME, **env_config), speed_ratio=1.0))
    worker = build_eval_worker_from_checkpoint(checkpoint_path)
    evaluate_local_worker(worker,
                          evaluation_num=1,
                          close_backend=True,
                          enable_replay=ENABLE_VIEWER)

    # Build a standalone single-agent evaluation policy
    env = gym.make(GYM_ENV_NAME, **algo_config.env_config)
    policy_map = build_eval_policy_from_checkpoint(checkpoint_path)
    policy_fn = build_policy_wrapper(
        policy_map, clip_actions=False, explore=False)
    for seed in (1, 1, 2):
        env.evaluate(policy_fn, seed=seed, horizon=env._max_episode_steps)
