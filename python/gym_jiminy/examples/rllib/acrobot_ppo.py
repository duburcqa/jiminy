"""Solve the official Open AI Gym Acrobot problem simulated in Jiminy using PPO
algorithm of Ray RLlib reinforcement learning framework.

It solves it consistently in less than 100000 timesteps in average.

.. warning::
    This script has been tested for pytorch~=2.5.0 and ray[rllib]~=2.38.0
"""

# ====================== Configure Python workspace =======================

import os
from copy import deepcopy

import numpy as np
import gymnasium as gym

import ray
from ray.tune.registry import register_env
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from gym_jiminy.rllib.ppo import PPOConfig
from gym_jiminy.rllib.utilities import (initialize,
                                        train,
                                        evaluate_from_runner,
                                        build_runner_from_checkpoint,
                                        build_module_from_checkpoint,
                                        build_module_wrapper)

# ============================== User parameters ==============================

GYM_ENV_NAME = "gym_jiminy.envs:acrobot"
GYM_ENV_KWARGS = {
    'continuous': True
}
ENABLE_RECORDING = False
ENABLE_VIEWER = "JIMINY_VIEWER_DISABLE" not in os.environ
DEBUG = False
SEED = 0
N_THREADS = 9
N_GPUS = 0

if __name__ == "__main__":
    # ==================== Initialize Ray and Tensorboard =====================

    # Start Ray and Tensorboard background processes
    logdir = initialize(
        num_cpus=N_THREADS, num_gpus=N_GPUS, debug=DEBUG, verbose=True)

    # Register the environment
    env_creator = lambda env_config: gym.make(GYM_ENV_NAME, **env_config)
    register_env("env", env_creator)

    # Mirroring map along X-axis
    obs_mirror_mat = np.diag([1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    action_mirror_mat = np.diag([-1.0])

    # ========================== Configure resources ==========================

    # Default PPO configuration
    algo_config = PPOConfig()

    # Resources settings
    algo_config.env_runners(
        # Number of CPUs to reserve per worker processes
        num_cpus_per_env_runner=1,
        # Number of GPUs to allocate for trainer process
        num_gpus_per_env_runner=0,
    )
    algo_config.learners(
        # Number of learner workers used for updating the policy
        num_learners=N_GPUS if N_GPUS > 1 else 0,
        # Number of CPUs allocated per learner worker
        num_cpus_per_learner=1,
        # Number of GPUs allocated per learner worker
        num_gpus_per_learner=min(1, N_GPUS),
    )
    algo_config.env_runners(
        # Number of rollout worker processes for parallel sampling
        num_env_runners=N_THREADS - 1,
        # Number of environments per worker processes
        num_envs_per_env_runner=1,
    )

    # ========================= Configure monitoring ==========================

    # Debugging and monitoring settings
    algo_config.fault_tolerance(
        # Whether to attempt to continue training if a worker crashes
        restart_failed_env_runners=False
    )
    algo_config.debugging(
        # Set the log level for the whole learning process
        log_level="DEBUG" if DEBUG else "ERROR",
        # Monitor system resource metrics (requires `psutil` and `gputil`)
        log_sys_usage=True,
    )
    algo_config.reporting(
        # Smooth metrics over this many episodes
        metrics_num_episodes_for_smoothing=100,
        # Wait for metric batches for this duration
        metrics_episode_collection_timeout_s=180,
        # Minimum training timesteps to accumulate between each reporting
        min_train_timesteps_per_iteration=0
    )

    algo_config.evaluation(
        # Evaluate every `evaluation_interval` training iterations
        evaluation_interval=20,
        # Number of "evaluation steps" based on specified unit
        evaluation_duration=10,
        # The unit with which to count the evaluation duration
        evaluation_duration_unit="episodes",
        # Number of parallel workers to use for evaluation
        evaluation_num_env_runners=1,
        # Whether to run evaluation in parallel to a Algorithm.train()
        evaluation_parallel_to_training=True
    )

    # =========================== Configure rollout ===========================

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
        env_config=dict(
            **GYM_ENV_KWARGS,
            viewer_kwargs=dict(
                display_com=False,
                display_dcm=False,
                display_f_external=False,
            )
        ),
    )

    # Rollout settings
    algo_config.env_runners(
        # Number of collected samples per environments
        rollout_fragment_length="auto",
        # Whether to rollout "complete_episodes" or "truncate_episodes"
        batch_mode="truncate_episodes",
        # Whether to LZ4 compress individual observations
        compress_observations=False,
    )
    algo_config.debugging(
        # Set the random seed of each worker processes based on worker_index
        seed=SEED
    )

    # ===================== Configure learning algorithm ======================

    # Batch settings settings
    algo_config.training(
        # Sample batches are concatenated together into batches of this size
        train_batch_size_per_learner=2000/max(1, N_GPUS),
        # Learning rate
        lr=5.0e-4,
        # Discount factor of the MDP (0.991: ~1% after 500 step)
        gamma=0.991,
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

    # PPO-specific settings
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
        enable_adversarial_noise=False,
        temporal_barrier_threshold=6.0,
        temporal_barrier_reg=1.0,
        symmetric_policy_reg=0.1,
        symmetric_spec=(obs_mirror_mat, action_mirror_mat),
        enable_symmetry_surrogate_loss=False,
        caps_temporal_reg=0.005,
        caps_spatial_reg=0.1,
        caps_global_reg=0.001,
        l2_reg=1e-8,
    )

    # Optimization settings
    algo_config.training(
        # Learning rate schedule
        lr_schedule=None,
        # Minibatch size of each SGD epoch
        minibatch_size=250,
        # Number of SGD epochs to execute per training iteration
        num_epochs=8,
        # Whether to shuffle sequences in the batch when training
        shuffle_batch_per_epoch=True,
        # Clamp the norm of the gradient during optimization (None to disable)
        grad_clip=None,
    )

    # ================== Configure policy and value networks ==================

    # Model settings
    algo_config.rl_module(
        # Policy model configuration
        model_config=DefaultModelConfig(
            # Number of hidden layers for fully connected net
            fcnet_hiddens=[64, 64],
            # Nonlinearity for built-in fully connected net
            fcnet_activation="tanh",
            # Whether to share layers between the policy and value function
            vf_share_layers=False,
            # The last half of the output layer does not dependent on the input
            free_log_std=True,
        )
    )

    # Exploration settings.
    # The exploration strategy must be implemented at "policy"-level, by
    # overwritting `_forward_exploration` or/and `forward_inference`.
    algo_config.env_runners(
        # Whether to disable exploration completely
        explore=True,
    )

    # ========================= Run the optimization ==========================

    # Initialize the learning algorithm and train the agent
    result, checkpoint_path = train(
        algo_config,
        logdir,
        max_timesteps=200000,
        enable_evaluation_replay=ENABLE_VIEWER,
        enable_evaluation_record_video=ENABLE_RECORDING,
        debug=DEBUG)

    # ========================= Terminate Ray backend =========================

    ray.shutdown()

    # ===================== Enjoy a trained agent locally =====================

    # Build a standalone local evaluation worker (not requiring ray backend)
    register_env("env", env_creator)
    env_runner = build_runner_from_checkpoint(checkpoint_path)
    evaluate_from_runner(env_runner,
                         num_episodes=1,
                         close_backend=True,
                         enable_replay=ENABLE_VIEWER)

    # Build a standalone single-agent evaluation policy
    env = env_creator(algo_config.env_config)
    rl_module = build_module_from_checkpoint(checkpoint_path)
    policy_fn = build_module_wrapper(rl_module)
    for seed in (1, 1, 2):
        env.get_wrapper_attr("evaluate")(
            policy_fn,
            seed=seed,
            horizon=env.spec.max_episode_steps)  # type: ignore[union-attr]
