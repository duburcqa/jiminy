"""Solve the official Open AI Gym Acrobot problem simulated in Jiminy using PPO
algorithm with TorchRL reinforcement learning framework.

The proposed implementation runs at 1800 frames/s using CPU-only on Intel
i9-11900H (8/16 physical/logical cores).

.. seealso::
    https://github.com/pytorch/rl/blob/v0.1.1/tutorials/sphinx-tutorials/coding_ppo.py

.. warning::
    Pre-compiled binaries of torchrl only supports torch==2.0.0
"""
import os
import math
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from gym_jiminy.envs import AcrobotJiminyEnv
from gym_jiminy.common.wrappers import FrameRateLimiter
from tensordict.nn import TensorDictModule

from torchrl.collectors import MultiSyncDataCollector
from torchrl.modules import MLP, NoisyLinear, NormalParamWrapper
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement)
from torchrl.modules import (
    ProbabilisticActor, ValueOperator, IndependentNormal)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import (
    SerialEnv,
    EnvCreator,
    Compose,
    TransformedEnv,
    DoubleToFloat,
    StepCounter,
    RewardSum,
)


N_EPOCHS = 20
FRAMES_PER_MINIBATCH = 250
MAX_GRAD_NORM = 0.5
DEVICE = "cuda:0" if torch.cuda.device_count() else "cpu"
N_ENVS = 8
N_WORKERS = 8
SEED = 0


if __name__ == '__main__':
    # Fix weird issue with multiprocessing
    __spec__ = None

    # Enforce seed for most common libraries
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the learning environment
    gym_make = lambda: AcrobotJiminyEnv(continuous=True)
    eval_env_creator = EnvCreator(lambda: GymWrapper(
        FrameRateLimiter(gym_make(), speed_ratio=2.0), device=DEVICE))
    train_env_creator = EnvCreator(
        lambda: GymWrapper(gym_make(), device=DEVICE))

    # Instantiate the evaluation environment
    eval_env = eval_env_creator()
    eval_env.set_seed(SEED)
    eval_env.eval()

    # Configure processing steps
    eval_env = TransformedEnv(eval_env, Compose(
        DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
        StepCounter(max_steps=500),
        RewardSum(),
    ))

    # Make a copy of the now fully-initialized (composed) transform.
    # It is necessary to avoid pickling the env to which they are attached.
    transform = eval_env.transform.clone()

    # Instantiate the Actor and Critic
    obs_size = math.prod(eval_env.observation_spec["observation"].shape)
    act_size = math.prod(eval_env.action_spec.shape)

    actor = ProbabilisticActor(
        TensorDictModule(
            NormalParamWrapper(MLP(
                in_features=obs_size,
                out_features=(2 * act_size),
                num_cells=[64, 64],
                activation_class=torch.nn.LeakyReLU,
                activation_kwargs=dict(
                    negative_slope=0.01
                ),
                activate_last_layer=False,
                layer_class=NoisyLinear,
                layer_kwargs=dict(
                    std_init=0.1
                ),
                device=DEVICE,
            )),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=eval_env.action_spec,
        distribution_class=IndependentNormal,
        return_log_prob=True
    )

    critic = ValueOperator(
        MLP(
            in_features=obs_size,
            out_features=1,
            num_cells=[64, 64],
            activation_class=nn.Tanh,
            activate_last_layer=False,
            device=DEVICE,
        ),
        in_keys=["observation"],
    ).to(device=DEVICE)

    # TODO: Cleanup network weight init
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            m.bias.data.zero_()

    actor.module[0].module.apply(init_weights)
    critic.module.apply(init_weights)

    # Instantiate and configure the data collector.
    # After some benchmarks, it turns out combining `MultiSyncDataCollector`
    # plus `SerialEnv` runs faster that `SyncDataCollector` plus `ParallelEnv`.
    collector = MultiSyncDataCollector(
        N_WORKERS * (lambda: TransformedEnv(
            SerialEnv(
                N_ENVS // N_WORKERS, train_env_creator,
                allow_step_when_done=False,
            ) if N_ENVS // N_WORKERS > 1 else train_env_creator(),
            transform.clone()
        ),),
        actor,
        frames_per_batch=4000,
        total_frames=200000,
        exploration_type=ExplorationType.RANDOM,
        reset_at_each_iter=True,
        update_at_each_batch=True,
        device=DEVICE,
        storing_device=DEVICE,
        split_trajs=True
    )
    frames_per_batch = (
        collector.frames_per_batch_worker * collector.num_workers)
    collector.set_seed(SEED)

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=DEVICE),
        sampler=SamplerWithoutReplacement()
    )

    # Configure the training algorithm
    adv_module = GAE(
        value_network=critic,
        gamma=0.97,
        lmbda=0.92,
        average_gae=True
    ).to(device=DEVICE)
    loss_module = ClipPPOLoss(
        actor,
        critic,
        value_target_key=adv_module.value_target_key,
        clip_epsilon=0.3,
        entropy_bonus=True,
        entropy_coef=0.0,
        critic_coef=0.01,
        loss_critic_type="l2",
        separate_losses=False
    ).to(device=DEVICE)
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=5.0e-4,
        weight_decay=1.0e-4,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    # Monitoring facilities
    logs = defaultdict(list)
    pbar = tqdm(total=collector.total_frames, unit=" frames")

    # Main training loop
    for data_splitted in collector:
        # Sort data by 'traj_ids' to ensure repeatability
        data_masked = data_splitted[data_splitted["collector", "mask"]]
        _, indices = torch.sort(
            data_masked["collector", "traj_ids"], stable=True)
        data = data_masked[indices]

        # Perform several epoch on the same collected batch
        for epoch in range(N_EPOCHS):
            # (Re-)Compute advantage.
            # Note that it must be re-computed at each epoch as its value
            # depends on the value network, which is updated in the inner loop.
            with torch.no_grad():
                adv_module(data)

            # Store batch in intermediary replay buffer (discarding old batch).
            # Note that it must be at least once after computing the advantage.
            buffer.extend(data)

            # Loop over the whole batch by randomized minibatches
            for _ in range(frames_per_batch // FRAMES_PER_MINIBATCH):
                # Sample mini-batch
                sample = buffer.sample(FRAMES_PER_MINIBATCH)

                # Compute total PPO loss
                loss_vals = loss_module(sample)
                loss_val = sum(
                    value for key, value in loss_vals.items()
                    if key.startswith("loss")
                )

                # Aborting minibatches if something it wrong with the loss
                if loss_val > 1.0:
                    break

                # Back-propagate the gradient and update the policy
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), MAX_GRAD_NORM)
                optim.step()
                optim.zero_grad()

        # Monitor high-level training statistics
        pbar.update(frames_per_batch)
        episode_lengths = data["next", "step_count"][
            torch.logical_or(data['next', 'done'], data['next', 'truncated'])]
        for info in ("mean", "min", "max"):
            value = float('nan')
            if episode_lengths.numel() > 0:
                value = getattr(episode_lengths.double(), info)().item()
            logs[f"episode_length_{info}"].append(value)
        logs["lr"].append(optim.param_groups[0]["lr"])
        pbar.set_description(", ".join((
            f"Mean episode length: {logs['episode_length_mean'][-1]: 4.1f}",
        )))

        # Early stopping if reaching the expected episode reward threshold
        if logs["episode_length_mean"][-1] < 100.0:
            break

    # Stop the data collector
    collector.shutdown()

    # Display high-level training statistics
    fig, axes = plt.subplots(1, 1, squeeze=False)
    for info in ("mean", "min", "max"):
        axes.flat[0].plot(logs[f"episode_length_{info}"], label=info)
    axes.flat[0].set_title("Episode length (training)")
    axes.flat[0].legend(loc="best")
    plt.show()

    # Evaluate the performance of the agent by running a complete episode
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        actor.eval()
        eval_env.rollout(
            policy=actor,
            max_steps=500,
            # callback=lambda env, data: env.render(),
            break_when_any_done=True
        )
        actor.train()
        print(f"Episode length: {eval_env.num_steps}")
        # eval_env.replay()  # FIXME: it segfault for some reason...
