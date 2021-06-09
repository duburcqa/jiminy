from typing import Dict, List, Type, Union, Optional

import gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import (ppo_surrogate_loss,
                                                   kl_and_loss_stats,
                                                   setup_mixins,
                                                   PPOTorchPolicy)

from gym_jiminy.common.utils import sample

torch, nn = try_import_torch()


DEFAULT_CONFIG = PPOTrainer.merge_trainer_configs(
    DEFAULT_CONFIG,
    {
        "caps_temporal_reg": 0.0,
        "caps_spatial_reg": 0.0,
        "caps_global_reg": 0.0
    },
    _allow_unknown_configs=True)


def ppo_caps_init(policy: Policy,
                  obs_space: gym.spaces.Space,
                  action_space: gym.spaces.Space,
                  config: TrainerConfigDict) -> None:
    # Call base implementation
    setup_mixins(policy, obs_space, action_space, config)

    # Add previous observation in viewer requirements for CAPS loss computation
    # TODO: Remove update of `policy.model.view_requirements` after ray fix
    caps_view_requirements = {
        "_prev_obs": ViewRequirement(
            data_col="obs",
            space=obs_space,
            shift=-1,
            used_for_compute_actions=False)}
    policy.model.view_requirements.update(caps_view_requirements)
    policy.view_requirements.update(caps_view_requirements)


def ppo_caps_loss(policy: Policy,
                  model: ModelV2,
                  dist_class: Type[TorchDistributionWrapper],
                  train_batch: SampleBatch
                  ) -> Union[TensorType, List[TensorType]]:
    # Compute original ppo loss
    total_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)

    # Shallow copy the input batch.
    # Be careful accessing fields using the original batch to properly
    # keep track of acessed keys, which will be used to discard useless
    # components of policy's view requirements.
    train_batch_copy = train_batch.copy(shallow=True)

    # Extract mean of predicted action from logits.
    # No need to compute the perform model forward pass since the original
    # PPO loss is already doing it, so just getting back the last ouput.
    action_logits = model._last_output
    action_dist = dist_class(action_logits, model)
    action_mean = action_dist.deterministic_sample()

    # Generate noisy observation based on specified sensivity
    offset = 0
    observation_noisy = train_batch["obs"].clone()
    batch_dim = observation_noisy.shape[:-1]
    observation_space = policy.observation_space.original_space
    for v in observation_space.sensitivity.values():
        noise = sample(dist="normal", scale=v, shape=(*batch_dim, len(v)))
        slice_idx = slice(offset, offset + len(v))
        observation_noisy[..., slice_idx] += torch.from_numpy(
            noise).to(observation_noisy.device)
        offset += len(v)

    # Compute the mean action corresponding to the noisy observation
    train_batch_copy["obs"] = observation_noisy
    action_logits_noisy, _ = model(train_batch_copy)
    action_dist_noisy = dist_class(action_logits_noisy, model)
    action_mean_noisy = action_dist_noisy.deterministic_sample()

    # Compute the mean action corresponding to the previous observation
    train_batch_copy["obs"] = train_batch["_prev_obs"]
    action_logits_prev, _ = model(train_batch_copy)
    action_dist_prev = dist_class(action_logits_prev, model)
    action_mean_prev = action_dist_prev.deterministic_sample()

    # Minimize the difference between the successive action mean
    policy._mean_temporal_caps_loss = torch.mean(
        (action_mean_prev - action_mean) ** 2)

    # Minimize the difference between the original action mean and the
    # one corresponding to the noisy observation.
    policy._mean_spatial_caps_loss = torch.mean(
        (action_mean_noisy - action_mean) ** 2)

    # Minimize the magnitude of action mean
    policy._mean_global_caps_loss = torch.mean(action_mean ** 2)

    # Update total loss
    total_loss += (
        policy.config["caps_temporal_reg"] * policy._mean_temporal_caps_loss +
        policy.config["caps_spatial_reg"] * policy._mean_spatial_caps_loss +
        policy.config["caps_global_reg"] * policy._mean_global_caps_loss)

    return total_loss


def caps_stats(policy: Policy,
               train_batch: SampleBatch) -> Dict[str, TensorType]:
    # Compute original stats report
    stats_dict = kl_and_loss_stats(policy, train_batch)

    # Add spatial CAPS loss to the report
    stats_dict.update({
        "temporal_smoothness": policy._mean_temporal_caps_loss,
        "spatial_smoothness": policy._mean_spatial_caps_loss,
        "global_smoothness": policy._mean_global_caps_loss})

    return stats_dict


PPOTorchPolicy = PPOTorchPolicy.with_updates(
    before_loss_init=ppo_caps_init,
    loss_fn=ppo_caps_loss,
    stats_fn=caps_stats
)


def get_policy_class(
        config: TrainerConfigDict) -> Optional[Type[Policy]]:
    if config["framework"] == "torch":
        return PPOTorchPolicy


PPOTrainer = PPOTrainer.with_updates(
    default_config=DEFAULT_CONFIG,
    get_policy_class=get_policy_class
)

__all__ = [
    "DEFAULT_CONFIG",
    "PPOTorchPolicy",
    "PPOTrainer"
]
