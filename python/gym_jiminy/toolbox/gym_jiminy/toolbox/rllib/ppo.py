""" TODO: Write documentation.
"""
from typing import Dict, List, Type, Union, Optional

import gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper, TorchDiagGaussian)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import (
    ppo_surrogate_loss, kl_and_loss_stats, setup_mixins, PPOTorchPolicy)

torch, nn = try_import_torch()


DEFAULT_CONFIG = PPOTrainer.merge_trainer_configs(
    DEFAULT_CONFIG,
    {
        "symmetric_policy_reg": 0.0,
        "caps_temporal_reg": 0.0,
        "caps_spatial_reg": 0.0,
        "caps_global_reg": 0.0
    },
    _allow_unknown_configs=True)


def ppo_init(policy: Policy,
             obs_space: gym.spaces.Space,
             action_space: gym.spaces.Space,
             config: TrainerConfigDict) -> None:
    """ TODO: Write documentation.
    """
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

    # Initialize extra loss
    policy._mean_symmetric_policy_loss = 0.0
    policy._mean_temporal_caps_loss = 0.0
    policy._mean_spatial_caps_loss = 0.0
    policy._mean_global_caps_loss = 0.0


def ppo_loss(policy: Policy,
             model: ModelV2,
             dist_class: Type[TorchDistributionWrapper],
             train_batch: SampleBatch
             ) -> Union[TensorType, List[TensorType]]:
    """ TODO: Write documentation.
    """
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
    if issubclass(dist_class, TorchDiagGaussian):
        action_mean_true, _ = torch.chunk(action_logits, 2, dim=1)
    else:
        action_dist = dist_class(action_logits, model)
        action_mean_true = action_dist.deterministic_sample()

    if policy.config["caps_temporal_reg"] > 0.0:
        # Compute the mean action corresponding to the previous observation
        observation_prev = train_batch["_prev_obs"]
        train_batch_copy["obs"] = observation_prev
        action_logits_prev, _ = model(train_batch_copy)
        if issubclass(dist_class, TorchDiagGaussian):
            action_mean_prev, _ = torch.chunk(action_logits_prev, 2, dim=1)
        else:
            action_dist_prev = dist_class(action_logits_prev, model)
            action_mean_prev = action_dist_prev.deterministic_sample()

        # Minimize the difference between the successive action mean
        policy._mean_temporal_caps_loss = torch.mean(
            (action_mean_prev - action_mean_true) ** 2)

        # Add temporal smoothness loss to total loss
        total_loss += policy.config["caps_temporal_reg"] * \
            policy._mean_temporal_caps_loss

    if policy.config["caps_spatial_reg"] > 0.0 or \
            policy.config["symmetric_policy_reg"] > 0.0:
        # Generate noisy observation based on specified sensivity
        offset = 0
        observation_true = train_batch["obs"]
        observation_noisy = observation_true.clone()
        batch_dim = observation_true.shape[:-1]
        observation_space = policy.observation_space.original_space
        for scale in observation_space.sensitivity.values():
            scale = torch.from_numpy(scale.copy()).to(
                dtype=torch.float32, device=observation_true.device)
            unit_noise = torch.randn(
                (*batch_dim, len(scale)), device=observation_true.device)
            slice_idx = slice(offset, offset + len(scale))
            observation_noisy[..., slice_idx].addcmul_(scale, unit_noise)
            offset += len(scale)

        # Compute the mean action corresponding to the noisy observation
        train_batch_copy["obs"] = observation_noisy
        action_logits_noisy, _ = model(train_batch_copy)
        if issubclass(dist_class, TorchDiagGaussian):
            action_mean_noisy, _ = torch.chunk(action_logits_noisy, 2, dim=1)
        else:
            action_dist_noisy = dist_class(action_logits_noisy, model)
            action_mean_noisy = action_dist_noisy.deterministic_sample()

    if policy.config["caps_spatial_reg"] > 0.0:
        # Minimize the difference between the original action mean and the
        # one corresponding to the noisy observation.
        policy._mean_spatial_caps_loss = torch.mean(
            (action_mean_noisy - action_mean_true) ** 2)

        # Add spatial smoothness loss to total loss
        total_loss += policy.config["caps_spatial_reg"] * \
            policy._mean_spatial_caps_loss

    if policy.config["caps_global_reg"] > 0.0:
        # Minimize the magnitude of action mean
        policy._mean_global_caps_loss = torch.mean(action_mean_true ** 2)

        # Add global smoothness loss to total loss
        total_loss += policy.config["caps_global_reg"] * \
            policy._mean_global_caps_loss

    if policy.config["symmetric_policy_reg"] > 0.0:
        # Compute mirrorred observation
        offset = 0
        observation_mirror = torch.empty_like(observation_true)
        observation_space = policy.observation_space.original_space
        for mirror_mat in observation_space.mirror_mat.values():
            mirror_mat = torch.from_numpy(mirror_mat.T.copy()).to(
                dtype=torch.float32, device=observation_true.device)
            slice_idx = slice(offset, offset + len(mirror_mat))
            torch.mm(observation_true[..., slice_idx],
                     mirror_mat,
                     out=observation_mirror[..., slice_idx])
            offset += len(mirror_mat)

        # Compute the mirrored mean action corresponding to the mirrored action
        train_batch_copy["obs"] = observation_mirror
        action_logits_mirror, _ = model(train_batch_copy)
        if issubclass(dist_class, TorchDiagGaussian):
            action_mean_mirror, _ = torch.chunk(action_logits_mirror, 2, dim=1)
        else:
            action_dist_mirror = dist_class(action_logits_mirror, model)
            action_mean_mirror = action_dist_mirror.deterministic_sample()
        action_mirror_mat = policy.action_space.mirror_mat
        action_mirror_mat = torch.from_numpy(action_mirror_mat.T.copy()).to(
            dtype=torch.float32, device=observation_true.device)
        action_mean_mirror = action_mean_mirror @ action_mirror_mat

        # Minimize the assymetry of policy output
        policy._mean_symmetric_policy_loss = torch.mean(
            (action_mean_mirror - action_mean_true) ** 2)

        # Add policy symmetry loss to total loss
        total_loss += policy.config["symmetric_policy_reg"] * \
            policy._mean_symmetric_policy_loss

    return total_loss


def ppo_stats(policy: Policy,
              train_batch: SampleBatch) -> Dict[str, TensorType]:
    """ TODO: Write documentation.
    """
    # Compute original stats report
    stats_dict = kl_and_loss_stats(policy, train_batch)

    # Add spatial CAPS loss to the report
    if policy.config["symmetric_policy_reg"] > 0.0:
        stats_dict["symmetry"] = policy._mean_symmetric_policy_loss
    if policy.config["caps_temporal_reg"] > 0.0:
        stats_dict["temporal_smoothness"] = policy._mean_temporal_caps_loss
    if policy.config["caps_spatial_reg"] > 0.0:
        stats_dict["spatial_smoothness"] = policy._mean_spatial_caps_loss
    if policy.config["caps_global_reg"] > 0.0:
        stats_dict["global_smoothness"] = policy._mean_global_caps_loss

    return stats_dict


PPOTorchPolicy = PPOTorchPolicy.with_updates(
    before_loss_init=ppo_init,
    loss_fn=ppo_loss,
    stats_fn=ppo_stats,
    get_default_config=lambda: DEFAULT_CONFIG,
)


def get_policy_class(
        config: TrainerConfigDict) -> Optional[Type[Policy]]:
    """ TODO: Write documentation.
    """
    if config["framework"] == "torch":
        return PPOTorchPolicy
    return None


PPOTrainer = PPOTrainer.with_updates(
    default_config=DEFAULT_CONFIG,
    get_policy_class=get_policy_class
)

__all__ = [
    "DEFAULT_CONFIG",
    "PPOTorchPolicy",
    "PPOTrainer"
]
