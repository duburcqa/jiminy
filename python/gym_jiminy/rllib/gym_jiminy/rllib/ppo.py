""" TODO: Write documentation.
"""
from typing import Dict, List, Type, Union, Optional, Any, Tuple

import gym
import torch

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper, TorchDiagGaussian)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import (
    ppo_surrogate_loss, kl_and_loss_stats, setup_mixins, PPOTorchPolicy)


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

    # Convert to torch.Tensor observation sensitivity data
    observation_space = policy.observation_space.original_space
    for field, scale in observation_space.sensitivity.items():
        if not isinstance(scale, torch.Tensor):
            scale = torch.from_numpy(scale).to(dtype=torch.float32)
            observation_space.sensitivity[field] = scale

    # Transpose and convert to torch.Tensor the observation mirroring data
    for field, mirror_mat in observation_space.mirror_mat.items():
        if not isinstance(mirror_mat, torch.Tensor):
            mirror_mat = torch.from_numpy(
                mirror_mat.T.copy()).to(dtype=torch.float32)
            observation_space.mirror_mat[field] = mirror_mat

    # Transpose and convert to torch.Tensor the action mirroring data
    action_space = policy.action_space
    if not isinstance(action_space.mirror_mat, torch.Tensor):
        action_mirror_mat = torch.from_numpy(
            action_space.mirror_mat.T.copy()).to(dtype=torch.float32)
        action_space.mirror_mat = action_mirror_mat


def ppo_loss(policy: Policy,
             model: ModelV2,
             dist_class: Type[TorchDistributionWrapper],
             train_batch: SampleBatch
             ) -> Union[TensorType, List[TensorType]]:
    """ TODO: Write documentation.
    """
    # Extract some proxies from convenience
    observation_true = train_batch["obs"]
    device = observation_true.device

    # Initialize the set of training batches to forward to the model
    train_batches = {"original": train_batch}

    if policy.config["caps_temporal_reg"] > 0.0:
        # Shallow copy the original training batch.
        # Be careful accessing fields using the original batch to properly
        # keep track of acessed keys, which will be used to discard useless
        # components of policy's view requirements.
        train_batch_copy = train_batch.copy(shallow=True)

        # Replace current observation by the previous one
        observation_prev = train_batch["_prev_obs"]
        train_batch_copy["obs"] = observation_prev

        # Append the training batches to the set
        train_batches["prev"] = train_batch_copy

    if policy.config["caps_spatial_reg"] > 0.0:
        # Shallow copy the original training batch
        train_batch_copy = train_batch.copy(shallow=True)

        # Generate noisy observation based on specified sensivity
        offset = 0
        observation_noisy = observation_true.clone()
        batch_dim = observation_true.shape[:-1]
        observation_space = policy.observation_space.original_space
        for field, scale in observation_space.sensitivity.items():
            scale = scale.to(device)
            observation_space.sensitivity[field] = scale
            unit_noise = torch.randn((*batch_dim, len(scale)), device=device)
            slice_idx = slice(offset, offset + len(scale))
            observation_noisy[..., slice_idx].addcmul_(scale, unit_noise)
            offset += len(scale)

        # Replace current observation by the noisy one
        train_batch_copy["obs"] = observation_noisy

        # Append the training batches to the set
        train_batches["noisy"] = train_batch_copy

    if policy.config["symmetric_policy_reg"] > 0.0:
        # Shallow copy the original training batch
        train_batch_copy = train_batch.copy(shallow=True)

        # Compute mirrorred observation
        offset = 0
        observation_mirror = torch.empty_like(observation_true)
        observation_space = policy.observation_space.original_space
        for field, mirror_mat in observation_space.mirror_mat.items():
            mirror_mat = mirror_mat.to(device)
            observation_space.mirror_mat[field] = mirror_mat
            slice_idx = slice(offset, offset + len(mirror_mat))
            torch.mm(observation_true[..., slice_idx],
                     mirror_mat,
                     out=observation_mirror[..., slice_idx])
            offset += len(mirror_mat)

        # Replace current observation by the mirrored one
        train_batch_copy["obs"] = observation_mirror

        # Append the training batches to the set
        train_batches["mirrored"] = train_batch_copy

    # Compute the logits for all training batches at onces
    train_batch_all = {}
    for k in ['obs']:
        train_batch_all[k] = torch.cat([
            s[k] for s in train_batches.values()], dim=0)
    logits_all, _ = model(train_batch_all)
    values_all = model.value_function()
    logits = dict(zip(train_batches.keys(),
                      torch.chunk(logits_all, len(train_batches), dim=0)))
    values = dict(zip(train_batches.keys(),
                      torch.chunk(values_all, len(train_batches), dim=0)))

    # Compute original ppo loss.
    # pylint: disable=unused-argument,missing-function-docstring
    class FakeModel:
        """Fake model enabling doing all forward passes at once.
        """
        def __init__(self,
                     model: ModelV2,
                     logits: torch.Tensor,
                     value: torch.Tensor) -> None:
            self._logits = logits
            self._value = value
            self._model = model

        def __getattr__(self, name: str) -> Any:
            return getattr(self._model, name)

        def __call__(self, *args: Any, **kwargs: Any
                     ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            return self._logits, []

        def value_function(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return self._value

    fake_model = FakeModel(model, logits["original"], values["original"])
    total_loss = ppo_surrogate_loss(
        policy, fake_model, dist_class, train_batch)

    # Extract mean of predicted action from logits.
    # No need to compute the perform model forward pass since the original
    # PPO loss is already doing it, so just getting back the last ouput.
    action_logits = logits["original"]
    if issubclass(dist_class, TorchDiagGaussian):
        action_mean_true, _ = torch.chunk(action_logits, 2, dim=1)
    else:
        action_dist = dist_class(action_logits, model)
        action_mean_true = action_dist.deterministic_sample()

    if policy.config["caps_temporal_reg"] > 0.0:
        # Compute the mean action corresponding to the previous observation
        action_logits_prev = logits["prev"]
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

    if policy.config["caps_spatial_reg"] > 0.0:
        # Compute the mean action corresponding to the noisy observation
        action_logits_noisy = logits["noisy"]
        if issubclass(dist_class, TorchDiagGaussian):
            action_mean_noisy, _ = torch.chunk(action_logits_noisy, 2, dim=1)
        else:
            action_dist_noisy = dist_class(action_logits_noisy, model)
            action_mean_noisy = action_dist_noisy.deterministic_sample()

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
        # Compute the mirrored mean action corresponding to the mirrored action
        action_logits_mirror = logits["mirrored"]
        if issubclass(dist_class, TorchDiagGaussian):
            action_mean_mirror, _ = torch.chunk(action_logits_mirror, 2, dim=1)
        else:
            action_dist_mirror = dist_class(action_logits_mirror, model)
            action_mean_mirror = action_dist_mirror.deterministic_sample()
        action_mirror_mat = policy.action_space.mirror_mat.to(device)
        policy.action_space.mirror_mat = action_mirror_mat
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
