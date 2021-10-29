""" TODO: Write documentation.
"""
import math
import operator
from functools import reduce
from typing import Dict, List, Type, Union, Optional, Any, Tuple

import gym
import torch

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper, TorchDiagGaussian)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.torch_ops import l2_loss
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import (
    ppo_surrogate_loss, kl_and_loss_stats, setup_mixins, PPOTorchPolicy)


DEFAULT_CONFIG = PPOTrainer.merge_trainer_configs(
    DEFAULT_CONFIG,
    {
        "enable_adversarial_noise": False,
        "spatial_noise_scale": 1.0,
        "sgld_beta_inv": 1.0e-8,
        "sgld_n_steps": 10,
        "temporal_barrier_scale": 1.0,
        "temporal_barrier_threshold": math.inf,
        "temporal_barrier_reg": 0.0,
        "symmetric_policy_reg": 0.0,
        "caps_temporal_reg": 0.0,
        "caps_spatial_reg": 0.0,
        "caps_global_reg": 0.0,
        "l2_reg": 0.0
    },
    _allow_unknown_configs=True)


def get_action_mean(model: ModelV2,
                    dist_class: Type[TorchDistributionWrapper],
                    action_logits: torch.Tensor) -> torch.Tensor:
    """ TODO: Write documentation.
    """
    if issubclass(dist_class, TorchDiagGaussian):
        action_mean, _ = torch.chunk(action_logits, 2, dim=1)
    else:
        action_dist = dist_class(action_logits, model)
        action_mean = action_dist.deterministic_sample()
    return action_mean


def get_adversarial_observation_sgld(
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
        noise_scale: float,
        beta_inv: float,
        n_steps: int,
        action_true_mean: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
    """ TODO: Write documentation.
    """
    # Compute mean field action for true observation if not provided
    if action_true_mean is None:
        with torch.no_grad():
            action_true_logits, _ = model(train_batch)
            action_true_mean = get_action_mean(
                model, dist_class, action_true_logits)
    else:
        action_true_mean = action_true_mean.detach()

    # Shallow copy the original training batch.
    # Be careful accessing fields using the original batch to properly keep
    # track of accessed keys, which will be used to automatically discard
    # useless components of policy's view requirements.
    train_batch_copy = train_batch.copy(shallow=True)

    # Extract original observation
    observation_true = train_batch["obs"]

    # Define observation upper and lower bounds for clipping
    obs_lb_flat = observation_true - noise_scale
    obs_ub_flat = observation_true + noise_scale

    # Adjust the step size based on noise scale and number of steps
    step_eps = noise_scale / n_steps

    # Use Stochastic gradient Langevin dynamics (SGLD) to compute adversary
    # observation perturbation. It consists in find nearby observations that
    # maximize the mean action difference.
    observation_noisy = observation_true + step_eps * 2.0 * (
        torch.empty_like(observation_true).bernoulli_(p=0.5) - 0.5)
    for i in range(n_steps):
        # Make sure gradient computation is required
        observation_noisy.requires_grad_(True)

        # Compute mean field action for noisy observation
        train_batch_copy["obs"] = observation_noisy
        action_noisy_logits, _ = model(train_batch_copy)
        action_noisy_mean = get_action_mean(
            model, dist_class, action_noisy_logits)

        # Compute action different and associated gradient
        loss = torch.mean(torch.sum(
            (action_noisy_mean - action_true_mean) ** 2, dim=-1))
        loss.backward()

        # compute the noisy gradient for observation update
        noise_factor = math.sqrt(2.0 * step_eps * beta_inv) / (i + 2)
        observation_update = observation_noisy.grad + \
            noise_factor * torch.randn_like(observation_true)

        # Need to clear gradients before the backward() for policy_loss
        observation_noisy.detach_()

        # Project gradient to step boundary.
        # Note that `sign` is used to be agnostic to the norm of the gradient,
        # which would require to tune the learning rate or use an adaptive step
        # method. Alternatively, the normalized gradient could be used, but it
        # takes more iterations to converge in practice.
        # TODO: The update step should be `step_eps` but it was found that
        # using `noise_scale` converges faster.
        observation_noisy += observation_update.sign() * noise_scale

        # clip into the upper and lower bounds
        observation_noisy.clamp_(obs_lb_flat, obs_ub_flat)

    return observation_noisy


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
    policy._loss_temporal_barrier_reg = 0.0
    policy._loss_symmetric_policy_reg = 0.0
    policy._loss_caps_temporal_reg = 0.0
    policy._loss_caps_spatial_reg = 0.0
    policy._loss_caps_global_reg = 0.0
    policy._loss_l2_reg = 0.0

    # Extract original observation space
    try:
        observation_space = policy.observation_space.original_space
    except AttributeError as e:
        raise NotImplementedError(
            "Only 'Dict' original observation space is supported.") from e

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
    train_batches = {"true": train_batch}

    if policy.config["caps_temporal_reg"] > 0.0 or \
            policy.config["temporal_barrier_reg"] > 0.0:
        # Shallow copy the original training batch
        train_batch_copy = train_batch.copy(shallow=True)

        # Replace current observation by the previous one
        observation_prev = train_batch["_prev_obs"]
        train_batch_copy["obs"] = observation_prev

        # Append the training batches to the set
        train_batches["prev"] = train_batch_copy

    if policy.config["caps_spatial_reg"] > 0.0 and \
            policy.config["enable_adversarial_noise"]:
        # Shallow copy the original training batch
        train_batch_copy = train_batch.copy(shallow=True)

        # Compute adversarial observation maximizing action difference
        observation_worst = get_adversarial_observation_sgld(
            model, dist_class, train_batch,
            policy.config["spatial_noise_scale"],
            policy.config["sgld_beta_inv"],
            policy.config["sgld_n_steps"])

        # Replace current observation by the adversarial one
        train_batch_copy["obs"] = observation_worst

        # Append the training batches to the set
        train_batches["worst"] = train_batch_copy

    if policy.config["caps_global_reg"] > 0.0 or \
            not policy.config["enable_adversarial_noise"]:
        # Shallow copy the original training batch
        train_batch_copy = train_batch.copy(shallow=True)

        # Generate noisy observation
        observation_noisy = torch.normal(
            observation_true, policy.config["spatial_noise_scale"])

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
            field_shape = observation_space[field].shape
            field_size = reduce(operator.mul, field_shape)
            slice_idx = slice(offset, offset + field_size)

            mirror_mat = mirror_mat.to(device)
            observation_space.mirror_mat[field] = mirror_mat

            if len(field_shape) > 1:
                observation_true_slice = observation_true[:, slice_idx] \
                    .reshape((-1, field_shape[0], field_shape[1])) \
                    .swapaxes(1, 0) \
                    .reshape((field_shape[0], -1))
                observation_mirror_slice = mirror_mat @ observation_true_slice
                observation_mirror[:, slice_idx] = observation_mirror_slice \
                    .reshape((field_shape[0], -1, field_shape[1])) \
                    .swapaxes(1, 0) \
                    .reshape((-1, field_size))
            else:
                torch.mm(observation_true[..., slice_idx],
                         mirror_mat,
                         out=observation_mirror[..., slice_idx])

            offset += field_size

        # Replace current observation by the mirrored one
        train_batch_copy["obs"] = observation_mirror

        # Append the training batches to the set
        train_batches["mirrored"] = train_batch_copy

    # Compute the action_logits for all training batches at onces
    train_batch_all = {}
    for k in ["obs"]:
        train_batch_all[k] = torch.cat([
            s[k] for s in train_batches.values()], dim=0)
    action_logits_all, _ = model(train_batch_all)
    values_all = model.value_function()
    action_logits = dict(zip(train_batches.keys(), torch.chunk(
        action_logits_all, len(train_batches), dim=0)))
    values = dict(zip(train_batches.keys(), torch.chunk(
        values_all, len(train_batches), dim=0)))

    # Compute original ppo loss.
    # pylint: disable=unused-argument,missing-function-docstring
    class FakeModel:
        """Fake model enabling doing all forward passes at once.
        """
        def __init__(self,
                     model: ModelV2,
                     action_logits: torch.Tensor,
                     value: torch.Tensor) -> None:
            self._action_logits = action_logits
            self._value = value
            self._model = model

        def __getattr__(self, name: str) -> Any:
            return getattr(self._model, name)

        def __call__(self, *args: Any, **kwargs: Any
                     ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            return self._action_logits, []

        def value_function(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return self._value

    fake_model = FakeModel(model, action_logits["true"], values["true"])
    total_loss = ppo_surrogate_loss(
        policy, fake_model, dist_class, train_batch)

    # Extract mean of predicted action from action_logits.
    # No need to compute the perform model forward pass since the original
    # PPO loss is already doing it, so just getting back the last ouput.
    action_true_logits = action_logits["true"]
    action_true_mean = get_action_mean(model, dist_class, action_true_logits)

    # Compute the mean action corresponding to the previous observation
    if policy.config["caps_temporal_reg"] > 0.0 or \
            policy.config["temporal_barrier_reg"] > 0.0:
        action_prev_logits = action_logits["prev"]
        action_prev_mean = get_action_mean(
            model, dist_class, action_prev_logits)

    # Compute the mean action corresponding to the noisy observation
    if policy.config["caps_global_reg"] > 0.0 or \
            not policy.config["enable_adversarial_noise"]:
        action_noisy_logits = action_logits["noisy"]
        action_noisy_mean = get_action_mean(
            model, dist_class, action_noisy_logits)

    # Compute the mean action corresponding to the worst observation
    if policy.config["caps_spatial_reg"] > 0.0 and \
            policy.config["enable_adversarial_noise"]:
        action_worst_logits = action_logits["worst"]
        action_worst_mean = get_action_mean(
            model, dist_class, action_worst_logits)

    # Compute the mirrored mean action corresponding to the mirrored action
    if policy.config["symmetric_policy_reg"] > 0.0:
        action_mirror_logits = action_logits["mirrored"]
        action_mirror_mean = get_action_mean(
            model, dist_class, action_mirror_logits)
        action_mirror_mat = policy.action_space.mirror_mat.to(device)
        policy.action_space.mirror_mat = action_mirror_mat
        action_mirror_mean = action_mirror_mean @ action_mirror_mat

    if policy.config["caps_temporal_reg"] > 0.0 or \
            policy.config["temporal_barrier_reg"] > 0.0:
        # Compute action temporal delta
        action_delta = (action_prev_mean - action_true_mean).abs()

        # Minimize the difference between the successive action mean
        if policy.config["caps_temporal_reg"] > 0.0:
            policy._loss_caps_temporal_reg = torch.mean(action_delta)

        # Add temporal smoothness loss to total loss
        total_loss += policy.config["caps_temporal_reg"] * \
            policy._loss_caps_temporal_reg

        # Add temporal barrier loss to total loss:
        # exp(scale * (err - thr)) - 1.0 if err > thr else 0.0
        if policy.config["temporal_barrier_reg"] > 0.0:
            scale = policy.config["temporal_barrier_scale"]
            threshold = policy.config["temporal_barrier_threshold"]
            policy._loss_temporal_barrier_reg = torch.mean(torch.exp(
                torch.clamp(
                    scale * (action_delta - threshold), min=0.0, max=5.0
                    )) - 1.0)

        # Add spatial smoothness loss to total loss
        total_loss += policy.config["temporal_barrier_reg"] * \
            policy._loss_temporal_barrier_reg

    if policy.config["caps_spatial_reg"] > 0.0:
        # Minimize the difference between the original action mean and the
        # perturbed one.
        if policy.config["enable_adversarial_noise"]:
            policy._loss_caps_spatial_reg = torch.mean(
                torch.sum((action_worst_mean - action_true_mean) ** 2, dim=1))
        else:
            policy._loss_caps_spatial_reg = torch.mean(
                torch.sum((action_noisy_mean - action_true_mean) ** 2, dim=1))

        # Add spatial smoothness loss to total loss
        total_loss += policy.config["caps_spatial_reg"] * \
            policy._loss_caps_spatial_reg

    if policy.config["caps_global_reg"] > 0.0:
        # Minimize the magnitude of action mean
        policy._loss_caps_global_reg = torch.mean(action_noisy_mean ** 2)

        # Add global smoothness loss to total loss
        total_loss += policy.config["caps_global_reg"] * \
            policy._loss_caps_global_reg

    if policy.config["symmetric_policy_reg"] > 0.0:
        # Minimize the assymetry of policy output
        policy._loss_symmetric_policy_reg = torch.mean(
            (action_mirror_mean - action_true_mean) ** 2)

        # Add policy symmetry loss to total loss
        total_loss += policy.config["symmetric_policy_reg"] * \
            policy._loss_symmetric_policy_reg

    if policy.config["l2_reg"] > 0.0:
        # Add actor l2-regularization loss
        l2_reg_loss = 0.0
        for name, params in model.named_parameters():
            if not name.endswith("bias"):
                l2_reg_loss += l2_loss(params)
        policy._loss_l2_reg = l2_reg_loss

        # Add l2-regularization loss to total loss
        total_loss += policy.config["l2_reg"] * policy._loss_l2_reg

    return total_loss


def ppo_stats(policy: Policy,
              train_batch: SampleBatch) -> Dict[str, TensorType]:
    """ TODO: Write documentation.
    """
    # Compute original stats report
    stats_dict = kl_and_loss_stats(policy, train_batch)

    # Add spatial CAPS loss to the report
    if policy.config["symmetric_policy_reg"] > 0.0:
        stats_dict["symmetry"] = policy._loss_symmetric_policy_reg
    if policy.config["temporal_barrier_reg"] > 0.0:
        stats_dict["temporal_barrier"] = policy._loss_temporal_barrier_reg
    if policy.config["caps_temporal_reg"] > 0.0:
        stats_dict["temporal_smoothness"] = policy._loss_caps_temporal_reg
    if policy.config["caps_spatial_reg"] > 0.0:
        stats_dict["spatial_smoothness"] = policy._loss_caps_spatial_reg
    if policy.config["caps_global_reg"] > 0.0:
        stats_dict["global_smoothness"] = policy._loss_caps_global_reg
    if policy.config["l2_reg"] > 0.0:
        stats_dict["l2_reg"] = policy._loss_l2_reg

    return stats_dict


PPOTorchPolicy = PPOTorchPolicy.with_updates(
    before_loss_init=ppo_init,
    loss_fn=ppo_loss,
    stats_fn=ppo_stats,
    get_default_config=lambda: DEFAULT_CONFIG
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
