""" TODO: Write documentation.
"""
import math
import operator
from functools import reduce
from typing import Dict, List, Type, Union, Optional, Any, Tuple

import gym
import torch

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper, TorchDiagGaussian)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule, TorchPolicy
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import l2_loss
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

from ray.rllib.agents.ppo import (
    DEFAULT_CONFIG as _DEFAULT_CONFIG, PPOTrainer as _PPOTrainer)
from ray.rllib.agents.ppo.ppo_torch_policy import (
    PPOTorchPolicy as _PPOTorchPolicy)


DEFAULT_CONFIG = _PPOTrainer.merge_trainer_configs(
    _DEFAULT_CONFIG,
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


def get_action_mean(model: TorchModelV2,
                    dist_class: Type[TorchDistributionWrapper],
                    action_logits: torch.Tensor) -> torch.Tensor:
    """Compute the mean value of the actions based on action distribution
    logits and type of distribution.

    .. note:
        It performs deterministic sampling for all distributions except
        multivariate independent normal distribution, for which the mean can be
        very efficiently extracted as a view of the logits.
    """
    if issubclass(dist_class, TorchDiagGaussian):
        action_mean, _ = torch.chunk(action_logits, 2, dim=1)
    else:
        action_dist = dist_class(action_logits, model)
        action_mean = action_dist.deterministic_sample()
    return action_mean


def get_adversarial_observation_sgld(
        model: TorchModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
        noise_scale: float,
        beta_inv: float,
        n_steps: int,
        action_true_mean: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
    """Compute adversarial observation maximizing Mean Squared Error between
    the original and the perturbed mean action using Stochastic gradient
    Langevin dynamics algorithm (SGLD).
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


def _compute_mirrored_value(value: torch.Tensor,
                            space: gym.spaces.Space,
                            mirror_mat: Union[
                                Dict[str, torch.Tensor], torch.Tensor]
                            ) -> torch.Tensor:
    """Compute mirrored value from observation space based on provided
    mirroring transformation.
    """
    def _update_flattened_slice(data: torch.Tensor,
                                shape: Tuple[int, ...],
                                mirror_mat: torch.Tensor) -> torch.Tensor:
        """Mirror an array of flattened tensor using provided transformation
        matrix.
        """
        if len(shape) > 1:
            data = data.reshape((-1, *shape)) \
                       .swapaxes(1, 0) \
                       .reshape((shape[0], -1))
            data_mirrored = mirror_mat @ data
            return data_mirrored.reshape((shape[0], -1, shape[1])) \
                                .swapaxes(1, 0) \
                                .reshape((-1, *shape))
        return torch.mm(data, mirror_mat)

    if isinstance(mirror_mat, dict):
        offset = 0
        value_mirrored = []
        for field, slice_mirror_mat in mirror_mat.items():
            field_shape = space.original_space[field].shape
            field_size = reduce(operator.mul, field_shape)
            slice_idx = slice(offset, offset + field_size)
            slice_mirrored = _update_flattened_slice(
                value[:, slice_idx], field_shape, slice_mirror_mat)
            value_mirrored.append(slice_mirrored)
            offset += field_size
        return torch.cat(value_mirrored, dim=1)
    return _update_flattened_slice(value, space.shape, mirror_mat)


class PPOTorchPolicy(_PPOTorchPolicy):
    """Add regularization losses on top of the original loss of PPO.
    More specifically, it adds:
        - CAPS regularization, which combines the spatial and temporal
          difference betwen previous and current state
        - Global regularization, which is the average norm of the action
        - temporal barrier, which is exponential barrier loss when the
          normalized action is above a threshold (much like interior point
          methods).
        - symmetry regularization, which is the error between actions and
        symmetric actions associated with symmetric observations.
        - L2 regularization of policy network weights
    """
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
        """Initialize PPO Torch policy.

        It extracts observation mirroring transforms for symmetry computations.
        """
        # pylint: disable=non-parent-init-called,super-init-not-called

        # Update default config wich provided partial config
        config = dict(DEFAULT_CONFIG, **config)

        # Call base implementation. Note that `PPOTorchPolicy.__init__` is
        # bypassed because it calls `_initialize_loss_from_dummy_batch`
        # automatically, and mirroring matrices are not extracted at this
        # point. It is not possible to extract them since `self.device` is set
        # by `TorchPolicy.__init__`.
        TorchPolicy.__init__(
            self,
            obs_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"])

        # TODO: Remove update of `policy.model.view_requirements` after ray fix
        # https://github.com/ray-project/ray/pull/21043
        self.model.view_requirements["prev_obs"] = \
            self.view_requirements["prev_obs"]

        # Initialize mixins
        EntropyCoeffSchedule.__init__(self, config["entropy_coeff"],
                                      config["entropy_coeff_schedule"])
        LearningRateSchedule.__init__(self, config["lr"],
                                      config["lr_schedule"])

        # Current KL value
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value
        self.kl_target = self.config["kl_target"]

        # Extract and convert observation and acrtion mirroring transform
        self.obs_mirror_mat: Optional[Union[
            Dict[str, torch.Tensor], torch.Tensor]] = None
        self.action_mirror_mat: Optional[Union[
            Dict[str, torch.Tensor], torch.Tensor]] = None
        if config["symmetric_policy_reg"] > 0.0:
            is_obs_dict = hasattr(obs_space, "original_space")
            if is_obs_dict:
                obs_space = obs_space.original_space
            # Observation space
            if is_obs_dict:
                self.obs_mirror_mat = {}
                for field, mirror_mat in obs_space.mirror_mat.items():
                    obs_mirror_mat = torch.tensor(mirror_mat,
                                                  dtype=torch.float32,
                                                  device=self.device)
                    self.obs_mirror_mat[field] = obs_mirror_mat.T.contiguous()
            else:
                obs_mirror_mat = torch.tensor(obs_space.mirror_mat,
                                              dtype=torch.float32,
                                              device=self.device)
                self.obs_mirror_mat = obs_mirror_mat.T.contiguous()

            # Action space
            action_mirror_mat = torch.tensor(action_space.mirror_mat,
                                             dtype=torch.float32,
                                             device=self.device)
            self.action_mirror_mat = action_mirror_mat.T.contiguous()

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    def _get_default_view_requirements(self) -> None:
        """Add previous observation to view requirements for CAPS
        regularization.
        """
        view_requirements = super()._get_default_view_requirements()
        view_requirements["prev_obs"] = ViewRequirement(
            data_col=SampleBatch.OBS,
            space=self.observation_space,
            shift=-1,
            used_for_compute_actions=False,
            used_for_training=True)
        return view_requirements

    def loss(self,
             model: TorchModelV2,
             dist_class: Type[TorchDistributionWrapper],
             train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        """Compute PPO loss with additional regulizations.
        """
        with torch.no_grad():
            # Extract some proxies from convenience
            observation_true = train_batch["obs"]

            # Initialize the set of training batches to forward to the model
            train_batches = {"true": train_batch}

            if self.config["caps_temporal_reg"] > 0.0 or \
                    self.config["temporal_barrier_reg"] > 0.0:
                # Shallow copy the original training batch
                train_batch_copy = train_batch.copy(shallow=True)

                # Replace current observation by the previous one
                observation_prev = train_batch["prev_obs"]
                train_batch_copy["obs"] = observation_prev

                # Append the training batches to the set
                train_batches["prev"] = train_batch_copy

            if self.config["caps_spatial_reg"] > 0.0 and \
                    self.config["enable_adversarial_noise"]:
                # Shallow copy the original training batch
                train_batch_copy = train_batch.copy(shallow=True)

                # Compute adversarial observation maximizing action difference
                observation_worst = get_adversarial_observation_sgld(
                    model, dist_class, train_batch,
                    self.config["spatial_noise_scale"],
                    self.config["sgld_beta_inv"],
                    self.config["sgld_n_steps"])

                # Replace current observation by the adversarial one
                train_batch_copy["obs"] = observation_worst

                # Append the training batches to the set
                train_batches["worst"] = train_batch_copy

            if self.config["caps_global_reg"] > 0.0 or \
                    not self.config["enable_adversarial_noise"]:
                # Shallow copy the original training batch
                train_batch_copy = train_batch.copy(shallow=True)

                # Generate noisy observation
                observation_noisy = torch.normal(
                    observation_true, self.config["spatial_noise_scale"])

                # Replace current observation by the noisy one
                train_batch_copy["obs"] = observation_noisy

                # Append the training batches to the set
                train_batches["noisy"] = train_batch_copy

            if self.config["symmetric_policy_reg"] > 0.0:
                # Shallow copy the original training batch
                train_batch_copy = train_batch.copy(shallow=True)

                # Compute mirrorred observation
                assert self.obs_mirror_mat is not None
                observation_mirror = _compute_mirrored_value(
                    observation_true,
                    self.observation_space,
                    self.obs_mirror_mat)

                # Replace current observation by the mirrored one
                train_batch_copy["obs"] = observation_mirror

                # Append the training batches to the set
                train_batches["mirrored"] = train_batch_copy

        # Compute the action_logits for all training batches at onces
        train_batch_all = {
            "obs": torch.cat([
                s["obs"] for s in train_batches.values()], dim=0)}
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
                         model: TorchModelV2,
                         action_logits: torch.Tensor,
                         value: torch.Tensor) -> None:
                self._action_logits = action_logits
                self._value = value
                self._model = model

            def __getattr__(self, name: str) -> Any:
                return getattr(self._model, name)

            def __call__(
                    self, *args: Any, **kwargs: Any
                    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                return self._action_logits, []

            def value_function(
                    self, *args: Any, **kwargs: Any) -> torch.Tensor:
                return self._value

        fake_model = FakeModel(model, action_logits["true"], values["true"])
        total_loss = super().loss(fake_model, dist_class, train_batch)

        # Extract mean of predicted action from action_logits.
        # No need to compute the perform model forward pass since the original
        # PPO loss is already doing it, so just getting back the last ouput.
        action_true_logits = action_logits["true"]
        action_true_mean = get_action_mean(
            model, dist_class, action_true_logits)

        # Compute the mean action corresponding to the previous observation
        if self.config["caps_temporal_reg"] > 0.0 or \
                self.config["temporal_barrier_reg"] > 0.0:
            action_prev_logits = action_logits["prev"]
            action_prev_mean = get_action_mean(
                model, dist_class, action_prev_logits)

        # Compute the mean action corresponding to the noisy observation
        if self.config["caps_global_reg"] > 0.0 or \
                not self.config["enable_adversarial_noise"]:
            action_noisy_logits = action_logits["noisy"]
            action_noisy_mean = get_action_mean(
                model, dist_class, action_noisy_logits)

        # Compute the mean action corresponding to the worst observation
        if self.config["caps_spatial_reg"] > 0.0 and \
                self.config["enable_adversarial_noise"]:
            action_worst_logits = action_logits["worst"]
            action_worst_mean = get_action_mean(
                model, dist_class, action_worst_logits)

        # Compute the mirrored mean action corresponding to the mirrored action
        if self.config["symmetric_policy_reg"] > 0.0:
            assert self.action_mirror_mat is not None
            action_mirror_logits = action_logits["mirrored"]
            action_mirror_mean = get_action_mean(
                model, dist_class, action_mirror_logits)
            action_revert_mean = _compute_mirrored_value(
                action_mirror_mean,
                self.action_space,
                self.action_mirror_mat)

        # Update total loss
        stats = model.tower_stats
        if self.config["caps_temporal_reg"] > 0.0 or \
                self.config["temporal_barrier_reg"] > 0.0:
            # Compute action temporal delta
            action_delta = (action_prev_mean - action_true_mean).abs()

            if self.config["caps_temporal_reg"] > 0.0:
                # Minimize the difference between the successive action mean
                caps_temporal_reg = torch.mean(action_delta)

                # Add temporal smoothness loss to total loss
                stats["caps_temporal_reg"] = caps_temporal_reg
                total_loss += \
                    self.config["caps_temporal_reg"] * caps_temporal_reg

            if self.config["temporal_barrier_reg"] > 0.0:
                # Add temporal barrier loss to total loss:
                # exp(scale * (err - thr)) - 1.0 if err > thr else 0.0
                scale = self.config["temporal_barrier_scale"]
                threshold = self.config["temporal_barrier_threshold"]
                temporal_barrier_reg = torch.mean(torch.exp(
                    torch.clamp(
                        scale * (action_delta - threshold), min=0.0, max=5.0
                        )) - 1.0)

                # Add spatial smoothness loss to total loss
                stats["temporal_barrier_reg"] = temporal_barrier_reg
                total_loss += \
                    self.config["temporal_barrier_reg"] * temporal_barrier_reg

        if self.config["caps_spatial_reg"] > 0.0:
            # Minimize the difference between the original action mean and the
            # perturbed one.
            if self.config["enable_adversarial_noise"]:
                caps_spatial_reg = torch.mean(torch.sum(
                    (action_worst_mean - action_true_mean) ** 2, dim=1))
            else:
                caps_spatial_reg = torch.mean(torch.sum(
                    (action_noisy_mean - action_true_mean) ** 2, dim=1))

            # Add spatial smoothness loss to total loss
            stats["caps_spatial_reg"] = caps_spatial_reg
            total_loss += self.config["caps_spatial_reg"] * caps_spatial_reg

        if self.config["caps_global_reg"] > 0.0:
            # Minimize the magnitude of action mean
            caps_global_reg = torch.mean(action_noisy_mean ** 2)

            # Add global smoothness loss to total loss
            stats["caps_global_reg"] = caps_global_reg
            total_loss += self.config["caps_global_reg"] * caps_global_reg

        if self.config["symmetric_policy_reg"] > 0.0:
            # Minimize the assymetry of self output
            symmetric_policy_reg = torch.mean(
                (action_revert_mean - action_true_mean) ** 2)

            # Add policy symmetry loss to total loss
            stats["symmetric_policy_reg"] = symmetric_policy_reg
            total_loss += \
                self.config["symmetric_policy_reg"] * symmetric_policy_reg

        if self.config["l2_reg"] > 0.0:
            # Add actor l2-regularization loss
            l2_reg = 0.0
            assert isinstance(model, torch.nn.Module)
            for name, params in model.named_parameters():
                if not name.endswith("bias") and params.requires_grad:
                    l2_reg += l2_loss(params)

            # Add l2-regularization loss to total loss
            stats["l2_reg"] = l2_reg
            total_loss += self.config["l2_reg"] * l2_reg

        return total_loss

    def extra_grad_info(self,
                        train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Add regularization values to statistics.
        """
        stats_dict = super().extra_grad_info(train_batch)

        if self.config["symmetric_policy_reg"] > 0.0:
            stats_dict["symmetry"] = torch.mean(
                torch.stack(self.get_tower_stats("symmetric_policy_reg")))
        if self.config["temporal_barrier_reg"] > 0.0:
            stats_dict["temporal_barrier"] = torch.mean(
                torch.stack(self.get_tower_stats("temporal_barrier_reg")))
        if self.config["caps_temporal_reg"] > 0.0:
            stats_dict["temporal_smoothness"] = torch.mean(
                torch.stack(self.get_tower_stats("caps_temporal_reg")))
        if self.config["caps_spatial_reg"] > 0.0:
            stats_dict["spatial_smoothness"] = torch.mean(
                torch.stack(self.get_tower_stats("caps_spatial_reg")))
        if self.config["caps_global_reg"] > 0.0:
            stats_dict["global_smoothness"] = torch.mean(
                torch.stack(self.get_tower_stats("caps_global_reg")))
        if self.config["l2_reg"] > 0.0:
            stats_dict["l2_reg"] = torch.mean(
                torch.stack(self.get_tower_stats("l2_reg")))

        return convert_to_numpy(stats_dict)


class PPOTrainer(_PPOTrainer):
    """Custom PPO Trainer with additional regularization losses on top of the
    original surrogate loss. See `PPOTorchPolicy` for details.
    """
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        """Returns a default configuration for the Trainer.
        """
        return DEFAULT_CONFIG

    def get_default_policy_class(self,
                                 config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return PPOTorchPolicy
        raise RuntimeError("The only framework supported is 'torch'.")


__all__ = [
    "DEFAULT_CONFIG",
    "PPOTorchPolicy",
    "PPOTrainer"
]
