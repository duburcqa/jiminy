"""Implement several regularization losses on top of the original PPO algorithm
to encourage smoothness of the action and clustering of the behavior of the
policy without having to rework the reward function itself. It takes advantage
of the analytical gradient of the policy.
"""
import math
import operator
from functools import reduce, partial
from typing import Optional, Union, Type, List, Dict, Any, Tuple, cast

import gymnasium as gym
import torch

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig as _PPOConfig, PPO as _PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import (
    PPOTorchPolicy as _PPOTorchPolicy)
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import l2_loss
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


def get_action_mean(model: ModelV2,
                    dist_class: Union[partial, Type[ActionDistribution]],
                    action_logits: torch.Tensor) -> torch.Tensor:
    """Compute the mean value of the actions based on action distribution
    logits and type of distribution.

    .. note:
        It performs deterministic sampling for all distributions except
        multivariate independent normal distribution, for which the mean can be
        very efficiently extracted as a view of the logits.
    """
    # Extract wrapped distribution class
    dist_class_unwrapped: Type[ActionDistribution]
    if isinstance(dist_class, partial):
        dist_class_func = cast(Type[ActionDistribution], dist_class.func)
        assert issubclass(dist_class_func, ActionDistribution)
        dist_class_unwrapped = dist_class_func
    else:
        dist_class_unwrapped = dist_class

    # Efficient specialization for `TorchDiagGaussian` distribution
    if issubclass(dist_class_unwrapped, TorchDiagGaussian):
        action_mean, _ = torch.chunk(action_logits, 2, dim=1)
        return action_mean

    # Slow but generic fallback
    action_dist = dist_class(action_logits, model)
    return action_dist.deterministic_sample()


def get_adversarial_observation_sgld(
        model: ModelV2,
        dist_class: Type[ActionDistribution],
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
        with torch.torch.enable_grad():
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
                            space: gym.spaces.Box,
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
            field_shape = space.original_space[  # type: ignore[attr-defined]
                field].shape
            field_size = reduce(operator.mul, field_shape)
            slice_idx = slice(offset, offset + field_size)
            slice_mirrored = _update_flattened_slice(
                value[:, slice_idx], field_shape, slice_mirror_mat)
            value_mirrored.append(slice_mirrored)
            offset += field_size
        return torch.cat(value_mirrored, dim=1)
    return _update_flattened_slice(value, space.shape, mirror_mat)


class PPOConfig(_PPOConfig):
    """Provide additional parameters on top of the original PPO algorithm to
    configure several regularization losses. See `PPOTorchPolicy` for details.
    """
    def __init__(self, algo_class: Optional[Type["PPO"]] = None):
        super().__init__(algo_class=algo_class or PPO)

        self.spatial_noise_scale = 1.0
        self.enable_adversarial_noise = False
        self.sgld_beta_inv = 1e-8
        self.sgld_n_steps = 10
        self.temporal_barrier_scale = 10.0
        self.temporal_barrier_threshold = float('inf')
        self.temporal_barrier_reg = 0.0
        self.symmetric_policy_reg = 0.0
        self.caps_temporal_reg = 0.0
        self.caps_spatial_reg = 0.0
        self.caps_global_reg = 0.0
        self.l2_reg = 0.0

    @override(_PPOConfig)
    def training(
        self,
        *,
        enable_adversarial_noise: Optional[bool] = None,
        spatial_noise_scale: Optional[float] = None,
        sgld_beta_inv: Optional[float] = None,
        sgld_n_steps: Optional[int] = None,
        temporal_barrier_scale: Optional[float] = None,
        temporal_barrier_threshold: Optional[float] = None,
        temporal_barrier_reg: Optional[float] = None,
        symmetric_policy_reg: Optional[float] = None,
        caps_temporal_reg: Optional[float] = None,
        caps_spatial_reg: Optional[float] = None,
        caps_global_reg: Optional[float] = None,
        l2_reg: Optional[float] = None,
        **kwargs: Any,
    ) -> "PPOConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if enable_adversarial_noise is not None:
            self.enable_adversarial_noise = enable_adversarial_noise
        if spatial_noise_scale is not None:
            self.spatial_noise_scale = spatial_noise_scale
        if sgld_beta_inv is not None:
            self.sgld_beta_inv = sgld_beta_inv
        if sgld_n_steps is not None:
            self.sgld_n_steps = sgld_n_steps
        if temporal_barrier_scale is not None:
            self.temporal_barrier_scale = temporal_barrier_scale
        if temporal_barrier_threshold is not None:
            self.temporal_barrier_threshold = temporal_barrier_threshold
        if temporal_barrier_reg is not None:
            self.temporal_barrier_reg = temporal_barrier_reg
        if symmetric_policy_reg is not None:
            self.symmetric_policy_reg = symmetric_policy_reg
        if caps_temporal_reg is not None:
            self.caps_temporal_reg = caps_temporal_reg
        if caps_spatial_reg is not None:
            self.caps_spatial_reg = caps_spatial_reg
        if caps_global_reg is not None:
            self.caps_global_reg = caps_global_reg
        if l2_reg is not None:
            self.l2_reg = l2_reg

        return self


class PPO(_PPO):
    """Custom PPO algorithm with additional regularization losses on top of the
    original surrogate loss. See `PPOTorchPolicy` for details.
    """
    @classmethod
    @override(_PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        """Returns a default configuration for the algorithm.
        """
        return PPOConfig()

    @classmethod
    @override(_PPO)
    def get_default_policy_class(cls, config: AlgorithmConfig
                                 ) -> Optional[Type[Policy]]:
        """Returns a default Policy class to use, given a config.
        """
        framework = config.framework_str
        if framework == "torch":
            return PPOTorchPolicy
        raise ValueError(f"The framework {framework} is not supported.")


class PPOTorchPolicy(_PPOTorchPolicy):
    """Add regularization losses on top of the original loss of PPO.

    More specifically, it adds:
        - CAPS regularization, which combines the spatial and temporal
        difference between previous and current state
        - Global regularization, which is the average norm of the action
        - temporal barrier, which is exponential barrier loss when the
        normalized action is above a threshold (much like interior point
        methods).
        - symmetry regularization, which is the error between actions and
        symmetric actions associated with symmetric observations.
        - L2 regularization of policy network weights
    """
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: Union[PPOConfig, Dict[str, Any]]) -> None:
        """Initialize PPO Torch policy.

        It extracts observation mirroring transforms for symmetry computations.
        """
        # pylint: disable=non-parent-init-called,super-init-not-called

        # Convert any type of input dict input classical dictionary for compat
        config_dict: Dict[str, Any] = {**PPOConfig().to_dict(), **config}
        validate_config(config_dict)

        # Call base implementation. Note that `PPOTorchPolicy.__init__` is
        # bypassed because it calls `_initialize_loss_from_dummy_batch`
        # automatically, and mirroring matrices are not extracted at this
        # point. It is not possible to extract them since `self.device` is set
        # by `TorchPolicyV2.__init__`.
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config_dict,
            max_seq_len=config_dict["model"]["max_seq_len"],
        )

        # Initialize mixins
        ValueNetworkMixin.__init__(self, config_dict)
        LearningRateSchedule.__init__(
            self, config_dict["lr"], config_dict["lr_schedule"])
        EntropyCoeffSchedule.__init__(self,
                                      config_dict["entropy_coeff"],
                                      config_dict["entropy_coeff_schedule"])
        KLCoeffMixin.__init__(self, config_dict)

        # Extract and convert observation and action mirroring transform
        self.obs_mirror_mat: Optional[Union[
            Dict[str, torch.Tensor], torch.Tensor]] = None
        self.action_mirror_mat: Optional[Union[
            Dict[str, torch.Tensor], torch.Tensor]] = None
        if config_dict["symmetric_policy_reg"] > 0.0:
            # Observation space
            is_obs_dict = hasattr(observation_space, "original_space")
            if is_obs_dict:
                observation_space = observation_space.\
                    original_space  # type: ignore[attr-defined]
                self.obs_mirror_mat = {}
                for field, mirror_mat in observation_space.\
                        mirror_mat.items():  # type: ignore[attr-defined]
                    obs_mirror_mat = torch.tensor(mirror_mat,
                                                  dtype=torch.float32,
                                                  device=self.device)
                    self.obs_mirror_mat[field] = obs_mirror_mat.T.contiguous()
            else:
                obs_mirror_mat = torch.tensor(
                    observation_space.mirror_mat,  # type: ignore[attr-defined]
                    dtype=torch.float32,
                    device=self.device)
                self.obs_mirror_mat = obs_mirror_mat.T.contiguous()

            # Action space
            action_mirror_mat = torch.tensor(
                action_space.mirror_mat,  # type: ignore[attr-defined]
                dtype=torch.float32,
                device=self.device)
            self.action_mirror_mat = action_mirror_mat.T.contiguous()

        self._initialize_loss_from_dummy_batch()
        self.config: Dict[str, Any]

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

    @override(_PPOTorchPolicy)
    def loss(self,
             model: ModelV2,
             dist_class: Type[ActionDistribution],
             train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        """Compute PPO loss with additional regularizations.
        """
        with torch.no_grad():
            # Extract some proxies from convenience
            observation_true = train_batch["obs"]

            # Initialize the various training batches to forward to the model
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
                assert isinstance(self.observation_space, gym.spaces.Box)
                observation_mirror = _compute_mirrored_value(
                    observation_true,
                    self.observation_space,
                    self.obs_mirror_mat)

                # Replace current observation by the mirrored one
                train_batch_copy["obs"] = observation_mirror

                # Append the training batches to the set
                train_batches["mirrored"] = train_batch_copy

        # Compute the action_logits for all the training batches at onces
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
        class FakeModel:
            """Fake model enabling doing all forward passes at once.
            """
            # pylint: disable=unused-argument,missing-function-docstring
            def __init__(self,
                         model: ModelV2,
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
        # PPO loss is already doing it, so just getting back the last output.
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
            assert isinstance(self.action_space, gym.spaces.Box)
            action_mirror_logits = action_logits["mirrored"]
            action_mirror_mean = get_action_mean(
                model, dist_class, action_mirror_logits)
            action_revert_mean = _compute_mirrored_value(
                action_mirror_mean,
                self.action_space,
                self.action_mirror_mat)

        # Update total loss
        stats = model.tower_stats  # type: ignore[attr-defined]
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

    @override(_PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Add regularization statistics.
        """
        stats_dict = super().stats_fn(train_batch)

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


__all__ = [
    "PPOConfig",
    "PPOTorchPolicy",
    "PPO"
]
