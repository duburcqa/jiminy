"""Implement several regularization losses on top of the original PPO algorithm
to encourage smoothness of the action and clustering of the behavior of the
policy without having to rework the reward function itself. It takes advantage
of the analytical gradient of the policy.
"""
import math
from typing import (
    Optional, Union, Sequence, Type, Dict, Any, List, Tuple, cast)

import numpy as np
import torch

from ray.rllib import SampleBatch
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import (
    POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY, Learner)
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig as _PPOConfig,
    PPO as _PPO)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
    PPOTorchLearner as _PPOTorchLearner)
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch
from ray.rllib.connectors.learner.\
    add_next_observations_from_episodes_to_train_batch import (
        AddNextObservationsFromEpisodesToTrainBatch)
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.torch_utils import explained_variance, l2_loss
from ray.rllib.utils.annotations import override
from ray.rllib.utils.from_config import _NotProvided, NotProvided
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.rllib.utils.typing import TensorType, EpisodeType, ModuleID

from jiminy_py import tree


LEARNER_RESULTS_CURR_L2_REG_KEY = "curr_l2_reg"


ObsMirrorMat = Union[np.ndarray, Sequence[np.ndarray]]
ActMirrorMat = Union[np.ndarray, Sequence[np.ndarray]]


def copy_batch(batch: SampleBatch) -> SampleBatch:
    """Creates a shallow copy of a given batch.

    .. note::
        The original implementation for shallow copy `batch.copy(shallow=True)`
        is extremely slow, and as such, its uses must be avoided.

    :param batch: Batch to copy.
    """
    return SampleBatch(
        dict(batch),
        _time_major=batch.time_major,
        _zero_padded=batch.zero_padded,
        _max_seq_len=batch.max_seq_len,
        _num_grad_updates=batch.num_grad_updates)


def get_adversarial_observation_sgld(
        module: RLModule,
        batch: SampleBatch,
        fwd_out: Dict[str, TensorType],
        noise_scale: float,
        beta_inv: float,
        n_steps: int) -> torch.Tensor:
    """Compute adversarial observation maximizing Mean Squared Error between
    the original and the perturbed mean action using Stochastic gradient
    Langevin dynamics algorithm (SGLD).
    """
    # Compute mean field action for true observation
    action_dist_class_train = module.get_train_action_dist_cls()
    action_dist = action_dist_class_train.from_logits(
        fwd_out[Columns.ACTION_DIST_INPUTS])
    action_true_mean = action_dist.to_deterministic().sample()

    # Shallow copy the original training batch.
    # Be careful accessing fields using the original batch to properly keep
    # track of accessed keys, which will be used to automatically discard
    # useless components of policy's view requirements.
    batch_copy = copy_batch(batch)

    # Extract original observation
    observation_true = batch[Columns.OBS]

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
        with torch.enable_grad():
            # Make sure gradient computation is required
            observation_noisy.requires_grad_(True)

            # Compute mean field action for noisy observation.
            # Note that `forward_train` must be called in place of
            # `forward_inference` to force computing the gradient.
            batch_copy[Columns.OBS] = observation_noisy
            outs = module.forward_train(batch_copy)
            action_dist = action_dist_class_train.from_logits(
                outs[Columns.ACTION_DIST_INPUTS])
            action_noisy_mean = action_dist.to_deterministic().sample()

            # Compute action different and associated gradient
            objective = torch.mean(torch.sum(
                (action_noisy_mean - action_true_mean) ** 2, dim=-1))
            objective.backward()

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
        # using `noise_scale` converges MUCH faster.
        observation_noisy += observation_update.sign() * noise_scale

        # clip into the upper and lower bounds
        observation_noisy.clamp_(obs_lb_flat, obs_ub_flat)

    return observation_noisy


def _compute_mirrored_value(value: torch.Tensor,
                            mirror_mat_nested: Tuple[torch.Tensor, ...]
                            ) -> torch.Tensor:
    """Compute mirrored value from observation space based on provided
    mirroring transformation.
    """
    offset, data_mirrored_all = 0, []
    for mirror_mat in mirror_mat_nested:
        size, _ = mirror_mat.shape
        data_mirrored_all.append(
            value[:, offset:(offset + size)] @ mirror_mat)
        offset += size
    return torch.cat(data_mirrored_all, dim=1)


class AddNextStatesFromEpisodesToTrainBatch(ConnectorV2):
    """Gets the last hidden state from a running episode and adds it to the
    batch, in case of the stateful RL module.

    .. note:
        This connector is added automatically to the Learner connector pipeline
        by `gym_jiminy.rllib.ppo.PPO` algorithm.

    .. warning:
        This connector only supports Learner pipeline (as opposed to
        env-to-module pipeline).
    """
    @override(ConnectorV2)
    def __call__(self,
                 *,
                 rl_module: RLModule,
                 batch: Dict[str, Any],
                 episodes: List[EpisodeType],
                 explore: Optional[bool] = None,
                 shared_data: Optional[dict] = None,
                 **kwargs: Any) -> Any:
        # Early return if not stateful
        if not rl_module.is_stateful():
            return batch

        # If "obs" already in `batch`, early out.
        if Columns.STATE_OUT in batch:
            return batch

        for episode in self.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=False):
            state_outs = episode.get_extra_model_outputs(key=Columns.STATE_OUT)
            self.add_n_batch_items(
                batch,
                Columns.STATE_OUT,
                items_to_add=state_outs,
                num_items=len(episode),
                single_agent_episode=episode)
        return batch


class PPOConfig(_PPOConfig):
    """Provide additional parameters on top of the original PPO algorithm to
    configure several regularization losses.

    .. seealso:
        See `gym_jiminy.rllib.ppo.PPOTorchLearner` for details.
    """
    def __init__(self, algo_class: Optional[Type["PPO"]] = None):
        super().__init__(algo_class=algo_class or PPO)

        # Enable new API stack by default
        self.enable_rl_module_and_learner = True
        self.enable_env_runner_and_connector_v2 = True

        # Define additional parameters
        self.spatial_noise_scale = 1.0
        self.enable_adversarial_noise = False
        self.sgld_beta_inv = 1e-8
        self.sgld_n_steps = 6
        self.temporal_barrier_scale = 10.0
        self.temporal_barrier_threshold = float('inf')
        self.temporal_barrier_reg = 0.0
        self.symmetric_policy_reg = 0.0
        self.symmetric_spec: Tuple[ObsMirrorMat, ActMirrorMat] = ([], [])
        self.enable_symmetry_surrogate_loss = False
        self.caps_temporal_reg = 0.0
        self.caps_spatial_reg = 0.0
        self.caps_global_reg = 0.0
        self.l2_reg = 0.0

    @override(_PPOConfig)
    def training(
        self,
        *,
        enable_adversarial_noise: Union[_NotProvided, bool] = NotProvided,
        spatial_noise_scale: Union[_NotProvided, float] = NotProvided,
        sgld_beta_inv: Union[_NotProvided, float] = NotProvided,
        sgld_n_steps: Union[_NotProvided, int] = NotProvided,
        temporal_barrier_scale: Union[_NotProvided, float] = NotProvided,
        temporal_barrier_threshold: Union[_NotProvided, float] = NotProvided,
        temporal_barrier_reg: Union[_NotProvided, float] = NotProvided,
        symmetric_policy_reg: Union[_NotProvided, float] = NotProvided,
        symmetric_spec: Union[
            _NotProvided, Tuple[ObsMirrorMat, ActMirrorMat]] = NotProvided,
        enable_symmetry_surrogate_loss: Union[
            _NotProvided, bool] = NotProvided,
        caps_temporal_reg: Union[_NotProvided, float] = NotProvided,
        caps_spatial_reg: Union[_NotProvided, float] = NotProvided,
        caps_global_reg: Union[_NotProvided, float] = NotProvided,
        l2_reg: Union[_NotProvided, float] = NotProvided,
        **kwargs: Any,
    ) -> "PPOConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if not isinstance(enable_adversarial_noise, _NotProvided):
            self.enable_adversarial_noise = enable_adversarial_noise
        if not isinstance(spatial_noise_scale, _NotProvided):
            self.spatial_noise_scale = spatial_noise_scale
        if not isinstance(sgld_beta_inv, _NotProvided):
            self.sgld_beta_inv = sgld_beta_inv
        if not isinstance(sgld_n_steps, _NotProvided):
            self.sgld_n_steps = sgld_n_steps
        if not isinstance(temporal_barrier_scale, _NotProvided):
            self.temporal_barrier_scale = temporal_barrier_scale
        if not isinstance(temporal_barrier_threshold, _NotProvided):
            self.temporal_barrier_threshold = temporal_barrier_threshold
        if not isinstance(temporal_barrier_reg, _NotProvided):
            self.temporal_barrier_reg = temporal_barrier_reg
        if not isinstance(symmetric_policy_reg, _NotProvided):
            self.symmetric_policy_reg = symmetric_policy_reg
        if not isinstance(symmetric_spec, _NotProvided):
            self.symmetric_spec = symmetric_spec
        if not isinstance(enable_symmetry_surrogate_loss, _NotProvided):
            self.enable_symmetry_surrogate_loss = (
                enable_symmetry_surrogate_loss)
        if not isinstance(caps_temporal_reg, _NotProvided):
            self.caps_temporal_reg = caps_temporal_reg
        if not isinstance(caps_spatial_reg, _NotProvided):
            self.caps_spatial_reg = caps_spatial_reg
        if not isinstance(caps_global_reg, _NotProvided):
            self.caps_global_reg = caps_global_reg
        if not isinstance(l2_reg, _NotProvided):
            self.l2_reg = l2_reg

        return self

    @override(_PPOConfig)
    def get_default_learner_class(self) -> Union[Type[Learner], str]:
        if self.framework_str == "torch":
            return PPOTorchLearner
        raise ValueError(
            f"The framework {self.framework_str} is not supported. Please "
            "use 'torch'.")

    def validate(self) -> None:
        # Call base implementation
        super().validate()

        # Make sure that the learner class is valid
        assert issubclass(self.learner_class, PPOTorchLearner)


class PPO(_PPO):
    """Custom PPO algorithm with additional regularization losses on top of the
    original surrogate loss.

    .. seealso:
        See `gym_jiminy.rllib.ppo.PPOTorchLearner` for details.
    """
    @classmethod
    @override(_PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        """Returns a default configuration for the algorithm.
        """
        return PPOConfig()


class PPOTorchLearner(_PPOTorchLearner):
    """Compute regularization lossed for conditioning action policy.

    More specifically, it adds:
        - CAPS regularization, which combines the spatial and temporal
        difference between previous and current state.
        - Global regularization, which is the average norm of the action
        - temporal barrier, which is exponential barrier loss when the
        normalized action is above a threshold (much like interior point
        methods).
        - symmetry regularization, which is the error between actions and
        symmetric actions associated with symmetric observations.
        - symmetry surrogate loss, which is the surrogate loss associated
        with the symmetric (actions, observations) spaces. As the surrogate
        loss goal is to increase the likelihood of selecting higher reward
        actions given the current state, the symmetry surrogate loss enables
        equivalent likelihood increase for selecting the symmetric higher
        reward actions given the symmetric state.
        - L2 regularization of policy network weights.

    .. warning::
        One must define how to mirror observations and actions. For now, the
        mirroring transformations being supported are those that can be written
        in matrix form as a left-hand side product, i.e.
        `(obs|action)_mirrored = (obs|action) @ mirror_mat`. The mirroring
        matrices for the observations and actions must be specified by setting
        the extra attribute `mirror_mat: StructNested[np.ndarray]` in their
        respective `gym.Space`. If the observation and/or action spaces are
        flattened by some environment wrapper, then the mirroring transform is
        applied block-by-block on each leaf of the original data structure
        after unflattening the data. Besides, block matrix multiplication is
        supported for efficiency. This means that one can specify a sequence of
        matrix blocks instead of one big mirroring matrix. In such a case, it
        implements block matrix multiplication where slices of the observations
        and/or actions are successively mirrored.

    .. seealso::
        More insights on the regularization losses with their emerging
        properties, and on how to tune the parameters can be found in the
        reference articles:
            - A. Duburcq, F. Schramm, G. Boeris, N. Bredeche, and Y.
            Chevaleyre, “Reactive Stepping for Humanoid Robots using
            Reinforcement Learning: Application to Standing Push Recovery on
            the Exoskeleton Atalante”, in International Conference on
            Intelligent Robots and Systems (IROS), 2022
            - S. Mysore, B. Mabsout, R. Mancuso, and K. Saenko, “Regularizing
            action policies for smooth control with reinforcement learning”,
            IEEE International Conference on Robotics and Automation (ICRA),
            2021
            - M. Mittal, N. Rudin, V. Klemm, A. Allshire, and M. Hutter,
            “Symmetry considerations for learning task symmetric robot
            policies”, IEEE International Conference on Robotics and Automation
            (ICRA), 2024
    """
    @override(_PPOTorchLearner)
    def build(self) -> None:
        # pylint: disable=attribute-defined-outside-init

        # Call base implementation
        super().build()

        # Dict mapping module IDs to the respective L2-reg Scheduler instance
        assert isinstance(self.config, PPOConfig)
        self.l2_reg_schedulers_per_module: Dict[ModuleID, Scheduler]
        self.l2_reg_schedulers_per_module = LambdaDefaultDict(
            lambda module_id: Scheduler(
                fixed_value_or_schedule=(
                    self.config.get_config_for_module(
                        module_id).l2_reg),  # type: ignore[attr-defined]
                framework=self.framework,
                device=self._device))

        # Make sure that the environment is single-agent
        if self.config.is_multi_agent():
            raise RuntimeError("Multi-agent environments are not supported")

        # Make sure that the default connectors are enabled
        assert self.config.add_default_connectors_to_learner_pipeline

        # Prepend a "add-STATE_OUT-from-episodes-to-train-batch" connector
        assert self._learner_connector is not None
        self._learner_connector.prepend(
            AddNextStatesFromEpisodesToTrainBatch())

        # Prepend a "add-NEXT_OBS-from-episodes-to-train-batch" connector
        # piece right after the corresponding "add-OBS-..." default piece.
        self._learner_connector.insert_after(
            AddObservationsFromEpisodesToBatch,
            AddNextObservationsFromEpisodesToTrainBatch())

        # Extract module configuration
        module_config: PPOConfig = cast(
            PPOConfig, self.config.get_config_for_module(DEFAULT_MODULE_ID))

        # Extract flattened sequence of mirroring matrix blocks.
        # Convert recursively each mirror matrix to `torch.Tensor` with
        # dtype `torch.float32` which is stored on the expected device.
        if (module_config.symmetric_policy_reg or
                module_config.enable_symmetry_surrogate_loss):
            self.obs_mirror_mat_nested, self.action_mirror_mat_nested = (tuple(
                torch.tensor(
                    mirror_mat,
                    dtype=torch.float32,
                    device=self._device)
                for mirror_mat in tree.flatten(mirror_mat_nested))
                for mirror_mat_nested in module_config.symmetric_spec)

    @override(_PPOTorchLearner)
    def remove_module(self,
                      module_id: ModuleID,
                      **kwargs: Any) -> MultiRLModuleSpec:
        # Call base implementation
        marl_spec = super().remove_module(module_id, **kwargs)

        # Remove L2-regularization scheduler
        self.l2_reg_schedulers_per_module.pop(module_id, None)

        return marl_spec

    @override(_PPOTorchLearner)
    def after_gradient_based_update(
            self, *, timesteps: Dict[str, Any]) -> None:
        # Call base implementation
        super().after_gradient_based_update(timesteps=timesteps)

        # Update L2-regularization coefficient via Scheduler
        for module_id, _ in self.module._rl_modules.items():
            new_l2_reg = self.l2_reg_schedulers_per_module[module_id].update(
                timestep=timesteps.get(NUM_ENV_STEPS_SAMPLED_LIFETIME, 0))
            self.metrics.log_value(
                (module_id, LEARNER_RESULTS_CURR_L2_REG_KEY),
                new_l2_reg,
                window=1)

    @override(_PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: SampleBatch,
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        """Compute PPO loss with additional regularizations.
        """
        # pylint: disable=possibly-used-before-assignment

        # Extract the right RL Module
        rl_module = self.module[module_id].unwrapped()

        # Assert(s) for type checker
        assert isinstance(rl_module, TorchRLModule)

        # FIXME: 'Fixed' base PPO implemented.
        # Fix "broken" statistics computation keeping only most recent value
        # instead of keeping track of all the gradient updates at each training
        # iterations.
        # Fix explained variance wrong computed due to being computed unmasked.
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]

            def possibly_masked_mean(data: torch.Tensor) -> torch.Tensor:
                nonlocal mask
                return torch.mean(data[mask])
        else:
            possibly_masked_mean = torch.mean  # type: ignore[assignment]

        action_dist_class_train = rl_module.get_train_action_dist_cls()
        action_dist_class_explore = rl_module.get_exploration_action_dist_cls()
        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS])
        prev_action_dist = action_dist_class_explore.from_logits(
            batch[Columns.ACTION_DIST_INPUTS])
        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS]) -
            batch[Columns.ACTION_LOGP])

        if config.use_kl_loss:
            kl_coef = self.curr_kl_coeffs_per_module[module_id]
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=self._device)

        entropy_sched = self.entropy_coeff_schedulers_per_module[module_id]
        entropy_coef = entropy_sched.get_current_value()
        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = torch.min(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            batch[Postprocessing.ADVANTAGES] * torch.clamp(
                logp_ratio, 1 - config.clip_param, 1 + config.clip_param))
        mean_surrogate_loss = possibly_masked_mean(surrogate_loss)

        if config.use_critic:
            value_fn_out = rl_module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
            vf_target = batch[Postprocessing.VALUE_TARGETS]
            vf_loss = torch.pow(value_fn_out - vf_target, 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
            # ------------------------------ FIX ------------------------------
            if Columns.LOSS_MASK in batch:
                vf_target = vf_target[mask]
                value_fn_out = value_fn_out[mask]
            vf_explained_var = explained_variance(vf_target, value_fn_out)
            # -----------------------------------------------------------------
        else:
            z = torch.tensor(0.0, device=self._device)
            vf_explained_var = mean_vf_unclipped_loss = mean_vf_loss = z

        total_loss = (
            - mean_surrogate_loss + config.vf_loss_coeff * mean_vf_loss -
            entropy_coef * mean_entropy)
        if config.use_kl_loss:
            total_loss += kl_coef * mean_kl_loss

        self.metrics.log_dict(
            {
                # ---------------------------- FIX ----------------------------
                key: value.item()
                # -------------------------------------------------------------
                for key, value in (
                    (POLICY_LOSS_KEY, -mean_surrogate_loss),
                    (VF_LOSS_KEY, mean_vf_loss),
                    (LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
                     mean_vf_unclipped_loss),
                    (LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY, vf_explained_var),
                    (ENTROPY_KEY, mean_entropy),
                    (LEARNER_RESULTS_KL_KEY, mean_kl_loss))
            },
            key=module_id,
            # ------------------------------ FIX ------------------------------
            reduce="mean",
            window=float("inf"),
            clear_on_reduce=True,
            # -----------------------------------------------------------------
            )

        # Extract some proxies from convenience
        observation_true = batch[Columns.OBS]
        batch_size = len(batch)

        # Extract mean of predicted action from action_logits.
        # No need to perform model forward pass since it was already done by
        # some connector in the learning pipeline, so just retrieving the value
        # from the training batch.
        action_true_mean = curr_action_dist.to_deterministic().sample()

        # Define various training batches to forward to the model
        batch_all = {}
        if config.caps_temporal_reg > 0.0 or config.temporal_barrier_reg > 0.0:
            # Shallow copy the original training batch
            batch_copy = copy_batch(batch)

            # Replace current observation and state by the next one
            batch_copy[Columns.OBS] = batch.pop(Columns.NEXT_OBS)
            if Columns.STATE_IN in batch_copy:
                batch_copy[Columns.STATE_IN] = batch.pop(Columns.STATE_OUT)

            # Append the training batches to the set
            batch_all["next"] = batch_copy
        if config.caps_spatial_reg > 0.0 or config.caps_global_reg > 0.0:
            # Shallow copy the original training batch
            batch_copy = copy_batch(batch)

            if config.enable_adversarial_noise:
                # Compute adversarial observation maximizing action difference
                observation_noisy = get_adversarial_observation_sgld(
                    rl_module,
                    batch,
                    fwd_out,
                    config.spatial_noise_scale,
                    config.sgld_beta_inv,
                    config.sgld_n_steps)
            else:
                # Generate noisy observation
                observation_noisy = torch.normal(
                    observation_true, config.spatial_noise_scale)

            # Replace current observation by the adversarial one
            batch_copy[Columns.OBS] = observation_noisy

            # Append the training batches to the set
            batch_all["noisy"] = batch_copy
        if config.symmetric_policy_reg > 0.0:
            # Shallow copy the original training batch
            batch_copy = copy_batch(batch)

            # Compute mirrored observation
            observation_mirrored = _compute_mirrored_value(
                observation_true, self.obs_mirror_mat_nested)

            # Replace current observation by the mirrored one
            batch_copy[Columns.OBS] = observation_mirrored

            # Append the training batches to the set
            batch_all["mirrored"] = batch_copy

        if batch_all:
            # Compute the actions for all the training batches at onces
            batch_cat = {
                field: torch.cat([
                    batch[field] for batch in batch_all.values()], dim=0)
                for field in (Columns.OBS, Columns.STATE_IN)
                if field in batch}
            outs_cat = rl_module.forward_train(batch_cat)
            action_logits_cat = outs_cat[Columns.ACTION_DIST_INPUTS]

            # Split the stacked actions in separated chunks
            batch_names = batch_all.keys()
            action_logits_all = dict(zip(batch_names, torch.split(
                action_logits_cat, batch_size, dim=0)))
            action_dist_cat = action_dist_class_train.from_logits(
                outs_cat[Columns.ACTION_DIST_INPUTS])
            actions_cat = action_dist_cat.to_deterministic().sample()
            actions_all = dict(zip(batch_names, torch.split(
                actions_cat, batch_size, dim=0)))

        # Update total loss
        if config.caps_temporal_reg > 0.0 or config.temporal_barrier_reg > 0.0:
            # Compute action temporal delta
            action_delta = (actions_all["next"] - action_true_mean).abs()

            if config.caps_temporal_reg > 0.0:
                # Minimize the difference between the successive action mean
                mean_caps_temporal_reg = possibly_masked_mean(action_delta)

                # Add temporal smoothness loss to total loss
                total_loss += config.caps_temporal_reg * mean_caps_temporal_reg
                self.metrics.log_value(
                    (module_id, "temporal_smoothness"),
                    mean_caps_temporal_reg.item(),
                    reduce="mean",
                    window=float("inf"),
                    clear_on_reduce=True)

            if config.temporal_barrier_reg > 0.0:
                # Add temporal barrier loss to total loss:
                # exp(scale * (err - thr)) - 1.0 if err > thr else 0.0
                temporal_barrier_reg = torch.exp(torch.clamp(
                    config.temporal_barrier_scale * (
                        action_delta - config.temporal_barrier_threshold),
                    min=0.0, max=5.0)) - 1.0
                mean_temporal_barrier_reg = possibly_masked_mean(
                    temporal_barrier_reg)

                # Add spatial smoothness loss to total loss
                total_loss += (
                    config.temporal_barrier_reg * mean_temporal_barrier_reg)
                self.metrics.log_value(
                    (module_id, "temporal_barrier"),
                    mean_temporal_barrier_reg.item(),
                    reduce="mean",
                    window=float("inf"),
                    clear_on_reduce=True)

        if config.caps_spatial_reg > 0.0:
            # Minimize the difference between the original action mean and the
            # perturbed one.
            caps_spatial_reg = torch.sum(
                (actions_all["noisy"] - action_true_mean) ** 2, dim=1)
            mean_caps_spatial_reg = possibly_masked_mean(caps_spatial_reg)

            # Add spatial smoothness loss to total loss
            total_loss += config.caps_spatial_reg * mean_caps_spatial_reg
            self.metrics.log_value(
                (module_id, "spatial_smoothness"),
                mean_caps_spatial_reg.item(),
                reduce="mean",
                window=float("inf"),
                clear_on_reduce=True)

        if config.caps_global_reg > 0.0:
            # Minimize the magnitude of action mean.
            # Note that noisy actions are used instead of the true ones. This
            # is on-purpose, as it extends the range of regularization beyond
            # the mean field.
            caps_global_reg = actions_all["noisy"] ** 2
            mean_caps_global_reg = possibly_masked_mean(caps_global_reg)

            # Add global smoothness loss to total loss
            total_loss += config.caps_global_reg * mean_caps_global_reg
            self.metrics.log_value(
                (module_id, "global_smoothness"),
                mean_caps_global_reg.item(),
                reduce="mean",
                window=float("inf"),
                clear_on_reduce=True)

        if config.symmetric_policy_reg > 0.0:
            # Compute the mirrored true action
            action_mirrored_mean = _compute_mirrored_value(
                action_true_mean, self.action_mirror_mat_nested)

        if (config.symmetric_policy_reg > 0.0 and
                not config.enable_symmetry_surrogate_loss):
            # Minimize the asymmetry of self output
            symmetric_policy_reg = (
                actions_all["mirrored"] - action_mirrored_mean) ** 2
            mean_symmetric_policy_reg = possibly_masked_mean(
                symmetric_policy_reg)

            # Add policy symmetry loss to total loss
            total_loss += (
                config.symmetric_policy_reg * mean_symmetric_policy_reg)
            self.metrics.log_value(
                (module_id, "symmetry"),
                mean_symmetric_policy_reg.item(),
                reduce="mean",
                window=float("inf"),
                clear_on_reduce=True)

        if (config.symmetric_policy_reg > 0.0 and
                config.enable_symmetry_surrogate_loss):
            # Get the mirror policy probability distribution
            # i.e. `action -> pi(action | obs_mirrored)``
            action_mirrored_dist = action_dist_class_train.from_logits(
                action_logits_all["mirrored"])

            # Compute probability of "mirrored action under true observation"
            # wrt the action distribution under mirrored observation.
            action_symmetry_logp = action_mirrored_dist.logp(
                action_mirrored_mean)

            # Compute the probablity ratio between reverted and true actions.
            action_symmetry_proba_ratio = torch.exp(
                action_symmetry_logp - batch[Columns.ACTION_LOGP])

            # Compute the surrogate symmetry loss.
            # In principle, each sample should be weighted by the ratio between
            # the probability of the mirrored observation and the probability
            # of the true observation at time t under the current policy.
            # Computing this term is intractable, or at the very least very
            # challenging. Consequently, it is simply ignored, assuming both
            # probability are equals. However, this hypothesis only makes sense
            # if the policy is symmetric, which is characterized by the
            # probability ratio previously computed. To get around this
            # limitation, this idea is to take inspiration from PPO-clip, which
            # computes a pessimistic estimate objective defined as the minimum
            # between the original objective and the one for which the
            # probability ratio is clipped around 1.0 to remove the incentive
            # to increase the divergence. With this approach, when the
            # divergence is large, only changes that encourages to reduce the
            # divergence to improve the objective are taken into account while
            # all the others are filtered out.
            advantages_true = batch[Columns.ADVANTAGES]
            symmetry_surrogate_loss = torch.min(
                advantages_true * action_symmetry_proba_ratio,
                advantages_true * torch.clamp(
                    action_symmetry_proba_ratio,
                    1.0 - config.clip_param,
                    1.0 + config.clip_param))
            mean_symmetry_surrogate_loss = possibly_masked_mean(
                symmetry_surrogate_loss)

            # Add symmetry surrogate loss to total loss
            total_loss -= (
                config.symmetric_policy_reg * mean_symmetry_surrogate_loss)
            self.metrics.log_value(
                (module_id, "symmetry_surrogate"),
                -mean_symmetry_surrogate_loss.item(),
                reduce="mean",
                window=float("inf"),
                clear_on_reduce=True)

        l2_ref_coeff = self.l2_reg_schedulers_per_module[
            module_id].get_current_value()
        if l2_ref_coeff > 0.0:
            # Add actor l2-regularization loss
            l2_reg = torch.tensor(0.0, device=self._device)
            for name, params in rl_module.named_parameters():
                if name.endswith(".weight") and params.requires_grad:
                    l2_reg += l2_loss(params)

            # Add l2-regularization loss to total loss
            total_loss += l2_ref_coeff * l2_reg
            self.metrics.log_value(
                (module_id, "l2_reg"),
                l2_reg.item(),
                reduce="mean",
                window=float("inf"),
                clear_on_reduce=True)

        return total_loss


__all__ = [
   "PPOConfig",
   "PPOTorchLearner",
   "PPO"
]
