"""Implement several regularization losses on top of the original PPO algorithm
to encourage smoothness of the action and clustering of the behavior of the
policy without having to rework the reward function itself. It takes advantage
of the analytical gradient of the policy.
"""
import math
import operator
from functools import reduce
from typing import Optional, Union, Type, Dict, Any, List, Tuple, cast

import numpy as np
import gymnasium as gym
import torch
from torch.nn import functional as F

from ray.rllib import SampleBatch
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner import Learner
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig as _PPOConfig, PPO as _PPO
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
    PPOTorchLearner as _PPOTorchLearner)
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch
from ray.rllib.connectors.learner.\
    add_next_observations_from_episodes_to_train_batch import (
        AddNextObservationsFromEpisodesToTrainBatch)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.from_config import _NotProvided, NotProvided
from ray.rllib.utils.typing import TensorType, EpisodeType, ModuleID

from jiminy_py import tree
from gym_jiminy.common.bases import BasePipelineWrapper
from gym_jiminy.common.wrappers import FlattenObservation, FlattenAction
from gym_jiminy.common.utils import zeros


def get_adversarial_observation_sgld(
        module: RLModule,
        batch: SampleBatch,
        noise_scale: float,
        beta_inv: float,
        n_steps: int) -> torch.Tensor:
    """Compute adversarial observation maximizing Mean Squared Error between
    the original and the perturbed mean action using Stochastic gradient
    Langevin dynamics algorithm (SGLD).
    """
    # Compute mean field action for true observation
    action_dist_class = module.get_train_action_dist_cls()
    action_dist = action_dist_class.from_logits(
        batch[Columns.ACTION_DIST_INPUTS]).to_deterministic()
    action_true_mean = action_dist.sample()

    # Shallow copy the original training batch.
    # Be careful accessing fields using the original batch to properly keep
    # track of accessed keys, which will be used to automatically discard
    # useless components of policy's view requirements.
    batch_copy = batch.copy(shallow=True)

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
            action_dist = action_dist_class.from_logits(
                outs[Columns.ACTION_DIST_INPUTS]).to_deterministic()
            action_noisy_mean = action_dist.sample()

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
                            shape_nested: Tuple[Tuple[int, ...], ...],
                            mirror_mat_nested: Tuple[torch.Tensor, ...]
                            ) -> torch.Tensor:
    """Compute mirrored value from observation space based on provided
    mirroring transformation.
    """
    batch_size, offset, data_mirrored_all = len(value), 0, []
    for shape, mirror_mat in zip(shape_nested, mirror_mat_nested):
        size = reduce(operator.mul, shape)
        data = value[:, offset:(offset + size)].reshape((batch_size, *shape))
        data_mirrored_all.append(data @ mirror_mat)
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
        if not isinstance(enable_symmetry_surrogate_loss, _NotProvided):
            self.enable_symmetry_surrogate_loss = \
                enable_symmetry_surrogate_loss
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

        # Early return if mirroring transforms are used in this context
        if (not module_config.symmetric_policy_reg and
                not module_config.enable_symmetry_surrogate_loss):
            return

        # Extract the original observation and acion spaces of the training
        # environment before and after flattening if applicable.
        # Note that keeping track of the "original" spaces before flattening is
        # necessary, because the original shape of the data must be restored
        # before mirroring the data by applying the block matrix product.
        # Ideally, jiminy should provide some "apply_observation_transform" and
        # "apply_reverse_observation_transform" helper methods. This way, one
        # could cast the final observation into the original observation space
        # (using `nan` as "fake" placeholder value for information that has
        # been lost in the process). Then, the original mirroring transform
        # could be applied. Finally, the observation could be projected back
        # into the final observation space. If some `nan` values are still
        # present in the end, then it means that some necessary bits of
        # information was missing, so that observations cannot be mirrored
        # losslessly. One option would be masking the unrecoverable information
        # by filtering out `nan` values when computing the difference between
        # the original and mirrored values.
        rl_module = self.module[DEFAULT_MODULE_ID]
        observation_space = rl_module.observation_space
        assert isinstance(observation_space, gym.spaces.Box)
        action_space = rl_module.action_space
        assert isinstance(action_space, gym.spaces.Box)

        config = self.config
        algo_class = cast(Type[PPO], config.algo_class)
        _, env_creator = algo_class._get_env_id_and_creator(config.env, config)
        env_runner_group = EnvRunnerGroup(
            env_creator=env_creator,
            validate_env=None,
            default_policy_class=algo_class.get_default_policy_class(config),
            config=config,
            num_env_runners=0,
            local_env_runner=True)

        worker = env_runner_group.local_env_runner
        assert isinstance(worker, SingleAgentEnvRunner)
        (env,) = worker.env.envs
        while isinstance(env, (BasePipelineWrapper, gym.Wrapper)):
            is_flatten_obs_wrapper = isinstance(env, FlattenObservation)
            is_flatten_action_wrapper = isinstance(env, FlattenAction)
            env = env.env
            if is_flatten_obs_wrapper:
                observation_space = env.observation_space
            if is_flatten_action_wrapper:
                action_space = env.action_space

        # Define helper to extract flattened sequence of mirroring matrices and
        # shape of all the leaves of the original space.
        def _extract_mirror_mat_and_shape(space: gym.Space) -> Tuple[
                Tuple[Tuple[int, ...], ...], Tuple[torch.Tensor, ...]]:
            """Extract the flattened sequence of mirroring matrix blocks and
            shapes of all the leaves of the original nested space.
            """
            # Extract flattened sequence of mirroring matrix blocks.
            # Convert recursively each mirror matrix to `torch.Tensor` with
            # dtype `torch.float32` which is stored on the expected device.
            mirror_mat_nested = tuple(
                torch.tensor(
                    space_leaf.mirror_mat,
                    dtype=torch.float32,
                    device=self._device)
                for space_leaf in tree.flatten(space))

            # Extract flattened sequence of original shapes
            shape_nested = tuple(
                np.atleast_1d(zeros(space_leaf)).shape
                for space_leaf in tree.flatten(space))

            return shape_nested, mirror_mat_nested

        self.obs_shape_nested, self.obs_mirror_mat_nested = (
            _extract_mirror_mat_and_shape(observation_space))
        self.action_shape_nested, self.action_mirror_mat_nested = (
            _extract_mirror_mat_and_shape(action_space))

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

        # Call base implementation
        total_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out)

        # Extract the right RL Module
        rl_module = self.module[module_id].unwrapped()

        # Assert(s) for type checker
        assert isinstance(rl_module, TorchRLModule)

        # Extract some proxies from convenience
        observation_true = batch[Columns.OBS]
        batch_size = len(batch)

        # Extract mean of predicted action from action_logits.
        # No need to perform model forward pass since it was already done by
        # some connector in the learning pipeline, so just retrieving the value
        # from the training batch.
        action_dist_class = rl_module.get_train_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]).to_deterministic()
        action_true_mean = action_dist.sample()

        # Define various training batches to forward to the model
        batch_all = {}
        if config.caps_temporal_reg > 0.0 or config.temporal_barrier_reg > 0.0:
            # Shallow copy the original training batch
            batch_copy = batch.copy(shallow=True)

            # Replace current observation and state by the next one
            batch_copy[Columns.OBS] = batch.pop(Columns.NEXT_OBS)
            if Columns.STATE_IN in batch_copy:
                batch_copy[Columns.STATE_IN] = batch.pop(Columns.STATE_OUT)

            # Append the training batches to the set
            batch_all["next"] = batch_copy
        if config.caps_spatial_reg > 0.0 or config.caps_global_reg > 0.0:
            # Shallow copy the original training batch
            batch_copy = batch.copy(shallow=True)

            if config.enable_adversarial_noise:
                # Compute adversarial observation maximizing action difference
                observation_noisy = get_adversarial_observation_sgld(
                    rl_module,
                    batch,
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
            batch_copy = batch.copy(shallow=True)

            # Compute mirrored observation
            observation_mirrored = _compute_mirrored_value(
                observation_true,
                self.obs_shape_nested,
                self.obs_mirror_mat_nested)

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
            action_dist_cat = action_dist_class.from_logits(
                outs_cat[Columns.ACTION_DIST_INPUTS]).to_deterministic()
            actions_cat = action_dist_cat.sample()
            actions_all = dict(zip(batch_names, torch.split(
                actions_cat, batch_size, dim=0)))

        # Update total loss
        if config.caps_temporal_reg > 0.0 or config.temporal_barrier_reg > 0.0:
            # Compute action temporal delta
            action_delta = F.l1_loss(
                actions_all["next"], action_true_mean, reduction='none')

            if config.caps_temporal_reg > 0.0:
                # Minimize the difference between the successive action mean
                caps_temporal_reg = torch.mean(action_delta)

                # Add temporal smoothness loss to total loss
                total_loss += config.caps_temporal_reg * caps_temporal_reg
                self.metrics.log_value(
                    (module_id, "temporal_smoothness"),
                    caps_temporal_reg.item(),
                    window=1)

            if config.temporal_barrier_reg > 0.0:
                # Add temporal barrier loss to total loss:
                # exp(scale * (err - thr)) - 1.0 if err > thr else 0.0
                temporal_barrier_reg = torch.mean(torch.exp(torch.clamp(
                    config.temporal_barrier_scale * (
                        action_delta - config.temporal_barrier_threshold),
                    min=0.0, max=5.0)) - 1.0)

                # Add spatial smoothness loss to total loss
                total_loss += (
                    config.temporal_barrier_reg * temporal_barrier_reg)
                self.metrics.log_value(
                    (module_id, "temporal_barrier"),
                    temporal_barrier_reg.item(),
                    window=1)

        if config.caps_spatial_reg > 0.0:
            # Minimize the difference between the original action mean and the
            # perturbed one.
            caps_spatial_reg = F.mse_loss(
                actions_all["noisy"], action_true_mean, reduction='sum'
                ) / batch_size

            # Add spatial smoothness loss to total loss
            total_loss += config.caps_spatial_reg * caps_spatial_reg
            self.metrics.log_value(
                (module_id, "spatial_smoothness"),
                caps_spatial_reg.item(),
                window=1)

        if config.caps_global_reg > 0.0:
            # Minimize the magnitude of action mean.
            # Note that noisy actions are used instead of the true ones. This
            # is on-purpose, as it extends the range of regularization beyond
            # the mean field.
            caps_global_reg = torch.sum(
                torch.square(actions_all["noisy"])) / batch_size

            # Add global smoothness loss to total loss
            total_loss += config.caps_global_reg * caps_global_reg
            self.metrics.log_value(
                (module_id, "global_smoothness"),
                caps_global_reg.item(),
                window=1)

        if config.symmetric_policy_reg > 0.0:
            # Compute the mirrored true action
            action_mirrored_mean = _compute_mirrored_value(
                action_true_mean,
                self.action_shape_nested,
                self.action_mirror_mat_nested)

        if (config.symmetric_policy_reg > 0.0 and
                not config.enable_symmetry_surrogate_loss):
            # Minimize the assymetry of self output
            symmetric_policy_reg = F.mse_loss(
                actions_all["mirrored"], action_mirrored_mean, reduction='sum'
                ) / batch_size

            # Add policy symmetry loss to total loss
            total_loss += config.symmetric_policy_reg * symmetric_policy_reg
            self.metrics.log_value(
                (module_id, "symmetry"),
                symmetric_policy_reg.item(),
                window=1)

        if (config.symmetric_policy_reg > 0.0 and
                config.enable_symmetry_surrogate_loss):
            # Get the mirror policy probability distribution
            # i.e. `action -> pi(action | obs_mirrored)``
            action_mirrored_dist = action_dist_class.from_logits(
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
            symmetry_surrogate_loss = - torch.mean(torch.min(
                advantages_true * action_symmetry_proba_ratio,
                advantages_true * torch.clamp(
                    action_symmetry_proba_ratio,
                    1.0 - config.clip_param,
                    1.0 + config.clip_param)))

            # Add symmetry surrogate loss to total loss
            total_loss += config.symmetric_policy_reg * symmetry_surrogate_loss
            self.metrics.log_value(
                (module_id, "symmetry_surrogate"),
                symmetry_surrogate_loss.item(),
                window=1)

        if config.l2_reg > 0.0:
            # Add actor l2-regularization loss
            l2_reg = torch.zeros((), device=self._device)
            for name, params in rl_module.named_parameters():
                if not name.endswith("bias") and params.requires_grad:
                    l2_reg += torch.mean(torch.square(params))

            # Add l2-regularization loss to total loss
            total_loss += config.l2_reg * l2_reg
            self.metrics.log_value(
                (module_id, "l2_reg"),
                l2_reg.item(),
                window=1)

        return total_loss


__all__ = [
   "PPOConfig",
   "PPOTorchLearner",
   "PPO"
]
