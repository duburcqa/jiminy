# mypy: ignore-errors
# pylint: skip-file
# flake8: noqa

import logging
from typing import Any, Dict, List, Optional, Union, Generator

import tree
import numpy as np

import ray
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.env.env_runner import ENV_STEP_FAILURE
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.policy.sample_batch import (
    MultiAgentBatch, SampleBatch, concat_samples)
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED_LIFETIME, WEIGHTS_SEQ_NO)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch
from ray.rllib.utils.postprocessing.value_predictions import (
    compute_value_targets)
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad_n_episodes,
    unpad_data_if_necessary)
from ray.rllib.utils.typing import EpisodeType, SampleBatchType


logger = logging.getLogger(__name__)


def synchronous_parallel_sample(
    *,
    worker_set: EnvRunnerGroup,
    max_agent_steps: Optional[int] = None,
    max_env_steps: Optional[int] = None,
    concat: bool = True,
    sample_timeout_s: Optional[float] = None,
    random_actions: bool = False,
    _uses_new_env_runners: bool = False,
    _return_metrics: bool = False,
) -> Union[List[SampleBatchType], SampleBatchType, List[EpisodeType], EpisodeType]:
    # Only allow one of `max_agent_steps` or `max_env_steps` to be defined.
    assert not (max_agent_steps is not None and max_env_steps is not None)

    agent_or_env_steps = 0
    max_agent_or_env_steps = max_agent_steps or max_env_steps or None
    sample_batches_or_episodes = []
    all_stats_dicts = []

    random_action_kwargs = {} if not random_actions else {"random_actions": True}

    # Stop collecting batches as soon as one criterium is met.
    while (max_agent_or_env_steps is None and agent_or_env_steps == 0) or (
        max_agent_or_env_steps is not None
        and agent_or_env_steps < max_agent_or_env_steps
    ):
        # No remote workers in the set -> Use local worker for collecting
        # samples.
        if worker_set.num_remote_workers() <= 0:
            sampled_data = [worker_set.local_env_runner.sample(**random_action_kwargs)]
            if _return_metrics:
                stats_dicts = [worker_set.local_env_runner.get_metrics()]
        # Loop over remote workers' `sample()` method in parallel.
        else:
            sampled_data = worker_set.foreach_worker(
                (
                    (lambda w: w.sample(**random_action_kwargs))
                    if not _return_metrics
                    else (lambda w: (w.sample(**random_action_kwargs), w.get_metrics()))
                ),
                local_env_runner=False,
                timeout_seconds=sample_timeout_s,
            )
            # Nothing was returned (maybe all workers are stalling) or no healthy
            # remote workers left: Break.
            # There is no point staying in this loop, since we will not be able to
            # get any new samples if we don't have any healthy remote workers left.
            if not sampled_data or worker_set.num_healthy_remote_workers() <= 0:
                if not sampled_data:
                    logger.warning(
                        "No samples returned from remote workers. If you have a "
                        "slow environment or model, consider increasing the "
                        "`sample_timeout_s` or decreasing the "
                        "`rollout_fragment_length` in `AlgorithmConfig.env_runners()."
                    )
                elif worker_set.num_healthy_remote_workers() <= 0:
                    logger.warning(
                        "No healthy remote workers left. Trying to restore workers ..."
                    )
                break

            if _return_metrics:
                stats_dicts = [s[1] for s in sampled_data]
                sampled_data = [s[0] for s in sampled_data]

        # Update our counters for the stopping criterion of the while loop.
        if _return_metrics:
            if max_agent_steps:
                agent_or_env_steps += sum(
                    int(agent_stat)
                    for stat_dict in stats_dicts
                    for agent_stat in stat_dict[NUM_AGENT_STEPS_SAMPLED].values()
                )
            else:
                # ---------------------------- FIX ----------------------------
                agent_or_env_steps += sum(
                    len(e) for episodes in sampled_data for e in episodes)
                # -------------------------------------------------------------
            sample_batches_or_episodes.extend(sampled_data)
            all_stats_dicts.extend(stats_dicts)
        else:
            for batch_or_episode in sampled_data:
                if max_agent_steps:
                    agent_or_env_steps += (
                        sum(e.agent_steps() for e in batch_or_episode)
                        if _uses_new_env_runners
                        else batch_or_episode.agent_steps()
                    )
                else:
                    agent_or_env_steps += (
                        sum(e.env_steps() for e in batch_or_episode)
                        if _uses_new_env_runners
                        else batch_or_episode.env_steps()
                    )
                sample_batches_or_episodes.append(batch_or_episode)
                # Break out (and ignore the remaining samples) if max timesteps (batch
                # size) reached. We want to avoid collecting batches that are too large
                # only because of a failed/restarted worker causing a second iteration
                # of the main loop.
                if (
                    max_agent_or_env_steps is not None
                    and agent_or_env_steps >= max_agent_or_env_steps
                ):
                    break

    if concat is True:
        # If we have episodes flatten the episode list.
        if _uses_new_env_runners:
            sample_batches_or_episodes = tree.flatten(sample_batches_or_episodes)
        # Otherwise we concatenate the `SampleBatch` objects
        else:
            sample_batches_or_episodes = concat_samples(sample_batches_or_episodes)

    if _return_metrics:
        return sample_batches_or_episodes, all_stats_dicts
    return sample_batches_or_episodes


import ray.rllib.execution.rollout_ops
ray.rllib.execution.rollout_ops.synchronous_parallel_sample = (
    synchronous_parallel_sample)


def SingleAgentEnvRunner_sample(
        self,
        *,
        num_timesteps: Optional[int] = None,
        num_episodes: Optional[int] = None,
        explore: bool,
        random_actions: bool = False,
        force_reset: bool = False) -> List[SingleAgentEpisode]:
    """Helper method to sample n timesteps or m episodes."""
    done_episodes_to_return: List[SingleAgentEpisode] = []

    # Have to reset the env (on all vector sub_envs).
    # -------------------------------- FIX --------------------------------
    if force_reset or self._needs_initial_reset:
    # ---------------------------------------------------------------------
        episodes = self._episodes = [None for _ in range(self.num_envs)]
        shared_data = self._shared_data = {}
        self._reset_envs(episodes, shared_data, explore)
        # We just reset the env. Don't have to force this again in the next
        # call to `self._sample_timesteps()`.
        self._needs_initial_reset = False
    else:
        episodes = self._episodes
        shared_data = self._shared_data

    # -------------------------------- FIX --------------------------------
    # ---------------------------------------------------------------------

    # Loop through `num_timesteps` timesteps or `num_episodes` episodes.
    ts = 0
    eps = 0
    while (
        (ts < num_timesteps) if num_timesteps is not None else (eps < num_episodes)
    ):
        # Act randomly.
        if random_actions:
            to_env = {
                Columns.ACTIONS: self.env.action_space.sample(),
            }
        # Compute an action using the RLModule.
        else:
            # Env-to-module connector (already cached).
            to_module = self._cached_to_module
            assert to_module is not None
            self._cached_to_module = None

            # RLModule forward pass: Explore or not.
            if explore:
                # Global env steps sampled are (roughly) this EnvRunner's lifetime
                # count times the number of env runners in the algo.
                global_env_steps_lifetime = (
                    self.metrics.peek(NUM_ENV_STEPS_SAMPLED_LIFETIME, default=0)
                    + ts
                ) * (self.config.num_env_runners or 1)
                to_env = self.module.forward_exploration(
                    to_module, t=global_env_steps_lifetime
                )
            else:
                to_env = self.module.forward_inference(to_module)

            # Module-to-env connector.
            to_env = self._module_to_env(
                rl_module=self.module,
                batch=to_env,
                episodes=episodes,
                explore=explore,
                shared_data=shared_data,
            )

        # Extract the (vectorized) actions (to be sent to the env) from the
        # module/connector output. Note that these actions are fully ready (e.g.
        # already unsquashed/clipped) to be sent to the environment) and might not
        # be identical to the actions produced by the RLModule/distribution, which
        # are the ones stored permanently in the episode objects.
        actions = to_env.pop(Columns.ACTIONS)
        actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, actions)
        # Try stepping the environment.
        results = self._try_env_step(actions_for_env)
        if results == ENV_STEP_FAILURE:
            return self._sample(
                num_timesteps=num_timesteps,
                num_episodes=num_episodes,
                explore=explore,
                random_actions=random_actions,
                force_reset=True,
            )
        observations, rewards, terminateds, truncateds, infos = results
        observations, actions = unbatch(observations), unbatch(actions)

        call_on_episode_start = set()
        for env_index in range(self.num_envs):
            extra_model_output = {k: v[env_index] for k, v in to_env.items()}
            extra_model_output[WEIGHTS_SEQ_NO] = self._weights_seq_no

            # Episode has no data in it yet -> Was just reset and needs to be called
            # with its `add_env_reset()` method.
            if not self._episodes[env_index].is_reset:
                episodes[env_index].add_env_reset(
                    observation=observations[env_index],
                    infos=infos[env_index],
                )
                call_on_episode_start.add(env_index)

            # Call `add_env_step()` method on episode.
            else:
                # Only increase ts when we actually stepped (not reset'd as a reset
                # does not count as a timestep).
                ts += 1
                episodes[env_index].add_env_step(
                    observation=observations[env_index],
                    action=actions[env_index],
                    reward=rewards[env_index],
                    infos=infos[env_index],
                    terminated=terminateds[env_index],
                    truncated=truncateds[env_index],
                    extra_model_outputs=extra_model_output,
                )

        # Env-to-module connector pass (cache results as we will do the RLModule
        # forward pass only in the next `while`-iteration.
        if self.module is not None:
            self._cached_to_module = self._env_to_module(
                episodes=episodes,
                explore=explore,
                rl_module=self.module,
                shared_data=shared_data,
            )

        for env_index in range(self.num_envs):
            # Call `on_episode_start()` callback (always after reset).
            if env_index in call_on_episode_start:
                self._make_on_episode_callback(
                    "on_episode_start", env_index, episodes
                )
            # Make the `on_episode_step` callbacks.
            else:
                self._make_on_episode_callback(
                    "on_episode_step", env_index, episodes
                )

            # Episode is done.
            if episodes[env_index].is_done:
                eps += 1

                # Make the `on_episode_end` callbacks (before finalizing the episode
                # object).
                self._make_on_episode_callback(
                    "on_episode_end", env_index, episodes
                )

                # Then finalize (numpy'ize) the episode.
                done_episodes_to_return.append(episodes[env_index].finalize())

                # -------------------------- FIX --------------------------
                # ---------------------------------------------------------

                # Create a new episode object with no data in it and execute
                # `on_episode_created` callback (before the `env.reset()` call).
                episodes[env_index] = SingleAgentEpisode(
                    observation_space=self.env.single_observation_space,
                    action_space=self.env.single_action_space,
                )

    # Return done episodes ...
    # TODO (simon): Check, how much memory this attribute uses.
    self._done_episodes_for_metrics.extend(done_episodes_to_return)
    # ... and all ongoing episode chunks.

    # Also, make sure we start new episode chunks (continuing the ongoing episodes
    # from the to-be-returned chunks).
    ongoing_episodes_to_return = []
    # Only if we are doing individual timesteps: We have to maybe cut an ongoing
    # episode and continue building it on the next call to `sample()`.
    if num_timesteps is not None:
        ongoing_episodes_continuations = [
            eps.cut(len_lookback_buffer=self.config.episode_lookback_horizon)
            for eps in self._episodes
        ]

        for eps in self._episodes:
            # Just started Episodes do not have to be returned. There is no data
            # in them anyway.
            if eps.t == 0:
                continue
            eps.validate()
            self._ongoing_episodes_for_metrics[eps.id_].append(eps)
            # Return finalized (numpy'ized) Episodes.
            ongoing_episodes_to_return.append(eps.finalize())

        # Continue collecting into the cut Episode chunks.
        self._episodes = ongoing_episodes_continuations

    self._increase_sampled_metrics(ts, len(done_episodes_to_return))

    # Return collected episode data.
    return done_episodes_to_return + ongoing_episodes_to_return


from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
SingleAgentEnvRunner._sample = SingleAgentEnvRunner_sample


def MiniBatchCyclicIterator__iter__(self):
    while (
        # Make sure each item in the total batch gets at least iterated over
        # `self._num_epochs` times.
        (
            self._num_total_minibatches == 0
            and min(self._num_covered_epochs.values()) < self._num_epochs
        )
        # Make sure we reach at least the given minimum number of mini-batches.
        or (
            self._num_total_minibatches > 0
            and self._minibatch_count < self._num_total_minibatches
        )
    ):
        minibatch = {}
        for module_id, module_batch in self._batch.policy_batches.items():

            if len(module_batch) == 0:
                raise ValueError(
                    f"The batch for module_id {module_id} is empty! "
                    "This will create an infinite loop because we need to cover "
                    "the same number of samples for each module_id."
                )
            s = self._start[module_id]  # start

            # TODO (sven): Fix this bug for LSTMs:
            #  In an RNN-setting, the Learner connector already has zero-padded
            #  and added a timerank to the batch. Thus, n_step would still be based
            #  on the BxT dimension, rather than the new B dimension (excluding T),
            #  which then leads to minibatches way too large.
            #  However, changing this already would break APPO/IMPALA w/o LSTMs as
            #  these setups require sequencing, BUT their batches are not yet time-
            #  ranked (this is done only in their loss functions via the
            #  `make_time_major` utility).
            #  Get rid of the _uses_new_env_runners c'tor arg, once this work is
            #  done.
            n_steps = self._minibatch_size

            samples_to_concat = []

            # get_len is a function that returns the length of a batch
            # if we are not slicing the batch in the batch dimension B, then
            # the length of the batch is simply the length of the batch
            # o.w the length of the batch is the length list of seq_lens.
            if module_batch._slice_seq_lens_in_B:
                assert module_batch.get(SampleBatch.SEQ_LENS) is not None, (
                    "MiniBatchCyclicIterator requires SampleBatch.SEQ_LENS"
                    "to be present in the batch for slicing a batch in the batch "
                    "dimension B."
                )

                def get_len(b: Dict[str, Any]) -> int:
                    return len(b[SampleBatch.SEQ_LENS])

                if self._uses_new_env_runners:
                    n_steps = int(
                        get_len(module_batch)
                        * (self._minibatch_size / len(module_batch))
                    )

            else:

                def get_len(b: Dict[str, Any]) -> int:
                    return len(b)

            # ---------------------------- FIX ----------------------------
            # Cycle through the batch until we have enough samples.
            while True:
                sample = module_batch[s:]
                try:
                    n_masked_steps = (
                        sample['loss_mask'].cumsum(dim=0) == n_steps
                        ).argwhere()[-1, 0] + 1
                    break
                except IndexError:
                    len_sample = sample['loss_mask'].sum()
                    if len_sample > 0:
                        samples_to_concat.append(sample)
                        n_masked_steps -= len_sample
                    s = 0
                    self._num_covered_epochs[module_id] += 1
                    if self._num_total_minibatches == 0:
                        if self._num_covered_epochs[module_id] == self._num_epochs:
                            return
                    # Shuffle the individual single-agent batch, if required.
                    # This should happen once per minibatch iteration in order to
                    # make each iteration go through a different set of minibatches.
                    if self._shuffle_batch_per_epoch:
                        module_batch.shuffle()
            # -------------------------------------------------------------

            e = s + n_masked_steps  # end
            if e > s:
                samples_to_concat.append(module_batch[s:e])

            # concatenate all the samples, we should have minibatch_size of sample
            # after this step
            minibatch[module_id] = concat_samples(samples_to_concat)
            # roll minibatch to zero when we reach the end of the batch
            self._start[module_id] = e

        # Note (Kourosh): env_steps is the total number of env_steps that this
        # multi-agent batch is covering. It should be simply inherited from the
        # original multi-agent batch.
        minibatch = MultiAgentBatch(minibatch, len(self._batch))
        yield minibatch

        self._minibatch_count += 1


from ray.rllib.utils.minibatch_utils import MiniBatchCyclicIterator
MiniBatchCyclicIterator.__iter__ = MiniBatchCyclicIterator__iter__


def GeneralAdvantageEstimation__call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs: Any) -> Dict[str, Any]:
    # Device to place all GAE result tensors (advantages and value targets) on.
    device = None

    # Extract all single-agent episodes.
    sa_episodes_list = list(
        self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
    )
    # Perform the value nets' forward passes.
    # TODO (sven): We need to check here in the pipeline already, whether a module
    #  should even be updated or not (which we usually do after(!) the Learner
    #  pipeline). This is an open TODO to move this filter into a connector as well.
    #  For now, we'll just check, whether `mid` is in batch and skip if it isn't.
    vf_preds = rl_module.foreach_module(
        func=lambda mid, module: (
            module.compute_values(batch[mid])
            if mid in batch and isinstance(module, ValueFunctionAPI)
            else None
        ),
        return_dict=True,
    )
    # Loop through all modules and perform each one's GAE computation.
    for module_id, module_vf_preds in vf_preds.items():
        # Skip those outputs of RLModules that are not implementers of
        # `ValueFunctionAPI`.
        if module_vf_preds is None:
            continue

        module = rl_module[module_id]
        device = module_vf_preds.device
        # Convert to numpy for the upcoming GAE computations.
        module_vf_preds = convert_to_numpy(module_vf_preds)

        # Collect (single-agent) episode lengths for this particular module.
        episode_lens = [
            len(e) for e in sa_episodes_list if e.module_id in [None, module_id]
        ]

        # Remove all zero-padding again, if applicable, for the upcoming
        # GAE computations.
        module_vf_preds = unpad_data_if_necessary(episode_lens, module_vf_preds)
        # Compute value targets.
        module_value_targets = compute_value_targets(
            values=module_vf_preds,
            rewards=unpad_data_if_necessary(
                episode_lens,
                convert_to_numpy(batch[module_id][Columns.REWARDS]),
            ),
            terminateds=unpad_data_if_necessary(
                episode_lens,
                convert_to_numpy(batch[module_id][Columns.TERMINATEDS]),
            ),
            truncateds=unpad_data_if_necessary(
                episode_lens,
                convert_to_numpy(batch[module_id][Columns.TRUNCATEDS]),
            ),
            gamma=self.gamma,
            lambda_=self.lambda_,
        )
        assert module_value_targets.shape[0] == sum(episode_lens)

        module_advantages = module_value_targets - module_vf_preds

        # Drop vf-preds, not needed in loss. Note that in the PPORLModule, vf-preds
        # are recomputed with each `forward_train` call anyway.
        # Standardize advantages (used for more stable and better weighted
        # policy gradient computations).
        # ------------------------------ FIX ------------------------------
        mask = convert_to_numpy(batch[module_id][Columns.LOSS_MASK])
        module_advantages_filtered = module_advantages[mask]
        module_advantages = (
            module_advantages - module_advantages_filtered.mean()
            ) / max(1e-4, module_advantages_filtered.std())
        # -----------------------------------------------------------------

        # Zero-pad the new computations, if necessary.
        if module.is_stateful():
            module_advantages = np.stack(
                split_and_zero_pad_n_episodes(
                    module_advantages,
                    episode_lens=episode_lens,
                    max_seq_len=module.model_config["max_seq_len"],
                ),
                axis=0,
            )
            module_value_targets = np.stack(
                split_and_zero_pad_n_episodes(
                    module_value_targets,
                    episode_lens=episode_lens,
                    max_seq_len=module.model_config["max_seq_len"],
                ),
                axis=0,
            )
        batch[module_id][Postprocessing.ADVANTAGES] = module_advantages
        batch[module_id][Postprocessing.VALUE_TARGETS] = module_value_targets

    # Convert all GAE results to tensors.
    if self._numpy_to_tensor_connector is None:
        self._numpy_to_tensor_connector = NumpyToTensor(
            as_learner_connector=True, device=device
        )
    tensor_results = self._numpy_to_tensor_connector(
        rl_module=rl_module,
        batch={
            mid: {
                Postprocessing.ADVANTAGES: module_batch[Postprocessing.ADVANTAGES],
                Postprocessing.VALUE_TARGETS: (
                    module_batch[Postprocessing.VALUE_TARGETS]
                ),
            }
            for mid, module_batch in batch.items()
            if vf_preds[mid] is not None
        },
        episodes=episodes,
    )
    # Move converted tensors back to `batch`.
    for mid, module_batch in tensor_results.items():
        batch[mid].update(module_batch)

    return batch


from ray.rllib.connectors.learner import GeneralAdvantageEstimation
GeneralAdvantageEstimation.__call__ = GeneralAdvantageEstimation__call__
