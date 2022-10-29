import copy
import logging
import numpy as np
from typing import Tuple, List, Sequence, Dict, Any

from gym.spaces import Space, flatdim

from ray.rllib.utils import try_import_torch, try_import_tf
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, TensorType
from ray.rllib.models.tf.misc import normc_initializer_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement


_, tf, _ = try_import_tf()
_, nn = try_import_torch()

logger = logging.getLogger(__name__)


class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self,
                 obs_space: Space,
                 action_space: Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = get_activation_fn(
            model_config.get("fcnet_activation"), framework="torch")
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        logger.debug("Constructing fcnet {} {}".format(hiddens, activation))
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(in_size=prev_layer_size,
                       out_size=size,
                       initializer=normc_initializer_torch(1.0),
                       activation_fn=activation))
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and self.num_outputs:
            layers.append(
                SlimFC(in_size=prev_layer_size,
                       out_size=self.num_outputs,
                       initializer=normc_initializer_torch(1.0),
                       activation_fn=activation))
            prev_layer_size = self.num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(in_size=prev_layer_size,
                           out_size=hiddens[-1],
                           initializer=normc_initializer_torch(1.0),
                           activation_fn=activation))
                prev_layer_size = hiddens[-1]
            if self.num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=self.num_outputs,
                    initializer=normc_initializer_torch(0.01),
                    activation_fn=None)
            else:
                self.num_outputs = (
                    [np.product(obs_space.shape)] + hiddens[-1:-1])[-1]

        # Create the feature networks
        self._shared_hidden_layers = None
        self._hidden_layers = None
        self._vf_hidden_layers = None
        if vf_share_layers:
            self._shared_hidden_layers = nn.Sequential(*layers)
        else:
            self._hidden_layers = nn.Sequential(*layers)
            self._vf_hidden_layers = copy.deepcopy(self._hidden_layers)

        # Non-shared value branch.
        self._value_branch = SlimFC(in_size=prev_layer_size,
                                    out_size=1,
                                    initializer=normc_initializer_torch(1.0),
                                    activation_fn=None)

        # Holds the current value output.
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self,
                input_dict: Dict[str, TensorType],
                state: Sequence[TensorType],
                seq_lens: TensorType
                ) -> Tuple[TensorType, Sequence[TensorType]]:
        obs = input_dict["obs_flat"].float()
        if self._shared_hidden_layers is not None:
            features = self._shared_hidden_layers(
                obs.reshape(obs.shape[0], -1))
        else:
            features = self._hidden_layers(obs.reshape(obs.shape[0], -1))
            vf_features = self._vf_hidden_layers(
                obs.reshape(obs.shape[0], -1))
        logits = self._logits(features) if self._logits else features
        self._cur_value = self._value_branch(vf_features).squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


class FrameStackingModel(TFModelV2):
    """A simple FC model that takes the last n observations, actions, and
    rewards as input.
    """
    def __init__(self,
                 obs_space: Space,
                 action_space: Space,
                 num_outputs: int,
                 model_config: Dict[str, Any],
                 name: str,
                 num_frames: int = 1) -> None:
        # Call base initializer first
        super().__init__(obs_space, action_space, None, model_config, name)

        # Backup some user arguments
        self.num_frames = num_frames
        self.num_outputs = num_outputs

        # Define some proxies for convenience
        sensor_space_start = 0
        for field, space in obs_space.original_space.spaces.items():
            if field != "sensors":
                sensor_space_start += flatdim(space)
            else:
                sensor_space_size = flatdim(space)
                sensor_space_end = sensor_space_start + sensor_space_size
                break
        self.sensor_space_range = [sensor_space_start, sensor_space_end]

        # Extract some user arguments
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        no_final_linear = model_config.get("no_final_linear")
        hiddens = model_config.get("fcnet_hiddens", [])
        vf_share_layers = model_config.get("vf_share_layers")

        # Specify the inputs
        if self.num_frames > 1:
            self.view_requirements["prev_n_obs"] = ViewRequirement(
                data_col=SampleBatch.OBS,
                shift="-{}:-1".format(num_frames),
                space=obs_space)
            self.view_requirements["prev_n_act"] = ViewRequirement(
                data_col=SampleBatch.ACTIONS,
                shift="-{}:-1".format(num_frames),
                space=action_space)
            self.view_requirements["prev_n_rew"] = ViewRequirement(
                data_col=SampleBatch.REWARDS,
                shift="-{}:-1".format(num_frames))

        # Buffer to store last computed value
        self._last_value = None

        # Define the input layer of the model
        stack_size = sensor_space_size + action_space.shape[0] + 1
        obs = tf.keras.layers.Input(
            shape=obs_space.shape, name="obs")
        if self.num_frames > 1:
            stack = tf.keras.layers.Input(
                shape=(self.num_frames, stack_size), name="stack")
            inputs = [obs, stack]
        else:
            inputs = obs

        # Build features extraction network
        # In:  (batch_size, n_features, n_timesteps)
        # Out: (batch_size, n_filters, n_timesteps - (kernel_size - 1))
        if self.num_frames >= 16:
            conv_1 = tf.keras.layers.Conv1D(
                filters=4,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="valid",
                name="conv_1")(stack)

            pool_1 = tf.keras.layers.AveragePooling1D(
                pool_size=2,
                strides=2,
                padding="valid",
                name="pool_1")(conv_1)

            conv_2 = tf.keras.layers.Conv1D(
                filters=8,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="valid",
                name="conv_2")(pool_1)

            pool_2 = tf.keras.layers.AveragePooling1D(
                pool_size=2,
                strides=2,
                padding="valid",
                name="pool_2")(conv_2)

            # Gather observation and extracted features as input
            flatten = tf.keras.layers.Flatten(name="flatten")(pool_2)
            features = tf.keras.layers.Dense(
                    units=8,
                    name="fc_features",
                    activation=activation,
                    kernel_initializer=normc_initializer_tf(1.0))(flatten)
            concat = tf.keras.layers.Concatenate(
                axis=-1, name="concat")([obs, features])
        elif self.num_frames > 1:
            # Gather current observation and previous stack as input
            features = tf.keras.layers.Flatten(name="flatten")(stack)
            concat = tf.keras.layers.Concatenate(
                axis=-1, name="concat")([obs, features])
        else:
            # Current observation is the only input
            concat = obs

        # concat = tf.keras.layers.GaussianNoise(0.1)(concat)

        # Create policy layers 0 to second-last.
        i = 1
        last_layer = concat
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                units=size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer_tf(1.0))(last_layer)
            i += 1

        # The last layer is adjusted to be of size num_outputs, but it is a
        # layer with activation.
        if no_final_linear:
            logits_out = tf.keras.layers.Dense(
                units=num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer_tf(0.01))(last_layer)
        # Finish the layers with the provided sizes (`hiddens`), plus a last
        # linear layer of size num_outputs.
        else:
            last_layer = tf.keras.layers.Dense(
                units=hiddens[-1],
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer_tf(1.0))(last_layer)
            logits_out = tf.keras.layers.Dense(
                units=num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer_tf(0.01))(last_layer)

        last_vf_layer = None
        if not vf_share_layers:
            # Build a dedicated hidden layers for the value net if requested
            i = 1
            last_vf_layer = concat
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    units=size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer_tf(1.0)
                    )(last_vf_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            units=1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer_tf(1.0))(
                last_vf_layer or last_layer)

        # Finish definition of the model
        self.base_model = tf.keras.Model(inputs, [logits_out, value_out])

    def forward(self,
                input_dict: Dict[str, tf.Tensor],
                states: List[Any],
                seq_lens: List[int]) -> Tuple[tf.Tensor, List[Any]]:
        obs = input_dict["obs_flat"]
        if self.num_frames > 1:
            stack = tf.concat((
                input_dict["prev_n_obs"][..., slice(*self.sensor_space_range)],
                input_dict["prev_n_act"],
                tf.expand_dims(input_dict["prev_n_rew"], axis=-1)
                ), axis=-1)
            inputs = [obs, stack]
        else:
            inputs = obs
        logits, self._last_value = self.base_model(inputs)
        mean, std = tf.split(logits, 2, axis=-1)
        std = std - 1.5
        logits = tf.concat((mean, std), axis=-1)
        return logits, states

    def value_function(self) -> tf.Tensor:
        return tf.reshape(self._last_value, [-1])

    def export_to_h5(self, h5_file: str) -> None:
        self.base_model.save_weights(h5_file)

    def import_from_h5(self, h5_file: str) -> None:
        self.base_model.load_weights(h5_file)
