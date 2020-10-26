import copy
import logging
import numpy as np
from typing import Dict, List

from gym.spaces import Space

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, TensorType
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict

_, nn = try_import_torch()

logger = logging.getLogger(__name__)


class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self,
                 obs_space: Space,
                 action_space: Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):
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
                       initializer=normc_initializer(1.0),
                       activation_fn=activation))
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and self.num_outputs:
            layers.append(
                SlimFC(in_size=prev_layer_size,
                       out_size=self.num_outputs,
                       initializer=normc_initializer(1.0),
                       activation_fn=activation))
            prev_layer_size = self.num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(in_size=prev_layer_size,
                           out_size=hiddens[-1],
                           initializer=normc_initializer(1.0),
                           activation_fn=activation))
                prev_layer_size = hiddens[-1]
            if self.num_outputs:
                self._logits = SlimFC(in_size=prev_layer_size,
                                      out_size=self.num_outputs,
                                      initializer=normc_initializer(0.01),
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
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)

        # Holds the current value output.
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self,
                input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
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
