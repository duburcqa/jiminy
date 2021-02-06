""" TODO: Write documentation.
"""
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
from gym import spaces

from ..bases import BaseControllerBlock
from ..envs import BaseJiminyEnv
from ..utils import set_value, fill, zeros, SpaceDictNested, FieldDictNested


def _stacked_space(space: spaces.Space,
                   num_stack: Sequence[int]) -> spaces.Space:
    """ TODO: Write documentation.
    """
    if isinstance(space, spaces.Box):
        return spaces.Box(
            low=np.tile(space.low, (num_stack, *np.ones_like(space.shape))),
            high=np.tile(space.high, (num_stack, *np.ones_like(space.shape))))
    if isinstance(space, spaces.Dict):
        value = OrderedDict()
        for field, subspace in dict.items(space.spaces):
            value[field] = _stacked_space(subspace, num_stack)
        return value
    if isinstance(space, spaces.Discrete):
        return spaces.MultiDiscrete(np.full((num_stack,), fill_value=space.n))
    raise NotImplementedError(
        f"Space of type {type(space)} is not supported.")


class GenericOrderHoldController(BaseControllerBlock):
    """Low-level First Order Hold controller.
    """
    def __init__(self,
                 env: BaseJiminyEnv,
                 update_ratio: int = 1,
                 order: int = 1,
                 **kwargs: Any) -> None:
        """
        :param update_ratio: Ratio between the update period of the controller
                             and the one of the subsequent controller.
        :param order: Order of the controller. It will change if number of
                      stacked next action required i.e. max(1, order).
        :param kwargs: Used arguments to allow automatic pipeline wrapper
                       generation.
        """
        # Backup user arguments
        self.order = order

        # Initialize the controller
        super().__init__(env, update_ratio)

        # Update counter and polynomial coefficients
        self.interp_ratio = 1
        self._n_prev_updates = 0
        self._action = zeros(_stacked_space(
                self.env.action_space, self.order + 1))
        self._k = deepcopy(self._action)
        self._Ainv = np.linalg.inv(np.stack([
            np.array([(i / max(1, order)) ** j for j in range(order + 1)
            ]) for i in range(order + 1)],axis=0))

    def _setup(self) -> None:
        """ TODO: Write documentation.
        """
        self.interp_ratio = int(self.env.step_dt / self.env.control_dt)
        self._n_prev_updates = self.interp_ratio - 1
        fill(self._k, 0.0)
        fill(self._action, 0.0)

    def _refresh_action_space(self) -> None:
        """Configure the action space of the controller.

        It is the same than the one of the environment, but repeated along the
        first axis if the order is larger than 1.
        """
        if self.order > 1:
            self.action_space = _stacked_space(
                self.env.action_space, self.order)
        else:
            self.action_space = self.env.action_space

    def get_fieldnames(self) -> FieldDictNested:
        if self.order > 1:
            return [[f"currentEffort{e}{i}" for e in self.robot.motors_names]
                    for i in range(self.order)]
        else:
            return [f"currentEffort{e}" for e in self.robot.motors_names]

    def compute_command(self,
                        measure: SpaceDictNested,
                        action: SpaceDictNested
                        ) -> np.ndarray:
        """ TODO: Write documentation.
        """
        if self._n_prev_updates == self.interp_ratio:
            self._n_prev_updates = 0
            self._action[0] = self._action[-1]
            self._action[-self.order:] = action
            set_value(self._k, self._Ainv @ self._action)
        self._n_prev_updates += 1
        coeffs = np.array([(self._n_prev_updates / self.interp_ratio) ** i
                           for i in range(self.order + 1)])
        return coeffs @ self._k
