
from typing import Sequence

import numpy as np

from ..bases import InterfaceJiminyEnv, BaseQuantityReward
from ..quantities import MaskedQuantity, AverageFrameSpatialVelocity

from .generic import radial_basis_function


class OdometryVelocityReward(BaseQuantityReward):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 env: InterfaceJiminyEnv,
                 target: Sequence[float],
                 cutoff: float) -> None:
        """ TODO: Write documentation.
        """
        # Backup some user argument(s)
        self.target = target
        self.cutoff = cutoff

        # Call base implementation
        super().__init__(
            env,
            "reward_odometry_velocity",
            (MaskedQuantity, dict(
                quantity=(AverageFrameSpatialVelocity, dict(frame_name="root_joint")),
                key=(0, 1, 5))),
            self._transform,
            is_normalized=True,
            is_terminal=False)

    def _transform(self, value: np.ndarray) -> np.ndarray:
        """ TODO: Write documentation.
        """
        return radial_basis_function(value - self.target, self.cutoff)
