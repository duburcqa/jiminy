from copy import deepcopy
from typing import Optional, Sequence

import numpy as np


class State:
    """Store the kinematics and dynamics data of the robot at a given time.
    """
    def __init__(self,
                 t: float,
                 q: np.ndarray,
                 v: Optional[np.ndarray] = None,
                 a: Optional[np.ndarray] = None,
                 tau: Optional[np.ndarray] = None,
                 contact_frames: Optional[Sequence[str]] = None,
                 f_ext: Optional[Sequence[np.ndarray]] = None,
                 copy: bool = False,
                 **kwargs):
        """
        :param t: Time.
        :param q: Configuration vector.
        :param v: Velocity vector.
        :param a: Acceleration vector.
        :param tau: Joint efforts.
        :param contact_frames: Name of the contact frames.
        :param f_ext: Joint external forces.
        :param copy: Force to copy the arguments.
        """
        # Time
        self.t = t
        # Configuration vector
        self.q = deepcopy(q) if copy else q
        # Velocity vector
        self.v = deepcopy(v) if copy else v
        # Acceleration vector
        self.a = deepcopy(a) if copy else a
        # Effort vector
        self.tau = deepcopy(tau) if copy else tau
        # Frame names of the contact points
        if copy:
            self.contact_frames = deepcopy(contact_frames)
        else:
            self.contact_frames = contact_frames
        # External forces
        self.f_ext = deepcopy(f_ext) if copy else f_ext

    def __repr__(self):
        """Convert the kinematics and dynamics data into string.

        :returns: The kinematics and dynamics data as a string.
        """
        msg = ""
        for key, val in self.__dict__.items():
            if val is not None:
                msg += f"{key} : {val}\n"
        return msg
