#!/usr/bin/env python

## @file jiminy_py/state.py

import numpy as np
from collections import defaultdict
from copy import copy as _copy, deepcopy


class State:
    """
    @brief      Object that contains the kinematics and dynamics state of the
                robot at a given time.
    """
    def __init__(self, t, q, v=None, a=None, tau=None, contact_frame=None, f_ext=None, copy=False, **kwargs):
        """
        @brief      Constructor

        @param[in]  q       Configuration vector
        @param[in]  v       Velocity vector
        @param[in]  a       Acceleration vector
        @param[in]  t       Time
        @param[in]  tau     Joint efforts
        @param[in]  contact_frame  Name of the contact frame.
        @param[in]  f_ext   External forces in the contact frame.
        @param[in]  copy    Force to copy the arguments

        @return     Instance of a state.
        """
        ## Time
        self.t = t
        ## Configuration vector
        self.q = _copy(q) if copy else q
        ## Velocity vector
        self.v = _copy(v) if copy else v
        ## Acceleration vector
        self.a = _copy(a) if copy else a
        ## Effort vector
        self.tau = _copy(tau) if copy else tau
        ## Frame name of the contact point, if nay
        self.contact_frame = contact_frame
        ## External forces
        self.f_ext = None
        if f_ext is not None:
            self.f_ext = deepcopy(f_ext) if copy else f_ext

    @staticmethod
    def todict(state_list):
        """
        @brief      Get the dictionary whose keys are the kinematics and dynamics
                    properties at several time steps from a list of State objects.

        @param[in]  state_list      List of State objects

        @return     Kinematics and dynamics state as a dictionary. Each property
                    is a 2D numpy array (row: state, column: time)
        """
        state_dict = {}
        state_dict['t'] = np.array([s.t for s in state_list])
        state_dict['q'] = np.stack([s.q for s in state_list], axis=-1)
        state_dict['v'] = np.stack([s.v for s in state_list], axis=-1)
        state_dict['a'] = np.stack([s.a for s in state_list], axis=-1)
        state_dict['tau'] = [s.tau for s in state_list]
        state_dict['contact_frame'] = [s.contact_frame for s in state_list]
        state_dict['f_ext'] = [s.f_ext for s in state_list]
        return state_dict

    @classmethod
    def fromdict(cls, state_dict):
        """
        @brief      Get a list of State objects from a dictionary whose keys are
                    the kinematics and dynamics properties at several time steps.

        @param[in]  state_dict      Dictionary whose keys are the kinematics and dynamics properties.
                                    Each property is a 2D numpy array (row: state, column: time)

        @return     List of State object
        """
        _state_dict = defaultdict(lambda: [None for i in range(len(state_dict['t']))], state_dict)
        state_list = []
        for i in range(len(state_dict['t'])):
            state_list.append(cls(**{k: v[..., i] if isinstance(v, np.ndarray) else v[i]
                                     for k,v in _state_dict.items()}))
        return state_list

    def __str__(self):
        """
        @brief      Convert the kinematics and dynamics properties into string

        @return     The kinematics and dynamics properties as a string
        """
        msg = ""
        for key, val in self.__dict__.items():
            if val is not None:
                msg += f"{key} : {val}\n"
        return msg

    def __repr__(self):
        return self.__str__()
