#!/usr/bin/env python

## @file jiminy_py/state.py

import numpy as np
from collections import defaultdict
from copy import copy, deepcopy


class State:
    """
    @brief      Object that contains the kinematics and dynamics state of the
                robot at a given time.
    """
    def __init__(self, q, v, a, t=None, f=None, tau=None, f_ext=None, **kwargs):
        """
        @brief      Constructor

        @param[in]  q       Configuration vector (with freeflyer if any) (1D numpy array)
        @param[in]  v       Velocity vector (1D numpy array)
        @param[in]  a       Acceleration vector (1D numpy array)
        @param[in]  t       Time
        @param[in]  f       Forces on the different bodies of the robot. Dictionary whose keys represent
                            a given foot orientation. For each orientation, a dictionary contains the
                            6D-force for each body (1D numpy array).
        @param[in]  tau     Joint torques. Dictionary whose keys represent a given foot orientation.
        @param[in]  f_ext   External forces represented in the frame of the Henke ankle. Dictionary
                            whose keys represent a given foot orientation.

        @return     Instance of a state.
        """
        ## Time
        self.t = copy(t)
        ## Configuration vector
        self.q = copy(q)
        ## Velocity vector
        self.v = copy(v)
        ## Acceleration vector
        self.a = copy(a)
        ## Forces on the different bodies of the robot
        self.f = {}
        if f is not None:
            self.f = deepcopy(f)
        ## Torque vector
        self.tau = {}
        if tau is not None:
            self.tau = deepcopy(tau)
        ## External forces represented in the frame of the Henke ankle
        self.f_ext = {}
        if f_ext is not None:
            self.f_ext = deepcopy(f_ext)

    @staticmethod
    def todict(state_list):
        """
        @brief      Get the dictionary whose keys are the kinematics and dynamics
                    properties at several time steps from a list of State objects.

        @param[in]  state_list      List of State objects

        @return     Kinematics and dynamics state as a dictionary. Each property
                    is a 2D numpy array (row: state, column: time)
        """
        state_dict = dict()
        state_dict['q'] = np.stack([s.q for s in state_list], axis=-1)
        state_dict['v'] = np.stack([s.v for s in state_list], axis=-1)
        state_dict['a'] = np.stack([s.a for s in state_list], axis=-1)
        state_dict['t'] = np.array([s.t for s in state_list])
        state_dict['f'] = [s.f for s in state_list]
        state_dict['tau'] = [s.tau for s in state_list]
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
        _state_dict = defaultdict(lambda: [None for i in range(state_dict['q'].shape[-1])], state_dict)
        state_list = []
        for i in range(state_dict['q'].shape[1]):
            state_list.append(cls(**{k: v[..., i] if isinstance(v, np.ndarray) else v[i]
                                     for k,v in _state_dict.items()}))
        return state_list

    def __repr__(self):
        """
        @brief      Convert the kinematics and dynamics properties into string

        @return     The kinematics and dynamics properties as a string
        """
        return "State(q=\n{!r},\nv=\n{!r},\na=\n{!r},\nt=\n{!r},\nf=\n{!r},\nf_ext=\n{!r})".format(
            self.q, self.v, self.a, self.t, self.f, self.f_ext)
