import numpy as np
from collections import defaultdict
from copy import copy as _copy, deepcopy
from typing import Optional, Union, List, Dict

from pinocchio import Force, StdVec_Force


class State:
    """
    @brief Store the kinematics and dynamics data of the robot at a given time.
    """
    def __init__(self,
                 t: float,
                 q: Union[np.ndarray, List[float]],
                 v: Optional[Union[np.ndarray, List[float]]] = None,
                 a: Optional[Union[np.ndarray, List[float]]] = None,
                 tau: Optional[Union[np.ndarray, List[float]]] = None,
                 contact_frame: str = None,
                 f_ext: Optional[Union[List[Force], StdVec_Force]] = None,
                 copy: bool = False,
                 **kwargs):
        """
        @brief Constructor.

        @param t  Time.
        @param q  Configuration vector.
        @param v  Velocity vector.
        @param a  Acceleration vector.
        @param tau  Joint efforts.
        @param contact_frame  Name of the contact frame.
        @param f_ext  External forces in the contact frame.
        @param copy  Force to copy the arguments.
        """
        # Time
        self.t = t
        # Configuration vector
        self.q = _copy(q) if copy else q
        # Velocity vector
        self.v = _copy(v) if copy else v
        # Acceleration vector
        self.a = _copy(a) if copy else a
        # Effort vector
        self.tau = _copy(tau) if copy else tau
        # Frame name of the contact point, if nay
        self.contact_frame = contact_frame
        # External forces
        self.f_ext = None
        if f_ext is not None:
            self.f_ext = deepcopy(f_ext) if copy else f_ext

    @staticmethod
    def todict(state_list: List['State']) -> Dict[
            str, Union[np.ndarray, List[Union[List[Force], StdVec_Force]]]]:
        """
        @brief Get the dictionary whose keys are the kinematics and dynamics
               data at several time steps from a list of State objects.

        @param state_list  List of State objects

        @return Kinematics and dynamics data as a dictionary.
                Each property is a 2D numpy array (row: state, column: time).
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
    def fromdict(cls,
                 state_dict: Dict[str, Union[
                     np.ndarray, List[Union[List[Force], StdVec_Force]]]]
                 ) -> List['State']:
        """
        @brief Get a list of State objects from a dictionary whose keys are the
               kinematics and dynamics data at several time steps.

        @param state_dict  Dictionary whose keys are the kinematics and
                           dynamics data. Each property is a 2D numpy
                           array (row: state, column: time).

        @return List of State.
        """
        _state_dict = defaultdict(
            lambda: [None for i in range(len(state_dict['t']))], state_dict)
        state_list = []
        for i in range(len(state_dict['t'])):
            state_list.append(cls(**{
                k: v[..., i] if isinstance(v, np.ndarray) else v[i]
                for k, v in _state_dict.items()}))
        return state_list

    def __repr__(self):
        """
        @brief Convert the kinematics and dynamics data into string.

        @return The kinematics and dynamics data as a string.
        """
        msg = ""
        for key, val in self.__dict__.items():
            if val is not None:
                msg += f"{key} : {val}\n"
        return msg
