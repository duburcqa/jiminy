#!/usr/bin/env python

## @file jiminy_py/log.py

import numpy as np
from csv import DictReader
import typing as tp

from .state import State
from .core import Engine, Robot

def extract_state_from_simulation_log(log_data:tp.Dict, robot:Robot):
    """
    @brief      Extract a trajectory object using from raw simulation data.

    @details    Extract time, joint positions and velocities evolution from log.
    .
    @remark     Note that the quaternion angular velocity vectors are expressed
                it body frame rather than world frame.

    @param[in]  log_data          Data from the log file, in a dictionnary.
    @param[in]  robot             Jiminy robot.

    @return     Trajectory dictionary. The actual trajectory corresponds to
                the field "evolution_robot" and it is a list of State object.
                The other fields are additional information.
    """
    t = log_data['Global.Time']
    qe = np.array([log_data[field] for field in log_data.keys()
                                   if 'currentFreeflyerPosition' in field or 'currentPosition' in field])
    dqe = np.array([log_data[field] for field in log_data.keys()
                                   if 'currentFreeflyerVelocity' in field or 'currentVelocity' in field])
    ddqe = np.array([log_data[field] for field in log_data.keys()
                                   if 'currentFreeflyerAcceleration' in field or 'currentAcceleration' in field])

    # Create state sequence
    evolution_robot = []
    for i in range(len(t)):
        evolution_robot.append(State(qe[:, i], dqe[:, i], ddqe[:, i], t[i]))

    return {'evolution_robot': evolution_robot,
            'robot': robot,
            'use_theoretical_model': False}


def read_log(filename:str) -> tp.Tuple[tp.Dict, tp.Dict]:
    """
    Read a logfile from jiminy. This function supports both text (csv)
    and binary log.

    Parameters:
        - filename: Name of the file to load.
    Retunrs:
        - A dictionnary containing the logged values, and a dictionnary
        containing the constants.
    """
    if is_log_binary(filename):
        # Read binary file using C++ parser.
        data_dict, constants_dict = Engine.read_log_binary(filename)
    else:
        # Read text csv file.
        constants_dict = {}
        with open(filename, 'r') as log:
            consts = next(log).split(', ')
            for c in consts:
                c_split = c.split('=')
                # Remove line end for last constant.
                constants_dict[c_split[0]] = c_split[1].strip('\n')
            # Read data from the log file, skipping the first line (the constants).
            data = {}
            reader = DictReader(log)
            for key, value in reader.__next__().items():
                data[key] = [value]
            for row in reader:
                for key, value in row.items():
                    data[key].append(value)
            for key, value in data.items():
                data[key] = np.array(value, dtype=np.float64)
        # Convert every element to array to provide same API as the C++ parser,
        # removing spaces present before the keys.
        data_dict = {k.strip() : np.array(v) for k,v in data.items()}
    return data_dict, constants_dict

def is_log_binary(filename):
    """
    Return true if the given filename appears to be binary.
    From https://stackoverflow.com/a/11301631/4820605.
    File is considered to be binary if it contains a NULL byte.
    """
    with open(filename, 'rb') as f:
        for block in f:
            if b'\0' in block:
                return True
    return False