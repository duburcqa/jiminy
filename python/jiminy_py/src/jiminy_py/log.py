#!/usr/bin/env python

## @file jiminy_py/log.py

import numpy as np

from .state import State


def extract_state_from_simulation_log(log_data, jiminy_model):
    """
    @brief      Extract a trajectory object using from raw simulation data.

    @details    Extract time, joint positions and velocities evolution from log.
    .
    @remark     Note that the quaternion angular velocity vectors are expressed
                it body frame rather than world frame.

    @param[in]  log_data          Data from the log file, in a dictionnary.
    @param[in]  jiminy_model        Jiminy model. Optional: None if omitted

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
            'jiminy_model': jiminy_model,
            'use_theoretical_model': False}
