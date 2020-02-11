#!/usr/bin/env python

## @file jiminy_py/log.py

import numpy as np

from .state import State


def extract_state_from_simulation_log(log_header, log_data, jiminy_model):
    """
    @brief      Extract a trajectory object using from raw simulation data.

    @details    Extract time, joint positions and velocities evolution from log.
    .
    @remark     Note that the quaternion angular velocity vectors are expressed
                it body frame rather than world frame.

    @param[in]  log_header          List of field names
    @param[in]  log_data            Data logged (2D numpy array: row = time, column = data)
    @param[in]  urdf_path           Full path of the URDF file
    @param[in]  jiminy_model        Jiminy model. Optional: None if omitted
    @param[in]  has_freeflyer       Whether the model has a freeflyer

    @return     Trajectory dictionary. The actual trajectory corresponds to
                the field "evolution_robot" and it is a list of State object.
                The other fields are additional information.
    """
    t = log_data[:,log_header.index('Global.Time')]
    qe = log_data[:,np.array(['currentFreeflyerPosition' in field
                              or 'currentPosition' in field for field in log_header])].T
    dqe = log_data[:,np.array(['currentFreeflyerVelocity' in field
                               or 'currentVelocity' in field for field in log_header])].T
    ddqe = log_data[:,np.array(['currentFreeflyerAcceleration' in field
                                or 'currentAcceleration' in field for field in log_header])].T

    # Create state sequence
    evolution_robot = []
    for i in range(len(t)):
        evolution_robot.append(State(qe[:, [i]], dqe[:, [i]], ddqe[:, [i]], t[i]))

    return {'evolution_robot': evolution_robot,
            'jiminy_model': jiminy_model,
            'use_theoretical_model': False}