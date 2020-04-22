# Utility functions for unit tests.

import numpy as np
from scipy.integrate import ode

from jiminy_py import core as jiminy


def load_urdf_default(urdf_path, motor_names, has_freeflyer = False):
    """
    @brief Create a jiminy.Robot from a URDF with several simplying hypothesis.

    @details The goal of this function is to ease creation of jiminy.Robot
             from a URDF by doing the following operations:
                - loading the URDF, deactivating joint position and velocity bounds.
                - adding motors as supplied, with no rotor inertia and not torque bound.
             These operations allow an unconstraint simulation of a linear system.
    @param[in] urdf_path Path to the URDF file
    @param[in] motor_names Name of the motors
    @param[in] has_freeflyer Optional, set the use of a freeflyer joint.
    """
    robot = jiminy.Robot()
    robot.initialize(urdf_path, has_freeflyer = has_freeflyer)
    for joint_name in motor_names:
        motor = jiminy.SimpleMotor(joint_name)
        robot.attach_motor(motor)
        motor.initialize(joint_name)

    # Configure robot
    model_options = robot.get_model_options()
    motor_options = robot.get_motors_options()
    model_options["joints"]["enablePositionLimit"] = False
    model_options["joints"]["enableVelocityLimit"] = False
    for m in motor_options:
        motor_options[m]['enableEffortLimit'] = False
        motor_options[m]['enableRotorInertia'] = False
    robot.set_model_options(model_options)
    robot.set_motors_options(motor_options)

    return robot

def integrate_dynamics(time, x0, dynamics):
    """
    @brief Integrate the dynamics funciton f(t, x) over timesteps time.
    @details This function solves an initial value problem, similar to
             scipy.solve_ivp, with specified stop points: namely, it solves
             dx = dynamics(t, x)
             x(t = time[0]) = x0
             evaluating x at times points in time (which do not need to
             be equally spaced). While this is also the goal of solve_ivp,
             this function suffers from severe accuracy limitations is our
             usecase: doing the simulation 'manually' with ode and a
             Runge-Kutta integrator yields much higher precision.
    @param[in] time List of time instant at which to evaluate the solution.
    @param[in] x0 Initial starting position.
    @param[in] dynamics Dynamics function, will signature dynamics(t,x) -> dx
    @return A len(time) x dim(x0) array: each line is the solution x at time time[i]
    """
    solver = ode(dynamics)
    solver.set_initial_value(x0, t = time[0])
    solver.set_integrator("dopri5")
    x_sol = [x0]
    for t in time[1:]:
        solver.integrate(t)
        x_sol.append(solver.y)

    return np.stack(x_sol, axis=0)