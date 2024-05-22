"""Utility functions for unit tests.
"""
import os
import numpy as np
from scipy.integrate import ode
from typing import Optional, Union, Dict, Sequence, Tuple, Callable

import jiminy_py.core as jiminy

from pinocchio import neutral


FunctionalControllerCallable = Callable[[
    float, np.ndarray, np.ndarray, jiminy.SensorMeasurementTree, np.ndarray
    ], None]


def load_urdf_default(urdf_name: str,
                      motor_names: Sequence[str] = (),
                      has_freeflyer: bool = False,
                      robot_name: str = "") -> jiminy.Robot:
    """Create a jiminy.Robot from a URDF with several simplifying hypothesis.

    The goal of this function is to ease creation of `jiminy.Robot` from a URDF
    by doing the following operations:
      - loading the URDF and deactivate position/velocity bounds
      - adding motors with no rotor inertia and no torque bounds
    These operations allow an unconstrained simulation of a linear system.

    :param urdf_name: Name to the URDF file.
    :param motor_names: Name of the motors.
    @param has_freeflyer: Set the use of a freeflyer joint.
                          Optional: no free-flyer by default.
    """
    # Get the urdf path and mesh search directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_root_dir = os.path.join(current_dir, "data")
    urdf_path = os.path.join(data_root_dir, urdf_name)

    # Create and initialize the robot
    robot = jiminy.Robot(robot_name)
    robot.initialize(urdf_path, has_freeflyer, [data_root_dir])

    # Add motors to the robot
    for joint_name in motor_names:
        motor = jiminy.SimpleMotor(joint_name)
        robot.attach_motor(motor)
        motor.initialize(joint_name)

    # Configure the robot
    robot_options = robot.get_options()
    motor_options = robot_options["motors"]
    for motor in robot.motors:
        motor_options[motor.name]["enableVelocityLimit"] = False
        motor_options[motor.name]['enableEffortLimit'] = False
        motor_options[motor.name]['enableArmature'] = False
    robot.set_options(robot_options)

    return robot


def setup_controller_and_engine(
        engine: jiminy.Engine,
        robot: jiminy.Robot,
        compute_command: Optional[FunctionalControllerCallable] = None,
        internal_dynamics: Optional[FunctionalControllerCallable] = None
        ) -> None:
    r"""Setup an engine to integrate the dynamics of a given robot, for a
    specific user-defined control law and internal dynamics.

    The goal of this function is to ease the configuration of `jiminy.Engine`
    by doing the following operations:
      - Wrapping the control law and internal dynamics as
        `jiminy.FunctionalController`.
      - Register the system robot/controller in the engine to
        integrate its dynamics.

    :param engine: Engine used to integrate the system dynamics.
    :param robot: Robot to control.
    :param compute_command:
        .. raw:: html

            Control law as a callable with signature:

        | compute_command\(
        |    **t**: float,
        |    **q**: np.ndarray,
        |    **v**: np.ndarray,
        |    **sensor_measurements**: jiminy_py.core.SensorMeasurementTree,
        |    **u_command**: np.ndarray
        |    \) -> None

        Optional: zero command torques by default.
    :param internal_dynamics:
        .. raw:: html

            Internal dynamics as a callable with signature:

        | internal_dynamics\(
        |    **t**: float,
        |    **q**: np.ndarray,
        |    **v**: np.ndarray,
        |    **sensor_measurements**: jiminy_py.core.SensorMeasurementTree,
        |    **u_command**: np.ndarray
        |    \) -> None

        Optional: No internal dynamics by default.
    """
    # Instantiate the controller
    robot.controller = jiminy.FunctionalController(
        compute_command, internal_dynamics)

    # Initialize the engine
    engine.add_robot(robot)

    # Enable telemetry by default
    engine_options = engine.get_options()
    telemetry_options = engine_options["telemetry"]
    telemetry_options["enableEffort"] = True
    telemetry_options["enableEnergy"] = True
    engine.set_options(engine_options)


def neutral_state(robot: jiminy.Model,
                  split: bool = False
                  ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Compute the neutral state of the robot, namely zero joint positions and
    unit quaternions regarding the configuration, and zero velocity vector.

    :param robot: Robot for which to compute the neutral state.
    """
    q0 = neutral(robot.pinocchio_model)
    v0 = np.zeros(robot.nv)
    if split:
        return q0, v0
    else:
        return np.concatenate((q0, v0))


def integrate_dynamics(time: np.ndarray,
                       x0: np.ndarray,
                       dynamics: Callable[[float, np.ndarray], np.ndarray]
                       ) -> np.ndarray:
    """Integrate the dynamics function f(t, x) over timesteps time.

    This function solves an initial value problem, similar to
    `scipy.solve_ivp`, with specified stop points: namely, it solves:

    .. code-block:: python

        dx = dynamics(t, x)
        x(t = time[0]) = x0

    evaluating x at times points in time (which do not need to be equally
    spaced). While this is also the goal of solve_ivp, this function suffers
    from severe accuracy limitations is our use case: doing the simulation
    'manually' with ode and a Runge-Kutta integrator yields much higher
    precision.

    :param time: Timesteps at which to evaluate the solution.
    :param x0: Initial state of the system.
    :param dynamics: Dynamics of the system, with signature:

                     .. code-block:: python

                         dynamics(t: float, x: np.ndarray) -> np:ndarray

    :return: 2D array for which the i-th line is the solution x at `time[i]`.
    """
    solver = ode(dynamics)
    solver.set_initial_value(x0, t=time[0])
    solver.set_integrator("dopri5")
    x_sol = [x0]
    for t in time[1:]:
        solver.integrate(t)
        x_sol.append(solver.y)
    return np.stack(x_sol, axis=0)


def simulate_and_get_state_evolution(
        engine: jiminy.Engine,
        tf: float,
        x0: Union[Dict[str, np.ndarray], np.ndarray],
        split: bool = False) -> Union[
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray]]:
    """Simulate the dynamics of the system and retrieve the state evolution
    over time.

    :param engine: List of time instant at which to evaluate the solution.
    :param tf: Duration of the simulation.
    :param x0: Initial state of the system.
    :param split: Whether to return q, v separately or as a state vector x.

    :returns: Pair containing first the sequence of time and second the
        associated sequence of states as a 2D array each line corresponds to a
        given time.
    """
    # Enable telemetry by default
    engine_options = engine.get_options()
    telemetry_options = engine_options["telemetry"]
    telemetry_options["enableEffort"] = True
    telemetry_options["enableEnergy"] = True
    engine.set_options(engine_options)

    # Run simulation
    if isinstance(x0, np.ndarray):
        robot, = engine.robots
        q0, v0 = x0[:robot.nq], x0[-robot.nv:]
    else:
        q0, v0 = {}, {}
        for robot in engine.robots:
            q0[robot.name] = x0[robot.name][:robot.nq]
            v0[robot.name] = x0[robot.name][-robot.nv:]
    engine.simulate(tf, q0, v0)

    # Get log data
    log_vars = engine.log_data["variables"]

    # Extract state evolution over time
    time = log_vars['Global.Time']
    if isinstance(x0, np.ndarray):
        robot, = engine.robots

        q_jiminy = np.stack([
            log_vars[field] for field in robot.log_position_fieldnames
            ], axis=-1)
        v_jiminy = np.stack([
            log_vars[field] for field in robot.log_velocity_fieldnames
            ], axis=-1)

        if split:
            return time, q_jiminy, v_jiminy

        x_jiminy = np.concatenate((q_jiminy, v_jiminy), axis=-1)
        return time, x_jiminy
    else:
        q_jiminy = [np.stack([
            log_vars['.'.join((robot.name, field))]
            for field in robot.log_position_fieldnames
        ], axis=-1) for robot in engine.robots]
        v_jiminy = [np.stack([
            log_vars['.'.join((robot.name, field))]
            for field in robot.log_velocity_fieldnames
        ], axis=-1) for robot in engine.robots]

        if split:
            return time, q_jiminy, v_jiminy

        x_jiminy = [np.concatenate((q, v), axis=-1)
                    for q, v in zip(q_jiminy, v_jiminy)]
        return time, x_jiminy
