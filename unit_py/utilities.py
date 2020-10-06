"""
@brief  Utility functions for unit tests.
"""
import os
import numpy as np
from scipy.integrate import ode

from jiminy_py import core as jiminy
from jiminy_py.core import ImuSensor as imu

from pinocchio import neutral


def load_urdf_default(urdf_name,
                      motor_names=(),
                      has_freeflyer=False):
    """
    @brief    Create a jiminy.Robot from a URDF with several simplying
              hypothesis.

    @details  The goal of this function is to ease creation of jiminy.Robot
              from a URDF by doing the following operations:
                 - loading the URDF, deactivating joint position and velocity
                   bounds.
                 - adding motors as supplied, with no rotor inertia and not
                   torque bound.
              These operations allow an unconstrained simulation of a linear
              system.

    @param[in]  urdf_name      Name to the URDF file.
    @param[in]  motor_names    Name of the motors.
    @param[in]  has_freeflyer  Set the use of a freeflyer joint.
                               Optional, no free-flyer by default.
    """
    # Get the urdf path and mesh search directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_root_dir = os.path.join(current_dir, "data")
    urdf_path = os.path.join(data_root_dir, urdf_name)

    # Create and initialize the robot
    robot = jiminy.Robot()
    robot.initialize(urdf_path, has_freeflyer, [data_root_dir])

    # Add motors to the robot
    for joint_name in motor_names:
        motor = jiminy.SimpleMotor(joint_name)
        robot.attach_motor(motor)
        motor.initialize(joint_name)

    # Configure the robot
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

def setup_controller_and_engine(engine,
                                robot,
                                compute_command=None,
                                internal_dynamics=None):
    """
    @brief Setup an engine to integrate the dynamics of a given robot,
           for a specific user-defined control law and internal dynamics.

    @details The goal of this function is to ease the configuration of
             jiminy.Engine by doing the following operations:
               - Wrapping the control law and internal dynamics in a
                 jiminy.ControllerFunctor.
               - Register the system robot/controller in the engine to
                 integrate its dynamics.

    @param engine  Engine used to integrate the system dynamics.
    @param robot  Robot to control.
    @param compute_command  Control law, which must be an function handle
                            with the following signature:
                                f(t, q, v, sensors_data, uMotors) -> None
                            Optional: zero command torques by default.
    @param internal_dynamics  Internal dynamics function handle with
                              signature:
                                  f(t, q, v, sensors_data, uFull) -> None
                              Optional: No internal dynamics by default.
    """
    # Instantiate the controller
    controller = jiminy.ControllerFunctor(
        compute_command, internal_dynamics)

    # Initialize the controller
    if controller is not None:
        controller.initialize(robot)

    # Initialize the engine
    if controller is not None:
        engine.initialize(robot, controller)
    else:
        engine.initialize(robot)

def neutral_state(robot):
    """
    @brief   Return the neutral state of the robot, namely zero joint
             positions and unit quaternions regarding the configuration, and
             zero velocity vector.

    @param robot  Robot for which to comput the neutral state.
    """
    q0 = neutral(robot.pinocchio_model)
    v0 = np.zeros(robot.nv)
    x0 = np.concatenate((q0, v0))
    return x0

def integrate_dynamics(time, x0, dynamics):
    """
    @brief Integrate the dynamics function f(t, x) over timesteps time.

    @details This function solves an initial value problem, similar to
             scipy.solve_ivp, with specified stop points: namely, it solves
                 dx = dynamics(t, x)
                 x(t = time[0]) = x0
             evaluating x at times points in time (which do not need to be
             equally spaced). While this is also the goal of solve_ivp, this
             function suffers from severe accuracy limitations is our usecase:
             doing the simulation 'manually' with ode and a Runge-Kutta
             integrator yields much higher precision.

    @param time  Timesteps at which to evaluate the solution.
    @param x0  Initial state of the system.
    @param dynamics  Dynamics of the system, with signature:
                         dynamics(t,x) -> dx.

    @return np.ndarray[len(time), dim(x0)]: each line is the solution x at
             time time[i]
    """
    solver = ode(dynamics)
    solver.set_initial_value(x0, t = time[0])
    solver.set_integrator("dopri5")
    x_sol = [x0]
    for t in time[1:]:
        solver.integrate(t)
        x_sol.append(solver.y)
    return np.stack(x_sol, axis=0)

def simulate_and_get_state_evolution(engine, tf, x0, split=False):
    """
    @brief Simulate the dynamics of the system and retrieve the state
           evolution over time.

    @param engine  List of time instant at which to evaluate the solution.
    @param tf  Duration of the simulation.
    @param x0  Initial state of the system.

    @return  - time: np.ndarray[len(time)]
             - state evoluion: np.ndarray[len(time), dim(x0)]: each line is the
               solution x at time time[i]
    """

    # Run simulation
    engine.simulate(tf, x0)

    # Get log data
    log_data, _ = engine.get_log()

    # Extract state evolution over time
    time = log_data['Global.Time']
    if isinstance(engine, jiminy.Engine):
        q_jiminy = np.stack([
            log_data['.'.join(['HighLevelController', s])]
            for s in engine.robot.logfile_position_headers], axis=-1)
        v_jiminy = np.stack([
            log_data['.'.join(['HighLevelController', s])]
            for s in engine.robot.logfile_velocity_headers], axis=-1)
        if split:
            return time, q_jiminy, v_jiminy
        else:
            x_jiminy = np.concatenate((q_jiminy, v_jiminy), axis=-1)
            return time, x_jiminy
    else:
        q_jiminy = [np.stack([
            log_data['.'.join(['HighLevelController', sys.name, s])]
            for s in sys.robot.logfile_position_headers
        ], axis=-1) for sys in engine.systems]
        v_jiminy = [np.stack([
            log_data['.'.join(['HighLevelController', sys.name, s])]
            for s in sys.robot.logfile_velocity_headers
        ], axis=-1) for sys in engine.systems]
        if split:
            return time, q_jiminy, v_jiminy
        else:
            x_jiminy = [np.concatenate((q, v), axis=-1)
                        for q, v in zip(q_jiminy, v_jiminy)]
            return time, x_jiminy

def simulate_and_get_imu_data_evolution(engine, tf, x0, split=False):
    # Run simulation
    engine.simulate(tf, x0)

    # Get log data
    log_data, _ = engine.get_log()

    # Extract state evolution over time
    time = log_data['Global.Time']
    if split:
        quat_jiminy = np.stack([log_data['.'.join(('PendulumLink', f))]
            for f in imu.fieldnames if f.startswith('Quat')
        ], axis=-1)
        gyro_jiminy = np.stack([log_data['.'.join(('PendulumLink', f))]
            for f in imu.fieldnames if f.startswith('Gyro')
        ], axis=-1)
        accel_jiminy = np.stack([log_data['.'.join(('PendulumLink', f))]
            for f in imu.fieldnames if f.startswith('Accel')
        ], axis=-1)
        return time, quat_jiminy, gyro_jiminy, accel_jiminy
    else:
        imu_jiminy = np.stack([log_data['.'.join(('PendulumLink', f))]
                for f in jiminy.ImuSensor.fieldnames
        ], axis=-1)
        return time, imu_jiminy
