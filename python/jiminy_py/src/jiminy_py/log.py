# mypy: disable-error-code="attr-defined, name-defined"
"""Utilities for extracting structured information from log data, from
reconstructing the robot to reading telemetry variables.
"""
import re
from bisect import bisect_right
from collections import OrderedDict
from typing import (
    Any, Callable, List, Dict, Optional, Sequence, Union, Literal, Type,
    overload)

import numpy as np

from . import core as jiminy
from . import tree
from .core import (  # pylint: disable=no-name-in-module
    EncoderSensor as encoder,
    EffortSensor as effort,
    ContactSensor as contact,
    ForceSensor as force,
    ImuSensor as imu)
from .dynamics import State, TrajectoryDataType


SENSORS_FIELDS: Dict[
        Type[jiminy.AbstractSensor], Union[List[str], Dict[str, List[str]]]
        ] = {
    encoder: encoder.fieldnames,
    effort: effort.fieldnames,
    contact: contact.fieldnames,
    force: {
        k: [e[len(k):] for e in force.fieldnames if e.startswith(k)]
        for k in ['F', 'M']
    },
    imu: {
        k: [e[len(k):] for e in imu.fieldnames if e.startswith(k)]
        for k in ['Quat', 'Gyro', 'Accel']
    }
}


FieldNested = Union[Dict[str, 'FieldNested'], Sequence['FieldNested'], str]


read_log = jiminy.core.Engine.read_log


@overload
def extract_variables_from_log(log_vars: Dict[str, np.ndarray],
                               fieldnames: FieldNested,
                               namespace: str = "",
                               *, as_dict: Literal[False] = False
                               ) -> List[np.ndarray]:
    ...


@overload
def extract_variables_from_log(log_vars: Dict[str, np.ndarray],
                               fieldnames: FieldNested,
                               namespace: str = "",
                               *, as_dict: Literal[True]
                               ) -> Dict[str, np.ndarray]:
    ...


def extract_variables_from_log(log_vars: Dict[str, np.ndarray],
                               fieldnames: FieldNested,
                               namespace: str = "",
                               *, as_dict: bool = False
                               ) -> Union[
                                   List[np.ndarray], Dict[str, np.ndarray]]:
    """Extract values associated with a set of variables in a specific
    namespace.

    :param log_vars: Logged variables as a dictionary.
    :param fieldnames: Structured fieldnames.
    :param namespace: Namespace of the fieldnames. Empty string to disable.
                      Optional: Empty by default.
    :param keep_structure: Whether to return a dictionary mapping flattened
                           fieldnames to values.
                           Optional: True by default.

    :returns:
        `np.ndarray` or None for each fieldname individually depending if it is
        found or not.
    """
    # Extract values from log if it exists
    if as_dict:
        keys: List[str] = []
    values: List[np.ndarray] = []
    for fieldname_path, fieldname in tree.flatten_with_path(fieldnames):
        # A key is the concatenation of namespace and full path of fieldname
        key = ".".join(filter(
            lambda key: isinstance(key, str) and key,  # type: ignore[arg-type]
            (namespace, *fieldname_path, fieldname)))

        # Raise an exception if the key does not exists and not fail safe
        if key not in log_vars:
            raise ValueError(f"Variable '{key}' not found in log file.")

        # Extract the value corresponding to the key
        if as_dict:
            keys.append(key)
        values.append(log_vars[key])

    # Return flat mapping from fieldnames (without prefix) to scalar values
    if as_dict:
        return OrderedDict(zip(keys, values))

    return values


def build_robot_from_log(
        log_data: Dict[str, Any],
        mesh_path_dir: Optional[str] = None,
        mesh_package_dirs: Sequence[str] = (),
        *, robot_name: Optional[str] = None
        ) -> jiminy.Robot:
    """Create and initialize a robot from single- or multi- robot simulations.

    .. note::
        It returns a valid and fully initialized robot, that can be used to
        perform new simulation if added to a Jiminy Engine, but the original
        controller is lost.

    .. note::
        Model options and `robot.pinocchio_model` will be the same as during
        the simulation until the next call to `reset` or `set_options` methods.

    .. note::
        In case of multi-robot simulation, one may use `build_robots_from_log`
        for building all robots at once without having the specify explicitly
        the name for each of them.

    :param log_data: Logged data (constants plus variables) as a dictionary.
    :param mesh_path_dir: Overwrite the common root of all absolute mesh paths.
                          It may be necessary to read logs generated on
                          different environments.
    :param mesh_package_dirs: Prepend custom mesh package search path
                              directories to the ones provided by log file. It
                              may be necessary to specify it to read log
                              generated on a different environment.
    :param robot_name: Name of the robot to build from log. If `None`, then the
                       name will be detected automatically in case of
                       single-robot simulations, otherwise it will raise an
                       exception.

    :returns: Reconstructed robot.
    """
    # Extract log constants
    log_constants = log_data["constants"]

    # Try to infer robot names from log constants if not specified
    if robot_name is None:
        robot_names = []
        for key in log_data["constants"].keys():
            robot_name_match = re.match(r"^(\w*?)\.?robot$", key)
            if robot_name_match is not None:
                robot_names.append(robot_name_match.group(1))
        if len(robot_names) > 1:
            raise ValueError(
                "In case of multi-robot simulations, the name of the robot to "
                "build must be specified. Please call `build_robots_from_log` "
                "to build all robots at once.")
        assert robot_names
        robot_name = robot_names[0]

    # Get binary serialized robot data
    robot_data = log_constants[".".join(filter(None, (robot_name, "robot")))]

    # Load robot
    return jiminy.build_robot_from_binary(
        robot_data, mesh_path_dir, mesh_package_dirs)


def build_robots_from_log(
        log_data: Dict[str, Any],
        mesh_path_dir: Optional[str] = None,
        mesh_package_dirs: Sequence[str] = (),
        ) -> Sequence[jiminy.Robot]:
    """Build all the robots in the log of the simulation.

    .. note::
        Internally, this function calls `build_robot_from_log` to build each
        available robot. Refer to `build_robot_from_log` for more information.

    :param log_data: Logged data (constants and variables) as a dictionary.
    :param mesh_path_dir: Overwrite the common root of all absolute mesh paths.
                          It which may be necessary to read log generated on a
                          different environment.
    :param mesh_package_dirs: Prepend custom mesh package search path
                              directories to the ones provided by log file. It
                              may be necessary to specify it to read log
                              generated on a different environment.

    :returns: Sequence of reconstructed robots.
    """
    # Try to infer robot names from log constants
    robot_names = []
    for key in log_data["constants"].keys():
        robot_name_match = re.match(r"^(\w*?)\.?robot$", key)
        if robot_name_match is not None:
            robot_names.append(robot_name_match.group(1))

    # Build all the robots sequentially
    robots = []
    for robot_name in robot_names:
        robot = build_robot_from_log(
            log_data, mesh_path_dir, mesh_package_dirs, robot_name=robot_name)
        robots.append(robot)

    return robots


def extract_trajectory_from_log(log_data: Dict[str, Any],
                                robot: Optional[jiminy.Robot] = None,
                                *, robot_name: Optional[str] = None
                                ) -> TrajectoryDataType:
    """Extract the minimal required information from raw log data in order to
    replay the simulation in a viewer.

    .. note::
        It extracts the required data for replay, namely temporal evolution of:
          - robot configuration: to display of the robot on the scene,
          - robot velocity: to update velocity-dependent markers such as DCM,
          - external forces: to update force-dependent markers.

    :param log_data: Logged data (constants and variables) as a dictionary.
    :param robot: Jiminy robot associated with the logged trajectory.
                  Optional: None by default. If None, then it will be
                  reconstructed from 'log_data' using `build_robot_from_log`.
    :param robot_name: Name of the robot to be constructed in the log. If it
                       is not known, call `build_robot_from_log`.

    :returns: Trajectory dictionary. The actual trajectory corresponds to the
              field "evolution_robot" and it is a list of State object. The
              other fields are additional information.
    """
    # Prepare robot namespace
    if robot is not None:
        if robot_name is None:
            robot_name = robot.name
        elif robot_name != robot.name:
            raise ValueError(
                "The name specified in `robot_name` and the name of the robot "
                "are differents.")
    elif robot_name is None:
        robot_name = ""

    # Handling of default argument(s)
    if robot is None:
        robot = build_robot_from_log(log_data, robot_name=robot_name)

    # Define some proxies for convenience
    log_vars = log_data["variables"]

    # Extract the joint positions, velocities and external forces over time
    positions = np.stack([
        log_vars.get(".".join(filter(None, (robot_name, field))), [])
        for field in robot.log_position_fieldnames], axis=-1)
    velocities = np.stack([
        log_vars.get(".".join(filter(None, (robot_name, field))), [])
        for field in robot.log_velocity_fieldnames], axis=-1)
    forces = np.stack([
        log_vars.get(".".join(filter(None, (robot_name, field))), [])
        for field in robot.log_f_external_fieldnames], axis=-1)

    # Determine which optional data are available
    has_positions = len(positions) > 0
    has_velocities = len(velocities) > 0
    has_forces = len(forces) > 0

    # Create state sequence
    evolution_robot = []
    q, v, f_ext = None, None, None
    for i, t in enumerate(log_vars["Global.Time"]):
        if has_positions:
            q = positions[i]
        if has_velocities:
            v = velocities[i]
        if has_forces:
            f_ext = [forces[i, (6 * (j - 1)):(6 * j)]
                     for j in range(1, robot.pinocchio_model.njoints)]
        evolution_robot.append(State(
            t=t, q=q, v=v, f_ext=f_ext))  # type: ignore[arg-type]

    return {"evolution_robot": evolution_robot,
            "robot": robot,
            "use_theoretical_model": False}


def extract_trajectories_from_log(
        log_data: Dict[str, Any],
        robots: Optional[Sequence[jiminy.Robot]] = None
        ) -> Dict[str, TrajectoryDataType]:
    """Extract the minimal required information from raw log data in order to
    replay the simulation in a viewer.

    .. note::
        This function calls `extract_trajectory_from_log` to extract each
        robot's trajectory. Refer to `extract_trajectory_from_log` for more
        information.

    :param log_data: Logged data (constants and variables) as a dictionary.
    :param robots: Sequend of Jiminy robots associated with the logged
                   trajectory.
                   Optional: None by default. If None, then it will be
                   reconstructed from 'log_data' using `build_robots_from_log`.

    :returns: Dictonary mapping each robot name to its corresponding
              trajectory.
    """
    # Handling of default argument(s)
    if robots is None:
        robots = build_robots_from_log(log_data)

    # Load the trajectory associated with each robot sequentially
    trajectories = {}
    for robot in robots:
        trajectories[robot.name] = extract_trajectory_from_log(
            log_data, robot, robot_name=robot.name)

    return trajectories


def update_sensor_measurements_from_log(
        log_data: Dict[str, Any],
        robot: jiminy.Model
        ) -> Callable[[float, np.ndarray, np.ndarray], None]:
    """Helper to make it easy to emulate sensor data update based on log data.

    .. note::
        It returns an update_hook that can forwarding the `Viewer.replay` to
        display sensor information such as contact forces for instance.

    :param log_data: Logged data (constants and variables) as a dictionary.
    :param robot: Jiminy robot associated with the logged trajectory.

    :returns: Callable taking update time in argument and returning nothing.
              Note that it does not through an exception if out-of-range, but
              rather clip to desired time to the available data range.
    """
    # Extract time from log
    log_vars = log_data["variables"]
    times = log_vars["Global.Time"]

    # Filter sensors whose data is available
    sensors_set, sensors_log = [], []
    for sensor_type, sensor_names in robot.sensor_names.items():
        sensor_fieldnames = getattr(jiminy, sensor_type).fieldnames
        for name in sensor_names:
            sensor = robot.get_sensor(sensor_type, name)
            log_fieldnames = [
                '.'.join((sensor.name, field)) for field in sensor_fieldnames]
            if log_fieldnames[0] in log_vars.keys():
                sensor_log = np.stack([
                    log_vars[field] for field in log_fieldnames], axis=-1)
                sensors_set.append(sensor)
                sensors_log.append(sensor_log)

    def update_hook(t: float,
                    q: np.ndarray,  # pylint: disable=unused-argument
                    v: np.ndarray  # pylint: disable=unused-argument
                    ) -> None:
        nonlocal times, sensors_set, sensors_log

        # Get surrounding indices in log data
        i = bisect_right(times, t)
        i_prev, i_next = max(i - 1, 0), min(i, len(times) - 1)

        # Early return if no more data is available
        if i_next == i_prev:
            return

        # Compute current time ratio
        ratio = (t - times[i_prev]) / (times[i_next] - times[i_prev])

        # Update sensors data
        for sensor, sensor_log, in zip(sensors_set, sensors_log):
            value_prev, value_next = sensor_log[i_prev], sensor_log[i_next]
            value = value_prev + (value_next - value_prev) * ratio
            sensor.data = value

    return update_hook
