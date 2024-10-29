# mypy: disable-error-code="attr-defined, name-defined"
"""Utilities for extracting structured information from log data, from
reconstructing the robot to reading telemetry variables.
"""
import re
from bisect import bisect_right
from itertools import zip_longest, starmap
from collections import OrderedDict
from typing import (
    Any, Callable, List, Dict, Optional, Sequence, Union, Literal, overload,
    cast)

import numpy as np

from . import core as jiminy
from . import tree
from .dynamics import State, Trajectory


FieldNested = Union[Dict[str, 'FieldNested'], Sequence['FieldNested'], str]
UpdateHook = Callable[[float, np.ndarray, Optional[np.ndarray]], None]

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
    :param as_dict: Whether to return a dictionary mapping flattened fieldnames
                    to values.
                    Optional: True by default.
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
            raise KeyError(f"Variable '{key}' not found in log file.")

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

    # Load robot from binary data
    return jiminy.load_from_binary(
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
                                ) -> Trajectory:
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

    :returns: Dictionary whose keys are the name of each robot and values are
              the corresponding `Trajectory` object.
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

    # Extract robot state data over time for all quantities available
    data: Dict[str, Union[Sequence[np.ndarray], np.ndarray]] = OrderedDict()
    for name in ("position",
                 "velocity",
                 "acceleration",
                 "effort",
                 "command",
                 "f_external",
                 "constraint"):
        fieldnames = getattr(robot, f"log_{name}_fieldnames")
        try:
            data[name] = extract_variables_from_log(
                log_vars, fieldnames, robot_name)
        except KeyError:
            data[name] = []

    # Add fictitious 'universe' external force and reshape data if available
    f_ext: np.ndarray = cast(np.ndarray, data["f_external"])
    if len(f_ext) > 0:
        f_ext = np.stack((*((np.zeros_like(f_ext[0]),) * 6), *f_ext), axis=-1)
        data["f_external"] = f_ext.reshape((len(f_ext), -1, 6), order='A')

    # Stack available data
    for key, values in data.items():
        if len(values) > 0 and not isinstance(values, np.ndarray):
            data[key] = np.stack(values, axis=-1)

    # Create state sequence
    states = tuple(starmap(
        State, zip_longest(log_vars["Global.Time"], *data.values())))

    # Create the trajectory
    return Trajectory(states, robot, use_theoretical_model=False)


def extract_trajectories_from_log(
        log_data: Dict[str, Any],
        robots: Optional[Sequence[jiminy.Robot]] = None
        ) -> Dict[str, Trajectory]:
    """Extract the minimal required information from raw log data in order to
    replay the simulation in a viewer.

    .. note::
        This function calls `extract_trajectory_from_log` to extract each
        robot's trajectory. Refer to `extract_trajectory_from_log` for more
        information.

    :param log_data: Logged data (constants and variables) as a dictionary.
    :param robots: Sequence of Jiminy robots associated with the logged
                   trajectory.
                   Optional: None by default. If None, then it will be
                   reconstructed from 'log_data' using `build_robots_from_log`.

    :returns: Dictionary mapping each robot name to its corresponding
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
        log_data: Dict[str, Any], robot: jiminy.Model) -> UpdateHook:
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
    sensor_data_map = []
    for sensor_group in robot.sensors.values():
        for sensor in sensor_group:
            log_sensor_name = ".".join((sensor.type, sensor.name))
            try:
                data = np.stack(extract_variables_from_log(
                    log_vars, sensor.fieldnames, log_sensor_name), axis=-1)
                sensor_data_map.append((sensor, data))
            except KeyError:
                pass

    def update_hook(t: float,
                    q: np.ndarray,  # pylint: disable=unused-argument
                    v: Optional[np.ndarray]  # pylint: disable=unused-argument
                    ) -> None:
        nonlocal times, sensor_data_map

        # Early return if no more data is available
        if t > times[-1]:
            return

        # Get surrounding indices in log data
        i = bisect_right(times, t)
        i_prev, i_next = max(i - 1, 0), min(i, len(times) - 1)

        # Compute linear interpolation ratio
        if i_next == i_prev:
            # Special case for final data point
            ratio = 1.0
        else:
            ratio = (t - times[i_prev]) / (times[i_next] - times[i_prev])

        # Update sensors data
        for sensor, data in sensor_data_map:
            value_prev, value_next = data[i_prev], data[i_next]
            sensor.data = value_prev + (value_next - value_prev) * ratio

    return update_hook
