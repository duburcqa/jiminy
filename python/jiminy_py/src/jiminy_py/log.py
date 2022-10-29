import os
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Callable, Tuple, Dict, Optional, Sequence, Union

import tree
import numpy as np

import pinocchio as pin

from . import core as jiminy
from .core import (EncoderSensor as encoder,
                   EffortSensor as effort,
                   ContactSensor as contact,
                   ForceSensor as force,
                   ImuSensor as imu)
from .dynamics import State, TrajectoryDataType


ENGINE_NAMESPACE = "HighLevelController"
SENSORS_FIELDS = {
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


FieldNested = Union[
    Dict[str, 'FieldNested'], Sequence['FieldNested'],
    str]  # type: ignore[misc]


read_log = jiminy.core.EngineMultiRobot.read_log


def extract_variables_from_log(log_vars: Dict[str, Any],
                               fieldnames: FieldNested,
                               namespace: Optional[str] = ENGINE_NAMESPACE,
                               *, as_dict: bool = False) -> Optional[Union[
                                   Tuple[Optional[np.ndarray], ...],
                                   Dict[str, Optional[np.ndarray]]]]:
    """Extract values associated with a set of variables in a specific
    namespace.

    :param log_vars: Logged variables as a dictionary.
    :param fieldnames: Structured fieldnames.
    :param namespace: Namespace of the fieldnames. None to disable.
                      Optional: ENGINE_TELEMETRY_NAMESPACE by default.
    :param keep_structure: Whether to return a dictionary mapping flattened
                           fieldnames to values.
                           Optional: True by default.

    :returns:
        `np.ndarray` or None for each fieldname individually depending if it is
        found or not.
    """
    # Key are the concatenation of the path and value
    keys = [
        ".".join(map(str, filter(
            lambda key: isinstance(key, str), (*fieldname_path, fieldname))))
        for fieldname_path, fieldname in tree.flatten_with_path(fieldnames)]

    # Extract value from log if it exists
    values = [
        log_vars.get(
            ".".join(filter(None, (namespace, key))), None) for key in keys]

    # Return None if no value was found
    if not values or all(elem is None for elem in values):
        return None

    if as_dict:
        # Return flat mapping from fieldnames (without prefix) to scalar values
        values = OrderedDict(zip(keys, values))

    return values


def build_robot_from_log(
        log_data: Dict[str, Any],
        mesh_package_dirs: Union[str, Sequence[str]] = ()) -> jiminy.Model:
    """Build robot from log.

    .. note::
        model options and `robot.pinocchio_model` are guarantee to be the same
        as during the simulation until the next call to `reset` method.

    .. note::
        It returns a valid and fully initialized robot, that can be used to
        perform new simulation if added to a Jiminy Engine, but the original
        controller is lost.

    .. warning::
        It does ot require the original URDF file to exist, but the original
        mesh paths (if any) must be valid since they are not bundle in the log
        archive for now.

    :param log_file: Logged data (constants and variables) as a dictionary.
    :param mesh_package_dirs: Prepend custom mesh package search path
                              directories to the ones provided by log file. It
                              may be necessary to specify it to read log
                              generated on a different environment.

    :returns: Reconstructed robot, and parsed log data as returned by
              `jiminy_py.log.read_log` method.
    """
    # Make sure provided 'mesh_package_dirs' is a list
    mesh_package_dirs = list(mesh_package_dirs)

    # Instantiate empty robot
    robot = jiminy.Robot()

    # Extract log constants
    log_constants = log_data["constants"]

    # Extract common info
    pinocchio_model = log_constants[
        ".".join((ENGINE_NAMESPACE, "pinocchio_model"))]

    try:
        # Extract geometry models
        collision_model = log_constants[
            ".".join((ENGINE_NAMESPACE, "collision_model"))]
        visual_model = log_constants[
            ".".join((ENGINE_NAMESPACE, "visual_model"))]

        # Initialize the model
        robot.initialize(pinocchio_model, collision_model, visual_model)
    except KeyError:
        # Extract initialization arguments
        urdf_data = log_constants[
            ".".join((ENGINE_NAMESPACE, "urdf_file"))]
        has_freeflyer = int(log_constants[
            ".".join((ENGINE_NAMESPACE, "has_freeflyer"))])
        mesh_package_dirs += log_constants.get(
            ".".join((ENGINE_NAMESPACE, "mesh_package_dirs")), [])

        # Make sure urdf data is available
        if len(urdf_data) <= 1:
            raise RuntimeError(
                "Impossible to build robot. The log is not persistent and the "
                "robot was not associated with a valid URDF file.")

        # Write urdf data in temporary file
        urdf_path = os.path.join(
            tempfile.gettempdir(),
            f"{next(tempfile._get_candidate_names())}.urdf")
        with open(urdf_path, "xb") as f:
            f.write(urdf_data.encode())

        # Initialize model
        robot.initialize(urdf_path, has_freeflyer, mesh_package_dirs)

        # Delete temporary file
        os.remove(urdf_path)

        # Load the options
        all_options = log_constants[
            ".".join((ENGINE_NAMESPACE, "options"))]
        robot.set_options(all_options["system"]["robot"])

        # Update model and data.
        # Dirty hack based on serialization/deserialization to update in-place.
        # Note that string archive cannot be used because it is not reliable
        # and fails on windows for some reason.
        buff = pin.serialization.StreamBuffer()
        pin.serialization.saveToBinary(pinocchio_model, buff)
        pin.serialization.loadFromBinary(robot.pinocchio_model, buff)
        pin.serialization.saveToBinary(pinocchio_model.createData(), buff)
        pin.serialization.loadFromBinary(robot.pinocchio_data, buff)

    return robot


def extract_trajectory_from_log(log_data: Dict[str, Any],
                                robot: Optional[jiminy.Model] = None
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

    :returns: Trajectory dictionary. The actual trajectory corresponds to the
              field "evolution_robot" and it is a list of State object. The
              other fields are additional information.
    """
    # Handling of default argument(s)
    if robot is None:
        robot = build_robot_from_log(log_data)

    # Define some proxies for convenience
    log_vars = log_data["variables"]

    # Extract the joint positions, velocities and external forces over time
    positions = np.stack([
        log_vars.get(".".join((ENGINE_NAMESPACE, field)), [])
        for field in robot.log_fieldnames_position], axis=-1)
    velocities = np.stack([
        log_vars.get(".".join((ENGINE_NAMESPACE, field)), [])
        for field in robot.log_fieldnames_velocity], axis=-1)
    forces = np.stack([
        log_vars.get(".".join((ENGINE_NAMESPACE, field)), [])
        for field in robot.log_fieldnames_f_external], axis=-1)

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
        evolution_robot.append(State(t=t, q=q, v=v, f_ext=f_ext))

    return {"evolution_robot": evolution_robot,
            "robot": robot,
            "use_theoretical_model": False}


def update_sensors_data_from_log(log_data: Dict[str, Any],
                                 robot: jiminy.Model
                                 ) -> Callable[[float], None]:
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
    for sensor_type, sensors_names in robot.sensors_names.items():
        sensor_fieldnames = getattr(jiminy, sensor_type).fieldnames
        for name in sensors_names:
            sensor = robot.get_sensor(sensor_type, name)
            log_fieldnames = [
                '.'.join((sensor.name, field)) for field in sensor_fieldnames]
            if log_fieldnames[0] in log_vars.keys():
                sensor_log = np.stack([
                    log_vars[field] for field in log_fieldnames], axis=-1)
                sensors_set.append(sensor)
                sensors_log.append(sensor_log)

    def update_hook(t: float, q: np.ndarray, v: np.ndarray) -> None:
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
