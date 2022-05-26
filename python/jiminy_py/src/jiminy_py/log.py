import os
import pathlib
import tempfile
from csv import DictReader
from bisect import bisect_right
from collections import OrderedDict
from typing import Callable, Tuple, Dict, Optional, Any, Sequence, Union

import h5py
import tree
import numpy as np

from . import core as jiminy
from .dynamics import State, TrajectoryDataType


FieldNested = Union[
    Dict[str, 'FieldNested'], Sequence['FieldNested'],
    str]  # type: ignore[misc]


def _is_log_binary(fullpath: str) -> bool:
    """Return True if the given fullpath appears to be binary log file.

    File is considered to be binary log if it contains a NULL byte.

    See https://stackoverflow.com/a/11301631/4820605 for reference.
    """
    try:
        with open(fullpath, 'rb') as f:
            for block in f:
                if b'\0' in block:
                    return True
    except IOError:
        pass
    return False


def read_log(fullpath: str,
             file_format: Optional[str] = None
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Read a logfile from jiminy.

    This function supports both text (csv) and binary log.

    :param fullpath: Name of the file to load.
    :param file_format: Name of the file to load.

    :returns: Pair of dictionaries containing respectively the logged values,
              and the constants.
    """
    # Handling of file file_format
    if file_format is None:
        file_ext = pathlib.Path(fullpath).suffix
        if file_ext == '.data':
            file_format = 'binary'
        elif file_ext == '.csv' or file_ext == '.txt':
            file_format = 'csv'
        elif file_ext == '.h5' or file_ext == '.hdf5':
            file_format = 'hdf5'
        if file_format is None and not _is_log_binary(fullpath):
            file_format = 'csv'
    if file_format is None:
        raise ValueError(
            "Impossible to determine the file format automatically. Please "
            "specify it manually.")
    if file_format not in ['binary', 'csv', 'hdf5']:
        raise ValueError(
            "Invalid 'file_format' argument. It must be either 'binary', "
            "'csv' or 'hdf5'.")

    if file_format == 'binary':
        # Read binary file using C++ parser.
        data_dict, const_dict = jiminy.Engine.read_log_binary(fullpath)
    elif file_format == 'csv':
        # Read text csv file.
        const_dict = {}
        with open(fullpath, 'r') as log:
            const_str = next(log).split(', ')
            for c in const_str:
                # Extract the name and value of the constant.
                # Note that newline is stripped at the end of the value.
                name, value = c.split('=')
                value = value.strip('\n')

                # Gather the constants in a dictionary.
                const_dict[name] = value.strip('\n')

            # Read data from the log file, skipping the first line, namely the
            # constants definition.
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
        data_dict = {k.strip(): np.array(v) for k, v in data.items()}
    elif file_format == 'hdf5':
        def parse_constant(key: str, value: str) -> Any:
            """Process some particular constants based on its name or type.
            """
            if isinstance(value, bytes):
                return value.decode()
            return value

        with h5py.File(fullpath, 'r') as f:
            # Load constants
            const_dict = {}
            for key, dataset in f['constants'].items():
                const_dict[key] = parse_constant(key, dataset[()])
            for key, value in dict(f['constants'].attrs).items():
                const_dict[key] = parse_constant(key, value)

            # Extract time
            time = f['Global.Time'][()] * f['Global.Time'].attrs['unit']

            # Load variables (1D time-series)
            data_dict = {'Global.Time': time}
            for key, value in f['variables'].items():
                data_dict[key] = value['value'][()]

    return data_dict, const_dict


def extract_data_from_log(log_data: Dict[str, np.ndarray],
                          fieldnames: FieldNested,
                          namespace: Optional[str] = 'HighLevelController',
                          *,
                          as_dict: bool = False) -> Optional[Union[
                              Tuple[Optional[np.ndarray], ...],
                              Dict[str, Optional[np.ndarray]]]]:
    """Extract values associated with a set of fieldnames in a specific
    namespace.

    :param log_data: Data from the log file, in a dictionnary.
    :param fieldnames: Structured fieldnames.
    :param namespace: Namespace of the fieldnames. None to disable.
                      Optional: 'HighLevelController' by default.
    :param keep_structure: Whether or not to return a dictionary mapping
                           flattened fieldnames to values.
                           Optional: True by default.

    :returns:
        `np.ndarray` or None for each fieldname individually depending if it is
        found or not. It
    """
    # Key are the concatenation of the path and value
    keys = [
        ".".join(map(str, filter(
            lambda key: isinstance(key, str), (*fieldname_path, fieldname))))
        for fieldname_path, fieldname in tree.flatten_with_path(fieldnames)]

    # Extract value from log if it exists
    values = [
        log_data.get(".".join(filter(None, (namespace, key))), None)
        for key in keys]

    # Return None if no value was found
    if not values or all(elem is None for elem in values):
        return None

    if as_dict:
        # Return flat mapping from fieldnameswithout prefix  to scalar values
        values = OrderedDict(zip(keys, values))

    return values


def extract_trajectory_data_from_log(log_data: Dict[str, np.ndarray],
                                     robot: jiminy.Model
                                     ) -> TrajectoryDataType:
    """Extract the minimal required information from raw log data in order to
    replay the simulation in a viewer.

    .. note::
        It extracts the required data for replay, namely temporal evolution of:
          - robot configuration: to display of the robot on the scene,
          - robot velocity: to update velocity-dependent markers such as DCM,
          - external forces: to update force-dependent markers.

    :param log_data: Data from the log file, in a dictionnary.
    :param robot: Jiminy robot.

    :returns: Trajectory dictionary. The actual trajectory corresponds to the
              field "evolution_robot" and it is a list of State object. The
              other fields are additional information.
    """
    times = log_data["Global.Time"]
    try:
        # Extract the joint positions, velocities and external forces over time
        positions = np.stack([
            log_data.get(".".join(("HighLevelController", field)), [])
            for field in robot.logfile_position_headers], axis=-1)
        velocities = np.stack([
            log_data.get(".".join(("HighLevelController", field)), [])
            for field in robot.logfile_velocity_headers], axis=-1)
        forces = np.stack([
            log_data.get(".".join(("HighLevelController", field)), [])
            for field in robot.logfile_f_external_headers], axis=-1)

        # Determine available data
        has_positions = len(positions) > 0
        has_velocities = len(velocities) > 0
        has_forces = len(forces) > 0

        # Determine whether to use the theoretical or flexible model
        use_theoretical_model = not robot.is_flexible

        # Create state sequence
        evolution_robot = []
        q, v, f_ext = None, None, None
        for i, t in enumerate(times):
            if has_positions:
                q = positions[i]
            if has_velocities:
                v = velocities[i]
            if has_forces:
                f_ext = [forces[i, (6 * (j - 1)):(6 * j)]
                         for j in range(1, robot.pinocchio_model.njoints)]
            evolution_robot.append(State(t=t, q=q, v=v, f_ext=f_ext))

        traj_data = {"evolution_robot": evolution_robot,
                     "robot": robot,
                     "use_theoretical_model": use_theoretical_model}
    except KeyError:  # The current options are inconsistent with log data
        # Toggle flexibilities
        model_options = robot.get_model_options()
        dyn_options = model_options["dynamics"]
        dyn_options["enableFlexibleModel"] = not robot.is_flexible
        robot.set_model_options(model_options)

        # Get viewer data
        traj_data = extract_trajectory_data_from_log(log_data, robot)

        # Restore back flexibilities
        dyn_options["enableFlexibleModel"] = not robot.is_flexible
        robot.set_model_options(model_options)

    return traj_data


def build_robot_from_log(
        log_constants: Dict[str, str],
        mesh_package_dirs: Union[str, Sequence[str]] = ()) -> jiminy.Model:
    """Build robot from log constants.

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

    :param log_file: Path of the simulation log file, in any format.
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

    # Extract common info
    pinocchio_model = log_constants["HighLevelController.pinocchio_model"]
    all_options = jiminy.load_config_json_string(
        log_constants["HighLevelController.options"])

    try:
        # Extract geometry models
        collision_model = log_constants["HighLevelController.collision_model"]
        visual_model = log_constants["HighLevelController.visual_model"]

        # Initialize the model
        robot.initialize(pinocchio_model, collision_model, visual_model)
        robot.set_options(all_options["system"]["robot"])
    except KeyError:
        # Extract initialization arguments
        urdf_data = log_constants["HighLevelController.urdf_file"]
        has_freeflyer = int(log_constants["HighLevelController.has_freeflyer"])
        if "HighLevelController.mesh_package_dirs" in log_constants.keys():
            mesh_package_dirs += log_constants[
                "HighLevelController.mesh_package_dirs"].split(";")

        # Create temporary dump file
        fd, tmp_path = tempfile.mkstemp()
        os.write(fd, urdf_data.encode())
        os.close(fd)

        # Initialize model
        robot.initialize(tmp_path, has_freeflyer, mesh_package_dirs)
        robot.set_options(all_options["system"]["robot"])

        # Update model and data.
        # Dirty hack based on serialization/deserialization to update in-place,
        # and string archive cannot be used because it is not reliable and
        # fails on windows for some reason.
        pinocchio_model.saveToBinary(tmp_path)
        robot.pinocchio_model.loadFromBinary(tmp_path)
        pinocchio_model.createData().saveToBinary(tmp_path)
        robot.pinocchio_data.loadFromBinary(tmp_path)

        # Delete temporary file
        os.remove(tmp_path)

    return robot


def emulate_sensors_data_from_log(log_data: Dict[str, np.ndarray],
                                  robot: jiminy.Model
                                  ) -> Callable[[float], None]:
    """Helper to make it easy to emulate sensor data update based on log data.

    .. note::
        It returns an update_hook that can forwarding the `Viewer.replay` to
        display sensor information such as contact forces for instance.

    :param log_data: Data from the log file, in a dictionnary.
    :param robot: Jiminy robot.

    :returns: Callable taking update time in argument and returning nothing.
              Note that it does not through an exception if out-of-range, but
              rather clip to desired time to the available data range.
    """
    # Extract time from log
    times = log_data['Global.Time']

    # Filter sensors whose data is available
    sensors_set, sensors_log = [], []
    for sensor_type, sensors_names in robot.sensors_names.items():
        sensor_fieldnames = getattr(jiminy, sensor_type).fieldnames
        for name in sensors_names:
            sensor = robot.get_sensor(sensor_type, name)
            log_fieldnames = [
                '.'.join((sensor.name, field)) for field in sensor_fieldnames]
            if log_fieldnames[0] in log_data.keys():
                sensor_log = np.stack([
                    log_data[field] for field in log_fieldnames], axis=-1)
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
