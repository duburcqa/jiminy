import os
import pathlib
import fnmatch
import tempfile
import argparse
from csv import DictReader
from textwrap import dedent
from itertools import cycle
from bisect import bisect_right
from collections import OrderedDict
from typing import Callable, Tuple, Dict, Optional, Any, Sequence, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing_extensions import TypedDict

from . import core as jiminy
from .state import State


class TrajectoryDataType(TypedDict, total=False):
    # List of State objects of increasing time.
    evolution_robot: Sequence[State]
    # Jiminy robot. None if omitted.
    robot: Optional[jiminy.Robot]
    # Whether to use theoretical or actual model
    use_theoretical_model: bool


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
            time = f['Global.Time'][()] / f['Global.Time'].attrs["unit"]

            # Load variables (1D time-series)
            data_dict = {'Global.Time': time}
            for key, value in f['variables'].items():
                data_dict[key] = value['value'][()]

    return data_dict, const_dict


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


def build_robot_from_log_constants(
        log_constants: Dict[str, str],
        mesh_package_dirs: Union[str, Sequence[str]] = ()) -> jiminy.Robot:
    """Build robot from log constants.

    .. note::
        model options and `robot.pinocchio_model` are guarentee to be the same
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
    :param mesh_package_dirs: Prepend custom mesh package seach path
                              directories to the ones provided by log file. It
                              may be necessary to specify it to read log
                              generated on a different environment.

    :returns: Reconstructed robot, and parsed log data as returned by
              `jiminy_py.log.read_log` method.
    """
    # Make sure provided 'mesh_package_dirs' is a list
    mesh_package_dirs = list(mesh_package_dirs)

    # Extract robot info
    pinocchio_model_str = log_constants[
        "HighLevelController.pinocchio_model"]
    urdf_file = log_constants["HighLevelController.urdf_file"]
    has_freeflyer = int(log_constants["HighLevelController.has_freeflyer"])
    if "HighLevelController.mesh_package_dirs" in log_constants.keys():
        mesh_package_dirs += log_constants[
            "HighLevelController.mesh_package_dirs"].split(";")
    all_options = jiminy.load_config_json_string(
        log_constants["HighLevelController.options"])

    # Create temporary URDF file
    fd, urdf_path = tempfile.mkstemp(suffix=".urdf")
    os.write(fd, urdf_file.encode())
    os.close(fd)

    # Build robot
    robot = jiminy.Robot()
    robot.initialize(urdf_path, has_freeflyer, mesh_package_dirs)
    robot.set_options(all_options["system"]["robot"])
    robot.pinocchio_model.loadFromString(pinocchio_model_str)

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
            if any(field.startswith(sensor.name) for field in log_data.keys()):
                sensor_log = np.stack([
                    log_data['.'.join((sensor.name, field))]
                    for field in sensor_fieldnames], axis=-1)
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


def plot_log():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=dedent("""
            Plot data from a jiminy log file using matplotlib.
            Specify a list of fields to plot, separated by a colon for \
            plotting on the same subplot.

            Example: h1 h2:h3:h4 generates two subplots, one with h1, one \
                     with h2, h3, and h4.

            Wildcard token '*' can be used. In such a case:
            - If *h2* matches several fields:
            each field will be plotted individually in subplots.
            - If :*h2* or :*h2*:*h3*:*h4* matches several fields:
            each field will be plotted in the same subplot.
            - If *h2*:*h3*:*h4* matches several fields:
            each match of h2, h3, and h4 will be plotted jointly in subplots.

            Note that if the number of matches for h2, h3, h4 differs, only \
            the minimum number will be plotted.

            Enter no plot command (only the file name) to view the list of \
            fields available inside the file."
        """))
    parser.add_argument("input", help="Input logfile.")
    parser.add_argument(
        "-c", "--compare", type=str, default=None, help=dedent("""
            Colon-separated list of comparison log files.

            The same data as the original log will be plotted in the same \
            subplot, with different line styes. These logfiles must be of the \
            same length and contain the same header as the original log file.

            Note that you can click on the figure top legend to show / hide \
            data from specific files.
        """))
    main_arguments, plotting_commands = parser.parse_known_args()

    # Load log file
    log_data, _ = read_log(main_arguments.input)

    # If no plotting commands, display the list of headers instead
    if len(plotting_commands) == 0:
        print("Available data:", *map(
            lambda s: f"- {s}", log_data.keys()), sep="\n")
        exit(0)

    # Load comparision logs, if any.
    compare_data = OrderedDict()
    if main_arguments.compare is not None:
        for fullpath in main_arguments.compare.split(':'):
            compare_data[fullpath], _ = read_log(fullpath)

    # Define linestyle cycle that will be used for comparison logs
    linestyles = ["--", "-.", ":"]

    # Parse plotting arguments.
    plotted_elements = []
    for cmd in plotting_commands:
        # Check that the command is valid, i.e. that all elements exits.
        # If it is the case, add it to the list.
        same_subplot = (cmd[0] == ':')
        headers = cmd.strip(':').split(':')

        # Expand each element according to wildcard expression
        matching_headers = []
        for h in headers:
            match = sorted(fnmatch.filter(log_data.keys(), h))
            if len(match) > 0:
                matching_headers.append(match)
            else:
                print(f"No matching headers for expression {h}")
        if len(matching_headers) == 0:
            continue

        # Compute number of subplots
        if same_subplot:
            plotted_elements.append([
                e for l_sub in matching_headers for e in l_sub])
        else:
            n_subplots = min([len(header) for header in matching_headers])
            for i in range(n_subplots):
                plotted_elements.append(
                    [header[i] for header in matching_headers])

    # Create figure.
    n_plot = len(plotted_elements)

    if n_plot == 0:
        print("Nothing to plot. Exiting...")
        return

    fig = plt.figure()

    # Create subplots, arranging them in a rectangular fashion.
    # Do not allow for n_cols to be more than n_rows + 2.
    n_cols = n_plot
    n_rows = 1
    while n_cols > n_rows + 2:
        n_rows = n_rows + 1
        n_cols = np.ceil(n_plot / (1.0 * n_rows))

    axes = []
    for i in range(n_plot):
        ax = fig.add_subplot(int(n_rows), int(n_cols), i+1)
        if i > 0:
            ax.get_shared_x_axes().join(axes[0], ax)
        axes.append(ax)

    # Store lines in dictionnary {file_name: plotted lines}, to enable to
    # toggle individually the visibility the data related to each of them.
    main_name = os.path.basename(main_arguments.input)
    plotted_lines = {main_name: []}
    for c in compare_data:
        plotted_lines[os.path.basename(c)] = []

    plt.gcf().canvas.set_window_title(main_arguments.input)
    t = log_data['Global.Time']

    # Plot each element.
    for ax, plotted_elem in zip(axes, plotted_elements):
        for name in plotted_elem:
            line = ax.step(t, log_data[name], label=name)
            plotted_lines[main_name].append(line[0])

            linecycler = cycle(linestyles)
            for c in compare_data:
                line = ax.step(compare_data[c]['Global.Time'],
                               compare_data[c][name],
                               next(linecycler),
                               color=line[0].get_color())
                plotted_lines[os.path.basename(c)].append(line[0])

    # Add legend and grid for each plot.
    for ax in axes:
        ax.set_xlabel('time (s)')
        ax.legend()
        ax.grid()

    # If a compare plot is present, add overall legend specifying line types
    plt.subplots_adjust(
        bottom=0.05,
        top=0.98,
        left=0.06,
        right=0.98,
        wspace=0.1,
        hspace=0.12)
    if len(compare_data) > 0:
        linecycler = cycle(linestyles)

        # Dictionnary: line in legend to log name
        legend_lines = {Line2D([0], [0], color='k'): main_name}
        for data_str in compare_data:
            legend_line_object = Line2D(
                [0], [0],  color='k', linestyle=next(linecycler))
            legend_lines[legend_line_object] = os.path.basename(data_str)
        legend = fig.legend(
            legend_lines.keys(), legend_lines.values(), loc='upper center',
            ncol=3)

        # Create a dict {picker: log name} for both the lines and the legend
        picker_to_name = {}
        for legline, name in zip(legend.get_lines(), legend_lines.values()):
            legline.set_picker(10)  # 10 pts tolerance
            picker_to_name[legline] = name
        for legline, name in zip(legend.get_texts(), legend_lines.values()):
            legline.set_picker(10)  # 10 pts tolerance
            picker_to_name[legline] = name

        # Increase top margin to fit legend
        fig.canvas.draw()
        legend_height = legend.get_window_extent().inverse_transformed(
            fig.transFigure).height
        plt.subplots_adjust(top=0.98-legend_height)

        # Make legend interactive
        def legend_clicked(event):
            file_name = picker_to_name[event.artist]
            for line in plotted_lines[file_name]:
                line.set_visible(not line.get_visible())
            fig.canvas.draw()
        fig.canvas.mpl_connect('pick_event', legend_clicked)

    plt.show()
