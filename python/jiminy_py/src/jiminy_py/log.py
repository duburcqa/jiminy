#!/usr/bin/env python

## @file jiminy_py/log.py

import argparse
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader

from .state import State
from .core import Engine, Robot


def extract_viewer_data_from_log(log_data, robot):
    """
    @brief      Extract the minimal required information from raw log data in
                order to replay the simulation in a viewer.

    @details    It extracts the time and joint positions evolution.
    .
    @remark     Note that the quaternion angular velocity vectors are expressed
                it body frame rather than world frame.

    @param[in]  log_data    Data from the log file, in a dictionnary.
    @param[in]  robot       Jiminy robot.

    @return     Trajectory dictionary. The actual trajectory corresponds to
                the field "evolution_robot" and it is a list of State object.
                The other fields are additional information.
    """

    # Get the current robot model options
    model_options = robot.get_model_options()

    # Extract the joint positions time evolution
    t = log_data["Global.Time"]
    try:
        qe = np.stack([log_data["HighLevelController." + s]
                       for s in robot.logfile_position_headers], axis=-1)
    except:
        model_options['dynamics']['enableFlexibleModel'] = not robot.is_flexible
        robot.set_model_options(model_options)
        qe = np.stack([log_data["HighLevelController." + s]
                       for s in robot.logfile_position_headers], axis=-1)

    # Determine whether the theoretical model of the flexible one must be used
    use_theoretical_model = not robot.is_flexible

    # Make sure that the flexibilities are enabled
    model_options['dynamics']['enableFlexibleModel'] = True
    robot.set_model_options(model_options)

    # Create state sequence
    evolution_robot = []
    for i in range(len(t)):
        evolution_robot.append(State(qe[i].T, None, None, t[i]))

    return {'evolution_robot': evolution_robot,
            'robot': robot,
            'use_theoretical_model': use_theoretical_model}

def is_log_binary(filename):
    """
    @brief   Return True if the given filename appears to be binary log file.

    @details File is considered to be binary log if it contains a NULL byte.
             From https://stackoverflow.com/a/11301631/4820605.
    """
    with open(filename, 'rb') as f:
        for block in f:
            if b'\0' in block:
                return True
    return False

def read_log(filename):
    """
    Read a logfile from jiminy. This function supports both text (csv)
    and binary log.

    Parameters:
        - filename: Name of the file to load.
    Retunrs:
        - A dictionnary containing the logged values, and a dictionnary
        containing the constants.
    """

    if is_log_binary(filename):
        # Read binary file using C++ parser.
        data_dict, constants_dict = Engine.read_log_binary(filename)
    else:
        # Read text csv file.
        constants_dict = {}
        with open(filename, 'r') as log:
            consts = next(log).split(', ')
            for c in consts:
                c_split = c.split('=')
                # Remove line end for last constant.
                constants_dict[c_split[0]] = c_split[1].strip('\n')
            # Read data from the log file, skipping the first line (the constants).
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
        data_dict = {k.strip() : np.array(v) for k,v in data.items()}
    return data_dict, constants_dict

def plot_log():
    description_str = \
        "Plot data from a jiminy log file using matplotlib.\n" + \
        "Specify a list of fields to plot, separated by a colon for plotting on the same subplot.\n\n" + \
        "Example: h1 h2:h3:h4 generates two subplots, one with h1, one with h2, h3, and h4.\n" + \
        "Wildcard token '*' can be used. In such a case:\n" + \
        "- If *h2* matches several fields : each field will be plotted individually in subplots. \n" + \
        "- If :*h2* or :*h2*:*h3*:*h4* matches several fields : each field will be plotted in the same subplot. \n" + \
        "- If *h2*:*h3*:*h4* matches several fields : each match of h2, h3, and h4 will be plotted jointly in subplots.\n" + \
        "  Note that if the number of matches for h2, h3, h4 differs, only the minimum number will be plotted.\n" + \
        "\nEnter no plot command (only the file name) to view the list of fields available inside the file."

    parser = argparse.ArgumentParser(description=description_str,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="Input logfile.")
    main_arguments, plotting_commands = parser.parse_known_args()

    # Load log file.
    log_data, _ = read_log(main_arguments.input)

    # If no plotting commands, display the list of headers instead.
    if len(plotting_commands) == 0:
        print("Available data:")
        print("\n - ".join(log_data.keys()))
        exit(0)

    # Parse plotting arguments.
    plotted_elements = []
    for cmd in plotting_commands:
        # Check that the command is valid, i.e. that all elements exits. If it is the case, add it to the list.
        same_subplot = (cmd[0] == ':')
        headers = cmd.strip(':').split(':')

        # Expand each element if wildcard tokens are present.
        matching_headers = []
        for h in headers:
            matching_headers.append(sorted(fnmatch.filter(log_data.keys(), h)))

        # Get minimum size for number of subplots.
        if same_subplot:
            plotted_elements.append([e for l_sub in matching_headers for e in l_sub])
        else:
            n_subplots = min([len(l) for l in matching_headers])
            for i in range(n_subplots):
                plotted_elements.append([l[i] for l in matching_headers])

    # Create figure.
    n_plot = len(plotted_elements)

    # Arrange plot in rectangular fashion: don't allow for n_cols to be more than n_rows + 2
    n_cols = n_plot
    n_rows = 1
    while n_cols > n_rows + 2:
        n_rows = n_rows + 1
        n_cols = int(np.ceil(n_plot / float(n_rows)))

    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex = True)
    if n_plot == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    plt.gcf().canvas.set_window_title(main_arguments.input)
    t = log_data['Global.Time']

    # Plot each element.
    for i in range(n_plot):
        for name in plotted_elements[i]:
            axs[i].plot(t, log_data[name], label = name)

    # Add legend and grid.
    for ax in axs:
        ax.set_xlabel('time (s)')
        ax.legend()
        ax.grid()
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.06, right=0.98, wspace=0.1, hspace=0.05)
    plt.show()
