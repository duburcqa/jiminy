#!/usr/bin/env python

## @file jiminy_py/log.py

import argparse
from csv import DictReader
from collections import OrderedDict
import fnmatch
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

from .core import Engine


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
            constants_str = next(log).split(', ')
            for c in constants_str:
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

    parser = argparse.ArgumentParser(description = description_str, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="Input logfile.")
    parser.add_argument("-c", "--compare", type=str, default=None, help="Colon-separated list of comparison log files:" +\
        " the same data as the original log will be plotted in the same subplot, with different line styes."+ \
        " These logfiles must be of the same length and contain the same header as the original log file.\n" +\
        " Note: you can click on the figure top legend to show / hide data from specific files.")
    main_arguments, plotting_commands = parser.parse_known_args()

    # Load log file.
    log_data, _ = read_log(main_arguments.input)

    # If no plotting commands, display the list of headers instead.
    if len(plotting_commands) == 0:
        print("Available data:")
        print("\n - ".join(log_data.keys()))
        exit(0)

    # Load comparision logs, if any.
    compare_data = OrderedDict()
    if main_arguments.compare is not None:
        for filename in main_arguments.compare.split(':'):
            compare_data[filename], _ = read_log(filename)
    # Define linestyle cycle that will be used for comparison logs.
    linestyles = ["--","-.",":"]

    # Parse plotting arguments.
    plotted_elements = []
    for cmd in plotting_commands:
        # Check that the command is valid, i.e. that all elements exits. If it is the case, add it to the list.
        same_subplot = (cmd[0] == ':')
        headers = cmd.strip(':').split(':')

        # Expand each element according to wildcard expression.
        matching_headers = []
        for h in headers:
            match = sorted(fnmatch.filter(log_data.keys(), h))
            if len(match) > 0:
                matching_headers.append(match)
            else:
                print(f"No matching headers for expression {h}")
        if len(matching_headers) == 0:
            continue

        # Compute number of subplots.
        if same_subplot:
            plotted_elements.append([e for l_sub in matching_headers for e in l_sub])
        else:
            n_subplots = min([len(l) for l in matching_headers])
            for i in range(n_subplots):
                plotted_elements.append([l[i] for l in matching_headers])

    # Create figure.
    n_plot = len(plotted_elements)

    if n_plot == 0:
        print(f"Nothing to plot. Exiting...")
        return

    fig = plt.figure()

    # Create subplots, arranging them in a rectangular fashion.
    # Do not allow for n_cols to be more than n_rows + 2
    n_cols = n_plot
    n_rows = 1
    while n_cols > n_rows + 2:
        n_rows = n_rows + 1
        n_cols = np.ceil(n_plot / (1.0 * n_rows))

    axs = []
    for i in range(n_plot):
        ax = fig.add_subplot(int(n_rows), int(n_cols), i+1)
        if i > 0:
            ax.get_shared_x_axes().join(axs[0], ax)
        axs.append(ax)

    # Store lines in dictionnary fine_name -> plotted lines, to toggle visibility.
    main_name = os.path.basename(main_arguments.input)
    plotted_lines = {main_name : []}
    for c in compare_data:
        plotted_lines[os.path.basename(c)] = []

    plt.gcf().canvas.set_window_title(main_arguments.input)
    t = log_data['Global.Time']
    # Plot each element.
    for ax, plotted_elem in zip(axs, plotted_elements):
        for name in plotted_elem:
            line = ax.plot(t, log_data[name], label = name)
            plotted_lines[main_name].append(line[0])

            linecycler = cycle(linestyles)
            for c in compare_data:
                l = ax.plot(compare_data[c]['Global.Time'], compare_data[c][name], next(linecycler), color=line[0].get_color())
                plotted_lines[os.path.basename(c)].append(l[0])

    # Add legend and grid for each plot.
    for ax in axs:
        ax.set_xlabel('time (s)')
        ax.legend()
        ax.grid()

    # If a compare plot is present, add overall legend specifying line types.
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.06, right=0.98, wspace=0.1, hspace=0.12)
    if len(compare_data) > 0:
        linecycler = cycle(linestyles)

        # Dictionnary: line in legend to log name.
        legend_lines = {Line2D([0], [0], color='k') : main_name}
        for c in compare_data:
            legend_lines[Line2D([0], [0],  color='k', linestyle=next(linecycler))] = os.path.basename(c)

        legend = fig.legend(legend_lines.keys(), legend_lines.values(), loc = 'upper center', ncol = 3)
        # Create a map (picker) -> (log name) for both the lines and the legend text.
        picker_to_name = {}
        for legline, name in zip(legend.get_lines(), legend_lines.values()):
            legline.set_picker(10)  # 10 pts tolerance
            picker_to_name[legline] = name
        for legline, name in zip(legend.get_texts(), legend_lines.values()):
            legline.set_picker(10)  # 10 pts tolerance
            picker_to_name[legline] = name

        # Increase top margin to fit legend
        fig.canvas.draw()
        legend_height = legend.get_window_extent().inverse_transformed(fig.transFigure).height
        plt.subplots_adjust(top = 0.98 - legend_height)

        # Make legend interactive
        def legend_clicked(event):
            file_name = picker_to_name[event.artist]
            for line in plotted_lines[file_name]:
                line.set_visible(not line.get_visible())
            fig.canvas.draw()
        fig.canvas.mpl_connect('pick_event', legend_clicked)

    plt.show()
