# Plot data from a jiminy log file. See inline help for more info.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from itertools import cycle
import fnmatch
import os

from jiminy_py.log import read_log

def main():
    description_str = "Plot data from a jiminy log file using matplotlib.\n" + \
                      "Simply specify a list of fields to plot, separated by a colon for plotting on the same subplot.\n" +\
                      "Example: h1 h2:h3 generates two subplots, one with h1, one with h2 and h3.\n" + \
                      "Regular expressions can be used. Enter no plot command (only the file name) to view the list of fields available inside the file."

    parser = argparse.ArgumentParser(description = description_str, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("input", help = "Input logfile.")
    parser.add_argument("-c", "--compare", type=str, default=None, help = "Colon-separated list of comparison log files:" +\
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
        headers = cmd.split(":")
        # Expand each element according to regular expression.
        matching_headers = []
        for h in headers:
            matching_headers.append(sorted(fnmatch.filter(log_data.keys(), h)))
        # Get minimum size for number of subplots.
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
        n_cols = np.ceil(n_plot / (1.0 * n_rows))

    fig, axs = plt.subplots(nrows=int(n_rows), ncols=int(n_cols), sharex = True)

    if n_plot == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    # Store lines in dictionnary fine_name -> plotted lines, to toggle visibility.
    main_name = os.path.basename(main_arguments.input)
    plotted_lines = {main_name : []}
    for c in compare_data:
        plotted_lines[os.path.basename(c)] = []

    plt.gcf().canvas.set_window_title(main_arguments.input)
    t = log_data['Global.Time']
    # Plot each element.
    for i in range(n_plot):
        for name in plotted_elements[i]:
            line = axs[i].plot(t, log_data[name], label = name)
            plotted_lines[main_name].append(line[0])

            linecycler = cycle(linestyles)
            for c in compare_data:
                l = axs[i].plot(t, compare_data[c][name], next(linecycler), color=line[0].get_color())
                plotted_lines[os.path.basename(c)].append(l[0])

    # Add legend and grid for each plot.
    for ax in axs:
        ax.set_xlabel('time (s)')
        ax.legend()
        ax.grid()

    # If a compare plot is present, add overall legend specifying line types.
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.06, right=0.98, wspace=0.1, hspace=0.05)
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

        def legend_clicked(event):
            file_name = picker_to_name[event.artist]
            for line in plotted_lines[file_name]:
                line.set_visible(not line.get_visible())
            fig.canvas.draw()
        # Make legend interactive
        fig.canvas.mpl_connect('pick_event', legend_clicked)
    plt.show()

if __name__ == "__main__":
    main()

