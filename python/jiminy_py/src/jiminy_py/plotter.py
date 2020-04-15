# Plot data from a jiminy log file. See inline help for more info.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from fnmatch import filter

from jiminy_py.core import Engine

def main():
    description_str = "Plot data from a jiminy log file using matplotlib.\n" + \
                      "Simply specify a list of fields to plot, separated by a colon for plotting on the same subplot.\n" +\
                      "Example: h1 h2:h3 generates two subplots, one with h1, one with h2 and h3.\n" + \
                      "Regular expressions can be used. Enter no plot command (only the file name) to view the list of fields available inside the file."

    parser = argparse.ArgumentParser(description = description_str, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("input", help = "Input logfile.")
    main_arguments, plotting_commands = parser.parse_known_args()

    # Load log file.
    log_data, _ = Engine.read_log(main_arguments.input)

    # If no plotting commands, display the list of headers instead.
    if len(plotting_commands) == 0:
        print("Available data:")
        print("\n - ".join(log_data.keys()))
        exit(0)

    # Parse plotting arguments.
    plotted_elements = []
    for cmd in plotting_commands:
        # Check that the command is valid, i.e. that all elements exits. If it is the case, add it to the list.
        headers = cmd.split(":")
        # Expand each element according to regular expression.
        matching_headers = []
        for h in headers:
            matching_headers.append(sorted(filter(log_data.keys(), h)))
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

    plt.gcf().canvas.set_window_title(main_arguments.input)
    t = log_data['Global.Time']
    # Plot each element.
    for i in range(n_plot):
        for name in plotted_elements[i]:
            axs[i].plot(t, log_data[name], label = name)
    # Add legend to upper left corner.
    for ax in axs:
        ax.legend(bbox_to_anchor=(1.0, 1.0), loc = 1)
        ax.grid()
    plt.subplots_adjust(bottom=0.05, top=0.98, left=0.06, right=0.98, wspace=0.1, hspace=0.05)
    plt.show()

if __name__ == "__main__":
    main()

