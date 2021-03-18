from functools import partial
from weakref import WeakKeyDictionary
from typing import Dict, Optional, Any, Tuple, List, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.widgets import Button
from matplotlib.backend_bases import Event
from typing_extensions import TypedDict


class TabData(TypedDict, total=True):
    """ TODO: Write documentation.
    """
    axes: List[Axes]
    legend_data: Tuple[List[Artist], List[str]]
    nav_stack: List[WeakKeyDictionary]
    nav_pos: int
    button: Button
    button_axcut: Axes


class TabbedFigure:
    """ TODO: Write documentation.
    """
    def __init__(self) -> None:
        """ TODO: Write documentation.
        """
        # Internal state buffers
        self.figure = plt.figure()
        self.legend: Optional[Legend] = None
        self.tabs_data = {}
        self.tab_active: Optional[TabData] = None

        # Register 'on resize' event callback to adjust layout
        self.figure.canvas.mpl_connect('resize_event', self.__adjust_layout)

    def __adjust_layout(self, event: Optional[Any] = None) -> None:
        """ TODO: Write documentation.
        """
        # TODO: It would be better to adjust whose parameters based on
        # `event` if provided.
        right_margin = 0.05
        if self.legend is not None:
            legend_extent = self.legend.get_window_extent()
            legend_width_rel = legend_extent.transformed(
                self.figure.transFigure.inverted()).width
            right_margin += legend_width_rel
        self.figure.subplots_adjust(
            bottom=0.1, top=0.9, left=0.05, right=1.0-right_margin,
            wspace=0.2, hspace=0.45)

    def __click(self, event: Event) -> None:
        """ TODO: Write documentation.
        """
        # Update buttons style
        for tab in self.tabs_data.values():
            button = tab["button"]
            if button.ax == event.inaxes:
                button.ax.set_facecolor('green')
                button.color = 'green'
                button_name = button.label.get_text().replace('\n', ' ')
            else:
                button.ax.set_facecolor('white')
                button.color = 'white'

        # Backup navigation history
        cur_stack = self.figure.canvas.toolbar._nav_stack
        self.tab_active["nav_stack"] = cur_stack._elements.copy()
        self.tab_active["nav_pos"] = cur_stack._pos

        # Update axes and title
        for ax in self.tab_active["axes"]:
            self.figure.delaxes(ax)
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        self.tab_active = self.tabs_data[button_name]
        self.figure.suptitle(button_name)
        for ax in self.tab_active["axes"]:
            self.figure.add_subplot(ax)
        handles, labels = self.tab_active["legend_data"]
        if labels:
            self.legend = self.figure.legend(
                handles, labels, loc='upper right',
                bbox_to_anchor=(0.99, 0.95))

        # Restore selected tab navigation history and toolbar state
        cur_stack._elements = self.tab_active["nav_stack"]
        cur_stack._pos = self.tab_active["nav_pos"]
        self.figure.canvas.toolbar._actions['forward'].setEnabled(
            cur_stack._pos < len(cur_stack) - 1)
        self.figure.canvas.toolbar._actions['back'].setEnabled(
            cur_stack._pos > 0)

        # Adjust layout
        self.__adjust_layout()

        # Refresh figure
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def add_tab(self,
                tab_name: str,
                time: np.ndarray,
                data: Union[np.ndarray, Dict[str, Union[
                    Dict[str, np.ndarray], np.ndarray]]],
                plot_method: Optional[Union[Callable[[
                    Axes, np.ndarray, np.ndarray], Any], str]] = None
                ) -> None:
        """ TODO: Write documentation.

        :param plot_method: Callable method taking axis object, time, and data
                            array in argument, or string instance method of
                            `matplotlib.axes.Axes`.
                            Optional: `step(..., where='post')` by default.
        """
        # Handle default arguments and converters
        if plot_method is None:
            plot_method = partial(Axes.step, where='post')
        elif isinstance(plot_method, str):
            plot_method = getattr(Axes, plot_method)

        # Initialize legend data
        legend_data = ([], [])

        if isinstance(data, dict):
            # Compute plot grid arrangement
            n_cols = len(data)
            n_rows = 1
            while n_cols > n_rows + 2:
                n_rows = n_rows + 1
                n_cols = int(np.ceil(len(data) / (1.0 * n_rows)))

            # Initialize axes, and early return if none
            axes = []
            ref_ax = None
            for i, plot_name in enumerate(data.keys()):
                uniq_label = '_'.join((tab_name, plot_name))
                ax = self.figure.add_subplot(
                    n_rows, n_cols, i+1, label=uniq_label)
                if self.tabs_data:
                    self.figure.delaxes(ax)
                if ref_ax is not None:
                    ax.get_shared_x_axes().join(ref_ax, ax)
                else:
                    ref_ax = ax
                axes.append(ax)
            if axes is None:
                return

            # Update their content
            for (plot_name, plot_data), ax in zip(data.items(), axes):
                if isinstance(plot_data, dict):
                    for line_name, line_data in plot_data.items():
                        plot_method(ax, time, line_data, label=line_name)
                else:
                    plot_method(ax, time, plot_data)
                ax.set_title(plot_name, fontsize='medium')
                ax.grid()
        else:
            # Draw single figure instead of subplot
            ax = self.figure.add_subplot(1, 1, 1, label=tab_name)
            plot_method(ax, time, data)
            if self.tabs_data:
                self.figure.delaxes(ax)
            axes = [ax]

        # Get unique legend for every subplots
        legend_data = ax.get_legend_handles_labels()

        # Resize existing buttons before adding new one
        buttons_width = 1.0 / (len(self.tabs_data) + 2)
        for i, tab in enumerate(self.tabs_data.values()):
            tab["button_axcut"].set_position(
                [buttons_width * (i + 0.5), 0.01, buttons_width, 0.05])

        # Add buttons to show/hide information
        button_idx = len(self.tabs_data)
        button_axcut = plt.axes(
            [buttons_width * (button_idx + 0.5), 0.01, buttons_width, 0.05])
        button = Button(button_axcut,
                        tab_name.replace(' ', '\n'),
                        color='white')

        # Register buttons events
        button.on_clicked(self.__click)

        # Create new tab data container
        self.tabs_data[tab_name] = {
            "axes": axes,
            "legend_data": legend_data,
            "nav_stack": [],
            "nav_pos": -1,
            "button": button,
            "button_axcut": button_axcut
        }

        # Check if it is the first tab to be added
        if self.tab_active is None:
            # Set new tab has the active one if none before
            self.tab_active = self.tabs_data[tab_name]

            # Show tab without blocking
            for ax in axes:
                ax.set_visible(True)
            self.figure.suptitle(tab_name)
            handles, labels = legend_data
            if labels:
                self.legend = self.figure.legend(handles, labels)
            button.ax.set_facecolor('green')
            button.color = 'green'

        # Update figure and show it without blocking
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.show()

    def remove_tab(self, tab_name: str) -> None:
        """ TODO: Write documentation.
        """
        # Remove desired tab
        tab = self.tabs_data.pop(tab_name)
        for ax in tab["axes"]:
            ax.remove()
        tab["buttont"].disconnect_events()
        tab["button_axcut"].remove()

    def clear(self) -> None:
        """ TODO: Write documentation.
        """
        # Remove every figure axes
        for tab_name in self.tabs_data.keys():
            self.remove_tab(tab_name)

    @classmethod
    def plot(cls,
             time: np.ndarray,
             tabs_data: Dict[str, Union[Dict[str, Union[
                    Dict[str, np.ndarray], np.ndarray]]]],
             **kwargs) -> "TabbedFigure":
        """ TODO: Write documentation.

        :param kwargs: Extra keyword arguments to forward to `add_tab` method.
        """
        tabbed_figure = cls()
        for name, data in tabs_data.items():
            tabbed_figure.add_tab(name, time, data, **kwargs)
        return tabbed_figure
