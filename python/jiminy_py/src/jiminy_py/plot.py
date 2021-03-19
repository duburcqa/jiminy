from functools import partial
from weakref import WeakKeyDictionary
from typing import Dict, Optional, Any, Tuple, List, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.widgets import Button
from matplotlib.backend_bases import Event, LocationEvent
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
    def __init__(self, sync_tabs: bool = False, **kwargs: Any) -> None:
        """ TODO: Write documentation.
        """
        # Backup user arguments
        self.sync_tabs = sync_tabs

        # Internal state buffers
        self.figure = plt.figure()
        self.legend: Optional[Legend] = None
        self.ref_ax: Optional[Axes] = None
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
        for tab in self.tabs_data.values():
            if self.sync_tabs or tab is self.tab_active:
                tab["nav_stack"] = cur_stack._elements.copy()
                tab["nav_pos"] = cur_stack._pos

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

        # Restore navigation history and toolbar state if necessary
        cur_stack._elements = self.tab_active["nav_stack"]
        cur_stack._pos = self.tab_active["nav_pos"]
        self.figure.canvas.toolbar._actions['forward'].setEnabled(
            cur_stack._pos < len(cur_stack) - 1)
        self.figure.canvas.toolbar._actions['back'].setEnabled(
            cur_stack._pos > 0)

        # Adjust layout
        self.__adjust_layout()

        # Refresh figure
        self.refresh()

    def refresh(self) -> None:
        # Refresh button size, in case the number of tabs has changed
        buttons_width = 1.0 / (len(self.tabs_data) + 1)
        for i, tab in enumerate(self.tabs_data.values()):
            tab["button_axcut"].set_position(
                [buttons_width * (i + 0.5), 0.01, buttons_width, 0.05])

        # Refresh figure on the spot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def add_tab(self,
                tab_name: str,
                time: np.ndarray,
                data: Union[np.ndarray, Dict[str, Union[
                    Dict[str, np.ndarray], np.ndarray]]],
                plot_method: Optional[Union[Callable[[
                    Axes, np.ndarray, np.ndarray], Any], str]] = None,
                **kwargs: Any) -> None:
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
            ref_ax = self.ref_ax if self.sync_tabs else None
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
            if self.sync_tabs:
                self.ref_ax = ref_ax
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

        # Add buttons to show/hide information
        uniq_label = '_'.join((tab_name, "button"))
        button_axcut = plt.axes([0.0, 0.0, 0.0, 0.0], label=uniq_label)
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
        self.refresh()
        self.figure.show()

    def set_active_tab(self, tab_name: str) -> None:
        event = LocationEvent("click", self.figure.canvas, 0, 0)
        event.inaxes = self.tabs_data[tab_name]["button"].ax
        self.__click(event)

    def remove_tab(self, tab_name: str) -> None:
        """ TODO: Write documentation.
        """
        # Reset current tab if it is the removed one
        tab = self.tabs_data.pop(tab_name)
        if tab is self.tab_active and self.tabs_data:
            self.set_active_tab(next(iter(self.tabs_data.keys())))

        # Change reference axis if to be deleted
        if any(ax is self.ref_ax for ax in tab["axes"]):
            if self.tabs_data:
                self.ref_ax = self.tab_active["axes"][0]
            else:
                self.ref_ax = None

        # Disable button
        tab["button"].disconnect_events()
        tab["button_axcut"].remove()

        # Remove axes and legend manually is not more tabs available
        if not self.tabs_data:
            if self.figure._suptitle is not None:
                self.figure._suptitle.remove()
                self._suptitle = None
            for ax in tab["axes"]:
                ax.remove()
            if self.legend is not None:
                self.legend.remove()
                self.legend = None

        # Refresh figure
        self.refresh()

    def clear(self) -> None:
        """ TODO: Write documentation.
        """
        # Remove every figure axes
        for tab_name in list(self.tabs_data.keys()):  # list to make copy
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
        tabbed_figure = cls(**kwargs)
        for name, data in tabs_data.items():
            tabbed_figure.add_tab(name, time, data, **kwargs)
        return tabbed_figure
