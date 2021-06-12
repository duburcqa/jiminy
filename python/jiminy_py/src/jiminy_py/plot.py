import pathlib
from math import ceil, sqrt, floor
from functools import partial
from weakref import WeakKeyDictionary
from typing import Dict, Optional, Any, Tuple, List, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.widgets import Button
from matplotlib.transforms import Bbox
from matplotlib.backend_bases import Event, LocationEvent
from matplotlib.backends.backend_pdf import PdfPages
from typing_extensions import TypedDict


EXPORT_DPI = 300


class ButtonBlit(Button):
    def _motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if event.inaxes == self.ax else self.color
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                self.ax.draw_artist(self.ax)
                self.ax.figure.canvas.blit(self.ax.bbox)


class TabData(TypedDict, total=True):
    """Internal data stored for each tab if `TabbedFigure`.
    """
    # Matpolib `Axes` handles of every subplots
    axes: List[Axes]
    # Matpolib Legend data as returned by `get_legend_handles_labels` method.
    # First element of pair is a list of Artist to legend, which usually are
    # `line2D` object, and the second element is the list of labels for each
    # of them.
    legend_data: Tuple[List[Artist], List[str]]
    # Matplotlib `NavigationToolbar2` navigation stack history. It is not meant
    # to be modified manually, but only copied/restored when needed.
    nav_stack: List[WeakKeyDictionary]
    # Matplotlib `NavigationToolbar2` navigation stack position, used
    # internally for undo/redo history. It is not meant to be modified
    # manually, but only copied/restored when needed.
    nav_pos: int
    # Button associated with the tab, on which to click to switch between tabs.
    button: ButtonBlit
    # Axe of the button, used internally to define the position and size of
    # the button.
    button_axcut: Axes


class TabbedFigure:
    """A windows with several tabs holding matplotlib subplots. It enables
    adding, modifying and removing tabs sequentially and conveniently.

    .. note::
        It has been designed to be cross-platform, and supported by any
        Matplotlib backend. So it can be used on-the-spot right after fresh
        install of Python and `jiminy_py`, without requiering elevated
        priviledge to install Qt4/5 or Tkinter.

    .. warning::
        It only supports plotting time-dependent data, the later corresponding
        to the horizontal axis of every subplots.
    """
    def __init__(self,
                 sync_tabs: bool = False,
                 window_title: str = "jiminy",
                 offscreen: bool = False,
                 **kwargs: Any) -> None:
        """Create empty tabbed figure.

        .. note::
            It will show-up on display automatically only adding the first tab.

        :param sync_tabs: Synchronize time window on every tabs (horizontal
                          axis), rather than independently for every tabs.
        :param window_title: Desired indow title.
                             Optional: "jiminy" by default.
        :param offscreen: Whether or not to enable display on screen of figure.
                          Optional: False by default.
        :param kwargs: Unused extra keyword arguments to enable forwarding.
        """
        # Backup user arguments
        self.sync_tabs = sync_tabs
        self.offscreen = offscreen

        # Internal state buffers
        self.figure = plt.figure()
        self.legend: Optional[Legend] = None
        self.ref_ax: Optional[Axes] = None
        self.tabs_data = {}
        self.tab_active: Optional[TabData] = None
        self.bbox_inches: Bbox = Bbox([[0.0, 0.0], [1.0, 1.0]])

        # Set window title
        self.figure.canvas.manager.set_window_title(window_title)

        # Set window size for offscreen rendering
        if self.offscreen:
            self.figure.set_size_inches(18, 12)

        # Register 'on resize' event callback to adjust layout
        self.figure.canvas.mpl_connect('resize_event', self.adjust_layout)

    def close(self) -> None:
        """Close figure.
        """
        plt.close(self.figure)

    def __del__(self) -> None:
        self.close()

    def adjust_layout(self,
                      event: Optional[Event] = None, *,
                      refresh_canvas: bool = False) -> None:
        """Optimize subplot grid and buttons width for best fit, then adjust
        layout based on window size.

        :param event: Event spent by figure `mpl_connect` 'resize_event'.
        :param refresh_canvas: Force redrawing figure canvas.
        """
        # Compute figure area for later export
        bbox = Bbox([[0.0, 0.065], [1.0, 1.0]])
        bbox_pixels = bbox.transformed(self.figure.transFigure)
        self.bbox_inches = bbox_pixels.transformed(
            self.figure.dpi_scale_trans.inverted())

        # Refresh button size, in case the number of tabs has changed
        buttons_width = 1.0 / (len(self.tabs_data) + 1)
        for i, tab in enumerate(self.tabs_data.values()):
            tab["button_axcut"].set_position(
                [buttons_width * (i + 0.5), 0.01, buttons_width, 0.05])

        # Re-arrange subplots in case figure aspect ratio has changed
        axes = self.tab_active["axes"]
        num_subplots = len(axes)
        figure_extent = self.figure.get_window_extent()
        figure_ratio = figure_extent.width / figure_extent.height
        num_rows_1 = max(1, floor(sqrt(num_subplots / figure_ratio)))
        num_cols_1 = ceil(num_subplots / num_rows_1)
        num_cols_2 = ceil(sqrt(num_subplots * figure_ratio))
        num_rows_2 = ceil(num_subplots / num_cols_2)
        if num_rows_1 * num_cols_1 < num_rows_2 * num_cols_2:
            num_rows, num_cols = map(int, (num_rows_1, num_cols_1))
        else:
            num_rows, num_cols = map(int, (num_rows_2, num_cols_2))
        for i, ax in enumerate(axes, 1):
            ax.change_geometry(num_rows, num_cols, i)

        # Adjust layout: namely margins between subplots
        right_margin = 0.03
        if self.legend is not None:
            legend_extent = self.legend.get_window_extent()
            legend_width_rel = legend_extent.transformed(
                self.figure.transFigure.inverted()).width
            right_margin += legend_width_rel
        self.figure.subplots_adjust(
            bottom=0.10, top=0.92, left=0.04, right=1.0-right_margin,
            wspace=0.40, hspace=0.30)

        # Refresh figure canvas if requested
        if refresh_canvas:
            self.refresh()

    def __click(self, event: Event, force_update: bool = False) -> None:
        """Event handler used internally to switch tab when a button is
        pressed.
        """
        # Get tab name to activate
        for tab in self.tabs_data.values():
            button = tab["button"]
            if button.ax == event.inaxes:
                tab_name = button.label.get_text().replace('\n', ' ')
                break

        # Early return if already active
        if not force_update and self.tab_active is self.tabs_data[tab_name]:
            return

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
        self.tab_active = self.tabs_data[tab_name]
        self.figure.suptitle(tab_name)
        for ax in self.tab_active["axes"]:
            self.figure.add_subplot(ax)
        handles, labels = self.tab_active["legend_data"]
        if labels:
            self.legend = self.figure.legend(
                handles, labels, loc='center right',
                bbox_to_anchor=(0.99, 0.5))

        # Restore navigation history and toolbar state if necessary
        cur_stack._elements = self.tab_active["nav_stack"]
        cur_stack._pos = self.tab_active["nav_pos"]
        self.figure.canvas.toolbar.set_history_buttons()

        # Update buttons style
        for tab in self.tabs_data.values():
            button = tab["button"]
            if tab is self.tab_active:
                button.ax.set_facecolor('green')
                button.color = 'green'
                button.hovercolor = 'green'
            else:
                button.ax.set_facecolor('white')
                button.color = 'white'
                button.hovercolor = '0.95'

        # Adjust layout and refresh figure
        self.adjust_layout(refresh_canvas=True)

    def refresh(self) -> None:
        """Refresh canvas drawing.
        """
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def add_tab(self,
                tab_name: str,
                time: np.ndarray,
                data: Union[np.ndarray, Dict[str, Union[
                    Dict[str, np.ndarray], np.ndarray]]],
                plot_method: Optional[Union[Callable[[
                    Axes, np.ndarray, np.ndarray], Any], str]] = None, *,
                refresh_canvas: bool = True,
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

        if isinstance(data, dict):
            # Compute plot grid arrangement
            n_cols = len(data)
            n_rows = 1
            while n_cols > n_rows + 2:
                n_rows = n_rows + 1
                n_cols = int(np.ceil(len(data) / n_rows))

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
        button = ButtonBlit(button_axcut,
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
            button.hovercolor = 'green'

        # Update figure and show it without blocking
        self.adjust_layout(refresh_canvas=refresh_canvas)
        if not self.offscreen:
            self.figure.show()

    def set_active_tab(self, tab_name: str) -> None:
        event = LocationEvent("click", self.figure.canvas, 0, 0)
        event.inaxes = self.tabs_data[tab_name]["button"].ax
        self.__click(event, force_update=True)

    def remove_tab(self,
                   tab_name: str, *,
                   refresh_canvas: bool = True) -> None:
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
        self.adjust_layout(refresh_canvas=refresh_canvas)

    def clear(self) -> None:
        """ TODO: Write documentation.
        """
        # Remove every figure axes
        for tab_name in list(self.tabs_data.keys()):  # list to make copy
            self.remove_tab(tab_name, refresh_canvas=False)
        self.refresh()

    def save_tab(self, image_path: str) -> None:
        """Export current tab, limiting the bounding box to the subplots.

        :param image_path: Desired location for generated image. Note that
                           only '.png' format is supported for now.
        """
        image_path = str(pathlib.Path(image_path).with_suffix('.png'))
        self.figure.savefig(
            image_path, format='png', dpi=EXPORT_DPI, transparent=False,
            bbox_inches=self.bbox_inches)

    def save_all_tabs(self, pdf_path: str) -> List[str]:
        """Export every tabs in a signle pdf, limiting systematically the
        bounding box to the subplots and legend.

        :param image_path: Desired location for generated pdf file.
        """
        pdf_path = str(pathlib.Path(pdf_path).with_suffix('.pdf'))
        with PdfPages(pdf_path) as pdf:
            for tab_name in self.tabs_data.keys():
                self.set_active_tab(tab_name)
                pdf.savefig(bbox_inches=self.bbox_inches)

    @classmethod
    def plot(cls,
             time: np.ndarray,
             tabs_data: Dict[str, Union[Dict[str, Union[
                    Dict[str, np.ndarray], np.ndarray]]]],
             image_path: Optional[str] = None,
             **kwargs) -> "TabbedFigure":
        """ TODO: Write documentation.

        :param image_path: It specified, the figure will be exported to pdf
                           without rendering on screen.
        :param kwargs: Extra keyword arguments to forward to `add_tab` method.
        """
        tabbed_figure = cls(**{"offscreen": image_path is not None, **kwargs})
        for name, data in tabs_data.items():
            tabbed_figure.add_tab(
                name, time, data, **kwargs, refresh_canvas=False)
        tabbed_figure.refresh()
        if image_path is not None:
            tabbed_figure.save_all_tabs(image_path)
        return tabbed_figure
