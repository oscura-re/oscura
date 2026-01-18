"""Keyboard navigation support for Oscura interactive visualizations.

This module provides keyboard navigation handlers for interactive plots
following WCAG 2.1 accessibility guidelines.


Example:
    >>> from oscura.visualization.keyboard import KeyboardNavigator
    >>> navigator = KeyboardNavigator(fig, ax)
    >>> navigator.connect()

References:
    - WCAG 2.1 Guideline 2.1: Keyboard Accessible
    - WAI-ARIA Authoring Practices for interactive widgets
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes

if TYPE_CHECKING:
    from matplotlib.backend_bases import KeyEvent
    from matplotlib.figure import Figure
    from matplotlib.text import Text


class KeyboardNavigator:
    """Keyboard navigation handler for interactive plots.

    : Interactive visualizations are fully keyboard-navigable.
    Provides keyboard shortcuts for panning, zooming, and navigation.

    Keyboard shortcuts:
        - Arrow keys: Pan left/right/up/down
        - +/-: Zoom in/out
        - Home: Reset to full view
        - Tab: Cycle through subplots
        - Escape: Cancel current operation
        - ?: Show help

    Args:
        fig: Matplotlib figure
        axes: Matplotlib axes or list of axes
        pan_step: Pan step as fraction of current range (default: 0.1)
        zoom_factor: Zoom factor for +/- keys (default: 1.2)

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from oscura.visualization.keyboard import KeyboardNavigator
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> nav = KeyboardNavigator(fig, ax)
        >>> nav.connect()
        >>> plt.show()

    References:
        ACC-003: Keyboard Navigation for Interactive Plots
    """

    def __init__(
        self,
        fig: Figure,
        axes: Axes | list[Axes],
        *,
        pan_step: float = 0.1,
        zoom_factor: float = 1.2,
    ) -> None:
        """Initialize keyboard navigator.

        Args:
            fig: Matplotlib figure
            axes: Single axes or list of axes
            pan_step: Pan step as fraction of range
            zoom_factor: Zoom factor for zoom operations
        """
        self.fig = fig
        self.axes_list = [axes] if isinstance(axes, Axes) else list(axes)
        self.current_axes_index = 0
        self.pan_step = pan_step
        self.zoom_factor = zoom_factor

        # Store original limits for reset
        self.original_limits = {}
        for i, ax in enumerate(self.axes_list):
            self.original_limits[i] = {
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
            }

        self._connection_id: int | None = None
        self._help_text: Text | None = None

    def connect(self) -> None:
        """Connect keyboard event handler to the figure.

        : Tab navigates between plot elements.
        Registers keyboard event callback with matplotlib.

        Example:
            >>> nav.connect()
            >>> # Now keyboard events are handled

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        self._connection_id = self.fig.canvas.mpl_connect("key_press_event", self._on_key)  # type: ignore[arg-type]
        self._highlight_active_axes()

    def disconnect(self) -> None:
        """Disconnect keyboard event handler.

        Example:
            >>> nav.disconnect()

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        if self._connection_id is not None:
            self.fig.canvas.mpl_disconnect(self._connection_id)
            self._connection_id = None

    def _on_key(self, event: KeyEvent) -> None:
        """Handle keyboard events.

        : Arrow keys move cursors, Enter selects/activates, Escape closes.

        Args:
            event: Matplotlib keyboard event

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        if event.key is None:
            return

        ax = self.axes_list[self.current_axes_index]

        # Arrow keys: Pan
        if event.key == "left":
            self._pan(ax, dx=-self.pan_step, dy=0)
        elif event.key == "right":
            self._pan(ax, dx=self.pan_step, dy=0)
        elif event.key == "up":
            self._pan(ax, dx=0, dy=self.pan_step)
        elif event.key == "down":
            self._pan(ax, dx=0, dy=-self.pan_step)

        # Zoom
        elif event.key == "+" or event.key == "=":
            self._zoom(ax, factor=1.0 / self.zoom_factor)
        elif event.key == "-" or event.key == "_":
            self._zoom(ax, factor=self.zoom_factor)

        # Reset view
        elif event.key == "home":
            self._reset_view(ax)

        # Tab: Cycle through axes
        elif event.key == "tab":
            self._cycle_axes(reverse=event.key == "shift+tab")

        # Help
        elif event.key == "?":
            self._show_help()

        # Escape: Close help or reset
        elif event.key == "escape":
            self._hide_help()

        else:
            return  # Unhandled key

        self.fig.canvas.draw_idle()

    def _pan(self, ax: Axes, dx: float, dy: float) -> None:
        """Pan the axes view.

        Args:
            ax: Axes to pan
            dx: Horizontal pan as fraction of range
            dy: Vertical pan as fraction of range

        References:
            ACC-003: Arrow keys move cursors
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        x_shift = dx * x_range
        y_shift = dy * y_range

        ax.set_xlim(xlim[0] + x_shift, xlim[1] + x_shift)
        ax.set_ylim(ylim[0] + y_shift, ylim[1] + y_shift)

    def _zoom(self, ax: Axes, factor: float) -> None:
        """Zoom the axes view.

        Args:
            ax: Axes to zoom
            factor: Zoom factor (>1 zooms out, <1 zooms in)

        References:
            ACC-003: +/- keys zoom in/out
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        x_range = (xlim[1] - xlim[0]) * factor
        y_range = (ylim[1] - ylim[0]) * factor

        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

    def _reset_view(self, ax: Axes) -> None:
        """Reset axes to original view.

        Args:
            ax: Axes to reset

        References:
            ACC-003: Home resets to full view
        """
        idx = self.axes_list.index(ax)
        original = self.original_limits[idx]
        ax.set_xlim(original["xlim"])
        ax.set_ylim(original["ylim"])

    def _cycle_axes(self, reverse: bool = False) -> None:
        """Cycle through axes with Tab key.

        Args:
            reverse: Cycle backwards (Shift+Tab)

        References:
            ACC-003: Tab navigates between plot elements
        """
        if len(self.axes_list) <= 1:
            return

        # Remove highlight from current axes
        self._unhighlight_axes(self.axes_list[self.current_axes_index])

        # Move to next/previous axes
        if reverse:
            self.current_axes_index = (self.current_axes_index - 1) % len(self.axes_list)
        else:
            self.current_axes_index = (self.current_axes_index + 1) % len(self.axes_list)

        # Highlight new axes
        self._highlight_active_axes()

    def _highlight_active_axes(self) -> None:
        """Highlight the currently active axes.

        : Focus indicators for selected element.

        References:
            ACC-003: Focus indicators for selected element
        """
        ax = self.axes_list[self.current_axes_index]
        # Add visual focus indicator
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)

    def _unhighlight_axes(self, ax: Axes) -> None:
        """Remove highlight from axes.

        Args:
            ax: Axes to unhighlight

        References:
            ACC-003: Focus indicators for selected element
        """
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    def _show_help(self) -> None:
        """Show keyboard shortcuts help.

        : ? key shows keyboard shortcuts help.

        References:
            ACC-003: ? key shows keyboard shortcuts help
        """
        if self._help_text is not None:
            return  # Already showing

        help_message = """
Keyboard Navigation Help
========================

Pan:
  ←/→/↑/↓  Pan left/right/up/down

Zoom:
  +/-      Zoom in/out

View:
  Home     Reset to full view

Navigation:
  Tab      Next subplot
  Shift+Tab Previous subplot

Help:
  ?        Show this help
  Esc      Close help

Press Esc to close this help.
"""
        # Add text box to figure
        self._help_text = self.fig.text(
            0.5,
            0.5,
            help_message,
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.95,
                "edgecolor": "black",
                "linewidth": 2,
            },
            zorder=1000,
        )
        self.fig.canvas.draw_idle()

    def _hide_help(self) -> None:
        """Hide keyboard shortcuts help.

        : Escape closes modals/menus.

        References:
            ACC-003: Escape closes modals/menus
        """
        if self._help_text is not None:
            self._help_text.remove()
            self._help_text = None
            self.fig.canvas.draw_idle()


def enable_keyboard_navigation(
    fig: Figure,
    axes: Axes | list[Axes] | None = None,
    **kwargs: Any,
) -> KeyboardNavigator:
    """Enable keyboard navigation for a figure.

    Convenience function to create and connect a KeyboardNavigator.

    Args:
        fig: Matplotlib figure
        axes: Axes to navigate (default: all axes in figure)
        **kwargs: Additional arguments passed to KeyboardNavigator

    Returns:
        Connected KeyboardNavigator instance

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from oscura.visualization.keyboard import enable_keyboard_navigation
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> nav = enable_keyboard_navigation(fig)
        >>> plt.show()

    References:
        ACC-003: Keyboard Navigation for Interactive Plots
    """
    if axes is None:
        axes = fig.get_axes()

    navigator = KeyboardNavigator(fig, axes, **kwargs)
    navigator.connect()
    return navigator


__all__ = [
    "KeyboardNavigator",
    "enable_keyboard_navigation",
]
