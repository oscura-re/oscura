"""Accessibility utilities for Oscura visualizations.

This module provides accessibility features for visualizations including
colorblind-safe palettes, alt-text generation, and keyboard navigation support.


Example:
    >>> from oscura.visualization.accessibility import (
    ...     get_colorblind_palette,
    ...     generate_alt_text,
    ...     KeyboardHandler
    ... )
    >>> palette = get_colorblind_palette("viridis")
    >>> alt_text = generate_alt_text(trace, "Time-domain waveform")

References:
    - Colorblind-safe palette design (Brettel 1997)
    - WCAG 2.1 accessibility guidelines
    - WAI-ARIA best practices
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


# Line style patterns for multi-line plots (ACC-001)
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

# Pass/fail symbols (ACC-001)
PASS_SYMBOL = "✓"
FAIL_SYMBOL = "✗"


def get_colorblind_palette(
    name: Literal["viridis", "cividis", "plasma", "inferno", "magma"] = "viridis",
) -> str:
    """Get colorblind-safe colormap name.

    : All visualizations use colorblind-safe palettes by default.
    Returns matplotlib colormap names that are perceptually uniform and colorblind-safe.

    Args:
        name: Colormap name. Options:
            - "viridis": Default, excellent for sequential data
            - "cividis": Optimized for colorblind users
            - "plasma": High contrast sequential
            - "inferno": Warm sequential
            - "magma": Dark to bright sequential

    Returns:
        Matplotlib colormap name string

    Raises:
        ValueError: If colormap name is not recognized

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from oscura.visualization.accessibility import get_colorblind_palette
        >>> cmap = get_colorblind_palette("viridis")
        >>> plt.plot([1, 2, 3], [1, 4, 2], color=plt.get_cmap(cmap)(0.5))

    References:
        ACC-001: Colorblind-Safe Visualization Palette
        Matplotlib perceptually uniform colormaps
    """
    valid_names = ["viridis", "cividis", "plasma", "inferno", "magma"]
    if name not in valid_names:
        raise ValueError(f"Unknown colormap: {name}. Valid options: {', '.join(valid_names)}")
    return name


def get_multi_line_styles(n_lines: int) -> list[tuple[tuple[float, float, float, float], str]]:
    """Get distinct line styles and colors for multi-line plots.

    : Multi-line plots use distinct line styles in addition to colors.
    Combines colorblind-safe colors with varied line styles for maximum distinguishability.

    Args:
        n_lines: Number of lines to style

    Returns:
        List of (color, linestyle) tuples where color is RGBA tuple

    Example:
        >>> from oscura.visualization.accessibility import get_multi_line_styles
        >>> import matplotlib.pyplot as plt
        >>> styles = get_multi_line_styles(4)
        >>> for i, (color, ls) in enumerate(styles):
        ...     plt.plot([1, 2, 3], [i, i+1, i+2], color=color, linestyle=ls)

    References:
        ACC-001: Colorblind-Safe Visualization Palette
    """
    # Use viridis colormap for colorblind-safe colors
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(n_lines - 1, 1)) for i in range(n_lines)]

    # Cycle through line styles
    styles: list[tuple[tuple[float, float, float, float], str]] = []
    for i in range(n_lines):
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        # Colors from colormap are RGBA tuples
        rgba_color = tuple(colors[i])
        styles.append((rgba_color, linestyle))  # type: ignore[arg-type]

    return styles


def format_pass_fail(
    passed: bool,
    *,
    use_color: bool = True,
    use_symbols: bool = True,
) -> str:
    """Format pass/fail status with symbols and optional colors.

    : Pass/fail uses symbols (✓/✗) not just red/green.
    Ensures accessibility by using symbols in addition to or instead of colors.

    Args:
        passed: Whether the test passed
        use_color: Include ANSI color codes (default: True)
        use_symbols: Include checkmark/cross symbols (default: True)

    Returns:
        Formatted string with symbol and/or color

    Example:
        >>> from oscura.visualization.accessibility import format_pass_fail
        >>> print(format_pass_fail(True))
        ✓ PASS
        >>> print(format_pass_fail(False))
        ✗ FAIL

    References:
        ACC-001: Colorblind-Safe Visualization Palette
    """
    if passed:
        symbol = PASS_SYMBOL if use_symbols else ""
        text = "PASS"
        color_code = "\033[92m" if use_color else ""  # Green
    else:
        symbol = FAIL_SYMBOL if use_symbols else ""
        text = "FAIL"
        color_code = "\033[91m" if use_color else ""  # Red

    reset_code = "\033[0m" if use_color else ""

    if use_symbols:
        return f"{color_code}{symbol} {text}{reset_code}"
    else:
        return f"{color_code}{text}{reset_code}"


def generate_alt_text(
    data: NDArray[np.float64] | dict[str, Any],
    plot_type: str,
    *,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Amplitude",
    sample_rate: float | None = None,
) -> str:
    """Generate descriptive alt-text for a plot.

    : Every plot has alt_text property describing content.
    Provides text-based summary for screen readers and accessibility tools.

    Args:
        data: Signal data array or statistics dictionary
        plot_type: Type of plot ("waveform", "spectrum", "histogram", "eye_diagram")
        title: Plot title (optional)
        x_label: X-axis label
        y_label: Y-axis label
        sample_rate: Sample rate in Hz (for time calculations)

    Returns:
        Descriptive alt-text string

    Example:
        >>> import numpy as np
        >>> from oscura.visualization.accessibility import generate_alt_text
        >>> signal = np.sin(2 * np.pi * 1e3 * np.linspace(0, 1e-3, 1000))
        >>> alt_text = generate_alt_text(signal, "waveform", title="1 kHz sine wave")
        >>> print(alt_text)
        1 kHz sine wave. Waveform plot showing Time vs Amplitude...

    References:
        ACC-002: Text Alternatives for Visualizations
        WCAG 2.1 Section 1.1.1 (Non-text Content)
    """
    parts = []

    # Add title if provided
    if title:
        parts.append(f"{title}.")

    # Describe plot type
    parts.append(f"{plot_type.replace('_', ' ').title()} plot showing {x_label} vs {y_label}.")

    # Add data statistics
    if isinstance(data, dict):
        # Already have statistics
        stats = data
    else:
        # Calculate statistics from array
        stats = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "n_samples": len(data),
        }

    # Format statistics
    if "n_samples" in stats:
        parts.append(f"Contains {stats['n_samples']} samples.")

    if "min" in stats and "max" in stats:
        parts.append(f"Range: {stats['min']:.3g} to {stats['max']:.3g} {y_label}.")

    if "mean" in stats:
        parts.append(f"Mean: {stats['mean']:.3g}.")

    if "std" in stats:
        parts.append(f"Standard deviation: {stats['std']:.3g}.")

    # Add duration if sample rate provided
    if sample_rate is not None and "n_samples" in stats:
        duration_s = stats["n_samples"] / sample_rate
        if duration_s < 1e-6:
            duration_str = f"{duration_s * 1e9:.2f} ns"
        elif duration_s < 1e-3:
            duration_str = f"{duration_s * 1e6:.2f} µs"
        elif duration_s < 1:
            duration_str = f"{duration_s * 1e3:.2f} ms"
        else:
            duration_str = f"{duration_s:.2f} s"
        parts.append(f"Duration: {duration_str}.")

    return " ".join(parts)


def add_plot_aria_attributes(
    fig: Figure,
    alt_text: str,
    *,
    role: str = "img",
    label: str | None = None,
) -> None:
    """Add ARIA attributes to matplotlib figure for accessibility.

    : HTML reports include aria-describedby for plots.
    Adds WAI-ARIA attributes to figure metadata for screen reader support.

    Args:
        fig: Matplotlib figure object
        alt_text: Alternative text description
        role: ARIA role (default: "img")
        label: ARIA label (optional)

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from oscura.visualization.accessibility import (
        ...     add_plot_aria_attributes,
        ...     generate_alt_text
        ... )
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> alt_text = generate_alt_text([1, 4, 2], "waveform")
        >>> add_plot_aria_attributes(fig, alt_text)

    References:
        ACC-002: Text Alternatives for Visualizations
        WAI-ARIA 1.2 specification
    """
    # Store as figure metadata
    if not hasattr(fig, "_oscura_accessibility"):
        fig._oscura_accessibility = {}  # type: ignore[attr-defined]

    fig._oscura_accessibility["alt_text"] = alt_text  # type: ignore[attr-defined]
    fig._oscura_accessibility["aria_role"] = role  # type: ignore[attr-defined]

    if label:
        fig._oscura_accessibility["aria_label"] = label  # type: ignore[attr-defined]


class KeyboardHandler:
    """Keyboard navigation handler for interactive plots.

    : Interactive visualizations are fully keyboard-navigable.
    Provides standard keyboard controls for plot interaction.

    Keyboard shortcuts:
        - Tab: Navigate between plot elements
        - Arrow keys: Move cursors/markers
        - Enter: Select/activate element
        - Escape: Close modals/menus
        - +/-: Zoom in/out
        - Home/End: Jump to start/end
        - Space: Toggle play/pause (for animations)

    Args:
        fig: Matplotlib figure to attach handlers to
        axes: Axes object for cursor/marker operations

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from oscura.visualization.accessibility import KeyboardHandler
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> handler = KeyboardHandler(fig, ax)
        >>> handler.enable()
        >>> plt.show()

    References:
        ACC-003: Keyboard Navigation for Interactive Plots
        WAI-ARIA Authoring Practices 1.2
    """

    def __init__(self, fig: Figure, axes: Axes) -> None:
        """Initialize keyboard handler.

        Args:
            fig: Matplotlib figure
            axes: Axes for operations
        """
        self.fig = fig
        self.axes = axes
        self.cursor_position: float = 0.0
        self.cursor_line: Any = None
        self.enabled: bool = False
        self._connection_id: int | None = None

        # Callback registry
        self.on_cursor_move: Callable[[float], None] | None = None
        self.on_select: Callable[[], None] | None = None
        self.on_escape: Callable[[], None] | None = None

    def enable(self) -> None:
        """Enable keyboard navigation.

        Connects keyboard event handlers to the figure.

        Example:
            >>> handler = KeyboardHandler(fig, ax)
            >>> handler.enable()

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        if not self.enabled:
            self._connection_id = self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
            self.enabled = True

    def disable(self) -> None:
        """Disable keyboard navigation.

        Disconnects keyboard event handlers.

        Example:
            >>> handler.disable()

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        if self.enabled and self._connection_id is not None:
            self.fig.canvas.mpl_disconnect(self._connection_id)
            self._connection_id = None
            self.enabled = False

    def _on_key_press(self, event: Any) -> None:
        """Handle keyboard events.

        Args:
            event: Matplotlib key press event

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        if event.key is None:
            return

        # Arrow keys: move cursor
        if event.key in ("left", "right"):
            self._move_cursor(event.key)

        # Enter: select/activate
        elif event.key == "enter":
            if self.on_select:
                self.on_select()

        # Escape: close/cancel
        elif event.key == "escape":
            if self.on_escape:
                self.on_escape()

        # +/-: zoom
        elif event.key in ("+", "="):
            self._zoom(1.2)
        elif event.key in ("-", "_"):
            self._zoom(0.8)

        # Home/End: jump to edges
        elif event.key == "home":
            self._jump_to_start()
        elif event.key == "end":
            self._jump_to_end()

    def _move_cursor(self, direction: str) -> None:
        """Move cursor left or right.

        Args:
            direction: "left" or "right"

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        xlim = self.axes.get_xlim()
        step = (xlim[1] - xlim[0]) * 0.01  # 1% of range

        if direction == "left":
            self.cursor_position = max(xlim[0], self.cursor_position - step)
        else:
            self.cursor_position = min(xlim[1], self.cursor_position + step)

        self._update_cursor()

        if self.on_cursor_move:
            self.on_cursor_move(self.cursor_position)

    def _update_cursor(self) -> None:
        """Update cursor line on plot.

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        ylim = self.axes.get_ylim()

        if self.cursor_line is None:
            # Create cursor line
            (self.cursor_line,) = self.axes.plot(
                [self.cursor_position, self.cursor_position],
                ylim,
                "r--",
                linewidth=2,
                label="Cursor",
            )
        else:
            # Update existing cursor
            self.cursor_line.set_xdata([self.cursor_position, self.cursor_position])

        self.fig.canvas.draw_idle()

    def _zoom(self, factor: float) -> None:
        """Zoom in or out.

        Args:
            factor: Zoom factor (>1 = zoom in, <1 = zoom out)

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        x_range = (xlim[1] - xlim[0]) / factor
        y_range = (ylim[1] - ylim[0]) / factor

        self.axes.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        self.axes.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

        self.fig.canvas.draw_idle()

    def _jump_to_start(self) -> None:
        """Jump cursor to start of plot.

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        xlim = self.axes.get_xlim()
        self.cursor_position = xlim[0]
        self._update_cursor()

        if self.on_cursor_move:
            self.on_cursor_move(self.cursor_position)

    def _jump_to_end(self) -> None:
        """Jump cursor to end of plot.

        References:
            ACC-003: Keyboard Navigation for Interactive Plots
        """
        xlim = self.axes.get_xlim()
        self.cursor_position = xlim[1]
        self._update_cursor()

        if self.on_cursor_move:
            self.on_cursor_move(self.cursor_position)


__all__ = [
    "FAIL_SYMBOL",
    "LINE_STYLES",
    "PASS_SYMBOL",
    "KeyboardHandler",
    "add_plot_aria_attributes",
    "format_pass_fail",
    "generate_alt_text",
    "get_colorblind_palette",
    "get_multi_line_styles",
]
