"""Waveform visualization functions.

This module provides time-domain waveform and multi-channel plots
with measurement annotations.


Example:
    >>> from oscura.visualization.waveform import plot_waveform, plot_multi_channel
    >>> plot_waveform(trace)
    >>> plot_multi_channel([ch1, ch2, ch3])

References:
    matplotlib best practices for scientific visualization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_waveform(
    trace: WaveformTrace,
    *,
    ax: Axes | None = None,
    time_unit: str = "auto",
    time_range: tuple[float, float] | None = None,
    show_grid: bool = True,
    color: str = "C0",
    label: str | None = None,
    show_measurements: dict[str, Any] | None = None,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "Amplitude",
    show: bool = True,
    save_path: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot time-domain waveform.

    Args:
        trace: Waveform trace to plot.
        ax: Matplotlib axes. If None, creates new figure.
        time_unit: Time unit ("s", "ms", "us", "ns", "auto").
        time_range: Optional (start, end) time range in seconds to display.
        show_grid: Show grid lines.
        color: Line color.
        label: Legend label.
        show_measurements: Dictionary of measurements to annotate.
        title: Plot title.
        xlabel: X-axis label (appended with time unit).
        ylabel: Y-axis label.
        show: If True, call plt.show() to display the plot.
        save_path: Path to save the figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches. Only used if ax is None.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If axes has no associated figure.

    Example:
        >>> import oscura as osc
        >>> trace = osc.load("signal.wfm")
        >>> fig = osc.plot_waveform(trace, time_unit="us", show=False)
        >>> fig.savefig("waveform.png")

        >>> # With custom styling
        >>> fig = osc.plot_waveform(trace,
        ...                        title="Captured Signal",
        ...                        xlabel="Time",
        ...                        ylabel="Voltage",
        ...                        color="blue")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Calculate time axis
    time = trace.time_vector

    # Auto-select time unit
    if time_unit == "auto":
        duration = time[-1] if len(time) > 0 else 0
        if duration < 1e-6:
            time_unit = "ns"
        elif duration < 1e-3:
            time_unit = "us"
        elif duration < 1:
            time_unit = "ms"
        else:
            time_unit = "s"

    time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    multiplier = time_multipliers.get(time_unit, 1.0)
    time_scaled = time * multiplier

    # Plot waveform
    ax.plot(time_scaled, trace.data, color=color, label=label, linewidth=0.8)

    # Apply time range if specified
    if time_range is not None:
        ax.set_xlim(time_range[0] * multiplier, time_range[1] * multiplier)

    # Labels
    ax.set_xlabel(f"{xlabel} ({time_unit})")
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    elif trace.metadata.channel_name:
        ax.set_title(f"Waveform - {trace.metadata.channel_name}")

    if show_grid:
        ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    # Add measurement annotations
    if show_measurements:
        _add_measurement_annotations(ax, trace, show_measurements, time_unit, multiplier)

    fig.tight_layout()

    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_multi_channel(
    traces: list[WaveformTrace | DigitalTrace],
    *,
    names: list[str] | None = None,
    shared_x: bool = True,
    share_x: bool | None = None,
    colors: list[str] | None = None,
    time_unit: str = "auto",
    show_grid: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot multiple channels in stacked subplots.

    Args:
        traces: List of traces to plot.
        names: Channel names for labels.
        shared_x: Share x-axis across subplots.
        share_x: Alias for shared_x (for compatibility).
        colors: List of colors for each trace. If None, uses default cycle.
        time_unit: Time unit ("s", "ms", "us", "ns", "auto").
        show_grid: Show grid lines.
        figsize: Figure size (width, height) in inches.
        title: Overall figure title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> fig = plot_multi_channel([ch1, ch2, ch3], names=["CLK", "DATA", "CS"])
        >>> plt.show()
    """
    # Handle share_x alias
    if share_x is not None:
        shared_x = share_x
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    n_channels = len(traces)

    if names is None:
        names = [f"CH{i + 1}" for i in range(n_channels)]

    if figsize is None:
        figsize = (10, 2 * n_channels)

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=figsize,
        sharex=shared_x,
    )

    if n_channels == 1:
        axes = [axes]

    # Auto-select time unit from first trace
    if time_unit == "auto" and len(traces) > 0:
        ref_trace = traces[0]
        duration = len(ref_trace.data) * ref_trace.metadata.time_base
        if duration < 1e-6:
            time_unit = "ns"
        elif duration < 1e-3:
            time_unit = "us"
        elif duration < 1:
            time_unit = "ms"
        else:
            time_unit = "s"

    time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    multiplier = time_multipliers.get(time_unit, 1.0)

    for i, (trace, name, ax) in enumerate(zip(traces, names, axes, strict=False)):
        time = trace.time_vector * multiplier
        color = colors[i] if colors is not None and i < len(colors) else f"C{i}"

        if isinstance(trace, WaveformTrace):
            ax.plot(time, trace.data, color=color, linewidth=0.8)
            ax.set_ylabel("V")
        else:
            # Digital trace - step plot
            ax.step(time, trace.data.astype(int), color=color, where="post", linewidth=1.0)
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["L", "H"])

        ax.set_ylabel(name, rotation=0, ha="right", va="center")

        if show_grid:
            ax.grid(True, alpha=0.3)

        # Only show x-label on bottom plot
        if i == n_channels - 1:
            ax.set_xlabel(f"Time ({time_unit})")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


def plot_xy(
    x_trace: WaveformTrace | NDArray[np.float64],
    y_trace: WaveformTrace | NDArray[np.float64],
    *,
    ax: Axes | None = None,
    color: str = "C0",
    marker: str = "",
    alpha: float = 0.7,
    title: str | None = None,
) -> Figure:
    """Plot X-Y (Lissajous) diagram.

    Args:
        x_trace: X-axis waveform.
        y_trace: Y-axis waveform.
        ax: Matplotlib axes.
        color: Line/marker color.
        marker: Marker style.
        alpha: Transparency.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If axes has no associated figure.

    Example:
        >>> fig = plot_xy(ch1, ch2)  # Phase relationship
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    x_data = x_trace.data if isinstance(x_trace, WaveformTrace) else x_trace
    y_data = y_trace.data if isinstance(y_trace, WaveformTrace) else y_trace

    # Ensure same length
    min_len = min(len(x_data), len(y_data))
    x_data = x_data[:min_len]
    y_data = y_data[:min_len]

    ax.plot(x_data, y_data, color=color, marker=marker, alpha=alpha, linewidth=0.5)

    ax.set_xlabel("X (V)")
    ax.set_ylabel("Y (V)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def _add_measurement_annotations(
    ax: Axes,
    trace: WaveformTrace,
    measurements: dict[str, Any],
    time_unit: str,
    multiplier: float,
) -> None:
    """Add measurement annotations to plot."""
    # Create annotation text
    text_lines = []

    for name, value in measurements.items():
        if isinstance(value, dict):
            val = value.get("value", value)
            unit = value.get("unit", "")
            if isinstance(val, float) and not np.isnan(val):
                text_lines.append(f"{name}: {val:.4g} {unit}")
        elif isinstance(value, float) and not np.isnan(value):
            text_lines.append(f"{name}: {value:.4g}")

    if text_lines:
        text = "\n".join(text_lines)
        ax.annotate(
            text,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )


__all__ = [
    "plot_multi_channel",
    "plot_waveform",
    "plot_xy",
]
