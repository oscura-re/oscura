"""Digital timing diagram visualization.

This module provides timing diagrams for digital signals with
protocol decode overlay support.


Example:
    >>> from oscura.visualization.digital import plot_timing
    >>> fig = plot_timing([clk, data, cs], names=["CLK", "DATA", "CS"])
    >>> plt.show()

References:
    matplotlib best practices for digital waveform visualization
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from oscura.analyzers.protocols.base import Annotation


def plot_timing(
    traces: Sequence[WaveformTrace | DigitalTrace],
    *,
    names: list[str] | None = None,
    annotations: list[list[Annotation]] | None = None,
    time_unit: str = "auto",
    show_grid: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    time_range: tuple[float, float] | None = None,
    threshold: float | str = "auto",
) -> Figure:
    """Plot digital timing diagram with protocol decode overlay.

    Creates a stacked timing diagram showing digital waveforms with
    timing information and optional protocol decode annotations.

    Args:
        traces: List of traces to plot (analog or digital).
        names: Channel names for labels. If None, uses CH1, CH2, etc.
        annotations: List of protocol annotations per channel (optional).
        time_unit: Time unit ("s", "ms", "us", "ns", "auto").
        show_grid: Show vertical grid lines at time intervals.
        figsize: Figure size (width, height) in inches.
        title: Overall figure title.
        time_range: Optional (start, end) time range to display in seconds.
        threshold: Threshold for analog-to-digital conversion ("auto" or float).

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If traces list is empty.

    Example:
        >>> fig = plot_timing(
        ...     [clk_trace, data_trace, cs_trace],
        ...     names=["CLK", "DATA", "CS"],
        ...     annotations=[[], uart_annotations, []]
        ... )
        >>> plt.savefig("timing.png")

    References:
        IEEE 181-2011: Standard for Transitional Waveform Definitions
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(traces) == 0:
        raise ValueError("traces list cannot be empty")

    n_channels = len(traces)

    if names is None:
        names = [f"CH{i + 1}" for i in range(n_channels)]

    if len(names) != n_channels:
        raise ValueError(f"names length ({len(names)}) must match traces ({n_channels})")

    if figsize is None:
        figsize = (12, 1.5 * n_channels)

    # Convert analog traces to digital
    from oscura.analyzers.digital.extraction import to_digital

    digital_traces: list[DigitalTrace] = []
    for trace in traces:
        if isinstance(trace, WaveformTrace):
            digital_traces.append(to_digital(trace, threshold=threshold))  # type: ignore[arg-type]
        else:
            digital_traces.append(trace)

    # Auto-select time unit from first trace
    if time_unit == "auto" and len(digital_traces) > 0:
        ref_trace = digital_traces[0]
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

    # Create figure
    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=figsize,
        sharex=True,
    )

    if n_channels == 1:
        axes = [axes]

    # Determine time range
    if time_range is not None:
        start_time, end_time = time_range
    else:
        start_time = 0.0
        end_time = max(trace.duration for trace in digital_traces if len(trace.data) > 0)

    for i, (trace, name, ax) in enumerate(zip(digital_traces, names, axes, strict=False)):
        time = trace.time_vector * multiplier

        # Filter to time range
        if time_range is not None:
            start_idx = int(np.searchsorted(trace.time_vector, start_time))
            end_idx = int(np.searchsorted(trace.time_vector, end_time))
            time = time[start_idx:end_idx]
            data_slice = trace.data[start_idx:end_idx]
        else:
            data_slice = trace.data

        # Plot digital waveform as step function
        ax.step(
            time,
            data_slice.astype(int),
            where="post",
            color=f"C{i}",
            linewidth=1.5,
        )

        # Set up digital signal display
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontweight="bold")

        if show_grid:
            ax.grid(True, alpha=0.2, axis="x")

        # Add protocol annotations if provided
        if annotations is not None and i < len(annotations) and annotations[i]:
            _add_protocol_annotations(ax, annotations[i], multiplier, time_unit)

        # Remove x-axis labels except for bottom plot
        if i < n_channels - 1:
            ax.set_xticklabels([])

    # Set x-label on bottom plot
    axes[-1].set_xlabel(f"Time ({time_unit})")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def _add_protocol_annotations(
    ax: Axes,
    annotations: list[Annotation],
    multiplier: float,
    time_unit: str,
) -> None:
    """Add protocol decode annotations to timing diagram.

    Args:
        ax: Matplotlib axes to annotate.
        annotations: List of protocol annotations.
        multiplier: Time unit multiplier for display.
        time_unit: Time unit string.
    """
    for ann in annotations:
        # Get annotation time range
        start_time = ann.start_sample * multiplier if hasattr(ann, "start_sample") else 0
        end_time = ann.end_sample * multiplier if hasattr(ann, "end_sample") else start_time

        # Get annotation text and level
        if hasattr(ann, "data"):
            text = str(ann.data)
        elif hasattr(ann, "value"):
            text = str(ann.value)
        else:
            text = str(ann)

        # Determine annotation color based on type/level
        color = "lightblue"
        if hasattr(ann, "level"):
            level_str = str(ann.level).lower()
            if "error" in level_str or "warn" in level_str:
                color = "lightcoral"
            elif "data" in level_str or "byte" in level_str:
                color = "lightgreen"
            elif "start" in level_str or "stop" in level_str:
                color = "lightyellow"

        # Draw annotation box
        width = end_time - start_time if end_time > start_time else multiplier * 10
        rect = Rectangle(
            (start_time, 1.05),
            width,
            0.15,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Add text label
        mid_time = start_time + width / 2
        ax.text(
            mid_time,
            1.125,
            text,
            ha="center",
            va="center",
            fontsize=7,
            fontfamily="monospace",
        )


def plot_logic_analyzer(
    traces: Sequence[DigitalTrace],
    *,
    names: list[str] | None = None,
    bus_groups: dict[str, list[int]] | None = None,
    time_unit: str = "auto",
    show_grid: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot logic analyzer style multi-channel display with bus grouping.

    Creates a timing diagram optimized for logic analyzer visualization
    with support for bus grouping (showing multi-bit buses as hex values).

    Args:
        traces: List of digital traces.
        names: Channel names.
        bus_groups: Dictionary mapping bus names to channel indices.
            Example: {"DATA": [0, 1, 2, 3], "ADDR": [4, 5, 6, 7]}
        time_unit: Time unit for display.
        show_grid: Show vertical grid lines.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If traces list is empty.

    Example:
        >>> fig = plot_logic_analyzer(
        ...     traces,
        ...     names=[f"D{i}" for i in range(8)],
        ...     bus_groups={"DATA": [0, 1, 2, 3, 4, 5, 6, 7]}
        ... )

    References:
        Logic analyzer display conventions
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(traces) == 0:
        raise ValueError("traces list cannot be empty")

    # Convert to list for plot_timing
    traces_list: list[WaveformTrace | DigitalTrace] = list(traces)

    # If no bus groups, just use regular timing diagram
    if bus_groups is None:
        return plot_timing(
            traces_list,
            names=names,
            time_unit=time_unit,
            show_grid=show_grid,
            figsize=figsize,
            title=title,
        )

    # Implementation for bus grouping would go here
    # For MVP, delegate to plot_timing
    return plot_timing(
        traces_list,
        names=names,
        time_unit=time_unit,
        show_grid=show_grid,
        figsize=figsize,
        title=title,
    )


__all__ = [
    "plot_logic_analyzer",
    "plot_timing",
]
