"""Specialized plot types for protocol analysis and state visualization.

This module provides specialized visualizations including protocol timing
diagrams, state machine views, and bus transaction timelines.


Example:
    >>> from oscura.visualization.specialized import plot_protocol_timing
    >>> fig = plot_protocol_timing(decoded_packets, sample_rate=1e6)

References:
    - Wavedrom-style digital waveform rendering
    - State machine diagram standards
    - Bus protocol visualization best practices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
    from matplotlib import patches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ProtocolSignal:
    """Protocol signal for timing diagram.

    Attributes:
        name: Signal name
        data: Signal data (0/1 for digital, values for analog)
        type: Signal type ("digital", "clock", "bus", "analog")
        transitions: List of transition times
        annotations: Dict of time -> annotation text
    """

    name: str
    data: NDArray[np.float64]
    type: Literal["digital", "clock", "bus", "analog"] = "digital"
    transitions: list[float] | None = None
    annotations: dict[float, str] | None = None


@dataclass
class StateTransition:
    """State machine transition.

    Attributes:
        from_state: Source state name
        to_state: Target state name
        condition: Transition condition/label
        style: Line style ("solid", "dashed", "dotted")
    """

    from_state: str
    to_state: str
    condition: str = ""
    style: Literal["solid", "dashed", "dotted"] = "solid"


def plot_protocol_timing(
    signals: list[ProtocolSignal],
    sample_rate: float,
    *,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    style: Literal["wavedrom", "classic"] = "wavedrom",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot protocol timing diagram in wavedrom style.

    Creates a timing diagram showing digital signals, clock edges, and
    bus transactions with annotations for protocol events.

    Args:
        signals: List of ProtocolSignal objects to plot.
        sample_rate: Sample rate in Hz.
        time_range: Time range to plot (t_min, t_max) in seconds. None = full range.
        time_unit: Time unit for x-axis ("s", "ms", "us", "ns", "auto").
        style: Diagram style ("wavedrom" = clean digital, "classic" = traditional).
        figsize: Figure size (width, height). Auto-calculated if None.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If signals list is empty.

    Example:
        >>> sda = ProtocolSignal("SDA", sda_data, type="digital")
        >>> scl = ProtocolSignal("SCL", scl_data, type="clock")
        >>> fig = plot_protocol_timing(
        ...     [scl, sda],
        ...     sample_rate=1e6,
        ...     style="wavedrom",
        ...     title="I2C Transaction"
        ... )

    References:
        VIS-021: Specialized - Protocol Timing Diagram
        Wavedrom digital waveform rendering
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(signals) == 0:
        raise ValueError("signals list cannot be empty")

    n_signals = len(signals)

    # Auto-calculate figure size
    if figsize is None:
        width = 12
        height = max(4, n_signals * 0.8 + 1)
        figsize = (width, height)

    fig, axes = plt.subplots(
        n_signals,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.1},
    )

    if n_signals == 1:
        axes = [axes]

    # Determine time range
    if time_range is None:
        max_len = max(len(sig.data) for sig in signals)
        t_min = 0.0
        t_max = max_len / sample_rate
    else:
        t_min, t_max = time_range

    # Select time unit
    if time_unit == "auto":
        time_range_val = t_max - t_min
        if time_range_val < 1e-6:
            time_unit = "ns"
            time_mult = 1e9
        elif time_range_val < 1e-3:
            time_unit = "us"
            time_mult = 1e6
        elif time_range_val < 1:
            time_unit = "ms"
            time_mult = 1e3
        else:
            time_unit = "s"
            time_mult = 1.0
    else:
        time_mult = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(time_unit, 1.0)

    # Plot each signal
    for _idx, (signal, ax) in enumerate(zip(signals, axes, strict=False)):
        # Create time vector
        time = np.arange(len(signal.data)) / sample_rate * time_mult

        # Filter to time range
        mask = (time >= t_min * time_mult) & (time <= t_max * time_mult)
        time = time[mask]
        data = signal.data[mask]

        if style == "wavedrom":
            _plot_wavedrom_signal(ax, time, data, signal)
        else:
            _plot_classic_signal(ax, time, data, signal)

        # Add signal name label
        ax.set_ylabel(signal.name, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_ylim(-0.2, 1.3)

        # Remove y-axis ticks
        ax.set_yticks([])

        # Grid for timing
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")

        # Add annotations
        if signal.annotations:
            for t, text in signal.annotations.items():
                if t_min <= t <= t_max:
                    ax.annotate(
                        text,
                        xy=(t * time_mult, 1.2),
                        fontsize=8,
                        ha="center",
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
                    )

    # X-axis label only on bottom plot
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    fig.tight_layout()
    return fig


def _plot_wavedrom_signal(
    ax: Axes,
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    signal: ProtocolSignal,
) -> None:
    """Plot signal in wavedrom style (clean digital waveform)."""
    if signal.type == "clock":
        # Clock signal: square wave
        for i in range(len(time) - 1):
            level = 1 if data[i] > 0.5 else 0
            ax.plot(
                [time[i], time[i + 1]],
                [level, level],
                "b-",
                linewidth=1.5,
            )
            # Vertical transition
            if i < len(time) - 1:
                next_level = 1 if data[i + 1] > 0.5 else 0
                if level != next_level:
                    ax.plot(
                        [time[i + 1], time[i + 1]],
                        [level, next_level],
                        "b-",
                        linewidth=1.5,
                    )

    elif signal.type == "digital":
        # Digital signal: step function with transitions
        for i in range(len(time) - 1):
            level = 1 if data[i] > 0.5 else 0
            ax.plot(
                [time[i], time[i + 1]],
                [level, level],
                "k-",
                linewidth=1.5,
            )
            # Vertical transition with slight slant for visual clarity
            if i < len(time) - 1:
                next_level = 1 if data[i + 1] > 0.5 else 0
                if level != next_level:
                    transition_width = (time[i + 1] - time[i]) * 0.1
                    ax.plot(
                        [time[i + 1] - transition_width, time[i + 1]],
                        [level, next_level],
                        "k-",
                        linewidth=1.5,
                    )

    elif signal.type == "bus":
        # Bus signal: show as high-impedance or data values
        ax.fill_between(time, 0.3, 0.7, alpha=0.3, color="gray")
        ax.plot(time, np.full_like(time, 0.5), "k-", linewidth=0.5)

    else:
        # Analog signal
        ax.plot(time, data, "r-", linewidth=1.2)


def _plot_classic_signal(
    ax: Axes,
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    signal: ProtocolSignal,
) -> None:
    """Plot signal in classic style (traditional oscilloscope-like)."""
    ax.plot(time, data, "b-", linewidth=1.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_state_machine(
    states: list[str],
    transitions: list[StateTransition],
    *,
    initial_state: str | None = None,
    final_states: list[str] | None = None,
    layout: Literal["circular", "hierarchical", "force"] = "circular",
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> Figure:
    """Plot state machine diagram.

    Creates a state diagram showing states as nodes and transitions as
    directed edges with condition labels.

    Args:
        states: List of state names.
        transitions: List of StateTransition objects.
        initial_state: Initial state (marked with double circle).
        final_states: List of final states (marked with double circle).
        layout: Layout algorithm for state positioning.
        figsize: Figure size (width, height).
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> states = ["IDLE", "ACTIVE", "WAIT", "DONE"]
        >>> transitions = [
        ...     StateTransition("IDLE", "ACTIVE", "START"),
        ...     StateTransition("ACTIVE", "WAIT", "BUSY"),
        ...     StateTransition("WAIT", "ACTIVE", "RETRY"),
        ...     StateTransition("ACTIVE", "DONE", "COMPLETE"),
        ... ]
        >>> fig = plot_state_machine(
        ...     states, transitions, initial_state="IDLE", final_states=["DONE"]
        ... )

    References:
        VIS-022: Specialized - State Machine View
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate state positions using selected layout
    positions = _calculate_state_positions(states, layout)

    # Draw states as circles
    state_radius = 0.15

    for state, (x, y) in positions.items():
        # Draw state circle
        circle = patches.Circle(
            (x, y),
            state_radius,
            fill=True,
            facecolor="lightblue",
            edgecolor="black",
            linewidth=2.0,
        )
        ax.add_patch(circle)

        # Mark initial state with double circle
        if state == initial_state:
            outer_circle = patches.Circle(
                (x, y),
                state_radius * 1.2,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
            ax.add_patch(outer_circle)

        # Mark final states with double circle
        if final_states and state in final_states:
            inner_circle = patches.Circle(
                (x, y),
                state_radius * 0.8,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
            ax.add_patch(inner_circle)

        # Add state label
        ax.text(
            x,
            y,
            state,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Draw transitions as arrows
    for trans in transitions:
        if trans.from_state not in positions or trans.to_state not in positions:
            continue

        x1, y1 = positions[trans.from_state]
        x2, y2 = positions[trans.to_state]

        # Calculate arrow start/end on circle perimeter
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1e-6:
            # Self-loop
            _draw_self_loop(ax, x1, y1, state_radius, trans.condition)
            continue

        # Normalize
        dx_norm = dx / dist
        dy_norm = dy / dist

        # Arrow start/end on circle edges
        arrow_start_x = x1 + dx_norm * state_radius
        arrow_start_y = y1 + dy_norm * state_radius
        arrow_end_x = x2 - dx_norm * state_radius
        arrow_end_y = y2 - dy_norm * state_radius

        # Line style
        linestyle = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
        }.get(trans.style, "-")

        # Draw arrow
        ax.annotate(
            "",
            xy=(arrow_end_x, arrow_end_y),
            xytext=(arrow_start_x, arrow_start_y),
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.5,
                "linestyle": linestyle,
                "color": "black",
            },
        )

        # Add transition label
        if trans.condition:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y,
                trans.condition,
                fontsize=8,
                ha="center",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "gray",
                    "alpha": 0.9,
                },
            )

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    if title:
        ax.set_title(title, fontsize=14, pad=20)

    fig.tight_layout()
    return fig


def _calculate_state_positions(
    states: list[str],
    layout: str,
) -> dict[str, tuple[float, float]]:
    """Calculate state positions using layout algorithm."""
    n_states = len(states)
    positions = {}

    if layout == "circular":
        # Arrange states in a circle
        angle_step = 2 * np.pi / n_states
        for i, state in enumerate(states):
            angle = i * angle_step
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            positions[state] = (x, y)

    elif layout == "hierarchical":
        # Arrange in rows (simplified hierarchical)
        states_per_row = int(np.ceil(np.sqrt(n_states)))
        for i, state in enumerate(states):
            row = i // states_per_row
            col = i % states_per_row
            x = (col + 0.5) / states_per_row
            y = 1.0 - (row + 0.5) / np.ceil(n_states / states_per_row)
            positions[state] = (x, y)

    else:  # force-directed (simplified)
        # Use random positions as a placeholder for true force-directed layout
        np.random.seed(42)
        for i, state in enumerate(states):  # noqa: B007
            x = 0.2 + 0.6 * np.random.rand()
            y = 0.2 + 0.6 * np.random.rand()
            positions[state] = (x, y)

    return positions


def _draw_self_loop(
    ax: Axes,
    x: float,
    y: float,
    radius: float,
    label: str,
) -> None:
    """Draw self-loop transition on state."""
    # Draw arc above state
    arc = patches.Arc(
        (x, y + radius),
        width=radius * 1.5,
        height=radius * 1.5,
        angle=0,
        theta1=0,
        theta2=180,
        linewidth=1.5,
        edgecolor="black",
        fill=False,
    )
    ax.add_patch(arc)

    # Add arrow head
    ax.annotate(
        "",
        xy=(x - radius * 0.7, y + radius * 0.3),
        xytext=(x - radius * 0.5, y + radius * 0.5),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "black"},
    )

    # Add label
    if label:
        ax.text(
            x,
            y + radius * 2.2,
            label,
            fontsize=8,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
        )


__all__ = [
    "ProtocolSignal",
    "StateTransition",
    "plot_protocol_timing",
    "plot_state_machine",
]
