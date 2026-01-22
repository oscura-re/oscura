"""Advanced digital logic visualizations.

State-of-the-art visualization tools for digital logic analysis including:
- Logic analyzer-style timeline displays
- Multi-channel bus views with bus decoding
- Timing diagram annotations
- IC timing validation overlays
- Eye diagrams for signal quality

Example:
    >>> from oscura.visualization.digital_advanced import plot_logic_analyzer_view
    >>> plot_logic_analyzer_view(channels, title="8-bit Data Bus")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from oscura.core.types import DigitalTrace, WaveformTrace


def plot_logic_analyzer_view(
    channels: dict[str, WaveformTrace | DigitalTrace],
    *,
    title: str | None = None,
    time_range: tuple[float, float] | None = None,
    group_buses: dict[str, list[str]] | None = None,
    show_hex: bool = True,
    show_cursors: bool = True,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[Figure, Axes]:
    """Create logic analyzer-style timeline display.

    Args:
        channels: Dictionary mapping channel names to traces.
        title: Optional plot title.
        time_range: Optional (start, end) time range in seconds.
        group_buses: Optional dict mapping bus names to channel lists.
        show_hex: Show hexadecimal values for buses.
        show_cursors: Show timing cursors.
        figsize: Figure size in inches.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> channels = {f"D{i}": trace for i, trace in enumerate(data_lines)}
        >>> group_buses = {"DATA": [f"D{i}" for i in range(8)]}
        >>> fig, ax = plot_logic_analyzer_view(channels, group_buses=group_buses)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine time range
    first_trace = next(iter(channels.values()))
    time_base = first_trace.metadata.time_base
    total_time = len(first_trace.data) * time_base

    t_start: float
    t_end: float
    if time_range is None:
        t_start, t_end = 0.0, total_time
    else:
        t_start, t_end = time_range

    # Calculate display window
    start_idx = int(t_start / time_base)
    end_idx = int(t_end / time_base)

    # Create time array
    time_array = np.arange(start_idx, end_idx) * time_base

    # Plot channels from bottom to top
    y_offset = 0.0
    channel_positions: dict[str, float] = {}

    # Handle bus grouping
    if group_buses:
        for bus_name, bus_channels in group_buses.items():
            # Combine bus channels into values
            bus_values = _combine_bus_channels(
                {name: channels[name] for name in bus_channels}, start_idx, end_idx
            )

            # Plot bus as combined signal with hex labels
            _plot_bus_signal(ax, time_array, bus_values, y_offset, bus_name, show_hex=show_hex)
            channel_positions[bus_name] = y_offset
            y_offset += 1.5  # Extra spacing for buses

            # Remove individual channels from main list
            for ch_name in bus_channels:
                channels.pop(ch_name, None)
    else:
        group_buses = {}

    # Plot remaining individual channels
    for ch_name, trace in channels.items():
        # Extract data in window
        data = np.asarray(trace.data[start_idx:end_idx])

        # Convert to digital if needed
        if hasattr(data, "dtype") and data.dtype == bool:
            digital_data = data.astype(float)
        else:
            # Threshold analog signal
            threshold = (np.max(data) + np.min(data)) / 2
            digital_data = (data >= threshold).astype(float)

        # Plot digital waveform
        _plot_digital_waveform(ax, time_array, digital_data, y_offset, ch_name)
        channel_positions[ch_name] = y_offset
        y_offset += 1

    # Style the plot
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Channel", fontsize=12, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_yticks([])

    # Format time axis
    _format_time_axis(ax, t_start, t_end)

    # Add cursors if requested
    if show_cursors:
        _add_timing_cursors(ax, t_start, t_end, y_offset)

    plt.tight_layout()
    return fig, ax


def plot_timing_diagram_with_annotations(
    signals: dict[str, WaveformTrace | DigitalTrace],
    *,
    timing_params: dict[str, tuple[float, float, str]] | None = None,
    title: str | None = None,
    reference_edges: dict[str, str] | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, Axes]:
    """Plot timing diagram with measurement annotations.

    Args:
        signals: Dictionary mapping signal names to traces.
        timing_params: Dict of {name: (start_time, end_time, label)}.
        title: Optional plot title.
        reference_edges: Dict mapping signal names to edge types.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> signals = {"CLK": clk, "DATA": data}
        >>> timing_params = {"t_su": (10e-9, 40e-9, "Setup = 30ns")}
        >>> fig, ax = plot_timing_diagram_with_annotations(signals, timing_params=timing_params)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot signals
    y_offset = 0
    signal_positions = {}

    for sig_name, trace in signals.items():
        # Get time array
        time_base = trace.metadata.time_base
        time_array = np.arange(len(trace.data)) * time_base
        data = np.asarray(trace.data)

        # Convert to digital
        if data.dtype == bool:
            digital_data = data.astype(float)
        else:
            threshold = (np.max(data) + np.min(data)) / 2
            digital_data = (data >= threshold).astype(float)

        # Offset for stacked view
        plot_data = digital_data + y_offset

        # Plot with nice edges
        ax.plot(time_array, plot_data, linewidth=2, color="royalblue")
        ax.fill_between(time_array, y_offset, plot_data, alpha=0.2, color="royalblue")

        # Add label
        ax.text(
            -time_array[-1] * 0.02,
            y_offset + 0.5,
            sig_name,
            ha="right",
            va="center",
            fontweight="bold",
        )

        signal_positions[sig_name] = y_offset
        y_offset += 2

    # Add timing annotations
    if timing_params:
        for t_start, t_end, label in timing_params.values():
            _add_timing_annotation(ax, t_start, t_end, y_offset - 1, label)

    # Style
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yticks([])

    plt.tight_layout()
    return fig, ax


def plot_ic_timing_validation(
    signals: dict[str, WaveformTrace | DigitalTrace],
    ic_name: str,
    measured_timings: dict[str, float],
    *,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[Figure, Axes]:
    """Plot timing diagram with IC specification overlay.

    Args:
        signals: Dictionary of signals.
        ic_name: IC part number (e.g., "74LS74").
        measured_timings: Measured timing parameters.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> signals = {"CLK": clk, "D": data, "Q": output}
        >>> measured = {"t_su": 25e-9, "t_h": 5e-9, "t_pd": 40e-9}
        >>> fig, ax = plot_ic_timing_validation(signals, "74LS74", measured)
    """
    from oscura.analyzers.digital.ic_database import validate_ic_timing

    # Validate timings
    validation = validate_ic_timing(ic_name, measured_timings)

    # Plot timing diagram with custom figsize
    fig, ax = plot_timing_diagram_with_annotations(
        signals, title=f"{ic_name} Timing Validation", figsize=figsize
    )

    # Add validation results as text box
    results_text = _format_validation_results(validation)
    ax.text(
        0.98,
        0.98,
        results_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        family="monospace",
    )

    return fig, ax


def plot_multi_ic_timing_path(
    ic_chain: list[tuple[str, dict[str, WaveformTrace | DigitalTrace]]],
    *,
    title: str = "Multi-IC Timing Path Analysis",
    figsize: tuple[float, float] = (16, 10),
) -> tuple[Figure, Axes]:
    """Plot timing analysis for cascaded ICs.

    Args:
        ic_chain: List of (ic_name, signals) tuples for each IC in chain.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).

    Example:
        >>> chain = [
        ...     ("74LS74", {"CLK": clk1, "Q": q1}),
        ...     ("74LS00", {"A": q1, "Y": y1}),
        ...     ("74LS74", {"D": y1, "Q": q2}),
        ... ]
        >>> fig, ax = plot_multi_ic_timing_path(chain)
    """
    fig, ax = plt.subplots(figsize=figsize)

    y_offset = 0.0

    for ic_name, signals in ic_chain:
        # Plot this IC's signals
        for sig_name, trace in signals.items():
            time_base = trace.metadata.time_base
            time_array = np.arange(len(trace.data)) * time_base
            data = np.asarray(trace.data)

            # Convert to digital
            if data.dtype == bool:
                digital_data = data.astype(float)
            else:
                threshold = (np.max(data) + np.min(data)) / 2
                digital_data = (data >= threshold).astype(float)

            # Plot
            plot_data = digital_data + y_offset
            ax.plot(time_array, plot_data, linewidth=2, label=f"{ic_name}.{sig_name}")

            y_offset += 1.5

        # Add IC boundary
        ax.axhline(y_offset, color="gray", linestyle="--", alpha=0.5)
        y_offset += 0.5

    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])

    plt.tight_layout()
    return fig, ax


def plot_bus_eye_diagram(
    bus_traces: list[WaveformTrace],
    *,
    symbol_period: float,
    num_symbols: int = 100,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot eye diagram for bus signal quality analysis.

    Args:
        bus_traces: List of traces (one per bit).
        symbol_period: Symbol period in seconds.
        num_symbols: Number of symbols to overlay.
        title: Optional plot title.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).
    """
    fig, axes = plt.subplots(len(bus_traces), 1, figsize=figsize, sharex=True)
    if len(bus_traces) == 1:
        axes = [axes]

    for idx, trace in enumerate(bus_traces):
        ax = axes[idx]

        # Extract eye diagram data
        sample_rate = trace.metadata.sample_rate
        samples_per_symbol = int(symbol_period * sample_rate)

        data = np.asarray(trace.data)

        # Overlay symbols
        for i in range(num_symbols):
            start = i * samples_per_symbol
            end = start + samples_per_symbol * 2  # Two symbol periods

            if end > len(data):
                break

            segment = data[start:end]
            time_segment = np.arange(len(segment)) / sample_rate

            ax.plot(time_segment * 1e9, segment, alpha=0.1, color="blue")

        ax.set_ylabel(f"Bit {idx}", fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ns)", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig, axes


# =============================================================================
# Helper Functions
# =============================================================================


def _plot_digital_waveform(
    ax: Axes, time: NDArray[np.float64], data: NDArray[np.float64], y_offset: float, label: str
) -> None:
    """Plot a single digital waveform."""
    # Offset data
    plot_data = data + y_offset

    # Plot with thick lines
    ax.plot(time, plot_data, linewidth=2, color="royalblue", drawstyle="steps-post")

    # Fill area
    ax.fill_between(time, y_offset, plot_data, alpha=0.2, color="royalblue", step="post")

    # Add label
    ax.text(
        time[0] - (time[-1] - time[0]) * 0.02,
        y_offset + 0.5,
        label,
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=10,
    )


def _plot_bus_signal(
    ax: Axes,
    time: NDArray[np.float64],
    values: NDArray[np.uint32],
    y_offset: float,
    label: str,
    *,
    show_hex: bool = True,
) -> None:
    """Plot a bus signal with hex value labels."""
    # Plot as multi-level signal
    plot_data = values.astype(float) / np.max(values) if np.max(values) > 0 else values
    plot_data = plot_data + y_offset

    ax.plot(time, plot_data, linewidth=2, color="green", drawstyle="steps-post")
    ax.fill_between(time, y_offset, plot_data, alpha=0.2, color="green", step="post")

    # Add bus label
    ax.text(
        time[0] - (time[-1] - time[0]) * 0.02,
        y_offset + 0.5,
        label,
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="green",
    )

    # Add hex values at transitions
    if show_hex:
        transitions = np.where(np.diff(values) != 0)[0]
        for trans_idx in transitions[:10]:  # Limit to first 10 transitions
            value = values[trans_idx + 1]
            ax.text(
                time[trans_idx + 1],
                y_offset + 0.5,
                f"0x{value:02X}",
                ha="left",
                va="bottom",
                fontsize=8,
                color="green",
            )


def _combine_bus_channels(
    bus_channels: dict[str, WaveformTrace | DigitalTrace], start_idx: int, end_idx: int
) -> NDArray[np.uint32]:
    """Combine individual bus lines into values."""
    # Sort channels by name (assume D0, D1, D2, etc.)
    sorted_channels = sorted(bus_channels.items(), key=lambda x: x[0])

    # Initialize result
    sorted_channels[0][1]
    num_samples = end_idx - start_idx
    result = np.zeros(num_samples, dtype=np.uint32)

    # Combine bits
    for bit_idx, (_ch_name, trace) in enumerate(sorted_channels):
        data = np.asarray(trace.data[start_idx:end_idx])

        # Convert to digital
        if data.dtype == bool:
            digital_data = data.astype(np.uint32)
        else:
            threshold = (np.max(data) + np.min(data)) / 2
            digital_data = (data >= threshold).astype(np.uint32)

        result |= (digital_data << bit_idx).astype(np.uint32)

    return result


def _format_time_axis(ax: Axes, t_start: float, t_end: float) -> None:
    """Format time axis with appropriate units."""
    duration = t_end - t_start

    # Get current tick locations
    ticks = ax.get_xticks()

    if duration < 1e-6:  # Nanoseconds
        ax.set_xlabel("Time (ns)", fontweight="bold")
        ax.set_xticks(ticks)  # Set ticks before labels
        ax.set_xticklabels([f"{t * 1e9:.1f}" for t in ticks])
    elif duration < 1e-3:  # Microseconds
        ax.set_xlabel("Time (μs)", fontweight="bold")
        ax.set_xticks(ticks)  # Set ticks before labels
        ax.set_xticklabels([f"{t * 1e6:.1f}" for t in ticks])
    elif duration < 1:  # Milliseconds
        ax.set_xlabel("Time (ms)", fontweight="bold")
        ax.set_xticks(ticks)  # Set ticks before labels
        ax.set_xticklabels([f"{t * 1e3:.1f}" for t in ticks])
    else:  # Seconds
        ax.set_xlabel("Time (s)", fontweight="bold")


def _add_timing_cursors(ax: Axes, t_start: float, t_end: float, y_max: float) -> None:
    """Add timing measurement cursors."""
    # Add two cursors at 1/4 and 3/4 of time range
    cursor1_time = t_start + (t_end - t_start) * 0.25
    cursor2_time = t_start + (t_end - t_start) * 0.75

    ax.axvline(cursor1_time, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axvline(cursor2_time, color="red", linestyle="--", alpha=0.7, linewidth=1.5)

    # Add delta time label
    delta_t = cursor2_time - cursor1_time
    mid_y = y_max / 2

    ax.annotate(
        "",
        xy=(cursor2_time, mid_y),
        xytext=(cursor1_time, mid_y),
        arrowprops={"arrowstyle": "<->", "color": "red", "lw": 2},
    )

    ax.text(
        (cursor1_time + cursor2_time) / 2,
        mid_y + 0.3,
        f"Δt = {delta_t * 1e9:.1f} ns",
        ha="center",
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.8},
    )


def _add_timing_annotation(
    ax: Axes, t_start: float, t_end: float, y_pos: float, label: str
) -> None:
    """Add timing measurement annotation."""
    ax.annotate(
        "",
        xy=(t_end, y_pos),
        xytext=(t_start, y_pos),
        arrowprops={"arrowstyle": "<->", "color": "red", "lw": 2},
    )

    ax.text(
        (t_start + t_end) / 2,
        y_pos + 0.3,
        label,
        ha="center",
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.8},
    )


def _format_validation_results(validation: dict[str, dict[str, Any]]) -> str:
    """Format IC timing validation results."""
    lines = ["Timing Validation:\n"]

    for param, result in validation.items():
        passes = result.get("passes")
        if passes is None:
            continue

        measured = result["measured"]
        spec = result["spec"]
        error = result["error"]

        status = "✓ PASS" if passes else "✗ FAIL"
        lines.append(
            f"{param}: {status}\n  Measured: {measured * 1e9:.1f}ns\n  Spec: {spec * 1e9:.1f}ns\n  Error: {error * 100:.1f}%\n"
        )

    return "".join(lines)


def generate_all_vintage_logic_plots(
    result: Any,
    traces: dict[str, WaveformTrace | DigitalTrace],
    *,
    output_dir: str | Path | None = None,
    save_formats: list[str] | None = None,
) -> dict[str, tuple[Figure, Axes] | Figure]:
    """Generate complete visualization suite for vintage logic analysis.

    Creates all relevant plots based on analysis results. Optionally saves
    figures to disk in multiple formats.

    Args:
        result: VintageLogicAnalysisResult object.
        traces: Dictionary of channel names to traces.
        output_dir: If provided, saves all figures to this directory.
        save_formats: Formats to save ("png", "svg", "pdf"). Default: ["png"].

    Returns:
        Dictionary mapping plot names to Figure/Axes tuples.

    Example:
        >>> from oscura.analyzers.digital.vintage import analyze_vintage_logic
        >>> result = analyze_vintage_logic(traces)
        >>> plots = generate_all_vintage_logic_plots(result, traces, output_dir="./plots")
        >>> # plots = {
        >>> #     "logic_analyzer": (fig, ax),
        >>> #     "timing_validation": (fig, ax),
        >>> #     ...
        >>> # }
    """
    from oscura.visualization.figure_manager import FigureManager

    plots: dict[str, tuple[Figure, Axes] | Figure] = {}

    # Initialize figure manager if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig_manager = FigureManager(output_path)
        if save_formats is None:
            save_formats = ["png"]
    else:
        fig_manager = None
        save_formats = []

    # 1. Logic analyzer view
    try:
        fig, ax = plot_logic_analyzer_view(
            traces,
            title=f"Logic Analyzer View - {result.detected_family}",
            show_cursors=True,
        )
        plots["logic_analyzer"] = (fig, ax)
        if fig_manager:
            fig_manager.save_figure(fig, "logic_analyzer", formats=save_formats)
    except Exception:
        pass  # Skip if plot fails

    # 2. IC timing validation plots for each identified IC
    for idx, ic_result in enumerate(result.identified_ics):
        try:
            # Create signals dictionary for timing validation
            fig, ax = plot_ic_timing_validation(
                signals=traces,
                ic_name=ic_result.ic_name,
                measured_timings=ic_result.timing_params,
            )
            plot_name = f"timing_validation_{ic_result.ic_name}_{idx}"
            plots[plot_name] = (fig, ax)
            if fig_manager:
                fig_manager.save_figure(fig, plot_name, formats=save_formats)
        except Exception:
            pass  # Skip if plot fails

    # 3. Multi-IC timing path visualization
    if result.timing_paths:
        for idx, path_result in enumerate(result.timing_paths):
            try:
                fig, ax = plot_multi_ic_timing_path(path_result)
                plot_name = f"timing_path_{idx}"
                plots[plot_name] = (fig, ax)
                if fig_manager:
                    fig_manager.save_figure(fig, plot_name, formats=save_formats)
            except Exception:
                pass  # Skip if plot fails

    # 4. Timing diagram with annotations (for first 2-3 channels)
    if len(traces) >= 2:
        try:
            # Select first 2-3 channels
            selected_traces = dict(list(traces.items())[: min(3, len(traces))])

            # Create timing annotations from measurements
            timing_params: dict[str, tuple[float, float, str]] = {}
            for key, value in result.timing_measurements.items():
                # Extract channel names from key like "CLK→DATA_t_pd"
                if "→" in key:
                    label = key.split("_")[-1]  # Get timing parameter
                    timing_params[label] = (0.0, float(value), label)

            fig, ax = plot_timing_diagram_with_annotations(
                selected_traces,
                timing_params=timing_params or None,
                title="Timing Diagram with Measurements",
            )
            plots["timing_diagram"] = (fig, ax)
            if fig_manager:
                fig_manager.save_figure(fig, "timing_diagram", formats=save_formats)
        except Exception:
            pass  # Skip if plot fails

    return plots


__all__ = [
    "generate_all_vintage_logic_plots",
    "plot_bus_eye_diagram",
    "plot_ic_timing_validation",
    "plot_logic_analyzer_view",
    "plot_multi_ic_timing_path",
    "plot_timing_diagram_with_annotations",
]
