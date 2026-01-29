"""Power profile visualization.


This module provides comprehensive power visualization including
time-domain plots, energy accumulation, and multi-channel views.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _normalize_power_channels(
    power: NDArray[np.float64] | dict[str, NDArray[np.float64]],
) -> tuple[dict[str, NDArray[np.float64]], bool]:
    """Normalize power input into channels dictionary.

    Args:
        power: Single array or dict of arrays.

    Returns:
        Tuple of (channels dict, is_multi boolean).

    Example:
        >>> channels, is_multi = _normalize_power_channels(np.array([1, 2, 3]))
        >>> channels
        {'Power': array([1, 2, 3])}
    """
    if isinstance(power, dict):
        return power, True
    return {"Power": np.asarray(power, dtype=np.float64)}, False


def _validate_and_create_time_array(
    time_array: NDArray[np.float64] | None,
    sample_rate: float | None,
    trace_length: int,
) -> NDArray[np.float64]:
    """Validate inputs and create time array.

    Args:
        time_array: Optional explicit time array.
        sample_rate: Optional sample rate in Hz.
        trace_length: Length of power trace.

    Returns:
        Validated time array.

    Raises:
        ValueError: If neither time_array nor sample_rate provided.
        ValueError: If time_array length doesn't match trace.

    Example:
        >>> time = _validate_and_create_time_array(None, 1000.0, 100)
        >>> len(time)
        100
    """
    if time_array is None and sample_rate is None:
        raise ValueError("Either time_array or sample_rate must be provided")

    if time_array is None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when time_array is not provided")
        return np.arange(trace_length) / sample_rate

    time_array_validated = np.asarray(time_array, dtype=np.float64)
    if len(time_array_validated) != trace_length:
        raise ValueError(
            f"time_array length {len(time_array_validated)} doesn't match "
            f"power trace length {trace_length}"
        )
    return time_array_validated


def _compute_time_scale(time_array: NDArray[np.float64]) -> tuple[NDArray[np.float64], str]:
    """Compute time scaling factor and units.

    Args:
        time_array: Time array in seconds.

    Returns:
        Tuple of (scaled time array, unit string).

    Example:
        >>> time = np.array([0, 1e-6, 2e-6])
        >>> scaled, unit = _compute_time_scale(time)
        >>> unit
        'µs'
    """
    time_max = time_array[-1]

    if time_max < 1e-6:
        return time_array * 1e9, "ns"
    if time_max < 1e-3:
        return time_array * 1e6, "µs"
    if time_max < 1:
        return time_array * 1e3, "ms"
    return time_array, "s"


def _create_figure_layout(
    is_multi: bool,
    layout: str,
    n_channels: int,
    show_energy: bool,
    figsize: tuple[float, float],
) -> tuple[Figure, list[Axes]]:
    """Create figure and axes layout.

    Args:
        is_multi: Multiple channels flag.
        layout: 'stacked' or 'overlay'.
        n_channels: Number of channels.
        show_energy: Show energy plot flag.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes list).

    Example:
        >>> fig, axes = _create_figure_layout(True, 'stacked', 2, True, (12, 6))
        >>> len(axes)
        3
    """
    if is_multi and layout == "stacked":
        n_plots = n_channels + (1 if show_energy else 0)
        fig, axes_obj = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            return fig, [axes_obj]
        return fig, list(axes_obj)

    fig, ax_power = plt.subplots(figsize=figsize)
    return fig, [ax_power]


def _plot_stacked_channels(
    axes: list[Axes],
    channels: dict[str, NDArray[np.float64]],
    time_scaled: NDArray[np.float64],
    time_unit: str,
    statistics: dict[str, float] | None,
    show_average: bool,
    show_peak: bool,
    show_energy: bool,
    sample_rate: float | None,
) -> None:
    """Plot channels in stacked layout.

    Args:
        axes: List of axes objects.
        channels: Channel data dictionary.
        time_scaled: Scaled time array.
        time_unit: Time unit string.
        statistics: Optional statistics dictionary.
        show_average: Show average line flag.
        show_peak: Show peak marker flag.
        show_energy: Show energy plot flag.
        sample_rate: Sample rate in Hz.

    Example:
        >>> _plot_stacked_channels(axes, channels, time, 'ms', None, True, True, True, 1e6)
    """
    for idx, (name, trace) in enumerate(channels.items()):
        ax = axes[idx]
        ax.plot(time_scaled, trace * 1e3, linewidth=0.8, label=name)

        # Compute or use statistics
        if statistics is None or name not in statistics:
            avg = np.mean(trace)
            peak = np.max(trace)
        else:
            avg = statistics[name]["average"]  # type: ignore[index]
            peak = statistics[name]["peak"]  # type: ignore[index]

        # Annotations
        if show_average:
            ax.axhline(
                float(avg * 1e3),
                color="r",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Avg: {avg * 1e3:.2f} mW",
            )

        if show_peak:
            peak_idx = np.argmax(trace)
            ax.plot(
                time_scaled[peak_idx],
                peak * 1e3,
                "rv",
                markersize=8,
                label=f"Peak: {peak * 1e3:.2f} mW",
            )

        ax.set_ylabel(f"{name}\n(mW)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Energy accumulation plot
    if show_energy:
        ax_energy = axes[-1]
        for name, trace in channels.items():
            if sample_rate is not None:
                energy = np.cumsum(trace) / sample_rate * 1e6  # µJ
                ax_energy.plot(time_scaled, energy, linewidth=0.8, label=name)

        ax_energy.set_ylabel("Cumulative\nEnergy (µJ)")
        ax_energy.set_xlabel(f"Time ({time_unit})")
        ax_energy.legend(loc="upper left", fontsize=8)
        ax_energy.grid(True, alpha=0.3)


def _plot_overlay_channels(
    ax: Axes,
    channels: dict[str, NDArray[np.float64]],
    time_scaled: NDArray[np.float64],
    time_unit: str,
    statistics: dict[str, float] | None,
    show_average: bool,
    show_peak: bool,
    show_energy: bool,
    sample_rate: float | None,
) -> None:
    """Plot channels in overlay layout.

    Args:
        ax: Axes object.
        channels: Channel data dictionary.
        time_scaled: Scaled time array.
        time_unit: Time unit string.
        statistics: Optional statistics dictionary.
        show_average: Show average line flag.
        show_peak: Show peak marker flag.
        show_energy: Show energy plot flag.
        sample_rate: Sample rate in Hz.

    Example:
        >>> _plot_overlay_channels(ax, channels, time, 'ms', None, True, True, True, 1e6)
    """
    for name, trace in channels.items():
        ax.plot(time_scaled, trace * 1e3, linewidth=0.8, label=name)

    # Statistics for first channel (or combined if overlay)
    first_trace = next(iter(channels.values()))
    if statistics is None:
        avg_val = float(np.mean(first_trace))
        peak_val = float(np.max(first_trace))
        total_energy_val: float | None = (
            float(np.sum(first_trace) / sample_rate) if sample_rate else None
        )
    else:
        avg_val = float(statistics.get("average", float(np.mean(first_trace))))
        peak_val = float(statistics.get("peak", float(np.max(first_trace))))
        total_energy_val = statistics.get("energy", None)

    # Annotations
    if show_average:
        ax.axhline(
            avg_val * 1e3,
            color="r",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Avg: {avg_val * 1e3:.2f} mW",
        )

    if show_peak:
        peak_idx = np.argmax(first_trace)
        ax.plot(
            time_scaled[peak_idx],
            peak_val * 1e3,
            "rv",
            markersize=8,
            label=f"Peak: {peak_val * 1e3:.2f} mW",
        )

    ax.set_ylabel("Power (mW)")
    ax.set_xlabel(f"Time ({time_unit})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Energy overlay on secondary y-axis
    if show_energy and sample_rate is not None:
        ax2 = ax.twinx()
        energy = np.cumsum(first_trace) / sample_rate * 1e6  # µJ
        ax2.plot(time_scaled, energy, "g--", linewidth=1.5, alpha=0.6)
        ax2.set_ylabel("Cumulative Energy (µJ)", color="g")
        ax2.tick_params(axis="y", labelcolor="g")

        if total_energy_val is not None:
            ax2.text(
                0.98,
                0.98,
                f"Total: {total_energy_val * 1e6:.2f} µJ",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )


def _prepare_power_plot_data(
    power: NDArray[np.float64] | dict[str, NDArray[np.float64]],
    time_array: NDArray[np.float64] | None,
    sample_rate: float | None,
) -> tuple[dict[str, NDArray[np.float64]], bool, NDArray[np.float64], str]:
    """Prepare data for power plot.

    Args:
        power: Power data (array or dict).
        time_array: Optional time array.
        sample_rate: Optional sample rate.

    Returns:
        Tuple of (channels, is_multi, time_scaled, time_unit).
    """
    channels, is_multi = _normalize_power_channels(power)
    time_array_validated = _validate_and_create_time_array(
        time_array, sample_rate, len(next(iter(channels.values())))
    )
    time_scaled, time_unit = _compute_time_scale(time_array_validated)
    return channels, is_multi, time_scaled, time_unit


def _render_power_plots(
    axes: list[Axes],
    channels: dict[str, NDArray[np.float64]],
    time_scaled: NDArray[np.float64],
    time_unit: str,
    is_multi: bool,
    layout: str,
    statistics: dict[str, float] | None,
    show_average: bool,
    show_peak: bool,
    show_energy: bool,
    sample_rate: float | None,
) -> None:
    """Render power plots based on layout.

    Args:
        axes: List of axes.
        channels: Channel data.
        time_scaled: Scaled time array.
        time_unit: Time unit string.
        is_multi: Multiple channels flag.
        layout: Layout type.
        statistics: Optional statistics.
        show_average: Show average line.
        show_peak: Show peak marker.
        show_energy: Show energy plot.
        sample_rate: Sample rate.
    """
    if is_multi and layout == "stacked":
        _plot_stacked_channels(
            axes,
            channels,
            time_scaled,
            time_unit,
            statistics,
            show_average,
            show_peak,
            show_energy,
            sample_rate,
        )
    else:
        _plot_overlay_channels(
            axes[0],
            channels,
            time_scaled,
            time_unit,
            statistics,
            show_average,
            show_peak,
            show_energy,
            sample_rate,
        )


def _finalize_plot(
    fig: Figure,
    title: str | None,
    is_multi: bool,
    save_path: str | Path | None,
    show: bool,
) -> None:
    """Finalize plot with title, layout, save, and display.

    Args:
        fig: Figure object.
        title: Plot title.
        is_multi: Multiple channels flag.
        save_path: Optional save path.
        show: Display flag.

    Example:
        >>> _finalize_plot(fig, "Power Profile", False, None, True)
    """
    if title is None:
        title = "Power Profile" + (" (Multi-Channel)" if is_multi else "")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()


def plot_power_profile(
    power: NDArray[np.float64] | dict[str, NDArray[np.float64]],
    *,
    sample_rate: float | None = None,
    time_array: NDArray[np.float64] | None = None,
    statistics: dict[str, float] | None = None,
    show_average: bool = True,
    show_peak: bool = True,
    show_energy: bool = True,
    multi_channel_layout: str = "stacked",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Generate power profile plot with annotations.

    Time-domain power visualization with average/peak markers
    and optional energy accumulation overlay. Supports multi-channel stacked view.

    Args:
        power: Power trace in watts. Can be:
            - Array: Single channel power trace
            - Dict: Multiple channels {name: trace}
        sample_rate: Sample rate in Hz (required if time_array not provided)
        time_array: Optional explicit time array (overrides sample_rate)
        statistics: Optional pre-computed statistics dict from power_statistics()
            If provided, used for annotations. Otherwise computed automatically.
        show_average: Show average power horizontal line (default: True)
        show_peak: Show peak power marker (default: True)
        show_energy: Show cumulative energy overlay (default: True)
        multi_channel_layout: Layout for multiple channels:
            - 'stacked': Separate subplots stacked vertically (default)
            - 'overlay': All channels on same plot
        title: Plot title (default: "Power Profile")
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save figure
        show: Display the figure (default: True)

    Returns:
        Matplotlib Figure object for further customization

    Raises:
        ValueError: If neither sample_rate nor time_array provided
        ValueError: If time_array length doesn't match power trace

    Examples:
        >>> import numpy as np
        >>> power = np.random.rand(1000) * 0.5 + 0.3
        >>> fig = plot_power_profile(power, sample_rate=1e6, title="Power")

        >>> from oscura.analyzers.power import power_statistics
        >>> stats = power_statistics(power, sample_rate=1e6)
        >>> fig = plot_power_profile(power, sample_rate=1e6, statistics=stats)

        >>> power_channels = {
        ...     'VDD_CORE': np.random.rand(1000) * 0.5,
        ...     'VDD_IO': np.random.rand(1000) * 0.3,
        ... }
        >>> fig = plot_power_profile(power_channels, sample_rate=1e6)

    Notes:
        - Energy accumulation computed via cumulative sum
        - Multiple channels can be overlaid or stacked
        - Annotations include average, peak, and total energy
        - Time axis auto-scaled to appropriate units (ns/µs/ms/s)

    References:
        PWR-004: Power Profile Visualization
    """
    channels, is_multi, time_scaled, time_unit = _prepare_power_plot_data(
        power, time_array, sample_rate
    )
    fig, axes = _create_figure_layout(
        is_multi, multi_channel_layout, len(channels), show_energy, figsize
    )
    _render_power_plots(
        axes,
        channels,
        time_scaled,
        time_unit,
        is_multi,
        multi_channel_layout,
        statistics,
        show_average,
        show_peak,
        show_energy,
        sample_rate,
    )
    _finalize_plot(fig, title, is_multi, save_path, show)
    return fig
