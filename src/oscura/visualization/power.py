"""Power profile visualization.


This module provides comprehensive power visualization including
time-domain plots, energy accumulation, and multi-channel views.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


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

    : Time-domain power visualization with average/peak markers
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
        >>> # Simple power profile plot
        >>> import numpy as np
        >>> power = np.random.rand(1000) * 0.5 + 0.3  # 300-800 mW
        >>> fig = plot_power_profile(
        ...     power,
        ...     sample_rate=1e6,
        ...     title="Device Power Consumption"
        ... )

        >>> # With pre-computed statistics
        >>> from oscura.analyzers.power import power_statistics
        >>> stats = power_statistics(power, sample_rate=1e6)
        >>> fig = plot_power_profile(
        ...     power,
        ...     sample_rate=1e6,
        ...     statistics=stats,
        ...     show_energy=True
        ... )

        >>> # Multi-channel stacked view
        >>> power_channels = {
        ...     'VDD_CORE': np.random.rand(1000) * 0.5,
        ...     'VDD_IO': np.random.rand(1000) * 0.3,
        ...     'VDD_ANALOG': np.random.rand(1000) * 0.2,
        ... }
        >>> fig = plot_power_profile(
        ...     power_channels,
        ...     sample_rate=1e6,
        ...     multi_channel_layout='stacked'
        ... )

    Notes:
        - Energy accumulation computed via cumulative sum
        - Multiple channels can be overlaid or stacked
        - Annotations include average, peak, and total energy
        - Time axis auto-scaled to appropriate units (ns/µs/ms/s)

    References:
        PWR-004: Power Profile Visualization
    """
    # Handle multi-channel input
    if isinstance(power, dict):
        channels = power
        is_multi = True
    else:
        channels = {"Power": np.asarray(power, dtype=np.float64)}
        is_multi = False

    # Validate inputs
    if time_array is None and sample_rate is None:
        raise ValueError("Either time_array or sample_rate must be provided")

    # Generate time array
    first_trace = next(iter(channels.values()))
    if time_array is None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when time_array is not provided")
        time_array = np.arange(len(first_trace)) / sample_rate
    else:
        time_array = np.asarray(time_array, dtype=np.float64)
        if len(time_array) != len(first_trace):
            raise ValueError(
                f"time_array length {len(time_array)} doesn't match "
                f"power trace length {len(first_trace)}"
            )

    # Determine time scale and units
    time_max = time_array[-1]
    if time_max < 1e-6:
        time_scale = 1e9
        time_unit = "ns"
    elif time_max < 1e-3:
        time_scale = 1e6
        time_unit = "µs"
    elif time_max < 1:
        time_scale = 1e3
        time_unit = "ms"
    else:
        time_scale = 1
        time_unit = "s"

    time_scaled = time_array * time_scale

    # Create figure
    if is_multi and multi_channel_layout == "stacked":
        n_channels = len(channels)
        n_plots = n_channels + (1 if show_energy else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            axes = [axes]
    else:
        fig, ax_power = plt.subplots(figsize=figsize)
        axes = [ax_power]

    # Plot each channel
    if is_multi and multi_channel_layout == "stacked":
        # Stacked layout - one subplot per channel
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
                    avg * 1e3,
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

    else:
        # Overlay layout or single channel
        ax = axes[0]

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

    # Set title
    if title is None:
        title = "Power Profile" + (" (Multi-Channel)" if is_multi else "")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig
