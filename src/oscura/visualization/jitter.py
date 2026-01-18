"""Jitter Analysis Visualization Functions.

This module provides visualization functions for jitter analysis including
TIE histograms, bathtub curves, DDJ/DCD plots, and jitter trend analysis.

Example:
    >>> from oscura.visualization.jitter import plot_tie_histogram, plot_bathtub_full
    >>> fig = plot_tie_histogram(tie_data)
    >>> fig = plot_bathtub_full(bathtub_result)

References:
    - IEEE 802.3: Jitter measurement specifications
    - JEDEC JESD65B: High-Speed Interface Measurements
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

try:
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    HAS_MATPLOTLIB = True
    HAS_SCIPY = True
except ImportError:
    HAS_MATPLOTLIB = False
    HAS_SCIPY = False

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

__all__ = [
    "plot_bathtub_full",
    "plot_dcd",
    "plot_ddj",
    "plot_jitter_trend",
    "plot_tie_histogram",
]


def plot_tie_histogram(
    tie_data: NDArray[np.floating[Any]],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    time_unit: str = "auto",
    bins: int | str = "auto",
    show_gaussian_fit: bool = True,
    show_statistics: bool = True,
    show_rj_dj: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot Time Interval Error (TIE) histogram with statistical analysis.

    Creates a histogram of TIE values with optional Gaussian fit overlay
    and RJ/DJ decomposition indicators.

    Args:
        tie_data: Array of TIE values in seconds.
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size in inches.
        title: Plot title.
        time_unit: Time unit ("s", "ms", "us", "ns", "ps", "auto").
        bins: Number of bins or "auto" for automatic selection.
        show_gaussian_fit: Overlay Gaussian fit for RJ estimation.
        show_statistics: Show statistics box.
        show_rj_dj: Show RJ/DJ separation indicators.
        show: Display plot interactively.
        save_path: Save plot to file.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> tie = np.random.randn(10000) * 2e-12  # 2 ps RMS jitter
        >>> fig = plot_tie_histogram(tie, time_unit="ps")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Select time unit
    if time_unit == "auto":
        max_tie = np.max(np.abs(tie_data))
        if max_tie < 1e-12:
            time_unit = "fs"
            time_mult = 1e15
        elif max_tie < 1e-9:
            time_unit = "ps"
            time_mult = 1e12
        elif max_tie < 1e-6:
            time_unit = "ns"
            time_mult = 1e9
        else:
            time_unit = "us"
            time_mult = 1e6
    else:
        time_mult = {
            "s": 1,
            "ms": 1e3,
            "us": 1e6,
            "ns": 1e9,
            "ps": 1e12,
            "fs": 1e15,
        }.get(time_unit, 1e12)

    tie_scaled = tie_data * time_mult

    # Calculate statistics
    mean_val = np.mean(tie_scaled)
    std_val = np.std(tie_scaled)
    pp_val = np.ptp(tie_scaled)
    rms_val = np.sqrt(np.mean(tie_scaled**2))

    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        tie_scaled,
        bins=bins,
        density=True,
        color="#3498DB",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Gaussian fit overlay
    if show_gaussian_fit and HAS_SCIPY:
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
        y_fit = norm.pdf(x_fit, mean_val, std_val)
        ax.plot(
            x_fit, y_fit, "r-", linewidth=2, label=f"Gaussian Fit (sigma={std_val:.2f} {time_unit})"
        )

    # RJ/DJ indicators
    if show_rj_dj:
        # Mark ±3sigma region (RJ contribution)
        ax.axvline(
            mean_val - 3 * std_val, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7
        )
        ax.axvline(
            mean_val + 3 * std_val, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7
        )

        # Shade RJ region
        ax.axvspan(
            mean_val - 3 * std_val,
            mean_val + 3 * std_val,
            alpha=0.1,
            color="#E74C3C",
            label="±3sigma (99.7% RJ)",
        )

    # Statistics box
    if show_statistics:
        stats_text = (
            f"Mean: {mean_val:.2f} {time_unit}\n"
            f"RMS: {rms_val:.2f} {time_unit}\n"
            f"Std Dev: {std_val:.2f} {time_unit}\n"
            f"Peak-Peak: {pp_val:.2f} {time_unit}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9},
            fontfamily="monospace",
        )

    # Labels
    ax.set_xlabel(f"TIE ({time_unit})", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Time Interval Error Distribution", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_bathtub_full(
    positions: NDArray[np.floating[Any]],
    ber_left: NDArray[np.floating[Any]],
    ber_right: NDArray[np.floating[Any]],
    *,
    ber_total: NDArray[np.floating[Any]] | None = None,
    target_ber: float = 1e-12,
    eye_opening: float | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    show_target: bool = True,
    show_eye_opening: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot full bathtub curve with left/right BER and eye opening.

    Creates a bathtub curve showing bit error rate vs sampling position
    within the unit interval, with target BER marker and eye opening
    annotation.

    Args:
        positions: Sample positions in UI (0 to 1).
        ber_left: Left-side BER values.
        ber_right: Right-side BER values.
        ber_total: Total BER values (optional, computed if not provided).
        target_ber: Target BER for eye opening calculation.
        eye_opening: Pre-calculated eye opening in UI (optional).
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        show_target: Show target BER line.
        show_eye_opening: Annotate eye opening.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> pos = np.linspace(0, 1, 100)
        >>> ber_l = 0.5 * erfc((pos - 0) / 0.1 / np.sqrt(2))
        >>> ber_r = 0.5 * erfc((1 - pos) / 0.1 / np.sqrt(2))
        >>> fig = plot_bathtub_full(pos, ber_l, ber_r, target_ber=1e-12)
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

    # Compute total BER if not provided
    if ber_total is None:
        ber_total = ber_left + ber_right

    # Clip very small values for log plot
    ber_left_plot = np.clip(ber_left, 1e-18, 1)
    ber_right_plot = np.clip(ber_right, 1e-18, 1)
    ber_total_plot = np.clip(ber_total, 1e-18, 1)

    # Plot BER curves
    ax.semilogy(positions, ber_left_plot, "b-", linewidth=2, label="BER Left", alpha=0.8)
    ax.semilogy(positions, ber_right_plot, "r-", linewidth=2, label="BER Right", alpha=0.8)
    ax.semilogy(positions, ber_total_plot, "k-", linewidth=2.5, label="BER Total")

    # Target BER line
    if show_target:
        ax.axhline(
            target_ber,
            color="#27AE60",
            linestyle="--",
            linewidth=2,
            label=f"Target BER = {target_ber:.0e}",
        )

    # Eye opening annotation
    if show_eye_opening:
        # Find eye opening at target BER
        if eye_opening is None:
            # Find crossover points
            left_cross = np.where(ber_total_plot < target_ber)[0]
            if len(left_cross) > 0:
                left_edge = positions[left_cross[0]]
                right_edge = positions[left_cross[-1]]
                eye_opening = right_edge - left_edge
            else:
                eye_opening = 0

        if eye_opening > 0:
            # Draw eye opening bracket
            center = 0.5
            left_edge = center - eye_opening / 2
            right_edge = center + eye_opening / 2

            ax.annotate(
                "",
                xy=(right_edge, target_ber),
                xytext=(left_edge, target_ber),
                arrowprops={"arrowstyle": "<->", "color": "#27AE60", "lw": 2},
            )
            ax.text(
                center,
                target_ber * 0.1,
                f"Eye Opening: {eye_opening:.3f} UI",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#27AE60",
            )

    # Shading for bathtub
    ax.fill_between(positions, 1e-18, ber_total_plot, alpha=0.1, color="gray")

    # Labels
    ax.set_xlabel("Sample Position (UI)", fontsize=11)
    ax.set_ylabel("Bit Error Rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-15, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Bathtub Curve", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_ddj(
    patterns: list[str],
    jitter_values: NDArray[np.floating[Any]],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    time_unit: str = "ps",
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot Data-Dependent Jitter (DDJ) by bit pattern.

    Creates a bar chart showing jitter contribution for each bit pattern,
    useful for identifying pattern-dependent timing variations.

    Args:
        patterns: List of bit pattern strings (e.g., ["010", "011", "100"]).
        jitter_values: Jitter values for each pattern.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        time_unit: Time unit for display.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> patterns = ["000", "001", "010", "011", "100", "101", "110", "111"]
        >>> ddj = np.array([0, 2.1, -1.5, 0.5, 0.8, -0.3, 1.2, -0.8])  # ps
        >>> fig = plot_ddj(patterns, ddj, time_unit="ps")
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

    # Color bars based on sign
    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in jitter_values]

    # Bar chart
    x_pos = np.arange(len(patterns))
    ax.bar(x_pos, jitter_values, color=colors, edgecolor="black", linewidth=0.5)

    # Reference line at zero
    ax.axhline(0, color="gray", linestyle="-", linewidth=1)

    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(patterns, fontfamily="monospace", fontsize=10)
    ax.set_xlabel("Bit Pattern", fontsize=11)
    ax.set_ylabel(f"DDJ ({time_unit})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add DDJ pp annotation
    ddj_pp = np.ptp(jitter_values)
    ax.text(
        0.98,
        0.98,
        f"DDJ pk-pk: {ddj_pp:.2f} {time_unit}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9},
    )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Data-Dependent Jitter by Pattern", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_dcd(
    high_times: NDArray[np.floating[Any]],
    low_times: NDArray[np.floating[Any]],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    time_unit: str = "auto",
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot Duty Cycle Distortion (DCD) analysis.

    Creates overlaid histograms of high and low pulse times to visualize
    duty cycle distortion.

    Args:
        high_times: Array of high-state durations.
        low_times: Array of low-state durations.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        time_unit: Time unit.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.
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

    # Select time unit
    if time_unit == "auto":
        max_time = max(np.max(high_times), np.max(low_times))
        if max_time < 1e-9:
            time_unit = "ps"
            time_mult = 1e12
        elif max_time < 1e-6:
            time_unit = "ns"
            time_mult = 1e9
        else:
            time_unit = "us"
            time_mult = 1e6
    else:
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}.get(time_unit, 1e9)

    high_scaled = high_times * time_mult
    low_scaled = low_times * time_mult

    # Calculate statistics
    mean_high = np.mean(high_scaled)
    mean_low = np.mean(low_scaled)
    period = mean_high + mean_low
    duty_cycle = mean_high / period * 100
    dcd = (mean_high - mean_low) / 2

    # Determine common bins
    all_times = np.concatenate([high_scaled, low_scaled])
    bins = np.linspace(np.min(all_times) * 0.95, np.max(all_times) * 1.05, 50)

    # Plot histograms
    ax.hist(
        high_scaled,
        bins=bins,
        alpha=0.6,
        color="#E74C3C",
        label="High Time",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        low_scaled,
        bins=bins,
        alpha=0.6,
        color="#3498DB",
        label="Low Time",
        edgecolor="black",
        linewidth=0.5,
    )

    # Mean lines
    ax.axvline(mean_high, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(mean_low, color="#3498DB", linestyle="--", linewidth=2, alpha=0.8)

    # Statistics box
    stats_text = (
        f"Mean High: {mean_high:.2f} {time_unit}\n"
        f"Mean Low: {mean_low:.2f} {time_unit}\n"
        f"Duty Cycle: {duty_cycle:.1f}%\n"
        f"DCD: {dcd:.2f} {time_unit}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9},
        fontfamily="monospace",
    )

    ax.set_xlabel(f"Pulse Width ({time_unit})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Duty Cycle Distortion Analysis", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_jitter_trend(
    time_axis: NDArray[np.floating[Any]],
    jitter_values: NDArray[np.floating[Any]],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
    time_unit: str = "auto",
    jitter_unit: str = "auto",
    show_trend: bool = True,
    show_bounds: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot jitter trend over time.

    Creates a time series plot of jitter values with optional trend line
    and statistical bounds.

    Args:
        time_axis: Time values (e.g., cycle number or time in seconds).
        jitter_values: Jitter values at each time point.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        time_unit: Time axis unit.
        jitter_unit: Jitter axis unit.
        show_trend: Show linear trend line.
        show_bounds: Show ±3σ bounds.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.
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

    # Auto-select jitter unit
    if jitter_unit == "auto":
        max_jitter = np.max(np.abs(jitter_values))
        if max_jitter < 1e-9:
            jitter_unit = "ps"
            jitter_mult = 1e12
        elif max_jitter < 1e-6:
            jitter_unit = "ns"
            jitter_mult = 1e9
        else:
            jitter_unit = "us"
            jitter_mult = 1e6
    else:
        jitter_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}.get(jitter_unit, 1e12)

    jitter_scaled = jitter_values * jitter_mult

    # Plot jitter values
    ax.plot(time_axis, jitter_scaled, "b-", linewidth=0.8, alpha=0.7, label="Jitter")

    mean_val = np.mean(jitter_scaled)
    std_val = np.std(jitter_scaled)

    # Mean line
    ax.axhline(
        mean_val,
        color="gray",
        linestyle="-",
        linewidth=1,
        label=f"Mean: {mean_val:.2f} {jitter_unit}",
    )

    # Statistical bounds
    if show_bounds:
        ax.axhline(mean_val + 3 * std_val, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(
            mean_val - 3 * std_val,
            color="#E74C3C",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"±3sigma: {3 * std_val:.2f} {jitter_unit}",
        )
        ax.fill_between(
            time_axis, mean_val - 3 * std_val, mean_val + 3 * std_val, alpha=0.1, color="#E74C3C"
        )

    # Trend line
    if show_trend:
        z = np.polyfit(time_axis, jitter_scaled, 1)
        p = np.poly1d(z)
        ax.plot(
            time_axis,
            p(time_axis),
            "g-",
            linewidth=2,
            label=f"Trend: {z[0]:.2e} {jitter_unit}/unit",
        )

    ax.set_xlabel(f"Time ({time_unit})" if time_unit != "auto" else "Sample Index", fontsize=11)
    ax.set_ylabel(f"Jitter ({jitter_unit})", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Jitter Trend Analysis", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
