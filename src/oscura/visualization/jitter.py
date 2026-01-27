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


def _determine_tie_time_unit(
    tie_data: NDArray[np.floating[Any]], time_unit: str
) -> tuple[str, float]:
    """Determine time unit and multiplier for TIE display.

    Args:
        tie_data: TIE values in seconds.
        time_unit: Requested time unit or "auto".

    Returns:
        Tuple of (time_unit, time_multiplier).
    """
    if time_unit == "auto":
        max_tie = np.max(np.abs(tie_data))
        if max_tie < 1e-12:
            return "fs", 1e15
        elif max_tie < 1e-9:
            return "ps", 1e12
        elif max_tie < 1e-6:
            return "ns", 1e9
        else:
            return "us", 1e6
    else:
        time_mult_map = {
            "s": 1,
            "ms": 1e3,
            "us": 1e6,
            "ns": 1e9,
            "ps": 1e12,
            "fs": 1e15,
        }
        if time_unit in time_mult_map:
            return time_unit, time_mult_map[time_unit]
        else:
            # Fallback to ps for invalid unit
            return "ps", 1e12


def _calculate_tie_statistics(
    tie_scaled: NDArray[np.floating[Any]],
) -> tuple[float, float, float, float]:
    """Calculate TIE statistical metrics.

    Args:
        tie_scaled: Scaled TIE values.

    Returns:
        Tuple of (mean, std, peak-to-peak, rms).
    """
    mean_val = float(np.mean(tie_scaled))
    std_val = float(np.std(tie_scaled))
    pp_val = float(np.ptp(tie_scaled))
    rms_val = float(np.sqrt(np.mean(tie_scaled**2)))
    return mean_val, std_val, pp_val, rms_val


def _add_gaussian_fit(
    ax: Axes, bin_edges: NDArray[np.floating[Any]], mean_val: float, std_val: float, time_unit: str
) -> None:
    """Add Gaussian fit overlay to histogram.

    Args:
        ax: Matplotlib axes to plot on.
        bin_edges: Histogram bin edges.
        mean_val: Mean value.
        std_val: Standard deviation.
        time_unit: Time unit string for label.
    """
    if not HAS_SCIPY:
        return

    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
    y_fit = norm.pdf(x_fit, mean_val, std_val)
    ax.plot(
        x_fit, y_fit, "r-", linewidth=2, label=f"Gaussian Fit (sigma={std_val:.2f} {time_unit})"
    )


def _add_rj_dj_indicators(ax: Axes, mean_val: float, std_val: float) -> None:
    """Add RJ/DJ separation indicators to plot.

    Args:
        ax: Matplotlib axes to plot on.
        mean_val: Mean value.
        std_val: Standard deviation.
    """
    # Mark ±3sigma region (RJ contribution)
    ax.axvline(mean_val - 3 * std_val, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(mean_val + 3 * std_val, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7)

    # Shade RJ region
    ax.axvspan(
        mean_val - 3 * std_val,
        mean_val + 3 * std_val,
        alpha=0.1,
        color="#E74C3C",
        label="±3sigma (99.7% RJ)",
    )


def _add_statistics_box(
    ax: Axes, mean_val: float, rms_val: float, std_val: float, pp_val: float, time_unit: str
) -> None:
    """Add statistics text box to plot.

    Args:
        ax: Matplotlib axes to plot on.
        mean_val: Mean value.
        rms_val: RMS value.
        std_val: Standard deviation.
        pp_val: Peak-to-peak value.
        time_unit: Time unit string.
    """
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
        time_unit: Time unit ("s", "ms", "us", "ns", "ps", "fs", "auto").
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

    fig, ax = _setup_tie_figure(ax, figsize)
    time_unit, time_mult = _determine_tie_time_unit(tie_data, time_unit)
    tie_scaled = tie_data * time_mult
    mean_val, std_val, pp_val, rms_val = _calculate_tie_statistics(tie_scaled)

    counts, bin_edges, patches = _plot_tie_histogram_data(ax, tie_scaled, bins)
    _add_tie_overlays(ax, show_gaussian_fit, show_rj_dj, bin_edges, mean_val, std_val, time_unit)
    _format_tie_plot(ax, show_statistics, mean_val, rms_val, std_val, pp_val, time_unit, title)

    fig.tight_layout()
    _save_and_show_tie_plot(fig, save_path, show)

    return fig


def _setup_tie_figure(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Setup figure and axes for TIE plot.

    Args:
        ax: Existing axes or None.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).

    Raises:
        ValueError: If axes has no figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)
    return fig, ax


def _plot_tie_histogram_data(
    ax: Axes, tie_scaled: NDArray[np.floating[Any]], bins: int | str
) -> tuple[Any, NDArray[Any], Any]:
    """Plot histogram data.

    Args:
        ax: Matplotlib axes.
        tie_scaled: Scaled TIE data.
        bins: Bin specification.

    Returns:
        Tuple of (counts, bin_edges, patches) from matplotlib hist.
    """
    result: tuple[Any, NDArray[Any], Any] = ax.hist(
        tie_scaled,
        bins=bins,
        density=True,
        color="#3498DB",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    return result


def _add_tie_overlays(
    ax: Axes,
    show_gaussian_fit: bool,
    show_rj_dj: bool,
    bin_edges: NDArray[Any],
    mean_val: float,
    std_val: float,
    time_unit: str,
) -> None:
    """Add Gaussian fit and RJ/DJ overlays to TIE plot.

    Args:
        ax: Matplotlib axes.
        show_gaussian_fit: Whether to show Gaussian fit.
        show_rj_dj: Whether to show RJ/DJ indicators.
        bin_edges: Histogram bin edges.
        mean_val: Mean TIE value.
        std_val: Standard deviation.
        time_unit: Time unit string.
    """
    if show_gaussian_fit:
        _add_gaussian_fit(ax, bin_edges, mean_val, std_val, time_unit)
    if show_rj_dj:
        _add_rj_dj_indicators(ax, mean_val, std_val)


def _format_tie_plot(
    ax: Axes,
    show_statistics: bool,
    mean_val: float,
    rms_val: float,
    std_val: float,
    pp_val: float,
    time_unit: str,
    title: str | None,
) -> None:
    """Format TIE plot axes and labels.

    Args:
        ax: Matplotlib axes.
        show_statistics: Whether to show statistics box.
        mean_val: Mean value.
        rms_val: RMS value.
        std_val: Standard deviation.
        pp_val: Peak-to-peak value.
        time_unit: Time unit.
        title: Plot title.
    """
    if show_statistics:
        _add_statistics_box(ax, mean_val, rms_val, std_val, pp_val, time_unit)

    ax.set_xlabel(f"TIE ({time_unit})", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    final_title = title if title else "Time Interval Error Distribution"
    ax.set_title(final_title, fontsize=12, fontweight="bold")


def _save_and_show_tie_plot(fig: Figure, save_path: str | Path | None, show: bool) -> None:
    """Save and/or show TIE plot.

    Args:
        fig: Matplotlib figure.
        save_path: Path to save file.
        show: Whether to display interactively.
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


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

    fig, ax = _get_or_create_figure(ax, figsize)
    ber_total = ber_total if ber_total is not None else ber_left + ber_right

    # Plot BER curves
    ber_total_plot = _plot_bathtub_ber_curves(ax, positions, ber_left, ber_right, ber_total)

    # Optional annotations
    if show_target:
        _add_target_ber_line(ax, target_ber)

    if show_eye_opening:
        _add_eye_opening_annotation(ax, positions, ber_total_plot, target_ber, eye_opening)

    # Styling
    _style_bathtub_plot(ax, positions, ber_total_plot, title)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _get_or_create_figure(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Get existing figure or create new one."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)
    return fig, ax


def _plot_bathtub_ber_curves(
    ax: Axes,
    positions: NDArray[np.floating[Any]],
    ber_left: NDArray[np.floating[Any]],
    ber_right: NDArray[np.floating[Any]],
    ber_total: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Plot BER curves and return clipped total BER."""
    ber_left_plot = np.clip(ber_left, 1e-18, 1)
    ber_right_plot = np.clip(ber_right, 1e-18, 1)
    ber_total_plot = np.clip(ber_total, 1e-18, 1)

    ax.semilogy(positions, ber_left_plot, "b-", linewidth=2, label="BER Left", alpha=0.8)
    ax.semilogy(positions, ber_right_plot, "r-", linewidth=2, label="BER Right", alpha=0.8)
    ax.semilogy(positions, ber_total_plot, "k-", linewidth=2.5, label="BER Total")

    return ber_total_plot


def _add_target_ber_line(ax: Axes, target_ber: float) -> None:
    """Add target BER horizontal line."""
    ax.axhline(
        target_ber,
        color="#27AE60",
        linestyle="--",
        linewidth=2,
        label=f"Target BER = {target_ber:.0e}",
    )


def _add_eye_opening_annotation(
    ax: Axes,
    positions: NDArray[np.floating[Any]],
    ber_total_plot: NDArray[np.floating[Any]],
    target_ber: float,
    eye_opening: float | None,
) -> None:
    """Add eye opening annotation if applicable."""
    if eye_opening is None:
        eye_opening = _calculate_eye_opening(positions, ber_total_plot, target_ber)

    if eye_opening <= 0:
        return

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


def _calculate_eye_opening(
    positions: NDArray[np.floating[Any]],
    ber_total: NDArray[np.floating[Any]],
    target_ber: float,
) -> float:
    """Calculate eye opening at target BER."""
    left_cross = np.where(ber_total < target_ber)[0]
    if len(left_cross) > 0:
        left_edge = positions[left_cross[0]]
        right_edge = positions[left_cross[-1]]
        return float(right_edge - left_edge)
    return 0.0


def _style_bathtub_plot(
    ax: Axes,
    positions: NDArray[np.floating[Any]],
    ber_total_plot: NDArray[np.floating[Any]],
    title: str | None,
) -> None:
    """Apply styling to bathtub plot."""
    ax.fill_between(positions, 1e-18, ber_total_plot, alpha=0.1, color="gray")
    ax.set_xlabel("Sample Position (UI)", fontsize=11)
    ax.set_ylabel("Bit Error Rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-15, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(title or "Bathtub Curve", fontsize=12, fontweight="bold")


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

    # Validate input lengths match
    if len(patterns) != len(jitter_values):
        raise ValueError(
            f"Mismatched lengths: patterns has {len(patterns)} elements "
            f"but jitter_values has {len(jitter_values)} elements"
        )

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


def _determine_dcd_time_unit(
    high_times: NDArray[np.floating[Any]], low_times: NDArray[np.floating[Any]], time_unit: str
) -> tuple[str, float]:
    """Determine time unit and scaling for DCD plot.

    Args:
        high_times: High-state durations.
        low_times: Low-state durations.
        time_unit: Requested time unit or "auto".

    Returns:
        Tuple of (time_unit, time_multiplier).
    """
    if time_unit == "auto":
        max_time = max(np.max(high_times), np.max(low_times))
        if max_time < 1e-9:
            return "ps", 1e12
        elif max_time < 1e-6:
            return "ns", 1e9
        else:
            return "us", 1e6
    else:
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}.get(time_unit, 1e9)
        return time_unit, time_mult


def _compute_dcd_statistics(
    high_scaled: NDArray[np.floating[Any]], low_scaled: NDArray[np.floating[Any]]
) -> tuple[float, float, float, float]:
    """Compute DCD statistics.

    Args:
        high_scaled: Scaled high-state durations.
        low_scaled: Scaled low-state durations.

    Returns:
        Tuple of (mean_high, mean_low, duty_cycle, dcd).
    """
    mean_high = float(np.mean(high_scaled))
    mean_low = float(np.mean(low_scaled))
    period = mean_high + mean_low
    duty_cycle = mean_high / period * 100
    dcd = (mean_high - mean_low) / 2
    return mean_high, mean_low, duty_cycle, dcd


def _plot_dcd_histograms(
    ax: Axes,
    high_scaled: NDArray[np.floating[Any]],
    low_scaled: NDArray[np.floating[Any]],
    mean_high: float,
    mean_low: float,
) -> None:
    """Plot DCD histograms with mean lines.

    Args:
        ax: Matplotlib axes.
        high_scaled: Scaled high-state durations.
        low_scaled: Scaled low-state durations.
        mean_high: Mean high value.
        mean_low: Mean low value.
    """
    all_times = np.concatenate([high_scaled, low_scaled])
    bins = np.linspace(np.min(all_times) * 0.95, np.max(all_times) * 1.05, 50)

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

    ax.axvline(mean_high, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(mean_low, color="#3498DB", linestyle="--", linewidth=2, alpha=0.8)


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

    fig, ax = _get_or_create_figure(ax, figsize)

    # Scale times
    time_unit, time_mult = _determine_dcd_time_unit(high_times, low_times, time_unit)
    high_scaled = high_times * time_mult
    low_scaled = low_times * time_mult

    # Calculate statistics
    mean_high, mean_low, duty_cycle, dcd = _compute_dcd_statistics(high_scaled, low_scaled)

    # Plot histograms
    _plot_dcd_histograms(ax, high_scaled, low_scaled, mean_high, mean_low)

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

    fig, ax = _setup_jitter_trend_figure(ax, figsize)
    jitter_unit, jitter_mult = _determine_jitter_unit(jitter_values, jitter_unit)
    jitter_scaled = jitter_values * jitter_mult

    mean_val, std_val = _plot_jitter_data(ax, time_axis, jitter_scaled, jitter_unit)
    _add_jitter_bounds(ax, time_axis, mean_val, std_val, jitter_unit, show_bounds)
    _add_jitter_trend(ax, time_axis, jitter_scaled, jitter_unit, show_trend)
    _format_jitter_trend_plot(ax, time_unit, jitter_unit, title)

    fig.tight_layout()
    _save_and_show_jitter_trend(fig, save_path, show)

    return fig


def _setup_jitter_trend_figure(
    ax: Axes | None, figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    """Setup figure and axes for jitter trend plot.

    Args:
        ax: Existing axes or None.
        figsize: Figure size.

    Returns:
        Tuple of (figure, axes).

    Raises:
        ValueError: If axes has no figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)
    return fig, ax


def _determine_jitter_unit(
    jitter_values: NDArray[np.floating[Any]], jitter_unit: str
) -> tuple[str, float]:
    """Determine jitter unit and multiplier.

    Args:
        jitter_values: Jitter value array.
        jitter_unit: Requested unit or "auto".

    Returns:
        Tuple of (unit_str, multiplier).
    """
    if jitter_unit == "auto":
        max_jitter = np.max(np.abs(jitter_values))
        if max_jitter < 1e-9:
            return "ps", 1e12
        elif max_jitter < 1e-6:
            return "ns", 1e9
        else:
            return "us", 1e6
    else:
        jitter_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}.get(jitter_unit, 1e12)
        return jitter_unit, jitter_mult


def _plot_jitter_data(
    ax: Axes,
    time_axis: NDArray[np.floating[Any]],
    jitter_scaled: NDArray[np.floating[Any]],
    jitter_unit: str,
) -> tuple[float, float]:
    """Plot jitter data and mean line.

    Args:
        ax: Matplotlib axes.
        time_axis: Time array.
        jitter_scaled: Scaled jitter values.
        jitter_unit: Jitter unit string.

    Returns:
        Tuple of (mean_val, std_val).
    """
    ax.plot(time_axis, jitter_scaled, "b-", linewidth=0.8, alpha=0.7, label="Jitter")

    mean_val = float(np.mean(jitter_scaled))
    std_val = float(np.std(jitter_scaled))

    ax.axhline(
        mean_val,
        color="gray",
        linestyle="-",
        linewidth=1,
        label=f"Mean: {mean_val:.2f} {jitter_unit}",
    )

    return mean_val, std_val


def _add_jitter_bounds(
    ax: Axes,
    time_axis: NDArray[np.floating[Any]],
    mean_val: float,
    std_val: float,
    jitter_unit: str,
    show_bounds: bool,
) -> None:
    """Add statistical bounds to plot.

    Args:
        ax: Matplotlib axes.
        time_axis: Time array.
        mean_val: Mean value.
        std_val: Standard deviation.
        jitter_unit: Unit string.
        show_bounds: Whether to show bounds.
    """
    if not show_bounds:
        return

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


def _add_jitter_trend(
    ax: Axes,
    time_axis: NDArray[np.floating[Any]],
    jitter_scaled: NDArray[np.floating[Any]],
    jitter_unit: str,
    show_trend: bool,
) -> None:
    """Add trend line to plot.

    Args:
        ax: Matplotlib axes.
        time_axis: Time array.
        jitter_scaled: Scaled jitter values.
        jitter_unit: Unit string.
        show_trend: Whether to show trend.
    """
    if not show_trend:
        return

    z = np.polyfit(time_axis, jitter_scaled, 1)
    p = np.poly1d(z)
    ax.plot(
        time_axis, p(time_axis), "g-", linewidth=2, label=f"Trend: {z[0]:.2e} {jitter_unit}/unit"
    )


def _format_jitter_trend_plot(
    ax: Axes, time_unit: str, jitter_unit: str, title: str | None
) -> None:
    """Format jitter trend plot axes and labels.

    Args:
        ax: Matplotlib axes.
        time_unit: Time unit string.
        jitter_unit: Jitter unit string.
        title: Plot title.
    """
    ax.set_xlabel(f"Time ({time_unit})" if time_unit != "auto" else "Sample Index", fontsize=11)
    ax.set_ylabel(f"Jitter ({jitter_unit})", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    final_title = title if title else "Jitter Trend Analysis"
    ax.set_title(final_title, fontsize=12, fontweight="bold")


def _save_and_show_jitter_trend(fig: Figure, save_path: str | Path | None, show: bool) -> None:
    """Save and/or show jitter trend plot.

    Args:
        fig: Matplotlib figure.
        save_path: Path to save file.
        show: Whether to display interactively.
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
