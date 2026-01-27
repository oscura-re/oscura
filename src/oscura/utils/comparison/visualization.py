"""Visualization utilities for trace comparison.

This module provides visualization functions for comparing traces including
overlay plots, difference plots, and comparison heat maps.


Example:
    >>> from oscura.utils.comparison.visualization import (
    ...     plot_overlay,
    ...     plot_difference,
    ...     plot_comparison_heatmap
    ... )
    >>> fig = plot_overlay(trace1, trace2)
    >>> fig = plot_difference(trace1, trace2)

References:
    - Tufte, E. R. (2001). The Visual Display of Quantitative Information
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from oscura.core.types import WaveformTrace
    from oscura.utils.comparison.compare import ComparisonResult


def plot_overlay(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    labels: tuple[str, str] = ("Trace 1", "Trace 2"),
    title: str = "Trace Comparison - Overlay",
    highlight_differences: bool = True,
    difference_threshold: float | None = None,
    figsize: tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Figure:
    """Create overlay plot showing both traces.

    : Overlay plot with difference highlighting.

    Args:
        trace1: First trace
        trace2: Second trace
        labels: Labels for the two traces
        title: Plot title
        highlight_differences: Highlight regions where traces differ
        difference_threshold: Threshold for highlighting (default: auto)
        figsize: Figure size (width, height)
        **kwargs: Additional arguments passed to plot()

    Returns:
        Matplotlib Figure object

    Example:
        >>> from oscura.utils.comparison.visualization import plot_overlay
        >>> fig = plot_overlay(measured, reference,
        ...                     labels=("Measured", "Reference"))
        >>> plt.show()

    References:
        CMP-003: Overlay plot with difference highlighting
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Create time axis
    if hasattr(trace1, "metadata") and trace1.metadata.sample_rate is not None:
        sample_rate = trace1.metadata.sample_rate
        time = np.arange(min_len) / sample_rate
        xlabel = "Time (s)"
    else:
        time = np.arange(min_len, dtype=np.float64)
        xlabel = "Sample"

    # Plot traces
    ax.plot(time, data1, label=labels[0], alpha=0.7, linewidth=1.5, **kwargs)
    ax.plot(time, data2, label=labels[1], alpha=0.7, linewidth=1.5, **kwargs)

    # Highlight differences
    if highlight_differences:
        diff = np.abs(data1 - data2)
        if difference_threshold is None:
            # Auto threshold: mean + 2*std of difference
            difference_threshold = float(np.mean(diff) + 2 * np.std(diff))

        # Find regions with significant difference
        diff_mask = diff > difference_threshold
        if np.any(diff_mask):
            # Highlight regions with vertical spans
            in_region = False
            start_idx = 0
            for i in range(len(diff_mask)):
                if diff_mask[i] and not in_region:
                    start_idx = i
                    in_region = True
                elif not diff_mask[i] and in_region:
                    ax.axvspan(
                        time[start_idx],
                        time[i - 1],
                        alpha=0.2,
                        color="red",
                        label="Difference" if start_idx == 0 else "",
                    )
                    in_region = False
            # Handle last region
            if in_region:
                ax.axvspan(time[start_idx], time[-1], alpha=0.2, color="red")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_difference(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    title: str = "Trace Comparison - Difference",
    normalize: bool = False,
    show_statistics: bool = True,
    figsize: tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Figure:
    """Create difference plot (trace1 - trace2).

    : Difference trace visualization.

    Args:
        trace1: First trace
        trace2: Second trace
        title: Plot title
        normalize: Normalize difference to percentage
        show_statistics: Show statistics text box
        figsize: Figure size
        **kwargs: Additional arguments passed to plot()

    Returns:
        Matplotlib Figure object

    Example:
        >>> from oscura.utils.comparison.visualization import plot_difference
        >>> fig = plot_difference(measured, reference, normalize=True)
        >>> plt.show()

    References:
        CMP-003: Comparison Visualization
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Compute difference
    diff = data1 - data2

    if normalize:
        # Normalize to percentage of reference range
        ref_range = np.ptp(data2)
        if ref_range > 0:
            diff = (diff / ref_range) * 100.0
        ylabel = "Difference (%)"
    else:
        ylabel = "Difference"

    # Create time axis
    if hasattr(trace1, "metadata") and trace1.metadata.sample_rate is not None:
        sample_rate = trace1.metadata.sample_rate
        time = np.arange(min_len) / sample_rate
        xlabel = "Time (s)"
    else:
        time = np.arange(min_len, dtype=np.float64)
        xlabel = "Sample"

    # Plot difference
    ax.plot(time, diff, label="Difference", **kwargs)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)

    # Add statistics text box
    if show_statistics:
        max_diff = float(np.max(np.abs(diff)))
        rms_diff = float(np.sqrt(np.mean(diff**2)))
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))

        stats_text = (
            f"Max: {max_diff:.3f}\nRMS: {rms_diff:.3f}\nMean: {mean_diff:.3f}\nStd: {std_diff:.3f}"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            fontsize=9,
            family="monospace",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_comparison_heatmap(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    title: str = "Trace Comparison - Difference Heatmap",
    window_size: int = 100,
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> Figure:
    """Create difference heatmap showing where changes occur.

    : Difference heat map showing where changes occur.

    Args:
        trace1: First trace
        trace2: Second trace
        title: Plot title
        window_size: Window size for heatmap bins
        figsize: Figure size
        **kwargs: Additional arguments passed to imshow()

    Returns:
        Matplotlib Figure object

    Example:
        >>> from oscura.utils.comparison.visualization import plot_comparison_heatmap
        >>> fig = plot_comparison_heatmap(trace1, trace2, window_size=50)
        >>> plt.show()

    References:
        CMP-003: Difference heat map showing where changes occur
    """
    fig, (ax_heat, ax_trace) = _create_heatmap_axes(figsize)

    # Align data and compute difference
    data1, data2, diff, min_len = _prepare_diff_data(trace1, trace2)

    # Create and plot heatmap
    n_windows, window_size = _compute_window_params(min_len, window_size)
    heatmap_data = _build_heatmap(data1, data2, diff, n_windows, window_size, min_len)
    _plot_heatmap(ax_heat, heatmap_data, title, **kwargs)

    # Plot difference trace
    time, xlabel = _compute_time_axis(trace1, min_len)
    _plot_diff_trace(ax_trace, time, diff, xlabel)

    plt.tight_layout()
    return fig


def _create_heatmap_axes(figsize: tuple[float, float]) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create figure with heatmap and trace axes."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_heat = fig.add_subplot(gs[0])
    ax_trace = fig.add_subplot(gs[1], sharex=ax_heat)
    return fig, (ax_heat, ax_trace)


def _prepare_diff_data(
    trace1: WaveformTrace, trace2: WaveformTrace
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Prepare aligned data and compute difference."""
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)
    min_len = min(len(data1), len(data2))
    data1, data2 = data1[:min_len], data2[:min_len]
    diff = np.abs(data1 - data2)
    return data1, data2, diff, min_len


def _compute_window_params(min_len: int, window_size: int) -> tuple[int, int]:
    """Compute window parameters for heatmap."""
    n_windows = min_len // window_size
    if n_windows == 0:
        return 1, min_len
    return n_windows, window_size


def _build_heatmap(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    diff: NDArray[np.float64],
    n_windows: int,
    window_size: int,
    min_len: int,
) -> NDArray[np.float64]:
    """Build heatmap data from windowed differences."""
    heatmap_data = np.zeros((10, n_windows))
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, min_len)
        window_data1, window_diff = data1[start:end], diff[start:end]
        y_min, y_max = (
            min(np.min(data1[start:end]), np.min(data2[start:end])),
            max(np.max(data1[start:end]), np.max(data2[start:end])),
        )
        if y_max - y_min > 0:
            bins = np.linspace(y_min, y_max, 11)
            for sample_idx, y_val in enumerate(window_data1):
                bin_idx = max(0, min(9, np.digitize(y_val, bins) - 1))
                heatmap_data[bin_idx, i] += window_diff[sample_idx]
    return heatmap_data / window_size


def _plot_heatmap(ax: Axes, heatmap_data: NDArray[np.float64], title: str, **kwargs: Any) -> None:
    """Plot heatmap on axes."""
    im = ax.imshow(
        heatmap_data, aspect="auto", cmap="hot", origin="lower", interpolation="nearest", **kwargs
    )
    plt.colorbar(im, ax=ax, label="Average Difference")
    ax.set_ylabel("Amplitude Bin")
    ax.set_title(title)


def _compute_time_axis(trace: WaveformTrace, min_len: int) -> tuple[NDArray[np.float64], str]:
    """Compute time axis and label."""
    if hasattr(trace, "metadata") and trace.metadata.sample_rate is not None:
        return np.arange(min_len) / trace.metadata.sample_rate, "Time (s)"
    return np.arange(min_len, dtype=np.float64), "Sample"


def _plot_diff_trace(
    ax: Axes, time: NDArray[np.float64], diff: NDArray[np.float64], xlabel: str
) -> None:
    """Plot difference trace on axes."""
    ax.plot(time, diff, linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Difference")
    ax.grid(True, alpha=0.3)


def plot_comparison_summary(
    result: ComparisonResult,
    *,
    title: str = "Trace Comparison Summary",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Create comprehensive comparison summary figure.

    : Summary table of key differences.

    Args:
        result: ComparisonResult from compare_traces()
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> from oscura.utils.comparison import compare_traces
        >>> from oscura.utils.comparison.visualization import plot_comparison_summary
        >>> result = compare_traces(trace1, trace2)
        >>> fig = plot_comparison_summary(result)
        >>> plt.show()

    References:
        CMP-003: Summary table of key differences
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # Create statistics table
    _plot_statistics_table(fig, gs, result, title)

    # Plot difference trace
    if result.difference_trace is not None:
        _plot_difference_trace(fig, gs, result.difference_trace)
        _plot_difference_histogram(fig, gs, result.difference_trace)

    # Plot violation locations
    _plot_violations(fig, gs, result)

    plt.tight_layout()
    return fig


def _plot_statistics_table(
    fig: Figure,
    gs: GridSpec,
    result: ComparisonResult,
    title: str,
) -> None:
    """Plot statistics table at top of summary."""
    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.axis("off")

    stats_data = [
        ["Match Status", "PASS ✓" if result.match else "FAIL ✗"],
        ["Similarity Score", f"{result.similarity:.4f}"],
        ["Correlation", f"{result.correlation:.4f}"],
        ["Max Difference", f"{result.max_difference:.6f}"],
        ["RMS Difference", f"{result.rms_difference:.6f}"],
    ]

    if result.statistics:
        stats_data.extend(
            [
                ["Mean Difference", f"{result.statistics['mean_difference']:.6f}"],
                ["Violations", f"{result.statistics['num_violations']}"],
                ["Violation Rate", f"{result.statistics['violation_rate'] * 100:.2f}%"],
            ]
        )

    table = ax_stats.table(
        cellText=stats_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],  # type: ignore[arg-type]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code match status
    if result.match:
        table[(1, 1)].set_facecolor("#90EE90")  # Light green
    else:
        table[(1, 1)].set_facecolor("#FFB6C1")  # Light red

    ax_stats.set_title(title, fontsize=14, fontweight="bold", pad=20)


def _plot_difference_trace(
    fig: Figure,
    gs: GridSpec,
    difference_trace: Any,
) -> None:
    """Plot difference trace in middle row."""
    ax_overlay = fig.add_subplot(gs[1, :])
    n_samples = len(difference_trace.data)
    time = np.arange(n_samples)
    ax_overlay.plot(time, difference_trace.data, label="Difference")
    ax_overlay.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax_overlay.set_xlabel("Sample")
    ax_overlay.set_ylabel("Difference")
    ax_overlay.set_title("Difference Trace")
    ax_overlay.legend()
    ax_overlay.grid(True, alpha=0.3)


def _plot_difference_histogram(
    fig: Figure,
    gs: GridSpec,
    difference_trace: Any,
) -> None:
    """Plot histogram of differences."""
    ax_hist = fig.add_subplot(gs[2, 0])
    diff_data = difference_trace.data
    ax_hist.hist(diff_data, bins=50, edgecolor="black", alpha=0.7)
    ax_hist.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero difference")
    ax_hist.set_xlabel("Difference")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Difference Distribution")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)


def _plot_violations(
    fig: Figure,
    gs: GridSpec,
    result: ComparisonResult,
) -> None:
    """Plot violation locations."""
    ax_viol = fig.add_subplot(gs[2, 1])

    if result.violations is not None and len(result.violations) > 0:
        ax_viol.scatter(
            result.violations,
            np.ones_like(result.violations),
            marker="|",
            s=100,
            color="red",
            alpha=0.5,
        )
        ax_viol.set_xlim(0, len(result.difference_trace.data) if result.difference_trace else 1000)
        ax_viol.set_ylim(0.5, 1.5)
        ax_viol.set_xlabel("Sample Index")
        ax_viol.set_title(f"Violation Locations ({len(result.violations)} total)")
        ax_viol.set_yticks([])
    else:
        ax_viol.text(
            0.5,
            0.5,
            "No Violations",
            ha="center",
            va="center",
            fontsize=14,
            color="green",
        )
        ax_viol.axis("off")


__all__ = [
    "plot_comparison_heatmap",
    "plot_comparison_summary",
    "plot_difference",
    "plot_overlay",
]
