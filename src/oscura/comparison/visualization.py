"""Visualization utilities for trace comparison.

This module provides visualization functions for comparing traces including
overlay plots, difference plots, and comparison heat maps.


Example:
    >>> from oscura.comparison.visualization import (
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

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from oscura.comparison.compare import ComparisonResult
    from oscura.core.types import WaveformTrace


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
        >>> from oscura.comparison.visualization import plot_overlay
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
        >>> from oscura.comparison.visualization import plot_difference
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
        >>> from oscura.comparison.visualization import plot_comparison_heatmap
        >>> fig = plot_comparison_heatmap(trace1, trace2, window_size=50)
        >>> plt.show()

    References:
        CMP-003: Difference heat map showing where changes occur
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_heat = fig.add_subplot(gs[0])
    ax_trace = fig.add_subplot(gs[1], sharex=ax_heat)

    # Get data
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Compute difference
    diff = np.abs(data1 - data2)

    # Create windowed heatmap
    n_windows = min_len // window_size
    if n_windows == 0:
        n_windows = 1
        window_size = min_len

    heatmap_data = np.zeros((10, n_windows))
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, min_len)
        window_diff = diff[start:end]

        # Bin into 10 levels based on amplitude
        window_data1 = data1[start:end]
        window_data2 = data2[start:end]
        y_min = min(np.min(window_data1), np.min(window_data2))
        y_max = max(np.max(window_data1), np.max(window_data2))

        if y_max - y_min > 0:
            bins = np.linspace(y_min, y_max, 11)
            for sample_idx in range(len(window_diff)):
                y_val = window_data1[sample_idx]  # window_data1 is already sliced
                bin_idx = np.digitize(y_val, bins) - 1
                bin_idx = max(0, min(9, bin_idx))
                heatmap_data[bin_idx, i] += window_diff[sample_idx]

    # Normalize heatmap
    heatmap_data = heatmap_data / window_size

    # Plot heatmap
    im = ax_heat.imshow(
        heatmap_data,
        aspect="auto",
        cmap="hot",
        origin="lower",
        interpolation="nearest",
        **kwargs,
    )
    plt.colorbar(im, ax=ax_heat, label="Average Difference")

    ax_heat.set_ylabel("Amplitude Bin")
    ax_heat.set_title(title)

    # Plot difference trace below
    if hasattr(trace1, "metadata") and trace1.metadata.sample_rate is not None:
        sample_rate = trace1.metadata.sample_rate
        time = np.arange(min_len) / sample_rate
        xlabel = "Time (s)"
    else:
        time = np.arange(min_len, dtype=np.float64)
        xlabel = "Sample"

    ax_trace.plot(time, diff, linewidth=0.5)
    ax_trace.set_xlabel(xlabel)
    ax_trace.set_ylabel("Difference")
    ax_trace.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


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
        >>> from oscura.comparison import compare_traces
        >>> from oscura.comparison.visualization import plot_comparison_summary
        >>> result = compare_traces(trace1, trace2)
        >>> fig = plot_comparison_summary(result)
        >>> plt.show()

    References:
        CMP-003: Summary table of key differences
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # Statistics table
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

    # Overlay plot
    if result.difference_trace is not None:
        # Plot difference trace
        ax_overlay = fig.add_subplot(gs[1, :])
        n_samples = len(result.difference_trace.data)
        time = np.arange(n_samples)
        ax_overlay.plot(time, result.difference_trace.data, label="Difference")
        ax_overlay.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax_overlay.set_xlabel("Sample")
        ax_overlay.set_ylabel("Difference")
        ax_overlay.set_title("Difference Trace")
        ax_overlay.legend()
        ax_overlay.grid(True, alpha=0.3)

    # Histogram of differences
    if result.difference_trace is not None:
        ax_hist = fig.add_subplot(gs[2, 0])
        diff_data = result.difference_trace.data
        ax_hist.hist(diff_data, bins=50, edgecolor="black", alpha=0.7)
        ax_hist.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero difference")
        ax_hist.set_xlabel("Difference")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Difference Distribution")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

    # Violation locations
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

    plt.tight_layout()
    return fig


__all__ = [
    "plot_comparison_heatmap",
    "plot_comparison_summary",
    "plot_difference",
    "plot_overlay",
]
