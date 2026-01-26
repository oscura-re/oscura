"""Reverse Engineering Pipeline Visualization Module.

This module provides comprehensive visualization functions for reverse engineering
pipeline results, including message type distributions, field layouts, confidence
heatmaps, protocol detection scores, and pipeline performance metrics.

Functions:
    plot_re_summary: Multi-panel dashboard of RE pipeline results
    plot_message_type_distribution: Pie/bar chart of discovered message types
    plot_message_field_layout: Visual field layout with byte positions
    plot_field_confidence_heatmap: Heatmap of field inference confidence scores
    plot_protocol_candidates: Bar chart of protocol detection scores
    plot_crc_parameters: Visualization of detected CRC parameters
    plot_pipeline_timing: Performance metrics for each pipeline stage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from oscura.inference.crc_reverse import CRCParameters
    from oscura.inference.message_format import InferredField, MessageSchema
    from oscura.utils.pipeline.reverse_engineering import (
        MessageTypeInfo,
        ProtocolCandidate,
        REAnalysisResult,
    )

__all__ = [
    "plot_crc_parameters",
    "plot_field_confidence_heatmap",
    "plot_message_field_layout",
    "plot_message_type_distribution",
    "plot_pipeline_timing",
    "plot_protocol_candidates",
    "plot_re_summary",
]


def plot_re_summary(
    result: REAnalysisResult,
    *,
    figsize: tuple[float, float] = (16, 10),
    title: str | None = None,
) -> Figure:
    """Create multi-panel dashboard showing RE pipeline results overview.

    Displays a comprehensive summary of the reverse engineering analysis including:
    - Message type distribution
    - Protocol candidates
    - Pipeline timing
    - Key statistics

    Args:
        result: REAnalysisResult from pipeline analysis.
        figsize: Figure size (width, height) in inches.
        title: Optional custom title for the dashboard.

    Returns:
        Matplotlib figure object.

    Example:
        >>> from oscura.utils.pipeline.reverse_engineering import REPipeline
        >>> pipeline = REPipeline()
        >>> result = pipeline.analyze(data)
        >>> fig = plot_re_summary(result)
        >>> fig.savefig("re_summary.png", dpi=300)
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Summary statistics (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_summary_stats(ax1, result)

    # Panel 2: Message type distribution (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_message_types_panel(ax2, result.message_types)

    # Panel 3: Protocol candidates (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_protocol_panel(ax3, result.protocol_candidates)

    # Panel 4: Pipeline timing (bottom-left and center)
    ax4 = fig.add_subplot(gs[1, :2])
    _plot_timing_panel(ax4, result.statistics)

    # Panel 5: Warnings/info (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    _plot_warnings_panel(ax5, result)

    # Set main title
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Reverse Engineering Analysis Summary", fontsize=14, fontweight="bold")

    return fig


def _get_field_type_colors() -> dict[str, str]:
    """Get color map for field types."""
    return {
        "constant": "#4CAF50",  # Green
        "counter": "#2196F3",  # Blue
        "timestamp": "#9C27B0",  # Purple
        "length": "#FF9800",  # Orange
        "checksum": "#F44336",  # Red
        "data": "#607D8B",  # Gray
        "unknown": "#9E9E9E",  # Light gray
    }


def _draw_field_rectangles(
    ax: Any,
    fields: list[InferredField],
    total_bytes: int,
    bar_height: float,
    y_center: float,
    type_colors: dict[str, str],
) -> None:
    """Draw field rectangles with labels."""
    for field in fields:
        color = type_colors.get(field.field_type, "#9E9E9E")
        width = field.size / total_bytes

        rect = Rectangle(
            (field.offset / total_bytes, y_center - bar_height / 2),
            width,
            bar_height,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        x_center = (field.offset + field.size / 2) / total_bytes
        label = f"{field.name}\n({field.field_type})" if field.size > 2 else field.field_type[:3]

        ax.text(
            x_center,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=9 if field.size > 2 else 7,
            fontweight="bold",
            color="white",
        )


def _add_field_offsets(
    ax: Any,
    fields: list[InferredField],
    total_bytes: int,
    bar_height: float,
    y_center: float,
) -> None:
    """Add byte offset labels."""
    for field in fields:
        ax.text(
            field.offset / total_bytes,
            y_center - bar_height / 2 - 0.08,
            f"{field.offset}",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax.text(
        1.0,
        y_center - bar_height / 2 - 0.08,
        f"{total_bytes}",
        ha="center",
        va="top",
        fontsize=8,
    )


def _add_field_legend(ax: Any, fields: list[InferredField], type_colors: dict[str, str]) -> None:
    """Add legend for field types."""
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=color, label=ftype)
        for ftype, color in type_colors.items()
        if any(f.field_type == ftype for f in fields)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)


def _format_layout_axes(ax: Any, total_bytes: int, title: str | None) -> None:
    """Format axes for layout plot."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    else:
        ax.set_title(
            f"Message Field Layout ({total_bytes} bytes)", fontsize=12, fontweight="bold", pad=20
        )


def plot_message_type_distribution(
    message_types: list[MessageTypeInfo],
    *,
    figsize: tuple[float, float] = (12, 5),
    chart_type: str = "both",
    title: str | None = None,
) -> Figure:
    """Plot pie/bar chart of discovered message types.

    Args:
        message_types: List of MessageTypeInfo from RE analysis.
        figsize: Figure size (width, height) in inches.
        chart_type: Type of chart - "pie", "bar", or "both".
        title: Optional custom title.

    Returns:
        Matplotlib figure object.

    Example:
        >>> fig = plot_message_type_distribution(result.message_types)
    """
    if not message_types:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No message types detected", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    if chart_type == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        _plot_type_pie(ax1, message_types)
        _plot_type_bar(ax2, message_types)
    elif chart_type == "pie":
        fig, ax = plt.subplots(figsize=figsize)
        _plot_type_pie(ax, message_types)
    else:  # bar
        fig, ax = plt.subplots(figsize=figsize)
        _plot_type_bar(ax, message_types)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig.suptitle("Message Type Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_message_field_layout(
    schema: MessageSchema,
    *,
    figsize: tuple[float, float] = (14, 6),
    title: str | None = None,
    show_values: bool = True,
) -> Figure:
    """Create visual field layout diagram showing byte positions.

    Displays a horizontal layout of message fields with:
    - Field boundaries marked
    - Field types color-coded
    - Byte offsets labeled
    - Optional sample values

    Args:
        schema: MessageSchema with inferred field structure.
        figsize: Figure size (width, height) in inches.
        title: Optional custom title.
        show_values: Whether to show sample field values.

    Returns:
        Matplotlib figure object.

    Example:
        >>> from oscura.inference.message_format import infer_format
        >>> schema = infer_format(messages)
        >>> fig = plot_message_field_layout(schema)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not schema.fields:
        ax.text(0.5, 0.5, "No fields detected", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    type_colors = _get_field_type_colors()
    total_bytes = schema.total_size
    bar_height = 0.6
    y_center = 0.5

    _draw_field_rectangles(ax, schema.fields, total_bytes, bar_height, y_center, type_colors)
    _add_field_offsets(ax, schema.fields, total_bytes, bar_height, y_center)
    _add_field_legend(ax, schema.fields, type_colors)
    _format_layout_axes(ax, schema.total_size, title)

    return fig


def plot_field_confidence_heatmap(
    fields: list[InferredField],
    *,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> Figure:
    """Create heatmap showing confidence scores for field inferences.

    Displays a visual representation of how confident the inference
    algorithm is about each field's type and boundaries.

    Args:
        fields: List of InferredField objects with confidence scores.
        figsize: Figure size (width, height) in inches.
        title: Optional custom title.

    Returns:
        Matplotlib figure object.

    Example:
        >>> fig = plot_field_confidence_heatmap(schema.fields)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not fields:
        ax.text(0.5, 0.5, "No fields to display", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    # Create data matrix
    n_fields = len(fields)
    metrics = ["Confidence", "Entropy", "Variance"]
    data = np.zeros((len(metrics), n_fields))

    for i, field in enumerate(fields):
        data[0, i] = field.confidence
        # Normalize entropy (0-8 bits) to 0-1
        data[1, i] = min(field.entropy / 8.0, 1.0)
        # Normalize variance using log scale
        data[2, i] = min(np.log1p(field.variance) / 10.0, 1.0)

    # Create heatmap
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score (normalized)", fontsize=10)

    # Set labels
    field_names = [f.name for f in fields]
    ax.set_xticks(range(n_fields))
    ax.set_xticklabels(field_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(n_fields):
            value = data[i, j]
            color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    else:
        ax.set_title("Field Inference Confidence Heatmap", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


def plot_protocol_candidates(
    candidates: list[ProtocolCandidate],
    *,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    top_n: int = 10,
) -> Figure:
    """Create bar chart of protocol detection scores.

    Shows the confidence scores for each detected protocol candidate,
    with visual indicators for the evidence sources.

    Args:
        candidates: List of ProtocolCandidate objects.
        figsize: Figure size (width, height) in inches.
        title: Optional custom title.
        top_n: Maximum number of candidates to display.

    Returns:
        Matplotlib figure object.

    Example:
        >>> fig = plot_protocol_candidates(result.protocol_candidates)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not candidates:
        ax.text(0.5, 0.5, "No protocol candidates detected", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    # Sort by confidence and take top N
    sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)[:top_n]

    names = [c.name for c in sorted_candidates]
    confidences = [c.confidence for c in sorted_candidates]

    # Color based on confidence level
    colors = []
    for conf in confidences:
        if conf >= 0.8:
            colors.append("#4CAF50")  # Green - high confidence
        elif conf >= 0.5:
            colors.append("#FF9800")  # Orange - medium
        else:
            colors.append("#F44336")  # Red - low

    y_pos = np.arange(len(names))
    ax.barh(y_pos, confidences, color=colors, edgecolor="black")

    # Add evidence indicators
    for i, cand in enumerate(sorted_candidates):
        indicators = []
        if cand.port_hint:
            indicators.append("P")  # Port hint
        if cand.header_match:
            indicators.append("H")  # Header match
        if cand.matched_patterns:
            indicators.append(f"M{len(cand.matched_patterns)}")  # Pattern matches

        if indicators:
            ax.text(
                confidences[i] + 0.02,
                i,
                " ".join(indicators),
                va="center",
                fontsize=8,
                color="gray",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence Score")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")

    # Add legend for evidence indicators
    ax.text(
        0.95,
        -0.12,
        "P=Port hint, H=Header match, M#=Pattern matches",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        style="italic",
        color="gray",
    )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Protocol Detection Candidates", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_crc_parameters(
    params: CRCParameters,
    *,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> Figure:
    """Create visualization of detected CRC parameters.

    Shows the recovered CRC parameters in a visually informative format,
    including polynomial representation, configuration flags, and
    confidence metrics.

    Args:
        params: CRCParameters object with recovered CRC settings.
        figsize: Figure size (width, height) in inches.
        title: Optional custom title.

    Returns:
        Matplotlib figure object.

    Example:
        >>> from oscura.inference.crc_reverse import CRCReverser
        >>> reverser = CRCReverser()
        >>> params = reverser.reverse(messages)
        >>> fig = plot_crc_parameters(params)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _plot_crc_parameter_table(ax1, params)
    _plot_crc_confidence_gauge(ax2, params)
    _finalize_crc_plot(fig, title)
    return fig


def _plot_crc_parameter_table(ax: Any, params: CRCParameters) -> None:
    """Plot CRC parameter table.

    Args:
        ax: Axes object.
        params: CRC parameters.
    """
    ax.axis("off")
    param_lines = [
        f"Width: {params.width} bits",
        f"Polynomial: 0x{params.polynomial:0{params.width // 4}X}",
        f"Init Value: 0x{params.init:0{params.width // 4}X}",
        f"XOR Out: 0x{params.xor_out:0{params.width // 4}X}",
        f"Reflect In: {'Yes' if params.reflect_in else 'No'}",
        f"Reflect Out: {'Yes' if params.reflect_out else 'No'}",
    ]

    if params.algorithm_name:
        param_lines.insert(0, f"Algorithm: {params.algorithm_name}")

    ax.text(0.5, 0.98, "CRC Parameters", ha="center", fontsize=14, fontweight="bold")

    y_start = 0.9
    y_step = 0.12
    for i, line in enumerate(param_lines):
        y = y_start - i * y_step
        ax.text(0.1, y, line, fontsize=11, fontfamily="monospace", va="top")


def _plot_crc_confidence_gauge(ax: Any, params: CRCParameters) -> None:
    """Plot confidence gauge for CRC parameters.

    Args:
        ax: Axes object.
        params: CRC parameters.
    """
    ax.set_aspect("equal")

    # Background arc
    theta = np.linspace(0, np.pi, 100)
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, color="#E0E0E0", linewidth=20, solid_capstyle="round")

    # Confidence arc (colored)
    conf_angle = np.pi * params.confidence
    theta_conf = np.linspace(0, conf_angle, int(100 * params.confidence) + 1)
    x_conf = r * np.cos(theta_conf)
    y_conf = r * np.sin(theta_conf)

    color = _get_confidence_color(params.confidence)
    ax.plot(x_conf, y_conf, color=color, linewidth=20, solid_capstyle="round")

    # Add text annotations
    ax.text(
        0, 0.2, f"{params.confidence:.0%}", ha="center", va="center", fontsize=24, fontweight="bold"
    )
    ax.text(0, -0.1, "Confidence", ha="center", va="center", fontsize=12)
    ax.text(
        0,
        -0.4,
        f"Test Pass Rate: {params.test_pass_rate:.0%}",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")


def _get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level.

    Args:
        confidence: Confidence value (0-1).

    Returns:
        Hex color code.
    """
    if confidence >= 0.8:
        return "#4CAF50"  # Green
    if confidence >= 0.5:
        return "#FF9800"  # Orange
    return "#F44336"  # Red


def _finalize_crc_plot(fig: Figure, title: str | None) -> None:
    """Finalize CRC plot with title.

    Args:
        fig: Figure object.
        title: Optional title.
    """
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    else:
        fig.suptitle("CRC Parameter Recovery", fontsize=14, fontweight="bold")
    plt.tight_layout()


def plot_pipeline_timing(
    statistics: dict[str, Any],
    *,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> Figure:
    """Create performance metrics visualization for pipeline stages.

    Shows timing breakdown for each stage of the RE pipeline,
    identifying bottlenecks and processing efficiency.

    Args:
        statistics: Statistics dictionary from REAnalysisResult.
        figsize: Figure size (width, height) in inches.
        title: Optional custom title.

    Returns:
        Matplotlib figure object.

    Example:
        >>> fig = plot_pipeline_timing(result.statistics)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    stage_timing = statistics.get("stage_timing", {})

    if not stage_timing:
        ax1.text(0.5, 0.5, "No timing data available", ha="center", va="center", fontsize=14)
        ax1.set_axis_off()
        ax2.set_axis_off()
        return fig

    stages = list(stage_timing.keys())
    times = list(stage_timing.values())
    total_time = sum(times)

    # Left panel: Bar chart of stage times
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stages)))
    y_pos = np.arange(len(stages))

    bars = ax1.barh(y_pos, times, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s.replace("_", " ").title() for s in stages])
    ax1.set_xlabel("Time (seconds)")

    # Add time labels on bars
    for i, (_bar, t) in enumerate(zip(bars, times, strict=False)):
        pct = t / total_time * 100 if total_time > 0 else 0
        ax1.text(t + 0.01, i, f"{t:.3f}s ({pct:.1f}%)", va="center", fontsize=9)

    ax1.set_title("Stage Execution Time", fontsize=11, fontweight="bold")

    # Right panel: Pie chart of time distribution
    # Filter out very small stages for cleaner pie
    significant = [(s, t) for s, t in zip(stages, times, strict=False) if t / total_time > 0.02]
    if significant:
        pie_stages_tuple, pie_times_tuple = zip(*significant, strict=False)
        other_time = total_time - sum(pie_times_tuple)
        pie_stages: list[str] = list(pie_stages_tuple)
        pie_times: list[float] = list(pie_times_tuple)
        if other_time > 0:
            pie_stages = pie_stages + ["Other"]
            pie_times = pie_times + [other_time]

        wedges, texts, autotexts = ax2.pie(
            pie_times,
            labels=[s.replace("_", " ").title() for s in pie_stages],
            autopct="%1.1f%%",
            colors=plt.cm.viridis(np.linspace(0.2, 0.8, len(pie_stages))),
        )
        ax2.set_title("Time Distribution", fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax2.set_axis_off()

    # Add total time annotation
    fig.text(
        0.5,
        0.02,
        f"Total Pipeline Time: {total_time:.3f} seconds",
        ha="center",
        fontsize=10,
        style="italic",
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig.suptitle("Pipeline Performance Metrics", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


# ============================================================================
# Helper functions for panel plotting
# ============================================================================


def _plot_summary_stats(ax: Any, result: REAnalysisResult) -> None:
    """Plot summary statistics panel."""
    ax.axis("off")

    stats = [
        ("Flows Analyzed", str(result.flow_count)),
        ("Messages", str(result.message_count)),
        ("Message Types", str(len(result.message_types))),
        ("Protocol Candidates", str(len(result.protocol_candidates))),
        ("Field Schemas", str(len(result.field_schemas))),
        ("Duration", f"{result.duration_seconds:.2f}s"),
        ("Warnings", str(len(result.warnings))),
    ]

    ax.text(0.5, 0.95, "Analysis Summary", ha="center", fontsize=11, fontweight="bold")

    y_pos = 0.8
    for label, value in stats:
        ax.text(0.1, y_pos, f"{label}:", fontsize=9, fontweight="bold")
        ax.text(0.7, y_pos, value, fontsize=9, ha="right")
        y_pos -= 0.11


def _plot_message_types_panel(ax: Any, message_types: list[MessageTypeInfo]) -> None:
    """Plot message types panel."""
    if not message_types:
        ax.text(0.5, 0.5, "No types", ha="center", va="center")
        ax.set_title("Message Types", fontsize=10)
        return

    names = [mt.name[:15] for mt in message_types[:8]]
    counts = [mt.sample_count for mt in message_types[:8]]

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    ax.pie(counts, labels=names, colors=colors, autopct="%1.0f%%", textprops={"fontsize": 8})
    ax.set_title("Message Types", fontsize=10)


def _plot_protocol_panel(ax: Any, candidates: list[ProtocolCandidate]) -> None:
    """Plot protocol candidates panel."""
    if not candidates:
        ax.text(0.5, 0.5, "No candidates", ha="center", va="center")
        ax.set_title("Protocol Candidates", fontsize=10)
        return

    sorted_cand = sorted(candidates, key=lambda c: c.confidence, reverse=True)[:5]
    names = [c.name for c in sorted_cand]
    confs = [c.confidence for c in sorted_cand]

    y_pos = np.arange(len(names))
    colors = ["#4CAF50" if c >= 0.7 else "#FF9800" if c >= 0.4 else "#F44336" for c in confs]

    ax.barh(y_pos, confs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title("Protocol Candidates", fontsize=10)


def _plot_timing_panel(ax: Any, statistics: dict[str, Any]) -> None:
    """Plot timing panel."""
    stage_timing = statistics.get("stage_timing", {})

    if not stage_timing:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center")
        ax.set_title("Stage Timing", fontsize=10)
        return

    stages = list(stage_timing.keys())
    times = list(stage_timing.values())

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stages)))
    bars = ax.bar(range(len(stages)), times, color=colors)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace("_", "\n") for s in stages], fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title("Pipeline Stage Timing", fontsize=10)

    for bar, t in zip(bars, times, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.2f}s",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _plot_warnings_panel(ax: Any, result: REAnalysisResult) -> None:
    """Plot warnings/info panel."""
    ax.axis("off")

    ax.text(0.5, 0.95, "Status & Warnings", ha="center", fontsize=10, fontweight="bold")

    if result.warnings:
        y_pos = 0.8
        for warning in result.warnings[:5]:
            ax.text(0.05, y_pos, f"! {warning[:40]}...", fontsize=8, color="orange")
            y_pos -= 0.15
    else:
        ax.text(0.5, 0.5, "No warnings", ha="center", va="center", fontsize=10, color="green")


def _plot_type_pie(ax: Any, message_types: list[MessageTypeInfo]) -> None:
    """Plot message type pie chart."""
    names = [mt.name for mt in message_types]
    counts = [mt.sample_count for mt in message_types]

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    ax.pie(counts, labels=names, colors=colors, autopct="%1.1f%%")
    ax.set_title("Distribution by Sample Count")


def _plot_type_bar(ax: Any, message_types: list[MessageTypeInfo]) -> None:
    """Plot message type bar chart."""
    names = [mt.name for mt in message_types]
    counts = [mt.sample_count for mt in message_types]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax.bar(range(len(names)), counts, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Sample Count")
    ax.set_title("Message Count per Type")
