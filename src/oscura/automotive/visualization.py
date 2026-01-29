"""Automotive CAN bus visualization utilities.

This module provides visualization functions for CAN bus data, including
message timelines, signal plots, frequency analysis, and bus utilization.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessage, CANMessageList

__all__ = [
    "plot_bus_timeline",
    "plot_bus_utilization",
    "plot_message_distribution",
    "plot_message_frequency",
    "plot_signal_timeline",
]


def plot_bus_timeline(
    messages: CANMessageList,
    *,
    max_messages: int = 1000,
    figsize: tuple[float, float] = (12, 6),
) -> Any:
    """Plot CAN bus message timeline.

    Creates a scatter plot showing when each CAN ID appeared on the bus.
    Useful for understanding bus activity patterns and message timing.

    Args:
        messages: List of CAN messages to plot.
        max_messages: Maximum number of messages to plot (for performance).
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If messages list is empty.

    Example:
        >>> from oscura.automotive.loaders import load_automotive_log
        >>> messages = load_automotive_log("capture.blf")
        >>> plot_bus_timeline(messages)
    """
    if len(messages) == 0:
        raise ValueError("No messages to plot")

    # Limit messages for performance
    plot_messages = messages[:max_messages]

    # Ensure plot_messages is a list for iteration
    if isinstance(plot_messages, CANMessage):
        # Single message returned from indexing
        msg_list: list[CANMessage] = [plot_messages]
    else:
        # Must be list[CANMessage] from slice
        msg_list = plot_messages

    # Extract timestamps and IDs
    timestamps = [_get_timestamp(msg) for msg in msg_list]
    arb_ids = [_get_arbitration_id(msg) for msg in msg_list]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(timestamps, arb_ids, alpha=0.5, s=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CAN Arbitration ID")
    ax.set_title(f"CAN Bus Timeline ({len(msg_list)} messages)")
    ax.grid(True, alpha=0.3)

    # Format y-axis as hex
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"0x{int(x):03X}"))

    plt.tight_layout()
    return fig


def plot_message_frequency(
    messages: CANMessageList,
    *,
    top_n: int = 20,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot CAN message frequency distribution.

    Creates a bar chart showing the most frequent CAN IDs.

    Args:
        messages: List of CAN messages to analyze.
        top_n: Number of top IDs to display.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If messages list is empty.

    Example:
        >>> plot_message_frequency(messages, top_n=15)
    """
    if len(messages) == 0:
        raise ValueError("No messages to plot")

    # Count message occurrences
    id_counts = Counter(_get_arbitration_id(msg) for msg in messages)
    most_common = id_counts.most_common(top_n)

    ids = [f"0x{id_:03X}" for id_, _ in most_common]
    counts = [count for _, count in most_common]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(ids)), counts)
    ax.set_xlabel("CAN ID")
    ax.set_ylabel("Message Count")
    ax.set_title(f"Top {top_n} Most Frequent CAN IDs")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Color bars by frequency
    max_count = max(counts)
    for bar, count in zip(bars, counts, strict=False):
        bar.set_color(plt.cm.viridis(count / max_count))

    plt.tight_layout()
    return fig


def plot_signal_timeline(
    messages: CANMessageList,
    arb_id: int,
    byte_index: int,
    *,
    figsize: tuple[float, float] = (12, 4),
) -> Any:
    """Plot timeline of a specific byte in CAN messages.

    Useful for visualizing how a particular signal changes over time.

    Args:
        messages: List of CAN messages.
        arb_id: CAN arbitration ID to filter.
        byte_index: Index of byte to plot (0-7).
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If no messages found with specified ID or byte index.

    Example:
        >>> # Plot byte 0 of CAN ID 0x123
        >>> plot_signal_timeline(messages, 0x123, 0)
    """
    # Filter messages by ID
    filtered = [msg for msg in messages if _get_arbitration_id(msg) == arb_id]

    if len(filtered) == 0:
        raise ValueError(f"No messages found with ID 0x{arb_id:03X}")

    # Extract timestamps and byte values
    timestamps = []
    values = []

    for msg in filtered:
        msg_data = _get_data(msg)
        if len(msg_data) > byte_index:
            timestamps.append(_get_timestamp(msg))
            values.append(msg_data[byte_index])

    if len(values) == 0:
        raise ValueError(f"No messages with ID 0x{arb_id:03X} have byte at index {byte_index}")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, values, linewidth=1, marker=".", markersize=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Byte {byte_index} Value")
    ax.set_title(f"CAN ID 0x{arb_id:03X} - Byte {byte_index} Timeline")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bus_utilization(
    messages: CANMessageList,
    *,
    window_size: float = 1.0,
    figsize: tuple[float, float] = (12, 5),
) -> Any:
    """Plot CAN bus utilization over time.

    Calculates and plots message rate in messages/second over time windows.

    Args:
        messages: List of CAN messages.
        window_size: Time window size in seconds for calculating rate.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If messages list is empty or all messages have same timestamp.

    Example:
        >>> # Plot bus utilization with 0.5s windows
        >>> plot_bus_utilization(messages, window_size=0.5)
    """
    if len(messages) == 0:
        raise ValueError("No messages to plot")

    # Get time range - use iteration to handle type union
    first_msg: CANMessage = next(iter(messages))
    last_msg: CANMessage = messages.messages[-1]  # Access internal list directly
    start_time = _get_timestamp(first_msg)
    end_time = _get_timestamp(last_msg)
    duration = end_time - start_time

    if duration == 0:
        raise ValueError("All messages have the same timestamp")

    # Create time bins
    num_bins = int(duration / window_size) + 1
    bins = np.linspace(start_time, end_time, num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Count messages per bin
    timestamps = np.array([_get_timestamp(msg) for msg in messages])
    counts, _ = np.histogram(timestamps, bins=bins)

    # Convert to messages/second
    rates = counts / window_size

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(bin_centers, rates, linewidth=1.5)
    ax.fill_between(bin_centers, rates, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Message Rate (msg/s)")
    ax.set_title(f"CAN Bus Utilization (window={window_size}s)")
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_rate = np.mean(rates)
    ax.axhline(
        mean_rate, color="r", linestyle="--", alpha=0.5, label=f"Mean: {mean_rate:.1f} msg/s"
    )
    ax.legend()

    plt.tight_layout()
    return fig


def plot_message_distribution(
    messages: CANMessageList,
    *,
    figsize: tuple[float, float] = (10, 8),
) -> Any:
    """Plot distribution of CAN message properties.

    Creates a multi-panel plot showing:
    - DLC (data length) distribution
    - Standard vs Extended ID ratio
    - Message timing histogram

    Args:
        messages: List of CAN messages.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If messages list is empty.

    Example:
        >>> plot_message_distribution(messages)
    """
    if len(messages) == 0:
        raise ValueError("No messages to plot")

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    # 1. DLC distribution
    dlcs = [_get_dlc(msg) for msg in messages]
    dlc_counts = Counter(dlcs)
    dlc_values = sorted(dlc_counts.keys())
    dlc_counts_list = [dlc_counts[d] for d in dlc_values]

    ax1.bar(dlc_values, dlc_counts_list, color="steelblue")
    ax1.set_xlabel("Data Length Code (DLC)")
    ax1.set_ylabel("Count")
    ax1.set_title("Message DLC Distribution")
    ax1.set_xticks(range(9))
    ax1.grid(True, axis="y", alpha=0.3)

    # 2. ID type distribution
    std_count = sum(1 for msg in messages if not _is_extended(msg))
    ext_count = sum(1 for msg in messages if _is_extended(msg))

    ax2.bar(
        ["Standard (11-bit)", "Extended (29-bit)"],
        [std_count, ext_count],
        color=["green", "orange"],
    )
    ax2.set_ylabel("Count")
    ax2.set_title("ID Type Distribution")
    ax2.grid(True, axis="y", alpha=0.3)

    # 3. Inter-message timing histogram
    if len(messages) > 1:
        timestamps = [_get_timestamp(msg) for msg in messages]
        intervals = np.diff(timestamps)
        # Filter out very large gaps (likely recording breaks)
        intervals = intervals[intervals < np.percentile(intervals, 99)]

        ax3.hist(intervals * 1000, bins=50, color="purple", alpha=0.7)  # Convert to ms
        ax3.set_xlabel("Inter-message Time (ms)")
        ax3.set_ylabel("Count")
        ax3.set_title("Message Timing Distribution")
        ax3.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# Helper functions to access CANMessage attributes safely
def _get_timestamp(msg: CANMessage) -> float:
    """Get timestamp from CANMessage."""
    return float(msg.timestamp)


def _get_arbitration_id(msg: CANMessage) -> int:
    """Get arbitration_id from CANMessage."""
    return int(msg.arbitration_id)


def _get_data(msg: CANMessage) -> bytes:
    """Get data from CANMessage."""
    return bytes(msg.data)


def _get_dlc(msg: CANMessage) -> int:
    """Get DLC from CANMessage."""
    return int(msg.dlc)


def _is_extended(msg: CANMessage) -> bool:
    """Check if CANMessage has extended ID."""
    return bool(msg.is_extended)
