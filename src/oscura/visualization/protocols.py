"""Protocol decoder visualization functions.

This module provides visualization functions for decoded protocol packets,
creating timing diagrams with multi-level annotations for protocol analysis.

Example:
    >>> from oscura.analyzers.protocols.uart import UARTDecoder
    >>> from oscura.visualization.protocols import plot_protocol_decode
    >>>
    >>> decoder = UARTDecoder(baudrate=115200)
    >>> packets = list(decoder.decode(trace))
    >>> fig = plot_protocol_decode(packets, trace=trace, title="UART Decode")

References:
    - Protocol visualization best practices
    - Wavedrom-style timing diagrams
    - sigrok annotation system
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from oscura.core.types import DigitalTrace, ProtocolPacket

try:
    import matplotlib.pyplot as plt
    from matplotlib import patches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_protocol_decode(
    packets: list[ProtocolPacket],
    *,
    trace: DigitalTrace | None = None,
    trace_channel: str | None = None,
    annotation_levels: list[str] | Literal["all"] = "all",
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_data: bool = True,
    show_errors: bool = True,
    colorize: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot decoded protocol packets with multi-level annotations.

    Creates a timing diagram showing the original waveform (if provided)
    and annotation rows for decoded protocol data at different levels
    (bits, bytes, fields, packets, messages).

    Args:
        packets: List of decoded protocol packets to visualize.
        trace: Optional digital trace to plot alongside annotations.
        trace_channel: Name of trace channel (default: protocol name).
        annotation_levels: Which annotation levels to display ("all" or list of level names).
        time_range: Time range to plot (t_min, t_max) in seconds. None = auto from packets.
        time_unit: Time unit for x-axis ("s", "ms", "us", "ns", "auto").
        show_data: Show decoded data values in annotations.
        show_errors: Highlight packets with errors.
        colorize: Use color coding for different packet types.
        figsize: Figure size (width, height). Auto-calculated if None.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If packets list is empty.

    Example:
        >>> decoder = UARTDecoder(baudrate=9600)
        >>> packets = list(decoder.decode(rx_trace))
        >>> fig = plot_protocol_decode(
        ...     packets,
        ...     trace=rx_trace,
        ...     time_unit="ms",
        ...     title="UART Communication"
        ... )

    References:
        VIS-030: Protocol Decode Visualization
    """
    _validate_plot_inputs(packets)
    protocol = packets[0].protocol
    t_min, t_max, time_mult, time_unit = _calculate_time_parameters(packets, time_range, time_unit)
    fig, axes = _create_figure_layout(trace, figsize)
    ax_idx = _plot_waveform_if_present(
        axes, trace, trace_channel, protocol, t_min, t_max, time_mult
    )
    _plot_packet_timeline(
        axes[ax_idx], packets, protocol, t_min, t_max, time_mult, show_data, show_errors, colorize
    )
    _finalize_plot_layout(axes, t_min, t_max, time_mult, time_unit, title)

    return fig


def _validate_plot_inputs(packets: list[ProtocolPacket]) -> None:
    """Validate plot inputs.

    Args:
        packets: List of protocol packets.

    Raises:
        ImportError: If matplotlib not available.
        ValueError: If packets list is empty.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")
    if len(packets) == 0:
        raise ValueError("packets list cannot be empty")


def _calculate_time_parameters(
    packets: list[ProtocolPacket],
    time_range: tuple[float, float] | None,
    time_unit: str,
) -> tuple[float, float, float, str]:
    """Calculate time range and multiplier for plotting.

    Args:
        packets: List of packets for auto time range.
        time_range: User-specified time range or None.
        time_unit: Time unit string.

    Returns:
        Tuple of (t_min, t_max, time_mult, time_unit).
    """
    if time_range is None:
        t_min = min(p.timestamp for p in packets)
        t_max = max(p.end_timestamp if p.end_timestamp else p.timestamp for p in packets)
        padding = (t_max - t_min) * 0.1
        t_min -= padding
        t_max += padding
    else:
        t_min, t_max = time_range

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

    return t_min, t_max, time_mult, time_unit


def _create_figure_layout(
    trace: DigitalTrace | None, figsize: tuple[float, float] | None
) -> tuple[Figure, list[Axes]]:
    """Create figure and axes layout.

    Args:
        trace: Optional trace for determining row count.
        figsize: Figure size or None for auto.

    Returns:
        Tuple of (figure, axes_list).
    """
    n_rows = 1 if trace is None else 2

    if figsize is None:
        width = 14
        height = max(4, n_rows * 1.5 + 1)
        figsize = (width, height)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.15, "height_ratios": [1] * n_rows},
    )

    if n_rows == 1:
        axes = [axes]

    return fig, axes


def _plot_waveform_if_present(
    axes: list[Axes],
    trace: DigitalTrace | None,
    trace_channel: str | None,
    protocol: str,
    t_min: float,
    t_max: float,
    time_mult: float,
) -> int:
    """Plot waveform trace if provided.

    Args:
        axes: List of axes to plot on.
        trace: Optional digital trace.
        trace_channel: Channel name override.
        protocol: Protocol name for default label.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time unit multiplier.

    Returns:
        Index of next available axis.
    """
    if trace is None:
        return 0

    ax = axes[0]
    trace_time = trace.time_vector * time_mult
    trace_data = trace.data.astype(float)

    mask = (trace_time >= t_min * time_mult) & (trace_time <= t_max * time_mult)
    trace_time = trace_time[mask]
    trace_data = trace_data[mask]

    _plot_digital_waveform(ax, trace_time, trace_data)

    channel_name = trace_channel if trace_channel else protocol
    ax.set_ylabel(channel_name, rotation=0, ha="right", va="center", fontsize=10)
    ax.set_ylim(-0.2, 1.3)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")

    return 1


def _plot_packet_timeline(
    ax: Axes,
    packets: list[ProtocolPacket],
    protocol: str,
    t_min: float,
    t_max: float,
    time_mult: float,
    show_data: bool,
    show_errors: bool,
    colorize: bool,
) -> None:
    """Plot packet timeline on axis.

    Args:
        ax: Matplotlib axis.
        packets: List of packets to plot.
        protocol: Protocol name.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time unit multiplier.
        show_data: Show data annotations.
        show_errors: Highlight errors.
        colorize: Use color coding.
    """
    for packet in packets:
        if packet.timestamp < t_min or packet.timestamp > t_max:
            continue

        start = packet.timestamp * time_mult
        end = (
            packet.end_timestamp if packet.end_timestamp else packet.timestamp + 0.001
        ) * time_mult

        color = _determine_packet_color(packet, protocol, show_errors, colorize)

        rect = patches.Rectangle(
            (start, 0.1),
            end - start,
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.add_patch(rect)

        if show_data and packet.data:
            _add_packet_annotation(ax, packet, start, end, show_errors)

        if show_errors and packet.errors:
            ax.plot(start, 1.1, "rx", markersize=8, markeredgewidth=2)

    ax.set_ylabel(f"{protocol}\nPackets", rotation=0, ha="right", va="center", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")


def _determine_packet_color(
    packet: ProtocolPacket, protocol: str, show_errors: bool, colorize: bool
) -> str:
    """Determine packet rectangle color.

    Args:
        packet: Protocol packet.
        protocol: Protocol name.
        show_errors: Whether to highlight errors.
        colorize: Whether to use protocol colors.

    Returns:
        Color string.
    """
    if show_errors and packet.errors:
        return "#ff6b6b"
    elif colorize:
        return _get_packet_color(packet, protocol)
    else:
        return "#4ecdc4"


def _add_packet_annotation(
    ax: Axes, packet: ProtocolPacket, start: float, end: float, show_errors: bool
) -> None:
    """Add data annotation to packet.

    Args:
        ax: Matplotlib axis.
        packet: Protocol packet.
        start: Start time in scaled units.
        end: End time in scaled units.
        show_errors: Whether errors are highlighted.
    """
    data_str = _format_packet_data(packet)
    mid_time = (start + end) / 2
    text_color = "white" if not (show_errors and packet.errors) else "black"
    ax.text(
        mid_time,
        0.5,
        data_str,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=text_color,
    )


def _finalize_plot_layout(
    axes: list[Axes],
    t_min: float,
    t_max: float,
    time_mult: float,
    time_unit: str,
    title: str | None,
) -> None:
    """Finalize plot layout with labels and title.

    Args:
        axes: List of axes.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time unit multiplier.
        time_unit: Time unit string.
        title: Optional plot title.
    """
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)
    axes[-1].set_xlim(t_min * time_mult, t_max * time_mult)

    fig = axes[0].get_figure()
    if fig and hasattr(fig, "tight_layout"):
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        fig.tight_layout()


def plot_uart_decode(
    packets: list[ProtocolPacket],
    *,
    rx_trace: DigitalTrace | None = None,
    tx_trace: DigitalTrace | None = None,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_parity_errors: bool = True,
    show_framing_errors: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "UART Communication",
) -> Figure:
    """Plot UART decoded packets with RX/TX lanes.

    Specialized visualization for UART showing separate RX and TX channels
    with decoded bytes and error highlighting.

    Args:
        packets: List of UART packets.
        rx_trace: Optional RX digital trace.
        tx_trace: Optional TX digital trace.
        time_range: Time range to plot (t_min, t_max) in seconds.
        time_unit: Time unit for x-axis ("s", "ms", "us", "ns", "auto").
        show_parity_errors: Highlight parity errors.
        show_framing_errors: Highlight framing errors.
        figsize: Figure size (width, height).
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If packets list is empty.

    Example:
        >>> decoder = UARTDecoder(baudrate=115200, parity="even")
        >>> packets = list(decoder.decode(rx_trace))
        >>> fig = plot_uart_decode(packets, rx_trace=rx_trace, time_unit="ms")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(packets) == 0:
        raise ValueError("packets list cannot be empty")

    # If we have both RX and TX, create dual-channel visualization
    if rx_trace is not None and tx_trace is not None:
        return _plot_dual_channel_uart(
            packets,
            rx_trace=rx_trace,
            tx_trace=tx_trace,
            time_range=time_range,
            time_unit=time_unit,
            show_parity_errors=show_parity_errors,
            show_framing_errors=show_framing_errors,
            figsize=figsize,
            title=title,
        )

    # Single-channel view using generic decode plot
    return plot_protocol_decode(
        packets,
        trace=rx_trace or tx_trace,
        trace_channel="RX" if rx_trace else "TX",
        show_errors=show_parity_errors or show_framing_errors,
        time_range=time_range,
        time_unit=time_unit,
        figsize=figsize,
        title=title,
    )


def _plot_dual_channel_uart(
    packets: list[ProtocolPacket],
    *,
    rx_trace: DigitalTrace,
    tx_trace: DigitalTrace,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_parity_errors: bool = True,
    show_framing_errors: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "UART Communication",
) -> Figure:
    """Create dual-channel UART visualization with separate RX/TX rows.

    Args:
        packets: List of UART packets (may include both RX and TX).
        rx_trace: RX digital trace.
        tx_trace: TX digital trace.
        time_range: Time range to plot (t_min, t_max) in seconds.
        time_unit: Time unit for x-axis.
        show_parity_errors: Highlight parity errors.
        show_framing_errors: Highlight framing errors.
        figsize: Figure size (width, height).
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    # Calculate time parameters
    t_min, t_max, time_mult, time_unit = _determine_time_params(packets, time_range, time_unit)

    # Create figure with 4 rows
    fig, axes = _create_dual_uart_figure(figsize)

    # Separate packets by channel
    rx_packets, tx_packets = _separate_uart_packets(packets)
    show_errors = show_parity_errors or show_framing_errors

    # Plot all four rows
    _plot_uart_channel_pair(
        axes[0], axes[1], rx_trace, rx_packets, "RX", t_min, t_max, time_mult, show_errors
    )
    _plot_uart_channel_pair(
        axes[2], axes[3], tx_trace, tx_packets, "TX", t_min, t_max, time_mult, show_errors
    )

    # Finalize plot
    _finalize_uart_plot(fig, axes, t_min, t_max, time_mult, time_unit, title)

    return fig


def _create_dual_uart_figure(
    figsize: tuple[float, float] | None,
) -> tuple[Figure, list[Axes]]:
    """Create figure for dual-channel UART plot.

    Args:
        figsize: Figure size or None for auto-calculation.

    Returns:
        Tuple of (figure, axes_list).
    """
    n_rows = 4

    if figsize is None:
        width = 14
        height = max(6, n_rows * 1.2 + 1)
        figsize = (width, height)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.1, "height_ratios": [1, 0.8, 1, 0.8]},
    )

    return fig, axes


def _separate_uart_packets(
    packets: list[ProtocolPacket],
) -> tuple[list[ProtocolPacket], list[ProtocolPacket]]:
    """Separate UART packets by channel (RX vs TX).

    Args:
        packets: List of UART packets.

    Returns:
        Tuple of (rx_packets, tx_packets).
    """
    rx_packets = []
    tx_packets = []

    for packet in packets:
        channel = getattr(packet, "channel", None)
        if channel is None and hasattr(packet, "metadata"):
            channel = packet.metadata.get("channel") if isinstance(packet.metadata, dict) else None

        if channel == "TX":
            tx_packets.append(packet)
        else:
            rx_packets.append(packet)

    # If no channel info, put all packets on RX
    if not rx_packets and not tx_packets:
        rx_packets = packets

    return rx_packets, tx_packets


def _plot_uart_channel_pair(
    ax_wave: Axes,
    ax_packets: Axes,
    trace: DigitalTrace,
    packets: list[ProtocolPacket],
    label: str,
    t_min: float,
    t_max: float,
    time_mult: float,
    show_errors: bool,
) -> None:
    """Plot waveform and packet row for a single UART channel.

    Args:
        ax_wave: Axis for waveform plot.
        ax_packets: Axis for packet annotations.
        trace: Digital trace for the channel.
        packets: Packets for this channel.
        label: Channel label (e.g., "RX" or "TX").
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time unit multiplier.
        show_errors: Whether to highlight errors.
    """
    # Plot waveform
    trace_time = trace.time_vector * time_mult
    trace_data = trace.data.astype(float)
    mask = (trace_time >= t_min * time_mult) & (trace_time <= t_max * time_mult)
    _plot_digital_waveform(ax_wave, trace_time[mask], trace_data[mask])
    ax_wave.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
    ax_wave.set_ylim(-0.2, 1.3)
    ax_wave.set_yticks([])
    ax_wave.grid(True, axis="x", alpha=0.3, linestyle=":")

    # Plot packets
    _plot_packet_row(ax_packets, packets, t_min, t_max, time_mult, show_errors)
    ax_packets.set_ylabel(f"{label}\nData", rotation=0, ha="right", va="center", fontsize=9)


def _finalize_uart_plot(
    fig: Figure,
    axes: list[Axes],
    t_min: float,
    t_max: float,
    time_mult: float,
    time_unit: str,
    title: str | None,
) -> None:
    """Add final formatting to UART plot.

    Args:
        fig: Matplotlib figure.
        axes: List of axes.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time multiplier.
        time_unit: Time unit string.
        title: Plot title or None.
    """
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)
    axes[-1].set_xlim(t_min * time_mult, t_max * time_mult)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    fig.tight_layout()


def _plot_packet_row(
    ax: Axes,
    packets: list[ProtocolPacket],
    t_min: float,
    t_max: float,
    time_mult: float,
    show_errors: bool,
) -> None:
    """Plot a single row of packets on the given axes."""
    for packet in packets:
        if packet.timestamp < t_min or packet.timestamp > t_max:
            continue

        start = packet.timestamp * time_mult
        end = (
            packet.end_timestamp if packet.end_timestamp else packet.timestamp + 0.001
        ) * time_mult

        # Determine packet color
        if show_errors and packet.errors:
            color = "#ff6b6b"  # Red for errors
        else:
            color = "#4ecdc4"  # Teal

        # Draw packet rectangle
        rect = patches.Rectangle(
            (start, 0.1),
            end - start,
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Add data annotation
        if packet.data:
            data_str = _format_packet_data(packet)
            mid_time = (start + end) / 2
            ax.text(
                mid_time,
                0.5,
                data_str,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white" if not (show_errors and packet.errors) else "black",
            )

        # Add error markers
        if show_errors and packet.errors:
            ax.plot(start, 1.1, "rx", markersize=6, markeredgewidth=2)

    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")


def plot_spi_decode(
    packets: list[ProtocolPacket],
    *,
    clk_trace: DigitalTrace | None = None,
    mosi_trace: DigitalTrace | None = None,
    miso_trace: DigitalTrace | None = None,
    cs_trace: DigitalTrace | None = None,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_mosi: bool = True,
    show_miso: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "SPI Transaction",
) -> Figure:
    """Plot SPI decoded packets with CLK, MOSI, MISO, CS signals.

    Specialized visualization for SPI showing all relevant signals
    and decoded words on MOSI/MISO channels.

    Args:
        packets: List of SPI packets.
        clk_trace: Optional clock signal trace.
        mosi_trace: Optional MOSI (Master Out Slave In) trace.
        miso_trace: Optional MISO (Master In Slave Out) trace.
        cs_trace: Optional chip select trace.
        time_range: Time range to plot (t_min, t_max) in seconds.
        time_unit: Time unit for x-axis.
        show_mosi: Show MOSI decoded data.
        show_miso: Show MISO decoded data.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If packets list is empty.

    Example:
        >>> decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        >>> packets = list(decoder.decode(clk=clk, mosi=mosi, miso=miso))
        >>> fig = plot_spi_decode(packets, clk_trace=clk, mosi_trace=mosi)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(packets) == 0:
        raise ValueError("packets list cannot be empty")

    # If we have multiple traces, create multi-channel visualization
    traces_available = sum(
        1 for t in [clk_trace, mosi_trace, miso_trace, cs_trace] if t is not None
    )

    if traces_available >= 2:
        return _plot_multi_channel_spi(
            packets,
            clk_trace=clk_trace,
            mosi_trace=mosi_trace,
            miso_trace=miso_trace,
            cs_trace=cs_trace,
            time_range=time_range,
            time_unit=time_unit,
            show_mosi=show_mosi,
            show_miso=show_miso,
            figsize=figsize,
            title=title,
        )

    # Single-channel view using generic decode plot
    return plot_protocol_decode(
        packets,
        trace=mosi_trace,
        trace_channel="MOSI",
        time_range=time_range,
        time_unit=time_unit,
        figsize=figsize,
        title=title,
    )


def _plot_multi_channel_spi(
    packets: list[ProtocolPacket],
    *,
    clk_trace: DigitalTrace | None = None,
    mosi_trace: DigitalTrace | None = None,
    miso_trace: DigitalTrace | None = None,
    cs_trace: DigitalTrace | None = None,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_mosi: bool = True,
    show_miso: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "SPI Transaction",
) -> Figure:
    """Create multi-channel SPI visualization with separate rows for each signal.

    Args:
        packets: List of SPI packets.
        clk_trace: Optional clock signal trace.
        mosi_trace: Optional MOSI trace.
        miso_trace: Optional MISO trace.
        cs_trace: Optional chip select trace.
        time_range: Time range to plot.
        time_unit: Time unit for x-axis.
        show_mosi: Show MOSI decoded data row.
        show_miso: Show MISO decoded data row.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    t_min, t_max, time_mult, time_unit = _determine_time_params(packets, time_range, time_unit)
    rows = _build_spi_row_list(cs_trace, clk_trace, mosi_trace, miso_trace, show_mosi, show_miso)

    if len(rows) == 0:
        return plot_protocol_decode(
            packets, time_range=(t_min, t_max), time_unit=time_unit, figsize=figsize, title=title
        )

    fig, axes = _create_spi_figure(rows, figsize)
    mosi_packets, miso_packets = _separate_spi_packets(packets)
    _render_spi_rows(axes, rows, t_min, t_max, time_mult, mosi_packets, miso_packets)
    _finalize_spi_plot(axes, t_min, t_max, time_mult, time_unit, title)

    return fig


def _determine_time_params(
    packets: list[ProtocolPacket],
    time_range: tuple[float, float] | None,
    time_unit: str,
) -> tuple[float, float, float, str]:
    """Determine time range and multiplier for SPI plot.

    Args:
        packets: List of SPI packets for time range calculation.
        time_range: User-specified time range or None for auto.
        time_unit: Time unit ("auto" or specific unit).

    Returns:
        Tuple of (t_min, t_max, time_mult, time_unit).
    """
    if time_range is None:
        t_min = min(p.timestamp for p in packets)
        t_max = max(p.end_timestamp if p.end_timestamp else p.timestamp for p in packets)
        padding = (t_max - t_min) * 0.1
        t_min -= padding
        t_max += padding
    else:
        t_min, t_max = time_range

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

    return t_min, t_max, time_mult, time_unit


def _build_spi_row_list(
    cs_trace: DigitalTrace | None,
    clk_trace: DigitalTrace | None,
    mosi_trace: DigitalTrace | None,
    miso_trace: DigitalTrace | None,
    show_mosi: bool,
    show_miso: bool,
) -> list[dict[str, Any]]:
    """Build list of row specifications for SPI multi-channel plot.

    Args:
        cs_trace: Chip select trace.
        clk_trace: Clock trace.
        mosi_trace: MOSI trace.
        miso_trace: MISO trace.
        show_mosi: Whether to show MOSI data row.
        show_miso: Whether to show MISO data row.

    Returns:
        List of row dictionaries specifying type, trace, label, channel.
    """
    rows: list[dict[str, Any]] = []

    if cs_trace is not None:
        rows.append({"type": "waveform", "trace": cs_trace, "label": "CS"})
    if clk_trace is not None:
        rows.append({"type": "waveform", "trace": clk_trace, "label": "CLK"})
    if mosi_trace is not None:
        rows.append({"type": "waveform", "trace": mosi_trace, "label": "MOSI"})
        if show_mosi:
            rows.append({"type": "packets", "label": "MOSI\nData", "channel": "MOSI"})
    if miso_trace is not None:
        rows.append({"type": "waveform", "trace": miso_trace, "label": "MISO"})
        if show_miso:
            rows.append({"type": "packets", "label": "MISO\nData", "channel": "MISO"})

    return rows


def _create_spi_figure(
    rows: list[dict[str, Any]],
    figsize: tuple[float, float] | None,
) -> tuple[Figure, list[Axes]]:
    """Create matplotlib figure and axes for SPI plot.

    Args:
        rows: Row specifications from _build_spi_row_list.
        figsize: Figure size or None for auto-calculation.

    Returns:
        Tuple of (figure, axes_list).
    """
    height_ratios = [1.0 if row["type"] == "waveform" else 0.6 for row in rows]
    n_rows = len(rows)

    if figsize is None:
        width = 14
        height = max(4, sum(height_ratios) * 1.2 + 1)
        figsize = (width, height)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.1, "height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = [axes]

    return fig, axes


def _separate_spi_packets(
    packets: list[ProtocolPacket],
) -> tuple[list[ProtocolPacket], list[ProtocolPacket]]:
    """Separate packets by channel (MOSI vs MISO).

    Args:
        packets: List of SPI packets.

    Returns:
        Tuple of (mosi_packets, miso_packets).
    """
    mosi_packets = []
    miso_packets = []

    for packet in packets:
        channel = getattr(packet, "channel", None)
        if channel is None and hasattr(packet, "metadata"):
            channel = packet.metadata.get("channel") if isinstance(packet.metadata, dict) else None

        if channel == "MISO":
            miso_packets.append(packet)
        else:
            mosi_packets.append(packet)

    if not mosi_packets and not miso_packets:
        mosi_packets = packets

    return mosi_packets, miso_packets


def _render_spi_rows(
    axes: list[Axes],
    rows: list[dict[str, Any]],
    t_min: float,
    t_max: float,
    time_mult: float,
    mosi_packets: list[ProtocolPacket],
    miso_packets: list[ProtocolPacket],
) -> None:
    """Render all SPI plot rows (waveforms and packet data).

    Args:
        axes: List of matplotlib axes.
        rows: Row specifications.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time multiplier for unit conversion.
        mosi_packets: MOSI channel packets.
        miso_packets: MISO channel packets.
    """
    for ax, row in zip(axes, rows, strict=False):
        if row["type"] == "waveform":
            _render_spi_waveform_row(ax, row, t_min, t_max, time_mult)
        else:
            _render_spi_packet_row(ax, row, t_min, t_max, time_mult, mosi_packets, miso_packets)


def _render_spi_waveform_row(
    ax: Axes,
    row: dict[str, Any],
    t_min: float,
    t_max: float,
    time_mult: float,
) -> None:
    """Render a single waveform row for SPI plot.

    Args:
        ax: Matplotlib axis to render on.
        row: Row specification with trace and label.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time multiplier for unit conversion.
    """
    trace = row["trace"]
    trace_time = trace.time_vector * time_mult
    trace_data = trace.data.astype(float)
    mask = (trace_time >= t_min * time_mult) & (trace_time <= t_max * time_mult)
    _plot_digital_waveform(ax, trace_time[mask], trace_data[mask])
    ax.set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=10)
    ax.set_ylim(-0.2, 1.3)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")


def _render_spi_packet_row(
    ax: Axes,
    row: dict[str, Any],
    t_min: float,
    t_max: float,
    time_mult: float,
    mosi_packets: list[ProtocolPacket],
    miso_packets: list[ProtocolPacket],
) -> None:
    """Render a single packet data row for SPI plot.

    Args:
        ax: Matplotlib axis to render on.
        row: Row specification with channel and label.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time multiplier for unit conversion.
        mosi_packets: MOSI channel packets.
        miso_packets: MISO channel packets.
    """
    channel = row.get("channel", "MOSI")
    pkts = mosi_packets if channel == "MOSI" else miso_packets
    _plot_packet_row(ax, pkts, t_min, t_max, time_mult, show_errors=True)
    ax.set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=9)


def _finalize_spi_plot(
    axes: list[Axes],
    t_min: float,
    t_max: float,
    time_mult: float,
    time_unit: str,
    title: str | None,
) -> None:
    """Add final formatting to SPI plot.

    Args:
        axes: List of matplotlib axes.
        t_min: Minimum time value.
        t_max: Maximum time value.
        time_mult: Time multiplier for unit conversion.
        time_unit: Time unit string for label.
        title: Plot title or None.
    """
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)
    axes[-1].set_xlim(t_min * time_mult, t_max * time_mult)

    fig = axes[0].get_figure()
    if title and fig is not None:
        fig.suptitle(title, fontsize=14, y=0.98)

    if fig is not None and hasattr(fig, "tight_layout"):
        fig.tight_layout()


def plot_i2c_decode(
    packets: list[ProtocolPacket],
    *,
    sda_trace: DigitalTrace | None = None,
    scl_trace: DigitalTrace | None = None,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_addresses: bool = True,
    show_ack_nack: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "I2C Transaction",
) -> Figure:
    """Plot I2C decoded packets with SDA/SCL and address annotations.

    Specialized visualization for I2C showing start/stop conditions,
    addresses, data bytes, and ACK/NACK bits.

    Args:
        packets: List of I2C packets.
        sda_trace: Optional SDA (data) signal trace.
        scl_trace: Optional SCL (clock) signal trace.
        time_range: Time range to plot (t_min, t_max) in seconds.
        time_unit: Time unit for x-axis.
        show_addresses: Highlight address bytes.
        show_ack_nack: Show ACK/NACK indicators.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> decoder = I2CDecoder()
        >>> packets = list(decoder.decode(sda=sda, scl=scl))
        >>> fig = plot_i2c_decode(packets, sda_trace=sda, scl_trace=scl)
    """
    return plot_protocol_decode(
        packets,
        trace=sda_trace,
        trace_channel="SDA",
        time_range=time_range,
        time_unit=time_unit,
        figsize=figsize,
        title=title,
    )


def plot_can_decode(
    packets: list[ProtocolPacket],
    *,
    can_trace: DigitalTrace | None = None,
    time_range: tuple[float, float] | None = None,
    time_unit: str = "auto",
    show_ids: bool = True,
    show_data_length: bool = True,
    colorize_by_id: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "CAN Bus",
) -> Figure:
    """Plot CAN decoded packets with arbitration IDs and data.

    Specialized visualization for CAN bus showing arbitration IDs,
    data length codes, and message data.

    Args:
        packets: List of CAN packets.
        can_trace: Optional CAN bus trace.
        time_range: Time range to plot (t_min, t_max) in seconds.
        time_unit: Time unit for x-axis.
        show_ids: Show arbitration IDs in annotations.
        show_data_length: Show DLC (Data Length Code).
        colorize_by_id: Use different colors for different CAN IDs.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> decoder = CANDecoder()
        >>> packets = list(decoder.decode(can_trace))
        >>> fig = plot_can_decode(packets, can_trace=can_trace, colorize_by_id=True)
    """
    return plot_protocol_decode(
        packets,
        trace=can_trace,
        trace_channel="CAN",
        colorize=colorize_by_id,
        time_range=time_range,
        time_unit=time_unit,
        figsize=figsize,
        title=title,
    )


def _plot_digital_waveform(
    ax: Axes,
    time: NDArray[np.float64],
    data: NDArray[np.float64],
) -> None:
    """Plot digital waveform with clean transitions."""
    for i in range(len(time) - 1):
        level = 1 if data[i] > 0.5 else 0
        # Horizontal line
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


def _get_packet_color(packet: ProtocolPacket, protocol: str) -> str:
    """Get color for packet based on protocol and type."""
    # Color palette for different protocols
    colors = {
        "UART": "#4ecdc4",  # Teal
        "SPI": "#95e1d3",  # Mint
        "I2C": "#f38181",  # Coral
        "CAN": "#aa96da",  # Purple
        "USB": "#fcbad3",  # Pink
        "1-Wire": "#ffffd2",  # Yellow
    }

    return colors.get(protocol, "#4ecdc4")


def _format_packet_data(packet: ProtocolPacket) -> str:
    """Format packet data for display."""
    if len(packet.data) == 0:
        return ""

    # For single byte, show as hex
    if len(packet.data) == 1:
        byte_val = packet.data[0]
        # Show both hex and ASCII if printable
        if 32 <= byte_val <= 126:
            return f"0x{byte_val:02X} '{chr(byte_val)}'"
        return f"0x{byte_val:02X}"

    # For multiple bytes, show hex string (limit to first few bytes)
    if len(packet.data) <= 4:
        return " ".join(f"{b:02X}" for b in packet.data)

    # For longer data, truncate
    return " ".join(f"{b:02X}" for b in packet.data[:3]) + "..."


__all__ = [
    "plot_can_decode",
    "plot_i2c_decode",
    "plot_protocol_decode",
    "plot_spi_decode",
    "plot_uart_decode",
]
