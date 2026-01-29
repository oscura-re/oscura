"""Packet metrics for stream analysis.

This module provides throughput, jitter, and loss rate metrics
for packet stream analysis.


Example:
    >>> from oscura.analyzers.packet.metrics import throughput, jitter, loss_rate
    >>> rate = throughput(packets)
    >>> jitter_stats = jitter(packets)

References:
    RFC 3550 for jitter calculation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


@dataclass
class PacketInfo:
    """Packet information for metrics calculation.

    Attributes:
        timestamp: Packet arrival timestamp in seconds.
        size: Packet size in bytes.
        sequence: Optional sequence number for loss detection.
    """

    timestamp: float
    size: int
    sequence: int | None = None


@dataclass
class ThroughputResult:
    """Throughput measurement result.

    Attributes:
        bytes_per_second: Data rate in bytes/second.
        bits_per_second: Data rate in bits/second.
        packets_per_second: Packet rate.
        total_bytes: Total bytes in measurement period.
        total_packets: Total packets in measurement period.
        duration: Measurement duration in seconds.
    """

    bytes_per_second: float
    bits_per_second: float
    packets_per_second: float
    total_bytes: int
    total_packets: int
    duration: float


@dataclass
class JitterResult:
    """Jitter measurement result.

    Attributes:
        mean: Mean inter-arrival time.
        std: Standard deviation of inter-arrival time.
        min: Minimum inter-arrival time.
        max: Maximum inter-arrival time.
        jitter_rfc3550: RFC 3550 jitter estimate.
    """

    mean: float
    std: float
    min: float
    max: float
    jitter_rfc3550: float


@dataclass
class LossResult:
    """Packet loss measurement result.

    Attributes:
        loss_rate: Loss rate as fraction (0-1).
        loss_percentage: Loss rate as percentage.
        packets_lost: Estimated number of lost packets.
        packets_received: Number of received packets.
        gaps: List of (start_seq, end_seq) gap ranges.
    """

    loss_rate: float
    loss_percentage: float
    packets_lost: int
    packets_received: int
    gaps: list[tuple[int, int]]


@dataclass
class LatencyResult:
    """Request-response latency result.

    Attributes:
        mean: Mean latency in seconds.
        std: Standard deviation.
        min: Minimum latency.
        max: Maximum latency.
        p50: Median latency.
        p95: 95th percentile latency.
        p99: 99th percentile latency.
        samples: Number of samples.
    """

    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    samples: int


def throughput(
    packets: Sequence[PacketInfo] | Iterator[PacketInfo],
    *,
    window_size: float | None = None,
) -> ThroughputResult:
    """Calculate throughput and packet rate.

    Args:
        packets: Sequence or iterator of packets.
        window_size: If provided, use sliding window of this duration.
            If None, calculate over entire sequence.

    Returns:
        ThroughputResult with throughput metrics.

    Example:
        >>> packets = [PacketInfo(t, sz) for t, sz in data]
        >>> result = throughput(packets)
        >>> print(f"Throughput: {result.bits_per_second / 1e6:.2f} Mbps")
    """
    packet_list = list(packets)

    if len(packet_list) < 2:
        return ThroughputResult(
            bytes_per_second=0.0,
            bits_per_second=0.0,
            packets_per_second=0.0,
            total_bytes=sum(p.size for p in packet_list),
            total_packets=len(packet_list),
            duration=0.0,
        )

    total_bytes = sum(p.size for p in packet_list)
    total_packets = len(packet_list)

    # Sort by timestamp
    sorted_packets = sorted(packet_list, key=lambda p: p.timestamp)
    duration = sorted_packets[-1].timestamp - sorted_packets[0].timestamp

    if duration <= 0:
        duration = 1e-9  # Avoid division by zero

    bytes_per_second = total_bytes / duration
    packets_per_second = total_packets / duration

    return ThroughputResult(
        bytes_per_second=bytes_per_second,
        bits_per_second=bytes_per_second * 8,
        packets_per_second=packets_per_second,
        total_bytes=total_bytes,
        total_packets=total_packets,
        duration=duration,
    )


def jitter(
    packets: Sequence[PacketInfo] | Iterator[PacketInfo],
) -> JitterResult:
    """Calculate inter-arrival time jitter.

    Computes jitter statistics including RFC 3550 jitter estimate.

    Args:
        packets: Sequence or iterator of packets with timestamps.

    Returns:
        JitterResult with jitter metrics.

    Example:
        >>> result = jitter(packets)
        >>> print(f"Jitter: {result.std * 1000:.3f} ms")

    References:
        RFC 3550 Section A.8 for jitter calculation
    """
    packet_list = list(packets)

    if len(packet_list) < 2:
        return JitterResult(
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            jitter_rfc3550=0.0,
        )

    # Sort by timestamp
    sorted_packets = sorted(packet_list, key=lambda p: p.timestamp)

    # Calculate inter-arrival times
    timestamps = np.array([p.timestamp for p in sorted_packets])
    iat = np.diff(timestamps)

    # RFC 3550 jitter estimate (smoothed absolute deviation)
    # J(i) = J(i-1) + (|D(i-1,i)| - J(i-1))/16
    # where D(i-1,i) is the difference in inter-arrival times
    if len(iat) > 1:
        d = np.diff(iat)  # Deviation from expected IAT
        jitter_rfc = 0.0
        for deviation in np.abs(d):
            jitter_rfc = jitter_rfc + (deviation - jitter_rfc) / 16
    else:
        jitter_rfc = 0.0

    return JitterResult(
        mean=float(np.mean(iat)),
        std=float(np.std(iat)),
        min=float(np.min(iat)),
        max=float(np.max(iat)),
        jitter_rfc3550=float(jitter_rfc),
    )


def loss_rate(
    packets: Sequence[PacketInfo] | Iterator[PacketInfo],
) -> LossResult:
    """Detect and report packet loss from sequence numbers.

    Args:
        packets: Sequence or iterator of packets with sequence numbers.

    Returns:
        LossResult with loss metrics.

    Example:
        >>> result = loss_rate(packets)
        >>> print(f"Loss rate: {result.loss_percentage:.2f}%")
        >>> for start, end in result.gaps:
        ...     print(f"Gap: {start} to {end}")
    """
    packet_list = list(packets)

    # Filter packets with sequence numbers
    with_seq = [(p.sequence, p.timestamp) for p in packet_list if p.sequence is not None]

    if len(with_seq) < 2:
        return LossResult(
            loss_rate=0.0,
            loss_percentage=0.0,
            packets_lost=0,
            packets_received=len(packet_list),
            gaps=[],
        )

    # Sort by sequence number
    sorted_seqs = sorted(with_seq, key=lambda x: x[0])
    sequences = [s[0] for s in sorted_seqs]

    # Find gaps in sequence
    gaps: list[tuple[int, int]] = []
    packets_lost = 0

    for i in range(1, len(sequences)):
        expected = sequences[i - 1] + 1
        actual = sequences[i]

        if actual > expected:
            # Gap detected
            gaps.append((expected, actual - 1))
            packets_lost += actual - expected

    # Calculate loss rate
    total_expected = sequences[-1] - sequences[0] + 1
    packets_received = len(sequences)

    loss_frac = packets_lost / total_expected if total_expected > 0 else 0.0

    return LossResult(
        loss_rate=loss_frac,
        loss_percentage=loss_frac * 100,
        packets_lost=packets_lost,
        packets_received=packets_received,
        gaps=gaps,
    )


def latency(
    request_times: Sequence[float] | NDArray[np.floating[Any]],
    response_times: Sequence[float] | NDArray[np.floating[Any]],
) -> LatencyResult:
    """Calculate request-response latency statistics.

    Args:
        request_times: Array of request timestamps.
        response_times: Array of corresponding response timestamps.

    Returns:
        LatencyResult with latency statistics.

    Raises:
        ValueError: If request and response arrays have different lengths.

    Example:
        >>> result = latency(request_times, response_times)
        >>> print(f"Mean latency: {result.mean * 1000:.2f} ms")
        >>> print(f"P99 latency: {result.p99 * 1000:.2f} ms")
    """
    req = np.asarray(request_times)
    resp = np.asarray(response_times)

    if len(req) != len(resp):
        raise ValueError("Request and response arrays must have same length")

    if len(req) == 0:
        return LatencyResult(
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            samples=0,
        )

    latencies = resp - req

    # Filter out negative latencies (invalid pairings)
    valid = latencies >= 0
    latencies = latencies[valid]

    if len(latencies) == 0:
        return LatencyResult(
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            samples=0,
        )

    return LatencyResult(
        mean=float(np.mean(latencies)),
        std=float(np.std(latencies)),
        min=float(np.min(latencies)),
        max=float(np.max(latencies)),
        p50=float(np.percentile(latencies, 50)),
        p95=float(np.percentile(latencies, 95)),
        p99=float(np.percentile(latencies, 99)),
        samples=len(latencies),
    )


def windowed_throughput(
    packets: Sequence[PacketInfo],
    window_size: float,
    step_size: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate throughput over sliding windows.

    Args:
        packets: Sequence of packets.
        window_size: Window size in seconds.
        step_size: Step size in seconds (default: window_size / 2).

    Returns:
        (times, throughputs) - Center times and throughput values.

    Example:
        >>> times, rates = windowed_throughput(packets, window_size=1.0)
        >>> plt.plot(times, rates / 1e6)
        >>> plt.ylabel("Throughput (Mbps)")
    """
    if step_size is None:
        step_size = window_size / 2

    packet_list = sorted(packets, key=lambda p: p.timestamp)

    if len(packet_list) < 2:
        return np.array([]), np.array([])

    start_time = packet_list[0].timestamp
    end_time = packet_list[-1].timestamp

    times = []
    throughputs = []

    window_start = start_time

    while window_start + window_size <= end_time:
        window_end = window_start + window_size

        # Count bytes in window
        window_bytes = sum(p.size for p in packet_list if window_start <= p.timestamp < window_end)

        center_time = window_start + window_size / 2
        rate = window_bytes / window_size * 8  # bits/second

        times.append(center_time)
        throughputs.append(rate)

        window_start += step_size

    return np.array(times), np.array(throughputs)


__all__ = [
    "JitterResult",
    "LatencyResult",
    "LossResult",
    "PacketInfo",
    "ThroughputResult",
    "jitter",
    "latency",
    "loss_rate",
    "throughput",
    "windowed_throughput",
]
