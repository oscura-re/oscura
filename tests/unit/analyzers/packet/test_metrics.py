"""Comprehensive unit tests for packet metrics module.

Tests packet stream analysis metrics:
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.packet.metrics import (
    JitterResult,
    LatencyResult,
    LossResult,
    PacketInfo,
    ThroughputResult,
    jitter,
    latency,
    loss_rate,
    throughput,
    windowed_throughput,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.packet]


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestThroughput:
    """Test throughput calculation."""

    def test_basic_throughput(self):
        """Test basic throughput calculation."""
        packets = [
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=1.0, size=100),
            PacketInfo(timestamp=2.0, size=100),
        ]

        result = throughput(packets)

        assert isinstance(result, ThroughputResult)
        assert result.total_bytes == 300
        assert result.total_packets == 3
        assert result.bytes_per_second > 0
        assert result.bits_per_second == result.bytes_per_second * 8

    def test_throughput_with_iterator(self):
        """Test throughput with iterator input."""

        def packet_gen():
            for i in range(5):
                yield PacketInfo(timestamp=float(i), size=50)

        result = throughput(packet_gen())

        assert result.total_packets == 5
        assert result.total_bytes == 250

    def test_single_packet(self):
        """Test throughput with single packet."""
        packets = [PacketInfo(timestamp=0.0, size=100)]

        result = throughput(packets)

        assert result.total_packets == 1
        assert result.bytes_per_second == 0.0  # Can't calculate rate

    def test_empty_packets(self):
        """Test throughput with no packets."""
        result = throughput([])

        assert result.total_packets == 0
        assert result.bytes_per_second == 0.0

    def test_throughput_duration(self):
        """Test duration calculation."""
        packets = [
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=5.0, size=100),
        ]

        result = throughput(packets)

        assert result.duration == 5.0

    def test_unsorted_packets(self):
        """Test throughput with unsorted timestamps."""
        packets = [
            PacketInfo(timestamp=2.0, size=100),
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=1.0, size=100),
        ]

        result = throughput(packets)

        # Should sort internally
        assert result.duration == 2.0

    def test_packets_per_second(self):
        """Test packet rate calculation."""
        packets = [
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=0.5, size=100),
            PacketInfo(timestamp=1.0, size=100),
        ]

        result = throughput(packets)

        assert result.packets_per_second == 3.0

    def test_zero_duration_protection(self):
        """Test zero duration edge case."""
        packets = [
            PacketInfo(timestamp=1.0, size=100),
            PacketInfo(timestamp=1.0, size=100),
        ]

        result = throughput(packets)

        # Should handle without division by zero
        assert result.bytes_per_second > 0


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestJitter:
    """Test inter-arrival time jitter measurement."""

    def test_no_jitter(self):
        """Test jitter with perfectly regular packets."""
        packets = [PacketInfo(timestamp=float(i * 0.1), size=100) for i in range(10)]

        result = jitter(packets)

        assert isinstance(result, JitterResult)
        assert result.std < 1e-10  # Nearly zero jitter
        assert result.mean > 0

    def test_jitter_with_variation(self):
        """Test jitter with timestamp variation."""
        rng = np.random.default_rng(42)
        timestamps = [0.1 * i + rng.normal(0, 0.01) for i in range(20)]
        packets = [PacketInfo(timestamp=t, size=100) for t in timestamps]

        result = jitter(packets)

        assert result.std > 0
        assert result.jitter_rfc3550 >= 0

    def test_jitter_min_max(self):
        """Test jitter min/max values."""
        packets = [
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=0.1, size=100),
            PacketInfo(timestamp=0.25, size=100),
            PacketInfo(timestamp=0.35, size=100),
        ]

        result = jitter(packets)

        assert result.min <= result.max
        assert result.min > 0

    def test_rfc3550_jitter(self):
        """Test RFC 3550 jitter calculation."""
        packets = [PacketInfo(timestamp=0.1 * i, size=100) for i in range(10)]

        result = jitter(packets)

        assert result.jitter_rfc3550 >= 0

    def test_single_packet_jitter(self):
        """Test jitter with single packet."""
        packets = [PacketInfo(timestamp=0.0, size=100)]

        result = jitter(packets)

        assert result.mean == 0.0
        assert result.std == 0.0

    def test_two_packets_jitter(self):
        """Test jitter with two packets."""
        packets = [
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=0.1, size=100),
        ]

        result = jitter(packets)

        assert result.mean == 0.1
        assert result.jitter_rfc3550 >= 0

    def test_unsorted_jitter(self):
        """Test jitter with unsorted packets."""
        packets = [
            PacketInfo(timestamp=0.2, size=100),
            PacketInfo(timestamp=0.0, size=100),
            PacketInfo(timestamp=0.1, size=100),
        ]

        result = jitter(packets)

        # Should sort internally
        assert result.mean > 0

    def test_jitter_with_iterator(self):
        """Test jitter with iterator input."""

        def packet_gen():
            for i in range(5):
                yield PacketInfo(timestamp=0.1 * i, size=100)

        result = jitter(packet_gen())

        assert isinstance(result, JitterResult)


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestLossRate:
    """Test packet loss detection from sequence numbers."""

    def test_no_loss(self):
        """Test loss rate with no missing packets."""
        packets = [PacketInfo(timestamp=0.0, size=100, sequence=i) for i in range(10)]

        result = loss_rate(packets)

        assert isinstance(result, LossResult)
        assert result.loss_rate == 0.0
        assert result.packets_lost == 0
        assert len(result.gaps) == 0

    def test_single_gap(self):
        """Test loss rate with single gap."""
        packets = [
            PacketInfo(timestamp=0.0, size=100, sequence=0),
            PacketInfo(timestamp=0.1, size=100, sequence=1),
            PacketInfo(timestamp=0.2, size=100, sequence=2),
            PacketInfo(timestamp=0.3, size=100, sequence=5),  # Gap: 3, 4 missing
        ]

        result = loss_rate(packets)

        assert result.packets_lost == 2
        assert len(result.gaps) == 1
        assert result.gaps[0] == (3, 4)

    def test_multiple_gaps(self):
        """Test loss rate with multiple gaps."""
        packets = [
            PacketInfo(timestamp=0.0, size=100, sequence=0),
            PacketInfo(timestamp=0.1, size=100, sequence=3),  # Gap: 1, 2
            PacketInfo(timestamp=0.2, size=100, sequence=6),  # Gap: 4, 5
        ]

        result = loss_rate(packets)

        assert result.packets_lost == 4
        assert len(result.gaps) == 2

    def test_loss_percentage(self):
        """Test loss percentage calculation."""
        packets = [
            PacketInfo(timestamp=0.0, size=100, sequence=0),
            PacketInfo(timestamp=0.1, size=100, sequence=2),  # 1 missing
        ]

        result = loss_rate(packets)

        assert result.loss_percentage == pytest.approx(33.333, rel=0.01)

    def test_no_sequence_numbers(self):
        """Test loss rate with no sequence numbers."""
        packets = [PacketInfo(timestamp=0.0, size=100) for _ in range(5)]

        result = loss_rate(packets)

        assert result.packets_lost == 0
        assert len(result.gaps) == 0

    def test_partial_sequence_numbers(self):
        """Test with some packets having sequence numbers."""
        packets = [
            PacketInfo(timestamp=0.0, size=100, sequence=0),
            PacketInfo(timestamp=0.1, size=100),  # No sequence
            PacketInfo(timestamp=0.2, size=100, sequence=2),
        ]

        result = loss_rate(packets)

        # Should only analyze packets with sequence numbers
        assert isinstance(result, LossResult)

    def test_unsorted_sequences(self):
        """Test loss rate with unsorted sequence numbers."""
        packets = [
            PacketInfo(timestamp=0.0, size=100, sequence=2),
            PacketInfo(timestamp=0.1, size=100, sequence=0),
            PacketInfo(timestamp=0.2, size=100, sequence=1),
        ]

        result = loss_rate(packets)

        # Should sort by sequence number
        assert result.packets_lost == 0

    def test_single_packet_loss(self):
        """Test loss rate with single packet."""
        packets = [PacketInfo(timestamp=0.0, size=100, sequence=0)]

        result = loss_rate(packets)

        assert result.loss_rate == 0.0

    def test_empty_packets_loss(self):
        """Test loss rate with no packets."""
        result = loss_rate([])

        assert result.loss_rate == 0.0
        assert result.packets_lost == 0


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestLatency:
    """Test request-response latency calculation."""

    def test_basic_latency(self):
        """Test basic latency calculation."""
        request_times = np.array([0.0, 1.0, 2.0])
        response_times = np.array([0.1, 1.1, 2.1])

        result = latency(request_times, response_times)

        assert isinstance(result, LatencyResult)
        assert result.mean == pytest.approx(0.1)
        assert result.std == pytest.approx(0.0)
        assert result.samples == 3

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        request_times = np.linspace(0, 1, 100)
        response_times = request_times + 0.1  # Fixed 100ms latency

        result = latency(request_times, response_times)

        assert result.p50 == pytest.approx(0.1, rel=0.01)
        assert result.p95 == pytest.approx(0.1, rel=0.01)
        assert result.p99 == pytest.approx(0.1, rel=0.01)

    def test_latency_variation(self):
        """Test latency with variation."""
        request_times = np.array([0.0, 1.0, 2.0])
        response_times = np.array([0.05, 1.15, 2.1])  # Varying latencies

        result = latency(request_times, response_times)

        assert result.min < result.max
        assert result.std > 0

    def test_latency_min_max(self):
        """Test latency min/max values."""
        request_times = np.array([0.0, 1.0, 2.0])
        response_times = np.array([0.05, 1.2, 2.1])

        result = latency(request_times, response_times)

        assert result.min == pytest.approx(0.05)
        assert result.max == pytest.approx(0.2)

    def test_negative_latency_filtered(self):
        """Test that negative latencies are filtered out."""
        request_times = np.array([0.0, 1.0, 2.0])
        response_times = np.array([0.1, 0.5, 2.1])  # response[1] before request[1]

        result = latency(request_times, response_times)

        # Should filter out negative latency
        assert result.samples <= 3

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        request_times = np.array([0.0, 1.0])
        response_times = np.array([0.1])

        with pytest.raises(ValueError, match="same length"):
            latency(request_times, response_times)

    def test_empty_latency(self):
        """Test latency with empty arrays."""
        request_times = np.array([])
        response_times = np.array([])

        result = latency(request_times, response_times)

        assert result.samples == 0
        assert result.mean == 0.0

    def test_list_input(self):
        """Test latency with list input."""
        request_times = [0.0, 1.0, 2.0]
        response_times = [0.1, 1.1, 2.1]

        result = latency(request_times, response_times)

        assert result.mean == pytest.approx(0.1)

    def test_all_negative_latencies(self):
        """Test when all latencies are negative (invalid)."""
        request_times = np.array([1.0, 2.0, 3.0])
        response_times = np.array([0.0, 0.5, 1.0])  # All before requests

        result = latency(request_times, response_times)

        assert result.samples == 0


# =============================================================================
# Windowed Throughput Tests
# =============================================================================


@pytest.mark.unit
class TestWindowedThroughput:
    """Test windowed throughput calculation."""

    def test_basic_windowed_throughput(self):
        """Test basic windowed throughput."""
        packets = [PacketInfo(timestamp=0.1 * i, size=100) for i in range(100)]

        times, rates = windowed_throughput(packets, window_size=1.0)

        assert len(times) == len(rates)
        assert len(times) > 0
        assert all(r >= 0 for r in rates)

    def test_windowed_with_step(self):
        """Test windowed throughput with custom step."""
        packets = [PacketInfo(timestamp=0.1 * i, size=100) for i in range(50)]

        times, rates = windowed_throughput(packets, window_size=1.0, step_size=0.5)

        assert len(times) > 0

    def test_windowed_default_step(self):
        """Test windowed throughput with default step."""
        packets = [PacketInfo(timestamp=0.1 * i, size=100) for i in range(30)]

        times, rates = windowed_throughput(packets, window_size=1.0)

        # Default step should be window_size / 2
        assert len(times) > 0

    def test_windowed_empty_packets(self):
        """Test windowed throughput with no packets."""
        times, rates = windowed_throughput([], window_size=1.0)

        assert len(times) == 0
        assert len(rates) == 0

    def test_windowed_single_packet(self):
        """Test windowed throughput with single packet."""
        packets = [PacketInfo(timestamp=0.0, size=100)]

        times, rates = windowed_throughput(packets, window_size=1.0)

        assert len(times) == 0  # Can't create window

    def test_windowed_variable_rate(self):
        """Test windowed throughput with varying packet rate."""
        # Dense packets at start, sparse at end
        packets = []
        for i in range(50):
            packets.append(PacketInfo(timestamp=0.01 * i, size=100))
        for i in range(10):
            packets.append(PacketInfo(timestamp=1.0 + 0.1 * i, size=100))

        times, rates = windowed_throughput(packets, window_size=0.5)

        assert len(times) > 0
        # Rates should vary
        assert max(rates) > min(rates)

    def test_windowed_returns_arrays(self):
        """Test that windowed throughput returns numpy arrays."""
        packets = [PacketInfo(timestamp=0.1 * i, size=100) for i in range(30)]

        times, rates = windowed_throughput(packets, window_size=1.0)

        assert isinstance(times, np.ndarray)
        assert isinstance(rates, np.ndarray)


# =============================================================================
# Data Class Tests
# =============================================================================


@pytest.mark.unit
class TestPacketMetricsDataClasses:
    """Test packet metrics data classes."""

    def test_packet_info_creation(self):
        """Test PacketInfo creation."""
        pkt = PacketInfo(timestamp=1.0, size=100, sequence=5)

        assert pkt.timestamp == 1.0
        assert pkt.size == 100
        assert pkt.sequence == 5

    def test_packet_info_optional_sequence(self):
        """Test PacketInfo with optional sequence."""
        pkt = PacketInfo(timestamp=1.0, size=100)

        assert pkt.sequence is None

    def test_throughput_result_attributes(self):
        """Test ThroughputResult attributes."""
        result = ThroughputResult(
            bytes_per_second=1000.0,
            bits_per_second=8000.0,
            packets_per_second=10.0,
            total_bytes=5000,
            total_packets=50,
            duration=5.0,
        )

        assert result.bytes_per_second == 1000.0
        assert result.bits_per_second == 8000.0

    def test_jitter_result_attributes(self):
        """Test JitterResult attributes."""
        result = JitterResult(
            mean=0.1,
            std=0.01,
            min=0.09,
            max=0.11,
            jitter_rfc3550=0.005,
        )

        assert result.mean == 0.1
        assert result.jitter_rfc3550 == 0.005

    def test_loss_result_attributes(self):
        """Test LossResult attributes."""
        result = LossResult(
            loss_rate=0.05,
            loss_percentage=5.0,
            packets_lost=5,
            packets_received=95,
            gaps=[(3, 4), (10, 12)],
        )

        assert result.loss_percentage == 5.0
        assert len(result.gaps) == 2

    def test_latency_result_attributes(self):
        """Test LatencyResult attributes."""
        result = LatencyResult(
            mean=0.1,
            std=0.02,
            min=0.05,
            max=0.15,
            p50=0.09,
            p95=0.14,
            p99=0.15,
            samples=100,
        )

        assert result.p50 == 0.09
        assert result.samples == 100
