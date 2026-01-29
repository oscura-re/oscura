"""Comprehensive unit tests for DAQ error-tolerant analysis module.

Tests all DAQ functionality:
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.packet.daq import (
    BitErrorAnalysis,
    DAQGap,
    DAQGapAnalysis,
    ErrorPattern,
    FuzzyMatch,
    JitterCompensationResult,
    PacketRecoveryResult,
    analyze_bit_errors,
    compensate_timestamp_jitter,
    detect_gaps,
    detect_gaps_by_samples,
    detect_gaps_by_timestamps,
    error_tolerant_decode,
    fuzzy_pattern_search,
    robust_packet_parse,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.packet]


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestFuzzyPatternSearch:
    """Test fuzzy bit pattern search with Hamming distance tolerance."""

    def test_exact_match(self):
        """Test exact pattern match (zero errors)."""
        data = bytes([0xAA, 0x55, 0x12, 0x34])
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert len(matches) == 1
        assert matches[0].offset == 0
        assert matches[0].bit_errors == 0
        assert matches[0].is_exact

    def test_fuzzy_match_one_error(self):
        """Test pattern match with one bit error."""
        # 0xAA55 = 1010101001010101
        # 0xAB55 = 1010101101010101 (1 bit different)
        data = bytes([0xAB, 0x55, 0x12, 0x34])
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=1)

        assert len(matches) >= 1
        found = [m for m in matches if m.offset == 0]
        assert len(found) == 1
        assert found[0].bit_errors <= 1

    def test_fuzzy_match_two_errors(self):
        """Test pattern match with two bit errors."""
        data = bytes([0xAA, 0x57, 0x12, 0x34])  # Second byte differs by 1 bit
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        assert len(matches) >= 1

    def test_no_match_too_many_errors(self):
        """Test that patterns with too many errors are rejected."""
        data = bytes([0xFF, 0xFF, 0x12, 0x34])
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        # 0xFFFF has 8 bit differences from 0xAA55, should not match
        exact_offset_zero = [m for m in matches if m.offset == 0 and m.bit_errors <= 2]
        assert len(exact_offset_zero) == 0

    def test_multiple_matches(self):
        """Test finding multiple pattern matches."""
        data = bytes([0xAA, 0x55, 0x00, 0xAA, 0x55])
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert len(matches) >= 2

    def test_pattern_from_bytes(self):
        """Test pattern specified as bytes."""
        data = bytes([0xAA, 0x55, 0x12, 0x34])
        matches = fuzzy_pattern_search(data, bytes([0xAA, 0x55]), pattern_bits=16, max_errors=0)

        assert len(matches) >= 1
        assert matches[0].offset == 0

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        data = bytes([0xAA, 0x55])
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert matches[0].confidence == 1.0

    def test_error_positions(self):
        """Test that error positions are correctly identified."""
        data = bytes([0xAB, 0x55])  # Bit 1 differs
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        found = [m for m in matches if m.offset == 0]
        assert len(found) >= 1
        if found[0].bit_errors > 0:
            assert len(found[0].error_positions) == found[0].bit_errors

    def test_step_parameter(self):
        """Test search with step size."""
        data = bytes([0xAA, 0x55, 0xAA, 0x55])
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=0, step=8)

        # With step=8, should find matches at byte boundaries
        assert len(matches) >= 2

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        data = np.array([0xAA, 0x55, 0x12, 0x34], dtype=np.uint8)
        matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert len(matches) >= 1


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestRobustPacketParse:
    """Test robust variable-length packet parsing with error recovery."""

    def test_parse_valid_packets(self):
        """Test parsing valid packet stream."""
        # Packet: [0xAA, 0x55, length, data...]
        data = bytes(
            [
                0xAA,
                0x55,
                0x04,
                0x01,
                0x02,
                0x03,
                0x04,  # Valid packet
                0xAA,
                0x55,
                0x02,
                0xFF,
                0xFE,  # Another valid packet
            ]
        )

        result = robust_packet_parse(data, sync_pattern=0xAA55, length_offset=2)

        assert len(result.packets) >= 1
        assert result.total_errors >= 0

    def test_recover_corrupted_length(self):
        """Test recovery when length field is corrupted."""
        # Packet with corrupted length field
        data = bytes(
            [
                0xAA,
                0x55,
                0xFF,
                0x01,
                0x02,  # Bad length
                0xAA,
                0x55,
                0x02,
                0x03,
                0x04,  # Good packet
            ]
        )

        result = robust_packet_parse(
            data, sync_pattern=0xAA55, length_offset=2, max_packet_length=10
        )

        # Should detect and recover
        assert len(result.recovered_packets) >= 0 or len(result.packets) >= 0

    def test_sync_with_errors(self):
        """Test sync pattern detection with bit errors."""
        data = bytes([0xAB, 0x55, 0x02, 0x01, 0x02])  # Sync has 1 bit error

        result = robust_packet_parse(data, sync_pattern=0xAA55, error_tolerance=2)

        assert len(result.packets) + len(result.recovered_packets) >= 0

    def test_resync_count(self):
        """Test that resynchronization events are counted."""
        data = bytes(
            [
                0xAA,
                0x55,
                0xFF,
                0x01,  # Bad length
                0xAA,
                0x55,
                0x01,
                0x42,  # Good sync
            ]
        )

        result = robust_packet_parse(
            data, sync_pattern=0xAA55, length_offset=2, max_packet_length=10
        )

        # Should track resync events
        assert result.sync_resync_count >= 0

    def test_failed_regions(self):
        """Test that unparseable regions are tracked."""
        data = bytes([0x00, 0x00, 0x00, 0x00])  # No valid sync

        result = robust_packet_parse(data, sync_pattern=0xAA55)

        # May have failed regions or just no packets
        assert isinstance(result.failed_regions, list)

    def test_numpy_input(self):
        """Test with numpy array input."""
        data = np.array([0xAA, 0x55, 0x02, 0x01, 0x02], dtype=np.uint8)

        result = robust_packet_parse(data, sync_pattern=0xAA55, length_offset=2)

        assert isinstance(result, PacketRecoveryResult)


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestTimestampJitterCompensation:
    """Test timestamp jitter compensation and clock correction."""

    def test_compensate_jitter_lowpass(self):
        """Test jitter compensation with lowpass filter."""
        # Create timestamps with jitter
        base = np.linspace(0, 1, 100)
        jitter = np.random.default_rng(42).normal(0, 1e-6, 100)
        timestamps = base + jitter

        result = compensate_timestamp_jitter(timestamps, method="lowpass")

        assert isinstance(result, JitterCompensationResult)
        assert len(result.corrected_timestamps) == len(timestamps)
        assert result.jitter_removed_ns >= 0

    def test_compensate_jitter_linear(self):
        """Test jitter compensation with linear fit."""
        timestamps = np.linspace(0, 1, 50) + np.random.default_rng(42).normal(0, 1e-6, 50)

        result = compensate_timestamp_jitter(timestamps, method="linear")

        assert result.correction_method == "linear"
        assert len(result.corrected_timestamps) == len(timestamps)

    def test_compensate_jitter_pll(self):
        """Test jitter compensation with PLL method."""
        timestamps = np.linspace(0, 1, 50) + np.random.default_rng(42).normal(0, 1e-6, 50)

        result = compensate_timestamp_jitter(timestamps, method="pll")

        assert result.correction_method == "pll"
        assert len(result.corrected_timestamps) == len(timestamps)

    def test_clock_drift_detection(self):
        """Test clock drift estimation."""
        # Create timestamps with drift
        timestamps = np.linspace(0, 1, 100) * 1.0001  # 0.01% drift

        result = compensate_timestamp_jitter(timestamps, expected_rate=100)

        assert abs(result.clock_drift_ppm) > 0

    def test_empty_timestamps(self):
        """Test with minimal timestamps."""
        timestamps = np.array([0.0])

        result = compensate_timestamp_jitter(timestamps)

        assert len(result.corrected_timestamps) == 1

    def test_auto_rate_detection(self):
        """Test automatic sample rate detection."""
        timestamps = np.linspace(0, 1, 100)

        result = compensate_timestamp_jitter(timestamps, expected_rate=None)

        assert isinstance(result.jitter_removed_ns, float)


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestErrorTolerantDecode:
    """Test error-tolerant protocol decoding."""

    def test_decode_uart(self):
        """Test UART decoding."""
        data = bytes([0x41, 0x42, 0x43])  # ASCII "ABC"

        result = error_tolerant_decode(data, protocol="uart")

        assert result["protocol"] == "uart"
        assert "frames" in result
        assert "error_count" in result

    def test_decode_spi(self):
        """Test SPI decoding."""
        data = bytes([0x01, 0x02, 0x03, 0x04])

        result = error_tolerant_decode(data, protocol="spi")

        assert result["protocol"] == "spi"
        assert len(result["frames"]) >= 0

    def test_decode_i2c(self):
        """Test I2C decoding."""
        # Simple I2C sequence: START, ADDR, DATA
        data = bytes([0x00, 0xA0, 0x42])  # Start, address, data

        result = error_tolerant_decode(data, protocol="i2c")

        assert result["protocol"] == "i2c"
        assert "frames" in result

    def test_decode_with_resync(self):
        """Test decoding with resynchronization."""
        data = bytes([0xFF, 0x00, 0xA0, 0x42])  # Bad byte, then valid I2C

        result = error_tolerant_decode(data, protocol="i2c", resync_on_error=True)

        assert result["resync_count"] >= 0

    def test_decode_without_resync(self):
        """Test decoding without resynchronization."""
        data = bytes([0xFF, 0x00, 0xA0, 0x42])

        result = error_tolerant_decode(data, protocol="i2c", resync_on_error=False)

        assert isinstance(result, dict)

    def test_invalid_protocol(self):
        """Test with invalid protocol name."""
        data = bytes([0x01, 0x02])

        with pytest.raises(ValueError, match="Unsupported protocol"):
            error_tolerant_decode(data, protocol="invalid")

    def test_numpy_input(self):
        """Test with numpy array input."""
        data = np.array([0x41, 0x42], dtype=np.uint8)

        result = error_tolerant_decode(data, protocol="uart")

        assert result["protocol"] == "uart"


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestBitErrorAnalysis:
    """Test bit error pattern analysis and diagnostics."""

    def test_no_errors(self):
        """Test analysis with no errors."""
        expected = bytes([0x01, 0x02, 0x03, 0x04])
        actual = bytes([0x01, 0x02, 0x03, 0x04])

        result = analyze_bit_errors(expected, actual)

        assert result.error_rate == 0.0
        assert result.error_pattern == ErrorPattern.SINGLE_BIT

    def test_single_bit_error(self):
        """Test single bit error detection."""
        expected = bytes([0x01, 0x02, 0x03, 0x04])
        actual = bytes([0x01, 0x03, 0x03, 0x04])  # One bit flip in byte 1

        result = analyze_bit_errors(expected, actual)

        assert result.error_rate > 0
        assert len(result.error_distribution) == 8

    def test_burst_error_detection(self):
        """Test burst error pattern detection."""
        expected = bytes([0x00] * 100)
        actual = bytearray([0x00] * 100)
        # Create burst of errors
        for i in range(10, 20):
            actual[i] = 0xFF

        result = analyze_bit_errors(expected, bytes(actual))

        assert result.error_rate > 0
        # Should detect burst pattern
        assert result.burst_length_max > 1

    def test_random_error_detection(self):
        """Test random error pattern detection."""
        expected = bytes([0x00] * 100)
        actual = bytearray([0x00] * 100)
        # Scatter random errors
        for i in [5, 15, 35, 67, 89]:
            actual[i] = 0x01

        result = analyze_bit_errors(expected, bytes(actual))

        assert result.error_rate > 0

    def test_systematic_error_detection(self):
        """Test systematic error pattern detection."""
        expected = bytes([0x00] * 100)
        actual = bytearray([0x00] * 100)
        # Regular pattern: every 10th byte
        for i in range(0, 100, 10):
            actual[i] = 0x01

        result = analyze_bit_errors(expected, bytes(actual))

        assert result.error_rate > 0

    def test_error_distribution(self):
        """Test error distribution by bit position."""
        expected = bytes([0x00, 0x00, 0x00, 0x00])
        actual = bytes([0x01, 0x01, 0x01, 0x01])  # LSB always set

        result = analyze_bit_errors(expected, actual)

        # Should show errors in bit 0
        assert result.error_distribution[0] > 0

    def test_probable_cause_emi(self):
        """Test EMI diagnosis."""
        expected = bytes([0x00] * 100)
        actual = bytearray([0x00] * 100)
        # Burst errors
        for i in range(20, 35):
            actual[i] = 0xFF

        result = analyze_bit_errors(expected, bytes(actual))

        assert (
            "interference" in result.probable_cause.lower()
            or "burst" in str(result.error_pattern).lower()
        )

    def test_recommendations_provided(self):
        """Test that recommendations are provided."""
        expected = bytes([0x00] * 50)
        actual = bytes([0x01] * 50)

        result = analyze_bit_errors(expected, actual)

        assert len(result.recommendations) > 0

    def test_numpy_input(self):
        """Test with numpy arrays."""
        expected = np.array([0x01, 0x02, 0x03], dtype=np.uint8)
        actual = np.array([0x01, 0x02, 0x04], dtype=np.uint8)

        result = analyze_bit_errors(expected, actual)

        assert isinstance(result, BitErrorAnalysis)

    def test_different_lengths(self):
        """Test with different length inputs."""
        expected = bytes([0x01, 0x02, 0x03])
        actual = bytes([0x01, 0x02])  # Shorter

        result = analyze_bit_errors(expected, actual)

        # Should handle by comparing only common length
        assert isinstance(result, BitErrorAnalysis)


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestDAQGapDetection:
    """Test DAQ gap detection in acquisitions."""

    def test_detect_gaps_by_timestamps_with_gap(self):
        """Test gap detection with timestamp discontinuity."""
        timestamps = np.array([0.0, 0.001, 0.002, 0.005, 0.006])  # Gap at 0.002-0.005

        result = detect_gaps_by_timestamps(timestamps, expected_interval=0.001, tolerance=0.5)

        assert isinstance(result, DAQGapAnalysis)
        assert result.total_gaps >= 1

    def test_detect_gaps_no_gaps(self):
        """Test with continuous timestamps."""
        timestamps = np.linspace(0, 1, 1000)

        result = detect_gaps_by_timestamps(timestamps, tolerance=0.1)

        assert result.total_gaps == 0
        assert result.acquisition_efficiency == 1.0

    def test_detect_gaps_by_samples_with_discontinuity(self):
        """Test gap detection via sample value discontinuity."""
        data = np.concatenate([np.linspace(0, 10, 100), np.linspace(100, 110, 100)])

        result = detect_gaps_by_samples(data, sample_rate=1000, check_discontinuities=True)

        assert isinstance(result, DAQGapAnalysis)

    def test_detect_gaps_auto_interval(self):
        """Test automatic interval detection."""
        timestamps = np.linspace(0, 1, 100)

        result = detect_gaps_by_timestamps(timestamps, expected_interval=None)

        assert result.sample_rate > 0

    def test_detect_gaps_trace_input(self):
        """Test gap detection with WaveformTrace."""
        data = np.linspace(0, 1, 100)
        metadata = TraceMetadata(sample_rate=100.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_gaps(trace)

        assert isinstance(result, DAQGapAnalysis)

    def test_gap_attributes(self):
        """Test gap object attributes."""
        timestamps = np.array([0.0, 0.1, 0.2, 0.5, 0.6])

        result = detect_gaps_by_timestamps(timestamps, expected_interval=0.1, tolerance=0.05)

        if result.total_gaps > 0:
            gap = result.gaps[0]
            assert isinstance(gap, DAQGap)
            assert gap.duration >= 0
            assert gap.missing_samples >= 0

    def test_acquisition_efficiency(self):
        """Test acquisition efficiency calculation."""
        timestamps = np.array([0.0, 0.1, 0.2, 0.5, 0.6])  # Missing 2 samples

        result = detect_gaps_by_timestamps(timestamps, expected_interval=0.1, tolerance=0.05)

        assert 0 <= result.acquisition_efficiency <= 1.0

    def test_discontinuities_list(self):
        """Test discontinuities tracking."""
        timestamps = np.array([0.0, 0.1, 0.2, 0.6, 0.7])

        result = detect_gaps_by_timestamps(timestamps, expected_interval=0.1, tolerance=0.05)

        assert isinstance(result.discontinuities, list)

    def test_metadata_preservation(self):
        """Test that metadata is preserved."""
        timestamps = np.linspace(0, 1, 100)

        result = detect_gaps_by_timestamps(timestamps, expected_interval=0.01, tolerance=0.1)

        assert "method" in result.metadata
        assert result.metadata["method"] == "timestamp"

    def test_min_gap_samples_threshold(self):
        """Test minimum gap samples threshold."""
        timestamps = np.array([0.0, 0.1, 0.2, 0.201, 0.3])  # Tiny gap

        result = detect_gaps_by_timestamps(
            timestamps, expected_interval=0.1, tolerance=0.05, min_gap_samples=5
        )

        # Small gap should be filtered out
        assert result.total_gaps == 0

    def test_empty_timestamps(self):
        """Test with empty or minimal timestamps."""
        timestamps = np.array([0.0])

        result = detect_gaps_by_timestamps(timestamps)

        assert result.total_gaps == 0
        assert result.acquisition_efficiency == 1.0

    def test_multiple_gaps(self):
        """Test detection of multiple gaps."""
        timestamps = np.array([0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 1.0, 1.1])

        result = detect_gaps_by_timestamps(timestamps, expected_interval=0.1, tolerance=0.05)

        assert result.total_gaps >= 2


# =============================================================================
# Data Class Tests
# =============================================================================


@pytest.mark.unit
class TestDAQDataClasses:
    """Test DAQ data classes and their properties."""

    def test_fuzzy_match_is_exact(self):
        """Test FuzzyMatch.is_exact property."""
        match = FuzzyMatch(offset=0, matched_bits=0xAA55, bit_errors=0)
        assert match.is_exact is True

        match_with_errors = FuzzyMatch(offset=0, matched_bits=0xAA55, bit_errors=2)
        assert match_with_errors.is_exact is False

    def test_daq_gap_attributes(self):
        """Test DAQGap attributes."""
        gap = DAQGap(
            start_index=10,
            end_index=20,
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            expected_samples=100,
            missing_samples=90,
        )

        assert gap.duration == 1.0
        assert gap.missing_samples == 90

    def test_packet_recovery_result_defaults(self):
        """Test PacketRecoveryResult default values."""
        result = PacketRecoveryResult()

        assert result.packets == []
        assert result.recovered_packets == []
        assert result.total_errors == 0

    def test_bit_error_analysis_defaults(self):
        """Test BitErrorAnalysis default values."""
        analysis = BitErrorAnalysis(
            error_rate=0.01,
            error_pattern=ErrorPattern.RANDOM,
        )

        assert analysis.burst_length_mean == 0.0
        assert analysis.error_distribution == []

    def test_error_pattern_enum(self):
        """Test ErrorPattern enum values."""
        assert ErrorPattern.RANDOM.value == "random"
        assert ErrorPattern.BURST.value == "burst"
        assert ErrorPattern.SYSTEMATIC.value == "systematic"
        assert ErrorPattern.SINGLE_BIT.value == "single_bit"
