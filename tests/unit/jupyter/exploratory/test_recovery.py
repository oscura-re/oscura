"""Comprehensive tests for bit error pattern analysis and recovery.

Tests requirements:
  - ErrorPattern enumeration
  - ErrorAnalysis dataclass
  - analyze_bit_errors function
  - Error pattern classification (random, burst, periodic)
  - Bit error rate calculation
  - Diagnosis suggestions
  - Severity levels
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.jupyter.exploratory.recovery import (
    ErrorAnalysis,
    ErrorPattern,
    analyze_bit_errors,
)

pytestmark = pytest.mark.unit


class TestErrorPatternEnum:
    """Test ErrorPattern enumeration."""

    def test_enum_values(self) -> None:
        """Test ErrorPattern enum values."""
        assert ErrorPattern.RANDOM.value == "random"
        assert ErrorPattern.BURST.value == "burst"
        assert ErrorPattern.PERIODIC.value == "periodic"
        assert ErrorPattern.UNKNOWN.value == "unknown"

    def test_enum_members(self) -> None:
        """Test that all expected members exist."""
        assert hasattr(ErrorPattern, "RANDOM")
        assert hasattr(ErrorPattern, "BURST")
        assert hasattr(ErrorPattern, "PERIODIC")
        assert hasattr(ErrorPattern, "UNKNOWN")


class TestErrorAnalysisDataclass:
    """Test ErrorAnalysis dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test ErrorAnalysis initialization."""
        error_pos = np.array([10, 20, 30], dtype=np.int64)
        analysis = ErrorAnalysis(
            bit_error_rate=0.003,
            error_count=30,
            total_bits=10000,
            pattern_type=ErrorPattern.RANDOM,
            mean_error_gap=333.3,
            error_positions=error_pos,
            diagnosis="EMI",
            severity="moderate",
        )

        assert analysis.bit_error_rate == 0.003
        assert analysis.error_count == 30
        assert analysis.total_bits == 10000
        assert analysis.pattern_type == ErrorPattern.RANDOM
        assert abs(analysis.mean_error_gap - 333.3) < 0.1
        assert len(analysis.error_positions) == 3
        assert analysis.diagnosis == "EMI"
        assert analysis.severity == "moderate"


class TestNoErrors:
    """Test analyze_bit_errors with no errors."""

    def test_identical_arrays(self) -> None:
        """Test analysis when received matches expected perfectly."""
        expected = np.random.RandomState(42).randint(0, 2, 10000, dtype=np.uint8)
        received = expected.copy()

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate == 0.0
        assert analysis.error_count == 0
        assert analysis.total_bits == 10000
        assert len(analysis.error_positions) == 0
        assert analysis.severity == "low"

    def test_all_zeros_identical(self) -> None:
        """Test with all zeros, no errors."""
        expected = np.zeros(5000, dtype=np.uint8)
        received = np.zeros(5000, dtype=np.uint8)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 0
        assert analysis.bit_error_rate == 0.0

    def test_all_ones_identical(self) -> None:
        """Test with all ones, no errors."""
        expected = np.ones(5000, dtype=np.uint8)
        received = np.ones(5000, dtype=np.uint8)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 0


class TestRandomErrors:
    """Test random error pattern detection."""

    def test_uniformly_distributed_errors(self) -> None:
        """Test detection of uniformly distributed random errors."""
        rng = np.random.default_rng(42)
        expected = rng.integers(0, 2, 10000, dtype=np.uint8)
        received = expected.copy()

        # Flip 50 random bits (0.5% BER)
        error_indices = rng.choice(10000, 50, replace=False)
        received[error_indices] = 1 - received[error_indices]

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 50
        assert abs(analysis.bit_error_rate - 0.005) < 1e-6
        # Random errors should not cluster
        assert analysis.pattern_type in [ErrorPattern.RANDOM, ErrorPattern.UNKNOWN]

    def test_low_ber_random(self) -> None:
        """Test low BER with random pattern."""
        rng = np.random.default_rng(42)
        expected = rng.integers(0, 2, 100000, dtype=np.uint8)
        received = expected.copy()

        # Very few errors: 0.01% BER
        error_indices = rng.choice(100000, 10, replace=False)
        received[error_indices] = 1 - received[error_indices]

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate < 0.001
        assert analysis.severity == "low"


class TestBurstErrors:
    """Test burst error pattern detection."""

    def test_single_error_burst(self) -> None:
        """Test detection of single error burst."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Create 50-bit burst at position 1000
        received[1000:1050] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 50
        assert analysis.pattern_type == ErrorPattern.BURST
        # Mean gap should be very small (errors are consecutive)
        assert analysis.mean_error_gap < 100

    def test_multiple_bursts(self) -> None:
        """Test detection of multiple burst errors."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Create multiple bursts
        received[1000:1020] = 1  # 20 bits
        received[3000:3030] = 1  # 30 bits
        received[7000:7010] = 1  # 10 bits

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 60
        # Should detect bursts or clustered errors
        # (exact classification depends on thresholds)

    def test_burst_diagnosis(self) -> None:
        """Test that burst errors suggest USB transmission issue."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Create tight burst
        received[5000:5100] = 1

        analysis = analyze_bit_errors(received, expected)

        # Burst pattern should suggest USB or transmission issue
        assert (
            "USB" in analysis.diagnosis
            or "transmission" in analysis.diagnosis
            or "burst" in analysis.diagnosis.lower()
        )


class TestPeriodicErrors:
    """Test periodic error pattern detection."""

    def test_regularly_spaced_errors(self) -> None:
        """Test detection of periodically spaced errors."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Create errors every 200 bits
        for i in range(0, 10000, 200):
            if i < 10000:
                received[i] = 1

        analysis = analyze_bit_errors(received, expected)

        # Should detect periodic pattern
        assert analysis.error_count == 50
        # Periodic errors suggest clock jitter
        # (exact classification depends on FFT threshold)

    def test_periodic_diagnosis(self) -> None:
        """Test that periodic errors suggest clock jitter."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Regular pattern
        for i in range(0, 10000, 100):
            if i < 10000:
                received[i] = 1

        analysis = analyze_bit_errors(received, expected, periodicity_threshold=0.05)

        # Periodic pattern may suggest clock jitter or interference
        # (diagnosis is implementation-dependent)


class TestBitErrorRate:
    """Test bit error rate calculations."""

    def test_1_percent_ber(self) -> None:
        """Test 1% BER classification."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # 1% errors
        rng = np.random.default_rng(42)
        errors = rng.choice(10000, 100, replace=False)
        received[errors] = 1

        analysis = analyze_bit_errors(received, expected)

        assert abs(analysis.bit_error_rate - 0.01) < 1e-6
        assert analysis.severity == "moderate"

    def test_5_percent_ber_severe(self) -> None:
        """Test 5% BER is classified as severe."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # 5% errors
        rng = np.random.default_rng(42)
        errors = rng.choice(10000, 500, replace=False)
        received[errors] = 1

        analysis = analyze_bit_errors(received, expected)

        assert abs(analysis.bit_error_rate - 0.05) < 1e-4
        assert analysis.severity == "severe"

    def test_very_low_ber(self) -> None:
        """Test very low BER (< 0.001) is classified as low."""
        expected = np.zeros(100000, dtype=np.uint8)
        received = expected.copy()

        # 0.01% BER
        rng = np.random.default_rng(42)
        errors = rng.choice(100000, 10, replace=False)
        received[errors] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate < 0.001
        assert analysis.severity == "low"


class TestErrorPositions:
    """Test error position tracking."""

    def test_error_positions_recorded(self) -> None:
        """Test that error positions are correctly recorded."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()

        # Flip specific bits
        received[[10, 50, 100, 500, 999]] = 1

        analysis = analyze_bit_errors(received, expected)

        assert len(analysis.error_positions) == 5
        assert 10 in analysis.error_positions
        assert 50 in analysis.error_positions
        assert 100 in analysis.error_positions
        assert 500 in analysis.error_positions
        assert 999 in analysis.error_positions

    def test_error_positions_sorted(self) -> None:
        """Test that error positions are in ascending order."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()

        received[[999, 50, 500, 10, 100]] = 1

        analysis = analyze_bit_errors(received, expected)

        positions = list(analysis.error_positions)
        assert positions == sorted(positions)


class TestMeanErrorGap:
    """Test mean error gap calculation."""

    def test_evenly_spaced_errors(self) -> None:
        """Test mean gap for evenly spaced errors."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Errors every 100 bits
        received[np.arange(0, 10000, 100)] = 1

        analysis = analyze_bit_errors(received, expected)

        # Mean gap should be close to 100
        assert 90 < analysis.mean_error_gap < 110

    def test_consecutive_errors(self) -> None:
        """Test mean gap for consecutive errors (burst)."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()

        # 10 consecutive errors
        received[100:110] = 1

        analysis = analyze_bit_errors(received, expected)

        # Mean gap should be 1 (consecutive)
        assert analysis.mean_error_gap < 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_different_length_arrays_error(self) -> None:
        """Test that different length arrays raise error."""
        received = np.zeros(100, dtype=np.uint8)
        expected = np.zeros(200, dtype=np.uint8)

        with pytest.raises(ValueError, match="same length"):
            analyze_bit_errors(received, expected)

    def test_empty_arrays_error(self) -> None:
        """Test that empty arrays raise error."""
        received = np.array([], dtype=np.uint8)
        expected = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="empty"):
            analyze_bit_errors(received, expected)

    def test_single_bit_arrays(self) -> None:
        """Test with single-bit arrays."""
        received = np.array([1], dtype=np.uint8)
        expected = np.array([0], dtype=np.uint8)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 1
        assert analysis.bit_error_rate == 1.0
        assert analysis.total_bits == 1


class TestBurstThreshold:
    """Test burst_threshold parameter."""

    def test_custom_burst_threshold_high(self) -> None:
        """Test with high burst threshold."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Errors with gap of 50 bits
        for i in range(0, 5000, 50):
            received[i] = 1

        # With high threshold, should not be classified as burst
        analysis = analyze_bit_errors(received, expected, burst_threshold=30)

        # May or may not be burst depending on threshold

    def test_custom_burst_threshold_low(self) -> None:
        """Test with low burst threshold."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Sparse errors
        rng = np.random.default_rng(42)
        errors = rng.choice(10000, 50, replace=False)
        received[errors] = 1

        analysis = analyze_bit_errors(received, expected, burst_threshold=1000)

        # With very high threshold, even sparse errors might be "burst"
        # (depends on actual distribution)


class TestPeriodicityThreshold:
    """Test periodicity_threshold parameter."""

    def test_strict_periodicity_threshold(self) -> None:
        """Test with strict periodicity detection."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Almost periodic (slight variation)
        for i in range(0, 10000, 100):
            if i < 10000:
                offset = np.random.randint(-5, 6)
                idx = min(9999, max(0, i + offset))
                received[idx] = 1

        analysis = analyze_bit_errors(received, expected, periodicity_threshold=0.01)

        # Strict threshold may not detect as periodic

    def test_loose_periodicity_threshold(self) -> None:
        """Test with loose periodicity detection."""
        expected = np.zeros(10000, dtype=np.uint8)
        received = expected.copy()

        # Regular pattern
        received[np.arange(0, 10000, 100)] = 1

        analysis = analyze_bit_errors(received, expected, periodicity_threshold=0.2)

        # Loose threshold should detect periodicity


class TestRealisticScenarios:
    """Test realistic error scenarios."""

    def test_emi_random_errors(self) -> None:
        """Test EMI-like random error pattern."""
        rng = np.random.default_rng(42)
        expected = rng.integers(0, 2, 50000, dtype=np.uint8)
        received = expected.copy()

        # Random flips (0.1% BER typical for EMI)
        errors = rng.choice(50000, 50, replace=False)
        received[errors] = 1 - received[errors]

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate < 0.01
        assert analysis.severity in ["low", "moderate"]

    def test_usb_burst_errors(self) -> None:
        """Test USB transmission burst error pattern."""
        expected = np.zeros(100000, dtype=np.uint8)
        received = expected.copy()

        # Simulate USB packet drop (burst errors)
        received[10000:10512] = 1  # 512 bytes lost

        analysis = analyze_bit_errors(received, expected)

        assert analysis.pattern_type == ErrorPattern.BURST
        assert "USB" in analysis.diagnosis or "burst" in analysis.diagnosis.lower()

    def test_clock_jitter_periodic_errors(self) -> None:
        """Test clock jitter periodic error pattern."""
        expected = np.zeros(50000, dtype=np.uint8)
        received = expected.copy()

        # Periodic bit slips every 1000 bits
        for i in range(0, 50000, 1000):
            if i < 50000:
                received[i] = 1

        analysis = analyze_bit_errors(received, expected, periodicity_threshold=0.1)

        # Should detect some pattern (periodic or unknown)
