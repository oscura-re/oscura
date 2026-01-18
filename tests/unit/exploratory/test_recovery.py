"""Comprehensive unit tests for bit error analysis and recovery module.

This module provides extensive testing for bit error pattern analysis
and capture diagnostics functionality.


Test Coverage:
- analyze_bit_errors with various error patterns
- Random error detection (EMI)
- Burst error detection (USB issues)
- Periodic error detection (clock jitter)
- generate_error_visualization_data
- Edge cases: no errors, all errors, single error
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.exploratory.recovery import (
    ErrorPattern,
    analyze_bit_errors,
    generate_error_visualization_data,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Functions
# =============================================================================


def generate_random_errors(
    data: np.ndarray, error_rate: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with random bit errors."""
    rng = np.random.default_rng(seed)
    expected = data.copy()
    received = data.copy()

    n_errors = int(len(data) * error_rate)
    error_indices = rng.choice(len(data), size=n_errors, replace=False)
    received[error_indices] = 1 - received[error_indices]

    return expected, received


def generate_burst_errors(
    data: np.ndarray, n_bursts: int, burst_length: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with burst errors."""
    rng = np.random.default_rng(seed)
    expected = data.copy()
    received = data.copy()

    for _ in range(n_bursts):
        start = rng.integers(0, max(1, len(data) - burst_length))
        received[start : start + burst_length] = 1 - received[start : start + burst_length]

    return expected, received


def generate_periodic_errors(
    data: np.ndarray, period: int, phase: int = 0, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data with periodic errors."""
    expected = data.copy()
    received = data.copy()

    # Flip bits at periodic intervals
    error_indices = np.arange(phase, len(data), period)
    received[error_indices] = 1 - received[error_indices]

    return expected, received


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestAnalyzeBitErrors:
    """Test bit error pattern analysis (DAQ-005)."""

    def test_no_errors(self) -> None:
        """Test analysis when there are no bit errors."""
        data = np.random.randint(0, 2, size=1000, dtype=np.uint8)
        expected = data.copy()
        received = data.copy()

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate == 0.0
        assert analysis.error_count == 0
        assert analysis.total_bits == 1000
        assert analysis.pattern_type == ErrorPattern.RANDOM
        assert analysis.severity == "low"
        assert "No errors" in analysis.diagnosis

    def test_random_error_pattern_emi(self) -> None:
        """Test detection of random errors (EMI/noise)."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_random_errors(data, error_rate=0.0005)

        analysis = analyze_bit_errors(received, expected)

        # Should detect random pattern
        assert analysis.pattern_type == ErrorPattern.RANDOM
        assert analysis.bit_error_rate < 0.01
        assert "EMI" in analysis.diagnosis or "noise" in analysis.diagnosis
        assert analysis.severity == "low"

    def test_burst_error_pattern_usb(self) -> None:
        """Test detection of burst errors (USB transmission issues)."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_burst_errors(data, n_bursts=3, burst_length=50)

        analysis = analyze_bit_errors(received, expected)

        # Should detect burst pattern
        assert analysis.pattern_type == ErrorPattern.BURST
        assert "USB" in analysis.diagnosis or "Burst" in analysis.diagnosis
        assert analysis.mean_error_gap < 100  # Clustered errors

    def test_periodic_error_pattern_clock_jitter(self) -> None:
        """Test detection of periodic errors (clock jitter)."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_periodic_errors(data, period=100, phase=5)

        analysis = analyze_bit_errors(received, expected)

        # Should detect periodic pattern
        assert analysis.pattern_type == ErrorPattern.PERIODIC
        assert "periodic" in analysis.diagnosis.lower() or "clock" in analysis.diagnosis.lower()

    def test_high_ber_severe(self) -> None:
        """Test severe error diagnosis with high BER."""
        data = np.random.randint(0, 2, size=1000, dtype=np.uint8)
        expected, received = generate_random_errors(data, error_rate=0.05)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate > 0.01
        assert analysis.severity == "severe"
        assert "SEVERE" in analysis.diagnosis

    def test_moderate_ber(self) -> None:
        """Test moderate error diagnosis."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_random_errors(data, error_rate=0.005)

        analysis = analyze_bit_errors(received, expected)

        assert 0.001 < analysis.bit_error_rate <= 0.01
        assert analysis.severity == "moderate"
        assert "MODERATE" in analysis.diagnosis

    def test_low_ber_acceptable(self) -> None:
        """Test acceptable error rate."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_random_errors(data, error_rate=0.0005)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.bit_error_rate < 0.001
        assert analysis.severity == "low"
        assert "Acceptable" in analysis.diagnosis

    def test_single_error(self) -> None:
        """Test with single bit error."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()
        received[500] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 1
        assert analysis.pattern_type == ErrorPattern.RANDOM
        assert len(analysis.error_positions) == 1
        assert analysis.error_positions[0] == 500

    def test_all_errors(self) -> None:
        """Test when all bits are errors."""
        expected = np.zeros(100, dtype=np.uint8)
        received = np.ones(100, dtype=np.uint8)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 100
        assert analysis.bit_error_rate == 1.0
        assert analysis.severity == "severe"

    def test_error_positions_correctness(self) -> None:
        """Test that error positions are correctly identified."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()

        # Flip specific bits
        error_indices = [10, 50, 100, 500, 999]
        received[error_indices] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == len(error_indices)
        np.testing.assert_array_equal(np.sort(analysis.error_positions), np.sort(error_indices))

    def test_mean_error_gap_calculation(self) -> None:
        """Test mean error gap calculation."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()

        # Errors at positions 0, 100, 200, 300 (gaps of 100)
        received[[0, 100, 200, 300]] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.mean_error_gap == pytest.approx(100.0, abs=1.0)

    def test_mismatched_length_raises_error(self) -> None:
        """Test that mismatched array lengths raise ValueError."""
        received = np.zeros(100, dtype=np.uint8)
        expected = np.zeros(200, dtype=np.uint8)

        with pytest.raises(ValueError, match="same length"):
            analyze_bit_errors(received, expected)

    def test_empty_arrays_raise_error(self) -> None:
        """Test that empty arrays raise ValueError."""
        received = np.array([], dtype=np.uint8)
        expected = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_bit_errors(received, expected)

    def test_burst_threshold_parameter(self) -> None:
        """Test custom burst threshold parameter."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_burst_errors(data, n_bursts=5, burst_length=30)

        # With default threshold (100)
        analysis1 = analyze_bit_errors(received, expected, burst_threshold=100)

        # With stricter threshold (50)
        analysis2 = analyze_bit_errors(received, expected, burst_threshold=50)

        # Both should detect bursts, but classification may differ
        assert analysis1.pattern_type in [ErrorPattern.BURST, ErrorPattern.RANDOM]

    def test_periodicity_threshold_parameter(self) -> None:
        """Test custom periodicity threshold parameter."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_periodic_errors(data, period=100)

        # With strict threshold
        analysis = analyze_bit_errors(received, expected, periodicity_threshold=0.05)

        # Should detect periodic pattern
        assert analysis.pattern_type == ErrorPattern.PERIODIC

    def test_mixed_error_patterns(self) -> None:
        """Test with mixed error patterns."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected = data.copy()
        received = data.copy()

        # Add some bursts
        received[1000:1050] = 1 - received[1000:1050]
        received[5000:5050] = 1 - received[5000:5050]

        # Add some random errors
        rng = np.random.default_rng(42)
        random_errors = rng.choice(10000, size=20, replace=False)
        received[random_errors] = 1 - received[random_errors]

        analysis = analyze_bit_errors(received, expected)

        # Should detect some pattern (may be burst or mixed)
        assert analysis.error_count > 100
        assert analysis.pattern_type in [
            ErrorPattern.BURST,
            ErrorPattern.RANDOM,
            ErrorPattern.UNKNOWN,
        ]


# =============================================================================
# Visualization Data Generation
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestGenerateErrorVisualizationData:
    """Test error visualization data generation."""

    def test_visualization_with_errors(self) -> None:
        """Test generating visualization data with errors."""
        data = np.random.randint(0, 2, size=10000, dtype=np.uint8)
        expected, received = generate_random_errors(data, error_rate=0.001)

        analysis = analyze_bit_errors(received, expected)
        viz_data = generate_error_visualization_data(analysis, histogram_bins=50)

        # Check all expected keys present
        assert "histogram_counts" in viz_data
        assert "histogram_edges" in viz_data
        assert "timeline" in viz_data

        # Check histogram
        assert len(viz_data["histogram_counts"]) == 50
        assert len(viz_data["histogram_edges"]) == 51  # n+1 edges

        # Check timeline
        assert len(viz_data["timeline"]) == analysis.total_bits
        assert np.sum(viz_data["timeline"]) == analysis.error_count

    def test_visualization_no_errors(self) -> None:
        """Test visualization data with no errors."""
        data = np.random.randint(0, 2, size=1000, dtype=np.uint8)

        analysis = analyze_bit_errors(data, data)  # Same data = no errors
        viz_data = generate_error_visualization_data(analysis)

        # Should return empty arrays
        assert len(viz_data["histogram_counts"]) == 0
        assert len(viz_data["histogram_edges"]) == 0
        assert len(viz_data["timeline"]) == 0

    def test_visualization_custom_bins(self) -> None:
        """Test custom histogram bins."""
        data = np.random.randint(0, 2, size=5000, dtype=np.uint8)
        expected, received = generate_random_errors(data, error_rate=0.002)

        analysis = analyze_bit_errors(received, expected)

        # Test with different bin counts
        for n_bins in [10, 20, 100]:
            viz_data = generate_error_visualization_data(analysis, histogram_bins=n_bins)
            assert len(viz_data["histogram_counts"]) == n_bins
            assert len(viz_data["histogram_edges"]) == n_bins + 1

    def test_timeline_error_positions(self) -> None:
        """Test that timeline correctly marks error positions."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()

        # Known error positions
        error_positions = [10, 50, 100, 500, 999]
        received[error_positions] = 1

        analysis = analyze_bit_errors(received, expected)
        viz_data = generate_error_visualization_data(analysis)

        # Check that timeline has 1s at error positions
        for pos in error_positions:
            assert viz_data["timeline"][pos] == 1.0

        # Check that non-error positions are 0
        assert viz_data["timeline"][20] == 0.0
        assert viz_data["timeline"][200] == 0.0

    def test_histogram_distribution(self) -> None:
        """Test that histogram captures error distribution."""
        data = np.zeros(10000, dtype=np.uint8)
        received = data.copy()

        # Cluster errors in first half
        received[1000:1100] = 1

        analysis = analyze_bit_errors(received, data)
        viz_data = generate_error_visualization_data(analysis, histogram_bins=20)

        # Most errors should be in earlier bins
        counts = viz_data["histogram_counts"]
        assert np.sum(counts[:10]) > np.sum(counts[10:])


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestExploratoryRecoveryEdgeCases:
    """Test edge cases and robustness."""

    def test_very_small_arrays(self) -> None:
        """Test with very small arrays."""
        expected = np.array([0, 1, 0], dtype=np.uint8)
        received = np.array([0, 0, 0], dtype=np.uint8)

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 1
        assert analysis.total_bits == 3

    def test_alternating_pattern(self) -> None:
        """Test with alternating bit pattern."""
        expected = np.tile([0, 1], 5000).astype(np.uint8)
        received = expected.copy()

        # Flip some bits
        received[[100, 200, 300]] = 1 - received[[100, 200, 300]]

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 3

    def test_all_zeros_expected(self) -> None:
        """Test with all zeros expected."""
        expected = np.zeros(1000, dtype=np.uint8)
        received = expected.copy()
        received[[10, 50, 100]] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 3

    def test_all_ones_expected(self) -> None:
        """Test with all ones expected."""
        expected = np.ones(1000, dtype=np.uint8)
        received = expected.copy()
        received[[10, 50, 100]] = 0

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 3

    def test_very_sparse_errors(self) -> None:
        """Test with very sparse errors."""
        expected = np.zeros(100000, dtype=np.uint8)
        received = expected.copy()
        received[[1000, 50000, 99000]] = 1

        analysis = analyze_bit_errors(received, expected)

        assert analysis.error_count == 3
        assert analysis.bit_error_rate < 0.0001
        assert analysis.severity == "low"

    def test_coefficient_of_variation(self) -> None:
        """Test that evenly spaced errors are detected as periodic."""
        data = np.zeros(10000, dtype=np.uint8)
        received = data.copy()

        # Evenly spaced errors (perfectly periodic - every 100 bits)
        received[np.arange(100, 10000, 100)] = 1

        analysis = analyze_bit_errors(received, data)

        # Should be classified as periodic due to perfect regular spacing
        assert analysis.pattern_type == ErrorPattern.PERIODIC
