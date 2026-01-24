"""Tests for timing side-channel analysis.

Tests for timing attack detection module.
"""

import numpy as np
import pytest

from oscura.analyzers.side_channel.timing import TimingAnalyzer

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# TimingAnalyzer tests
# =============================================================================


def test_timing_analyzer_initialization():
    """Test TimingAnalyzer initialization."""
    analyzer = TimingAnalyzer(confidence_level=0.95, min_samples=100)
    assert analyzer.confidence_level == 0.95
    assert analyzer.min_samples == 100


def test_timing_analyzer_invalid_confidence():
    """Test TimingAnalyzer with invalid confidence level."""
    with pytest.raises(ValueError, match="confidence_level must be in"):
        TimingAnalyzer(confidence_level=1.5)

    with pytest.raises(ValueError, match="confidence_level must be in"):
        TimingAnalyzer(confidence_level=-0.1)


def test_timing_analyzer_invalid_min_samples():
    """Test TimingAnalyzer with invalid min_samples."""
    with pytest.raises(ValueError, match="min_samples must be >= 10"):
        TimingAnalyzer(min_samples=5)


def test_timing_analyzer_insufficient_samples():
    """Test TimingAnalyzer with insufficient samples."""
    analyzer = TimingAnalyzer(min_samples=100)
    timings = np.random.randn(50)
    inputs = np.random.randint(0, 256, 50, dtype=np.uint8)

    with pytest.raises(ValueError, match="Insufficient samples"):
        analyzer.analyze(timings, inputs)


def test_timing_analyzer_length_mismatch():
    """Test TimingAnalyzer with mismatched input lengths."""
    analyzer = TimingAnalyzer()
    timings = np.random.randn(100)
    inputs = np.random.randint(0, 256, 50, dtype=np.uint8)

    with pytest.raises(ValueError, match="length mismatch"):
        analyzer.analyze(timings, inputs)


def test_timing_analyzer_no_leak():
    """Test TimingAnalyzer with no timing leak."""
    np.random.seed(42)

    analyzer = TimingAnalyzer(confidence_level=0.95, min_samples=100)

    # Random timings independent of inputs
    timings = np.random.randn(500) * 10.0 + 100.0
    inputs = np.random.randint(0, 256, 500, dtype=np.uint8)

    result = analyzer.analyze(timings, inputs)

    # Should not detect leak
    assert isinstance(result.has_leak, bool)
    assert isinstance(result.leaks, list)
    assert isinstance(result.timing_statistics, dict)
    assert "mean" in result.timing_statistics
    assert "std" in result.timing_statistics


def test_timing_analyzer_with_leak():
    """Test TimingAnalyzer with synthetic timing leak."""
    np.random.seed(123)

    analyzer = TimingAnalyzer(confidence_level=0.95, min_samples=100)

    # Create timings with leak on bit 0
    n_samples = 500
    inputs = np.random.randint(0, 256, n_samples, dtype=np.uint8)
    timings = np.random.randn(n_samples) * 5.0 + 100.0

    # Add timing leak: bit 0 causes 20ns delay
    bit_0_set = (inputs & 1) != 0
    timings[bit_0_set] += 20.0

    result = analyzer.analyze(timings, inputs)

    # Should detect leak
    assert result.has_leak is True
    assert len(result.leaks) > 0

    # Check leak on bit 0
    bit_0_leaks = [leak for leak in result.leaks if leak.input_bit == 0]
    assert len(bit_0_leaks) > 0

    leak = bit_0_leaks[0]
    assert leak.is_significant
    assert leak.p_value < 0.05
    assert leak.mean_difference > 10.0  # Should detect the 20ns difference


def test_timing_analyzer_2d_inputs():
    """Test TimingAnalyzer with 2D input array."""
    np.random.seed(42)

    analyzer = TimingAnalyzer(min_samples=100)

    # Multi-byte inputs
    n_samples = 300
    inputs = np.random.randint(0, 256, (n_samples, 16), dtype=np.uint8)
    timings = np.random.randn(n_samples) * 5.0 + 100.0

    # Add leak on byte 5, bit 3
    bit_set = (inputs[:, 5] & (1 << 3)) != 0
    timings[bit_set] += 15.0

    result = analyzer.analyze(timings, inputs)

    # Should detect leak
    assert result.has_leak is True

    # Find leak on byte 5, bit 3
    target_leaks = [leak for leak in result.leaks if leak.input_byte == 5 and leak.input_bit == 3]
    assert len(target_leaks) > 0


def test_timing_leak_attributes():
    """Test TimingLeak has all expected attributes."""
    np.random.seed(42)

    analyzer = TimingAnalyzer(min_samples=100)

    inputs = np.random.randint(0, 256, 200, dtype=np.uint8)
    timings = np.random.randn(200) * 10.0 + 100.0

    # Add obvious leak
    timings[(inputs & 1) != 0] += 30.0

    result = analyzer.analyze(timings, inputs)

    assert len(result.leaks) > 0
    leak = result.leaks[0]

    assert hasattr(leak, "input_bit")
    assert hasattr(leak, "input_byte")
    assert hasattr(leak, "mean_difference")
    assert hasattr(leak, "t_statistic")
    assert hasattr(leak, "p_value")
    assert hasattr(leak, "confidence")
    assert hasattr(leak, "effect_size")
    assert hasattr(leak, "is_significant")


def test_timing_analyzer_partitioning():
    """Test TimingAnalyzer with custom partitioning."""
    np.random.seed(42)

    analyzer = TimingAnalyzer()

    inputs = np.random.randint(0, 256, 200, dtype=np.uint8)
    timings = np.random.randn(200) * 5.0 + 100.0

    # Add leak for high/low partition
    high_values = inputs >= 128
    timings[high_values] += 20.0

    # Custom partition function
    def partition_func(x):
        return x >= 128

    mean_diff, t_stat, p_val = analyzer.analyze_with_partitioning(timings, inputs, partition_func)

    assert mean_diff > 10.0  # Should detect difference
    assert p_val < 0.05  # Should be significant


def test_timing_analyzer_detect_outliers():
    """Test outlier detection."""
    np.random.seed(42)

    analyzer = TimingAnalyzer()

    # Normal distribution with outliers
    timings = np.random.randn(200) * 10.0 + 100.0
    timings[0] = 500.0  # Outlier
    timings[50] = -500.0  # Outlier

    outliers = analyzer.detect_outliers(timings, threshold=3.0)

    assert outliers[0] is True or outliers[0] is np.True_
    assert outliers[50] is True or outliers[50] is np.True_
    assert np.sum(outliers) >= 2  # At least the two outliers


def test_timing_analyzer_detect_outliers_no_variance():
    """Test outlier detection with constant values."""
    analyzer = TimingAnalyzer()

    # All same value
    timings = np.ones(100) * 100.0

    outliers = analyzer.detect_outliers(timings)

    # No outliers if all values identical
    assert np.sum(outliers) == 0


def test_timing_analyzer_mutual_information():
    """Test mutual information calculation."""
    np.random.seed(42)

    analyzer = TimingAnalyzer()

    # Independent variables should have low MI
    inputs = np.random.randint(0, 256, 500, dtype=np.uint8)
    timings = np.random.randn(500) * 10.0 + 100.0

    mi = analyzer.compute_mutual_information(timings, inputs, n_bins=10)

    assert isinstance(mi, float)
    assert mi >= 0.0  # MI is non-negative


def test_timing_analyzer_mutual_information_with_leak():
    """Test mutual information with timing leak."""
    np.random.seed(123)

    analyzer = TimingAnalyzer()

    # Create strong correlation
    inputs = np.random.randint(0, 256, 500, dtype=np.uint8)
    timings = inputs.astype(float) + np.random.randn(500) * 5.0

    mi = analyzer.compute_mutual_information(timings, inputs, n_bins=10)

    assert mi > 0.1  # Should have non-trivial MI


def test_timing_result_attributes():
    """Test TimingAttackResult has all expected attributes."""
    np.random.seed(42)

    analyzer = TimingAnalyzer(min_samples=100)

    inputs = np.random.randint(0, 256, 200, dtype=np.uint8)
    timings = np.random.randn(200) * 10.0 + 100.0

    result = analyzer.analyze(timings, inputs)

    assert hasattr(result, "has_leak")
    assert hasattr(result, "leaks")
    assert hasattr(result, "confidence")
    assert hasattr(result, "timing_statistics")

    assert isinstance(result.has_leak, bool)
    assert isinstance(result.leaks, list)
    assert isinstance(result.confidence, float)
    assert isinstance(result.timing_statistics, dict)


def test_timing_statistics_content():
    """Test timing statistics contain expected keys."""
    np.random.seed(42)

    analyzer = TimingAnalyzer(min_samples=100)

    inputs = np.random.randint(0, 256, 200, dtype=np.uint8)
    timings = np.random.randn(200) * 10.0 + 100.0

    result = analyzer.analyze(timings, inputs)

    stats = result.timing_statistics
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats

    # Check values are reasonable
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["std"] >= 0.0
