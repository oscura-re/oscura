"""Comprehensive unit tests for trace comparison functionality.

This module tests the basic comparison functions including difference
calculation, correlation, and similarity scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import IQTrace, TraceMetadata, WaveformTrace
from oscura.utils.comparison.compare import (
    ComparisonResult,
    compare_traces,
    correlation,
    difference,
    similarity_score,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sine_trace() -> WaveformTrace:
    """Create a sine wave trace for testing."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def cosine_trace() -> WaveformTrace:
    """Create a cosine wave trace for testing."""
    data = np.cos(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def constant_trace() -> WaveformTrace:
    """Create a constant trace for testing."""
    data = np.ones(1000) * 0.5
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def noisy_sine_trace() -> WaveformTrace:
    """Create a noisy sine wave trace for testing."""
    np.random.seed(42)
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    noise = np.random.normal(0, 0.1, 1000)
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data + noise, metadata=metadata)


@pytest.fixture
def iq_trace() -> IQTrace:
    """Create an I/Q trace for testing."""
    t = np.linspace(0, 1e-3, 1000)
    i_data = np.cos(2 * np.pi * 1e6 * t)
    q_data = np.sin(2 * np.pi * 1e6 * t)
    metadata = TraceMetadata(sample_rate=1e6)
    return IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestDifference:
    """Test trace difference calculation."""

    def test_difference_identical(self, sine_trace: WaveformTrace) -> None:
        """Test difference of identical traces returns zeros."""
        result = difference(sine_trace, sine_trace)
        np.testing.assert_allclose(result.data, 0, atol=1e-10)
        assert result.metadata.channel == "difference"

    def test_difference_offset(self, sine_trace: WaveformTrace) -> None:
        """Test difference with constant offset."""
        offset_data = sine_trace.data + 1.0
        trace2 = WaveformTrace(data=offset_data, metadata=sine_trace.metadata)
        result = difference(sine_trace, trace2)
        np.testing.assert_allclose(result.data, -1.0, rtol=1e-10)

    def test_difference_inverted(self, sine_trace: WaveformTrace) -> None:
        """Test difference with inverted trace."""
        inverted_data = -sine_trace.data
        trace2 = WaveformTrace(data=inverted_data, metadata=sine_trace.metadata)
        result = difference(sine_trace, trace2)
        np.testing.assert_allclose(result.data, 2 * sine_trace.data, rtol=1e-10)

    def test_difference_normalized(self, sine_trace: WaveformTrace) -> None:
        """Test normalized difference calculation."""
        offset_data = sine_trace.data + 0.1
        trace2 = WaveformTrace(data=offset_data, metadata=sine_trace.metadata)
        result = difference(sine_trace, trace2, normalize=True)
        # Normalized to percentage of reference range
        ref_range = np.ptp(trace2.data)
        expected = (-0.1 / ref_range) * 100.0
        assert np.allclose(result.data, expected)

    def test_difference_length_mismatch(self, sine_trace: WaveformTrace) -> None:
        """Test difference with different length traces."""
        short_data = sine_trace.data[:500]
        trace2 = WaveformTrace(data=short_data, metadata=sine_trace.metadata)
        result = difference(sine_trace, trace2)
        # Should truncate to shorter length
        assert len(result.data) == 500

    def test_difference_custom_channel_name(self, sine_trace: WaveformTrace) -> None:
        """Test difference with custom channel name."""
        result = difference(sine_trace, sine_trace, channel="my_diff")
        assert result.metadata.channel == "my_diff"


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestCorrelation:
    """Test cross-correlation functionality."""

    def test_self_correlation_peak(self, sine_trace: WaveformTrace) -> None:
        """Test self-correlation has peak at zero lag."""
        lags, corr = correlation(sine_trace, sine_trace)
        peak_idx = np.argmax(corr)
        # Peak should be at or near zero lag
        assert abs(lags[peak_idx]) < 10

    def test_correlation_modes(self, sine_trace: WaveformTrace) -> None:
        """Test different correlation modes."""
        # Full mode
        lags_full, corr_full = correlation(sine_trace, sine_trace, mode="full")
        assert len(corr_full) == 2 * len(sine_trace.data) - 1

        # Same mode
        lags_same, corr_same = correlation(sine_trace, sine_trace, mode="same")
        assert len(corr_same) == len(sine_trace.data)

        # Valid mode
        lags_valid, corr_valid = correlation(sine_trace, sine_trace, mode="valid")
        assert len(corr_valid) == 1

    def test_correlation_without_normalization(self, sine_trace: WaveformTrace) -> None:
        """Test correlation without normalization."""
        lags, corr = correlation(sine_trace, sine_trace, normalize=False)
        assert len(lags) == len(corr)
        # Unnormalized correlation should be larger
        assert np.max(corr) > 1.0

    def test_correlation_shifted_signal(self, sine_trace: WaveformTrace) -> None:
        """Test correlation with time-shifted signal."""
        # Create shifted version
        shifted_data = np.roll(sine_trace.data, 50)
        trace2 = WaveformTrace(data=shifted_data, metadata=sine_trace.metadata)

        lags, corr = correlation(sine_trace, trace2)
        peak_idx = np.argmax(corr)
        # Peak should be near lag=50 or lag=-50 (depending on direction)
        assert abs(abs(lags[peak_idx]) - 50) < 20

    def test_correlation_different_signals(
        self, sine_trace: WaveformTrace, constant_trace: WaveformTrace
    ) -> None:
        """Test correlation between different signals."""
        lags, corr = correlation(sine_trace, constant_trace)
        # Correlation with constant should be low
        assert np.max(np.abs(corr)) < 0.5


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestSimilarityScore:
    """Test similarity score calculation."""

    def test_identical_traces(self, sine_trace: WaveformTrace) -> None:
        """Test similarity of identical traces."""
        score = similarity_score(sine_trace, sine_trace)
        assert score > 0.99

    def test_different_traces(self, sine_trace: WaveformTrace, cosine_trace: WaveformTrace) -> None:
        """Test similarity of different traces."""
        score = similarity_score(sine_trace, cosine_trace)
        # Sine and cosine are orthogonal, should have low similarity
        assert score < 0.6

    def test_inverted_traces(self, sine_trace: WaveformTrace) -> None:
        """Test similarity of inverted traces."""
        inverted_data = -sine_trace.data
        trace2 = WaveformTrace(data=inverted_data, metadata=sine_trace.metadata)
        score = similarity_score(sine_trace, trace2)
        # Inverted traces should have very low similarity
        assert score < 0.1

    def test_similarity_methods(self, sine_trace: WaveformTrace) -> None:
        """Test different similarity methods."""
        noisy_data = sine_trace.data + np.random.normal(0, 0.01, len(sine_trace.data))
        trace2 = WaveformTrace(data=noisy_data, metadata=sine_trace.metadata)

        # Correlation method
        score_corr = similarity_score(trace2, sine_trace, method="correlation")
        assert 0.8 < score_corr < 1.0

        # RMS method
        score_rms = similarity_score(trace2, sine_trace, method="rms")
        assert 0.8 < score_rms < 1.0

        # MSE method
        score_mse = similarity_score(trace2, sine_trace, method="mse")
        assert 0.8 < score_mse < 1.0

        # Cosine method
        score_cosine = similarity_score(trace2, sine_trace, method="cosine")
        assert 0.8 < score_cosine < 1.0

    def test_similarity_without_normalization(self, sine_trace: WaveformTrace) -> None:
        """Test similarity without amplitude normalization.

        Note: Correlation-based similarity is inherently scale-invariant because
        Pearson correlation normalizes by standard deviations. Both scaled traces
        will have perfect correlation (1.0) regardless of amplitude scaling.
        """
        # Scale the trace
        scaled_data = sine_trace.data * 2.0
        trace2 = WaveformTrace(data=scaled_data, metadata=sine_trace.metadata)

        # With normalization (default)
        score_norm = similarity_score(trace2, sine_trace, normalize_amplitude=True)
        assert score_norm > 0.95

        # Without normalization - correlation is scale-invariant so should still be high
        score_no_norm = similarity_score(trace2, sine_trace, normalize_amplitude=False)
        assert score_no_norm >= 0.95  # Both should be high due to correlation's scale invariance

    def test_similarity_constant_traces(self, constant_trace: WaveformTrace) -> None:
        """Test similarity with constant traces."""
        # Identical constants
        score = similarity_score(constant_trace, constant_trace)
        assert score > 0.95

        # Different constants
        data2 = np.ones(1000) * 0.7
        trace2 = WaveformTrace(data=data2, metadata=constant_trace.metadata)
        score2 = similarity_score(constant_trace, trace2)
        # Should still be high after normalization
        assert score2 > 0.5


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestCompareTraces:
    """Test comprehensive trace comparison."""

    def test_compare_identical(self, sine_trace: WaveformTrace) -> None:
        """Test comparing identical traces."""
        result = compare_traces(sine_trace, sine_trace)
        assert result.match
        assert result.max_difference < 1e-10
        assert result.similarity > 0.99
        assert result.correlation > 0.99
        assert result.difference_trace is not None

    def test_compare_with_tolerance(self, sine_trace: WaveformTrace) -> None:
        """Test comparison with absolute tolerance."""
        noisy_data = sine_trace.data + np.random.normal(0, 0.001, len(sine_trace.data))
        trace2 = WaveformTrace(data=noisy_data, metadata=sine_trace.metadata)

        result = compare_traces(sine_trace, trace2, tolerance=0.01)
        assert result.match
        assert result.max_difference < 0.01

    def test_compare_with_percentage_tolerance(self, sine_trace: WaveformTrace) -> None:
        """Test comparison with percentage tolerance."""
        result = compare_traces(sine_trace, sine_trace, tolerance_pct=1.0)
        assert result.match

    def test_compare_different_lengths(self, sine_trace: WaveformTrace) -> None:
        """Test comparing traces of different lengths."""
        short_data = sine_trace.data[:500]
        trace2 = WaveformTrace(data=short_data, metadata=sine_trace.metadata)

        result = compare_traces(sine_trace, trace2)
        # Should work by truncating to shorter length
        assert isinstance(result, ComparisonResult)

    def test_compare_methods(self, sine_trace: WaveformTrace) -> None:
        """Test different comparison methods."""
        offset_data = sine_trace.data + 0.01
        trace2 = WaveformTrace(data=offset_data, metadata=sine_trace.metadata)

        # Absolute method
        result_abs = compare_traces(trace2, sine_trace, tolerance=0.02, method="absolute")
        assert result_abs.match

        # Relative method
        result_rel = compare_traces(trace2, sine_trace, tolerance_pct=1.0, method="relative")
        assert isinstance(result_rel, ComparisonResult)

        # Statistical method
        result_stat = compare_traces(trace2, sine_trace, method="statistical")
        assert isinstance(result_stat, ComparisonResult)

    def test_compare_without_difference_trace(self, sine_trace: WaveformTrace) -> None:
        """Test comparison without including difference trace."""
        result = compare_traces(sine_trace, sine_trace, include_difference=False)
        assert result.difference_trace is None

    def test_compare_violations(self, sine_trace: WaveformTrace) -> None:
        """Test violation detection in comparison."""
        # Create trace that violates tolerance in some places
        violated_data = sine_trace.data.copy()
        violated_data[100:110] += 1.0
        trace2 = WaveformTrace(data=violated_data, metadata=sine_trace.metadata)

        result = compare_traces(sine_trace, trace2, tolerance=0.1)
        assert not result.match
        assert result.violations is not None
        assert len(result.violations) > 0
        assert result.statistics is not None
        assert result.statistics["num_violations"] > 0

    def test_compare_statistics(self, sine_trace: WaveformTrace) -> None:
        """Test statistics in comparison result."""
        result = compare_traces(sine_trace, sine_trace)
        assert result.statistics is not None
        assert "mean_difference" in result.statistics
        assert "std_difference" in result.statistics
        assert "median_difference" in result.statistics
        assert "num_violations" in result.statistics
        assert "violation_rate" in result.statistics
        assert "p_value" in result.statistics

    def test_compare_constant_traces(self, constant_trace: WaveformTrace) -> None:
        """Test comparing constant traces."""
        result = compare_traces(constant_trace, constant_trace)
        assert result.match
        # Correlation may be NaN or undefined for constants, but shouldn't crash
        assert isinstance(result.correlation, float)


@pytest.mark.unit
@pytest.mark.comparison
class TestComparisonCompareEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_traces(self) -> None:
        """Test comparison with empty traces."""
        # Empty arrays should raise ValueError at trace creation time
        metadata = TraceMetadata(sample_rate=1e6)
        with pytest.raises(ValueError, match="data array cannot be empty"):
            WaveformTrace(data=np.array([]), metadata=metadata)

    def test_single_sample(self) -> None:
        """Test comparison with single sample traces."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace1 = WaveformTrace(data=np.array([1.0]), metadata=metadata)
        trace2 = WaveformTrace(data=np.array([1.0]), metadata=metadata)

        result = compare_traces(trace1, trace2)
        assert result.match

    def test_nan_values(self) -> None:
        """Test handling of NaN values in traces."""
        metadata = TraceMetadata(sample_rate=1e6)
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        trace = WaveformTrace(data=data_with_nan, metadata=metadata)

        # Should either handle NaN or raise appropriate error
        try:
            result = similarity_score(trace, trace)
            # If it succeeds, check it doesn't return NaN
            assert not np.isnan(result)
        except (ValueError, FloatingPointError):
            # Acceptable to raise on NaN
            pass

    def test_inf_values(self) -> None:
        """Test handling of infinite values."""
        metadata = TraceMetadata(sample_rate=1e6)
        data_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
        trace = WaveformTrace(data=data_with_inf, metadata=metadata)

        try:
            result = difference(trace, trace)
            # Difference should be well-defined even with inf
            assert isinstance(result, WaveformTrace)
        except (ValueError, FloatingPointError):
            # Acceptable to raise
            pass

    def test_very_large_values(self) -> None:
        """Test with very large amplitude values."""
        metadata = TraceMetadata(sample_rate=1e6)
        data_large = np.ones(1000) * 1e10
        trace = WaveformTrace(data=data_large, metadata=metadata)

        result = similarity_score(trace, trace)
        assert result > 0.99

    def test_very_small_values(self) -> None:
        """Test with very small amplitude values."""
        metadata = TraceMetadata(sample_rate=1e6)
        data_small = np.ones(1000) * 1e-10
        trace = WaveformTrace(data=data_small, metadata=metadata)

        result = similarity_score(trace, trace)
        assert result > 0.99
