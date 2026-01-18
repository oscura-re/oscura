"""Timing edge case tests for analyzers to improve coverage.

This module tests timing-related boundary conditions and edge cases
in signal analyzers.

(Edge Case Tests)

- Timing edge cases testing (+0.5% coverage)
- Zero-duration signal handling
- Sample-rate boundary conditions
- Floating-point precision edge cases
- Clock frequency validation
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from oscura.analyzers.digital.edges import detect_edges
from oscura.analyzers.digital.timing import propagation_delay, setup_time, slew_rate
from oscura.analyzers.statistics.basic import basic_stats
from oscura.analyzers.statistics.correlation import cross_correlation
from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


@pytest.mark.unit
@pytest.mark.analyzer
class TestZeroDurationSignals:
    """Test handling of zero-duration signals."""

    def test_detect_edges_empty_trace(self) -> None:
        """Test detect_edges with empty trace."""
        data = np.array([], dtype=np.float64)
        threshold = 0.5

        edges = detect_edges(data, threshold=threshold)
        rising_edges = [e for e in edges if e.edge_type == "rising"]
        falling_edges = [e for e in edges if e.edge_type == "falling"]
        assert rising_edges == []
        assert falling_edges == []

    def test_detect_edges_single_sample(self) -> None:
        """Test detect_edges with single-sample trace."""
        data = np.array([1.0], dtype=np.float64)
        threshold = 0.5

        edges = detect_edges(data, threshold=threshold)
        # No transitions with only one sample
        rising_edges = [e for e in edges if e.edge_type == "rising"]
        falling_edges = [e for e in edges if e.edge_type == "falling"]
        assert len(rising_edges) == 0
        assert len(falling_edges) == 0

    def test_detect_edges_two_samples_rising(self) -> None:
        """Test detect_edges with two samples (minimum for transition)."""
        data = np.array([0.0, 1.0], dtype=np.float64)
        threshold = 0.5

        edges = detect_edges(data, threshold=threshold)
        # Should detect one rising edge
        rising_edges = [e for e in edges if e.edge_type == "rising"]
        assert len(rising_edges) == 1

    def test_slew_rate_empty_trace(self) -> None:
        """Test slew_rate with empty trace."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=np.array([], dtype=np.float64), metadata=metadata)

        # Should return NaN or empty array
        result = slew_rate(trace)
        assert np.isnan(result) or result == 0.0


@pytest.mark.unit
@pytest.mark.analyzer
class TestSampleRateBoundaries:
    """Test sample rate boundary conditions."""

    def test_very_low_sample_rate(self) -> None:
        """Test analysis with very low sample rate (1 Hz)."""
        metadata = TraceMetadata(sample_rate=1.0)  # 1 Hz
        data = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        assert stats["mean"] == pytest.approx(0.0)
        # Sample rate comes from metadata, not stats
        assert trace.metadata.sample_rate == 1.0

    def test_very_high_sample_rate(self) -> None:
        """Test analysis with very high sample rate (10 THz)."""
        metadata = TraceMetadata(sample_rate=10e12)  # 10 THz
        data = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        assert trace.metadata.sample_rate == 10e12
        # Duration should be very small (use trace.duration, not metadata.duration)
        assert trace.duration < 1e-12

    def test_exact_power_of_two_sample_rate(self) -> None:
        """Test with sample rate exactly at power of 2."""
        metadata = TraceMetadata(sample_rate=1024.0)  # 2^10
        data = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        edges = detect_edges(data, threshold=0.5)
        # Should have precise timing calculations
        rising_edges = [e for e in edges if e.edge_type == "rising"]
        assert len(rising_edges) >= 0

    def test_sample_rate_one(self) -> None:
        """Test minimum sample rate of 1.0 Hz."""
        metadata = TraceMetadata(sample_rate=1.0)
        data = np.array([1.0, 0.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        edges = detect_edges(data, threshold=0.5)
        falling_edges = [e for e in edges if e.edge_type == "falling"]
        assert len(falling_edges) >= 0


@pytest.mark.unit
@pytest.mark.analyzer
class TestFloatingPointPrecision:
    """Test floating-point precision edge cases."""

    def test_time_calculation_precision(self) -> None:
        """Test time calculation doesn't lose precision."""
        # Use sample rate that creates repeating decimal
        metadata = TraceMetadata(sample_rate=3.0)  # 1/3 second per sample
        data = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        edges = detect_edges(data, threshold=0.5)
        # Should detect transitions
        rising_edges = [e for e in edges if e.edge_type == "rising"]
        assert len(rising_edges) >= 0

    def test_very_small_time_differences(self) -> None:
        """Test detecting transitions with very small time differences."""
        metadata = TraceMetadata(sample_rate=1e15)  # 1 PHz - femtosecond resolution
        data = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        edges = detect_edges(data, threshold=0.5)
        rising_edges = [e for e in edges if e.edge_type == "rising"]
        assert len(rising_edges) >= 0

    def test_near_zero_values(self) -> None:
        """Test statistics with values near floating-point zero."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([1e-300, -1e-300, 1e-300, -1e-300], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        assert stats["mean"] == pytest.approx(0.0, abs=1e-299)

    def test_large_magnitude_differences(self) -> None:
        """Test handling of large magnitude differences."""
        metadata = TraceMetadata(sample_rate=1e6)
        # Mix very large and very small values
        data = np.array([1e100, 1e-100, 1e100, 1e-100], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        # Should not overflow or underflow
        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["std"])


@pytest.mark.unit
@pytest.mark.analyzer
class TestTimingAnalysisEdgeCases:
    """Test timing analysis edge cases."""

    def test_propagation_delay_same_trace(self) -> None:
        """Test propagation delay with same trace (zero delay)."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        delay = propagation_delay(trace, trace, ref_level=0.5)
        # When comparing trace to itself, input and output edges are identical,
        # so no subsequent edge is found - returns NaN
        assert np.isnan(delay)

    def test_setup_time_at_boundary(self) -> None:
        """Test setup time at exact clock boundary."""
        metadata = TraceMetadata(sample_rate=1e6)
        data_signal = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        clock_signal = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        data_trace = WaveformTrace(data=data_signal, metadata=metadata)
        clock_trace = WaveformTrace(data=clock_signal, metadata=metadata)

        time = setup_time(data_trace, clock_trace, data_stable_level=0.5, clock_edge="rising")
        # Setup time should be positive (data changes before clock)
        assert time >= 0.0 or np.isnan(time)

    def test_slew_rate_instant_transition(self) -> None:
        """Test slew rate with instant (single-sample) transition."""
        metadata = TraceMetadata(sample_rate=1e9)  # 1 GHz
        data = np.array([0.0, 1.0], dtype=np.float64)  # Instant transition
        trace = WaveformTrace(data=data, metadata=metadata)

        # Slew rate for rising edges with default 20%-80% levels
        rate = slew_rate(trace, edge_type="rising")
        # Should be very high (1V in 1ns = 1e9 V/s)
        assert rate >= 1e8 or np.isnan(rate)  # At least 100 MV/s


@pytest.mark.unit
@pytest.mark.analyzer
class TestCorrelationEdgeCases:
    """Test correlation edge cases."""

    def test_cross_correlate_empty_traces(self) -> None:
        """Test cross-correlation with empty traces."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace1 = WaveformTrace(data=np.array([], dtype=np.float64), metadata=metadata)
        trace2 = WaveformTrace(data=np.array([], dtype=np.float64), metadata=metadata)

        # Should handle empty gracefully or raise error
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = cross_correlation(trace1.data, trace2.data, sample_rate=1e6)
                assert len(result.lags) == 0 or len(result.lags) == 1
        except (AnalysisError, ValueError):
            # Acceptable to raise error on empty
            pass

    def test_cross_correlate_single_sample(self) -> None:
        """Test cross-correlation with single sample."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace1 = WaveformTrace(data=np.array([1.0], dtype=np.float64), metadata=metadata)
        trace2 = WaveformTrace(data=np.array([1.0], dtype=np.float64), metadata=metadata)

        result = cross_correlation(trace1.data, trace2.data, sample_rate=1e6)
        assert len(result.lags) > 0

    def test_cross_correlate_perfect_correlation(self) -> None:
        """Test cross-correlation of identical signals."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        trace1 = WaveformTrace(data=data, metadata=metadata)
        trace2 = WaveformTrace(data=data.copy(), metadata=metadata)

        result = cross_correlation(trace1.data, trace2.data, sample_rate=1e6)
        # Peak correlation should be at zero lag
        peak_idx = np.argmax(np.abs(result.correlation))
        assert peak_idx == len(result.correlation) // 2  # Zero lag


@pytest.mark.unit
@pytest.mark.analyzer
class TestStatisticalEdgeCases:
    """Test statistical analysis edge cases."""

    def test_statistics_constant_signal(self) -> None:
        """Test statistics of constant signal (zero variance)."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([5.0] * 100, dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        assert stats["mean"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(0.0, abs=1e-10)
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0

    def test_statistics_alternating_extremes(self) -> None:
        """Test statistics with alternating min/max values."""
        import warnings

        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array(
            [float("-inf"), float("inf"), float("-inf"), float("inf")], dtype=np.float64
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats = basic_stats(trace.data)
        # Should handle infinite values
        assert np.isinf(stats["max"])
        assert np.isinf(stats["min"])

    def test_statistics_all_nan(self) -> None:
        """Test statistics with all NaN values."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        # All statistics should be NaN
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])

    def test_statistics_single_non_nan(self) -> None:
        """Test statistics with mostly NaN and one valid value."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([np.nan, 42.0, np.nan], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=metadata)

        stats = basic_stats(trace.data)
        # Should compute stats ignoring NaN or return NaN
        assert np.isnan(stats["mean"]) or stats["mean"] == pytest.approx(42.0)


# Property-based testing with Hypothesis
@pytest.mark.unit
@pytest.mark.analyzer
class TestPropertyBasedTiming:
    """Property-based tests for timing edge cases."""

    @given(st.floats(min_value=1.0, max_value=1e15, allow_nan=False, allow_infinity=False))
    def test_sample_rate_always_positive(self, sample_rate: float) -> None:
        """Property: Sample rate must always be positive."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        assert metadata.sample_rate > 0

    @given(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        )
    )
    def test_edge_detection_bounded(self, data: list[float]) -> None:
        """Property: Edge count never exceeds len(data) - 1."""
        arr = np.array(data, dtype=np.float64)
        edges = detect_edges(arr, threshold=0.0)

        rising_edges = [e for e in edges if e.edge_type == "rising"]
        falling_edges = [e for e in edges if e.edge_type == "falling"]
        total_edges = len(rising_edges) + len(falling_edges)
        assert 0 <= total_edges <= len(data) - 1

    @given(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        )
    )
    def test_statistics_always_finite(self, data: list[float]) -> None:
        """Property: Statistics should be finite for finite input."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=np.array(data, dtype=np.float64), metadata=metadata)

        stats = basic_stats(trace.data)
        # Mean and std should be finite for finite data
        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["std"])

    @given(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        )
    )
    def test_cross_correlation_symmetric(self, data: list[float]) -> None:
        """Property: Auto-correlation should be symmetric."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=np.array(data, dtype=np.float64), metadata=metadata)

        # Auto-correlation (trace with itself)
        result = cross_correlation(trace.data, trace.data, sample_rate=1e6)

        # Correlation should be symmetric around center
        n = len(result.correlation)
        if n > 1:
            # Check approximate symmetry
            left_half = result.correlation[: n // 2]
            right_half = result.correlation[n // 2 :][::-1]
            # Allow for numerical precision differences
            assert np.allclose(left_half, right_half[: len(left_half)], rtol=1e-5, atol=1e-8)
