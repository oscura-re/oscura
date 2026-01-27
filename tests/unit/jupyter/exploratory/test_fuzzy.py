"""Comprehensive tests for fuzzy matching module.

Tests requirements:
  - FuzzyTimingResult dataclass
  - fuzzy_timing_match with WaveformTrace and edge arrays
  - Timing deviation calculation
  - Jitter measurement
  - Outlier detection
  - Confidence scoring
  - FuzzyPatternResult dataclass
  - Edge extraction from traces
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.jupyter.exploratory.fuzzy import (
    FuzzyTimingResult,
    fuzzy_timing_match,
)

pytestmark = pytest.mark.unit


class TestFuzzyTimingResult:
    """Test FuzzyTimingResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test FuzzyTimingResult initialization."""
        result = FuzzyTimingResult(
            match=True,
            confidence=0.95,
            period=1e-6,
            deviation=0.05,
            jitter_rms=1e-9,
            outlier_count=2,
            outlier_indices=[5, 10],
        )

        assert result.match is True
        assert result.confidence == 0.95
        assert result.period == 1e-6
        assert result.deviation == 0.05
        assert result.jitter_rms == 1e-9
        assert result.outlier_count == 2
        assert result.outlier_indices == [5, 10]


class TestFuzzyTimingMatchEdgeArray:
    """Test fuzzy_timing_match with edge time arrays."""

    def test_perfect_periodic_edges(self) -> None:
        """Test matching with perfectly periodic edges."""
        # Perfect 1 MHz signal (1 us period)
        edges = np.arange(0, 100e-6, 1e-6)

        result = fuzzy_timing_match(
            edges,
            expected_period=1e-6,
            tolerance=0.1,
        )

        assert result.match is True
        assert result.confidence > 0.99
        assert abs(result.period - 1e-6) < 1e-9
        assert result.deviation < 0.01
        assert result.outlier_count == 0

    def test_edges_with_jitter(self) -> None:
        """Test matching with timing jitter."""
        # 1 MHz signal with 5% jitter
        base_period = 1e-6
        edges = np.cumsum(np.random.default_rng(42).normal(base_period, base_period * 0.05, 100))

        result = fuzzy_timing_match(
            edges,
            expected_period=base_period,
            tolerance=0.15,  # Allow 15% tolerance
        )

        assert result.match is True
        assert result.confidence > 0.5
        assert abs(result.period - base_period) < base_period * 0.1
        assert result.jitter_rms > 0

    def test_edges_with_outliers(self) -> None:
        """Test detection of timing outliers."""
        # Create edges with some outliers
        edges = np.arange(0, 50e-6, 1e-6)
        edges = edges.astype(float)
        edges[10] += 5e-6  # Large deviation
        edges[20] += 5e-6  # Another outlier

        result = fuzzy_timing_match(
            edges,
            expected_period=1e-6,
            tolerance=0.1,
        )

        assert result.outlier_count >= 2
        assert len(result.outlier_indices) >= 2
        assert 9 in result.outlier_indices or 10 in result.outlier_indices  # Edge index or interval

    def test_insufficient_edges(self) -> None:
        """Test with less than 2 edges."""
        edges = np.array([1e-6])

        result = fuzzy_timing_match(
            edges,
            expected_period=1e-6,
            tolerance=0.1,
        )

        assert result.match is False
        assert result.confidence == 0.0
        assert result.period == 0.0

    def test_empty_edges(self) -> None:
        """Test with empty edge array."""
        edges = np.array([])

        result = fuzzy_timing_match(
            edges,
            expected_period=1e-6,
            tolerance=0.1,
        )

        assert result.match is False
        assert result.confidence == 0.0

    def test_auto_period_detection(self) -> None:
        """Test automatic period detection when not provided."""
        # 2 MHz signal
        edges = np.arange(0, 100e-6, 0.5e-6)

        result = fuzzy_timing_match(edges, tolerance=0.1)

        assert abs(result.period - 0.5e-6) < 1e-9
        assert result.match is True

    def test_large_deviation_no_match(self) -> None:
        """Test that large deviations result in no match."""
        # 1 MHz signal
        edges = np.arange(0, 100e-6, 1e-6)

        # Expect 10 MHz (wrong by 10x)
        result = fuzzy_timing_match(
            edges,
            expected_period=0.1e-6,
            tolerance=0.1,
        )

        assert result.match is False
        assert result.confidence < 0.5
        assert result.deviation > 0.1


class TestFuzzyTimingMatchWithTrace:
    """Test fuzzy_timing_match with WaveformTrace objects."""

    def test_square_wave_trace(self) -> None:
        """Test timing match with square wave trace."""
        # Generate 1 kHz square wave at 1 MHz sample rate
        sample_rate = 1e6
        duration = 0.01  # 10 ms
        t = np.arange(0, duration, 1 / sample_rate)
        data = (np.sin(2 * np.pi * 1e3 * t) > 0).astype(float) * 3.3

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = fuzzy_timing_match(
            trace,
            expected_period=1e-3,  # 1 kHz = 1 ms period
            tolerance=0.15,
            sample_rate=sample_rate,
        )

        # Should detect edges and match period
        assert result.period > 0
        # Period should be roughly 0.5 ms (half period between edges)
        assert 0.0003 < result.period < 0.0007

    def test_trace_without_sample_rate_fails(self) -> None:
        """Test that trace without sample_rate parameter and missing metadata raises error."""
        data = np.array([0, 1, 0, 1, 0])
        # Create trace with sample_rate, but don't pass it to function
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        # Temporarily set sample_rate to None in metadata to trigger the error
        trace.metadata.sample_rate = None  # type: ignore[assignment]

        with pytest.raises(ValueError, match="sample_rate"):
            fuzzy_timing_match(trace, expected_period=1e-6)

    def test_trace_with_invalid_sample_rate(self) -> None:
        """Test that trace with invalid sample_rate raises error."""
        data = np.array([0, 1, 0, 1, 0])

        # TraceMetadata validates sample_rate in __post_init__, so it raises on creation
        with pytest.raises(ValueError, match="sample_rate"):
            TraceMetadata(sample_rate=0)

    def test_trace_with_few_edges(self) -> None:
        """Test trace with insufficient edges."""
        # Constant signal, no edges
        data = np.ones(1000)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1e6),
        )

        result = fuzzy_timing_match(
            trace,
            expected_period=1e-6,
            tolerance=0.1,
            sample_rate=1e6,
        )

        assert result.match is False
        assert result.confidence == 0.0

    def test_noisy_digital_signal(self) -> None:
        """Test timing match with noisy digital signal."""
        # Generate square wave with noise
        sample_rate = 1e6
        duration = 0.01
        t = np.arange(0, duration, 1 / sample_rate)
        clean = (np.sin(2 * np.pi * 1e3 * t) > 0).astype(float) * 3.3

        # Add small noise (10% of signal)
        rng = np.random.default_rng(42)
        noisy = clean + rng.normal(0, 0.33, len(clean))

        trace = WaveformTrace(
            data=noisy,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = fuzzy_timing_match(
            trace,
            expected_period=1e-3,
            tolerance=0.2,  # Higher tolerance for noise
            sample_rate=sample_rate,
        )

        # Should still detect edges despite noise
        assert result.period > 0


class TestToleranceSettings:
    """Test various tolerance settings."""

    def test_strict_tolerance(self) -> None:
        """Test with strict tolerance (1%)."""
        edges = np.arange(0, 100e-6, 1e-6)

        result = fuzzy_timing_match(
            edges,
            expected_period=1e-6,
            tolerance=0.01,  # 1% tolerance
        )

        assert result.match is True
        assert result.confidence > 0.98

    def test_loose_tolerance(self) -> None:
        """Test with loose tolerance (50%)."""
        # Signal with 30% deviation
        edges = np.arange(0, 100e-6, 1.3e-6)

        result = fuzzy_timing_match(
            edges,
            expected_period=1e-6,
            tolerance=0.5,  # 50% tolerance
        )

        assert result.match is True
        assert result.confidence > 0

    def test_confidence_decreases_with_deviation(self) -> None:
        """Test that confidence decreases as deviation increases."""
        base_edges = np.arange(0, 100e-6, 1e-6)

        # Test with increasing deviations
        confidences = []
        for deviation_pct in [0.01, 0.05, 0.10, 0.15]:
            edges = np.arange(0, 100e-6, 1e-6 * (1 + deviation_pct))
            result = fuzzy_timing_match(
                edges,
                expected_period=1e-6,
                tolerance=0.2,
            )
            confidences.append(result.confidence)

        # Confidence should decrease monotonically
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]


class TestJitterMeasurement:
    """Test jitter RMS calculation."""

    def test_zero_jitter(self) -> None:
        """Test jitter calculation with perfect periodicity."""
        edges = np.arange(0, 100e-6, 1e-6)

        result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)

        assert result.jitter_rms < 1e-12  # Essentially zero

    def test_measurable_jitter(self) -> None:
        """Test jitter calculation with realistic jitter."""
        # Add controlled jitter
        base_period = 1e-6
        rng = np.random.default_rng(42)
        periods = rng.normal(base_period, base_period * 0.01, 100)  # 1% jitter
        edges = np.cumsum(periods)

        result = fuzzy_timing_match(edges, expected_period=base_period, tolerance=0.1)

        # Should measure non-zero jitter
        assert result.jitter_rms > 0
        # Jitter should be roughly 1% of period
        assert result.jitter_rms < base_period * 0.05


class TestEdgeExtraction:
    """Test edge extraction from waveform traces."""

    def test_edge_extraction_from_clean_signal(self) -> None:
        """Test edge detection from clean digital signal."""
        # Create clean square wave
        sample_rate = 1e6
        t = np.arange(0, 0.001, 1 / sample_rate)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = fuzzy_timing_match(
            trace,
            expected_period=0.5e-3,  # Half period
            tolerance=0.15,
            sample_rate=sample_rate,
        )

        # Should detect edges
        assert result.period > 0

    def test_edge_extraction_threshold_calculation(self) -> None:
        """Test that edge extraction uses proper threshold."""
        # Create signal with non-zero baseline
        sample_rate = 1e6
        t = np.arange(0, 0.001, 1 / sample_rate)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 5.0, 1.0)

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = fuzzy_timing_match(
            trace,
            expected_period=0.5e-3,
            tolerance=0.15,
            sample_rate=sample_rate,
        )

        # Should still detect edges with appropriate threshold
        assert result.period > 0
