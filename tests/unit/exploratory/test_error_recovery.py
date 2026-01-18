"""Comprehensive unit tests for error recovery module.

This module provides extensive testing for error recovery mechanisms,
graceful degradation, partial decode support, and confidence-based error handling.


Test Coverage:
- recover_corrupted_data with various corruption types
- graceful_degradation for analysis failures
- partial_decode with segment processing
- ErrorContext capture and suggestions
- retry_with_adjustment with parameter tuning
- Edge cases: empty data, fully corrupted, no errors
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.exploratory.error_recovery import (
    ErrorContext,
    graceful_degradation,
    partial_decode,
    recover_corrupted_data,
    retry_with_adjustment,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Functions
# =============================================================================


def make_waveform_trace(data: np.ndarray, sample_rate: float = 1e6) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def inject_spikes(data: np.ndarray, n_spikes: int, spike_value: float = 100.0) -> np.ndarray:
    """Inject random corruption spikes into data."""
    corrupted = data.copy()
    spike_indices = np.random.choice(len(data), size=n_spikes, replace=False)
    corrupted[spike_indices] = spike_value
    return corrupted


def inject_bursts(data: np.ndarray, n_bursts: int, burst_length: int = 10) -> np.ndarray:
    """Inject burst corruption into data."""
    corrupted = data.copy()
    for _ in range(n_bursts):
        start = np.random.randint(0, max(1, len(data) - burst_length))
        corrupted[start : start + burst_length] = np.nan
    return corrupted


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestRecoverCorruptedData:
    """Test corrupted data recovery (ERROR-001)."""

    def test_no_corruption_returns_original(self) -> None:
        """Test that clean data passes through unchanged."""
        data = np.sin(np.linspace(0, 10 * np.pi, 1000))
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(trace)

        assert stats.corrupted_samples == 0
        assert stats.recovered_samples == 0
        assert stats.confidence == 1.0
        assert stats.recovery_method == "none"
        np.testing.assert_array_equal(recovered_trace.data, trace.data)

    def test_recover_spike_noise_interpolation(self) -> None:
        """Test recovery of spike noise using interpolation."""
        # Generate clean signal
        data = np.sin(np.linspace(0, 10 * np.pi, 1000))

        # Inject 10 spikes
        corrupted = inject_spikes(data, n_spikes=10, spike_value=100.0)
        trace = make_waveform_trace(corrupted)

        recovered_trace, stats = recover_corrupted_data(
            trace, corruption_threshold=3.0, recovery_method="interpolate"
        )

        # Check statistics
        assert stats.corrupted_samples >= 10
        assert stats.recovered_samples >= 10
        assert stats.recovery_method == "interpolate"
        assert 0.0 <= stats.confidence <= 1.0

        # Recovered data should be closer to original
        original_error = np.mean(np.abs(corrupted - data))
        recovered_error = np.mean(np.abs(recovered_trace.data - data))
        assert recovered_error < original_error

    def test_recover_nan_values(self) -> None:
        """Test recovery of NaN values."""
        data = np.ones(100)
        data[10:15] = np.nan
        data[50] = np.nan
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(trace, recovery_method="median")

        # All NaN should be detected and recovered
        assert stats.corrupted_samples >= 6
        assert not np.any(np.isnan(recovered_trace.data))

    def test_recover_inf_values(self) -> None:
        """Test recovery of Inf values."""
        data = np.ones(100) * 5.0
        data[20:25] = np.inf
        data[30] = -np.inf
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(trace, recovery_method="zero")

        # All Inf should be detected and recovered
        assert stats.corrupted_samples >= 6
        assert not np.any(np.isinf(recovered_trace.data))
        assert stats.recovery_method == "zero"

    def test_large_gap_unrecoverable(self) -> None:
        """Test that large gaps are marked unrecoverable."""
        data = np.ones(1000)
        # Create a large corrupted region
        data[100:250] = np.nan
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(
            trace, max_gap_samples=100, recovery_method="interpolate"
        )

        # Large gap should be unrecoverable
        assert stats.unrecoverable_samples > 0
        assert stats.confidence < 1.0

    def test_median_recovery_method(self) -> None:
        """Test median recovery method."""
        data = np.sin(np.linspace(0, 4 * np.pi, 500)) + 2.0
        corrupted = inject_spikes(data, n_spikes=5)
        trace = make_waveform_trace(corrupted)

        recovered_trace, stats = recover_corrupted_data(
            trace, recovery_method="median", corruption_threshold=3.0
        )

        assert stats.recovery_method == "median"
        assert stats.recovered_samples > 0

    def test_zero_recovery_method(self) -> None:
        """Test zero-fill recovery method."""
        data = np.ones(100) * 5.0
        data[40:45] = 100.0  # Outliers
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(
            trace, recovery_method="zero", corruption_threshold=2.0
        )

        assert stats.recovery_method == "zero"
        # Check that recovered regions contain zeros
        assert np.any(recovered_trace.data == 0.0)

    def test_empty_trace(self) -> None:
        """Test recovery with empty trace."""
        data = np.array([])
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(trace)

        assert stats.total_samples == 0
        assert stats.corrupted_samples == 0

    def test_constant_signal_no_mad(self) -> None:
        """Test recovery when signal has zero MAD."""
        data = np.ones(100) * 3.3
        data[50] = 100.0  # Single spike
        trace = make_waveform_trace(data)

        recovered_trace, stats = recover_corrupted_data(trace, corruption_threshold=3.0)

        # Should fallback to std and still detect outlier
        assert stats.corrupted_samples >= 1

    def test_confidence_calculation(self) -> None:
        """Test confidence score calculation."""
        data = np.sin(np.linspace(0, 4 * np.pi, 1000))
        corrupted = inject_spikes(data, n_spikes=3)
        trace = make_waveform_trace(corrupted)

        recovered_trace, stats = recover_corrupted_data(
            trace, recovery_method="interpolate", max_gap_samples=10
        )

        # Small isolated spikes should give high confidence
        assert stats.confidence > 0.5


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestGracefulDegradation:
    """Test graceful degradation (ERROR-002)."""

    def test_successful_full_analysis(self) -> None:
        """Test graceful degradation when analysis succeeds."""

        def mock_analysis(trace: WaveformTrace) -> dict:
            return {
                "mean": np.mean(trace.data),
                "std": np.std(trace.data),
                "max": np.max(trace.data),
            }

        data = np.random.randn(1000)
        trace = make_waveform_trace(data)

        result = graceful_degradation(mock_analysis, trace)

        assert result.quality_level == "full"
        assert len(result.available_features) == 3
        assert len(result.missing_features) == 0
        assert "mean" in result.result
        assert "std" in result.result

    def test_failed_analysis_with_fallback(self) -> None:
        """Test graceful degradation when analysis fails."""

        def failing_analysis(trace: WaveformTrace) -> dict:
            raise ValueError("Analysis failed")

        data = np.random.randn(100)
        trace = make_waveform_trace(data)

        result = graceful_degradation(
            failing_analysis, trace, required_features=[], optional_features=[]
        )

        assert result.quality_level in ["degraded", "minimal", "failed"]
        assert len(result.warnings) > 0
        assert any("failed" in w.lower() for w in result.warnings)

    def test_partial_feature_availability(self) -> None:
        """Test when some features succeed and others fail."""

        def partial_analysis(trace: WaveformTrace) -> dict:
            # This will fail but fallback will try trace attributes
            raise RuntimeError("Simulated failure")

        data = np.random.randn(100)
        trace = make_waveform_trace(data)
        trace.custom_attr = 42  # type: ignore[attr-defined]

        result = graceful_degradation(
            partial_analysis,
            trace,
            required_features=["data"],
            optional_features=["custom_attr", "missing_attr"],
        )

        # Should have degraded quality but some features available
        assert "data" in result.available_features or "custom_attr" in result.available_features
        assert result.quality_level in ["degraded", "minimal"]

    def test_empty_data_handling(self) -> None:
        """Test graceful degradation with empty data."""

        def analysis_needs_data(trace: WaveformTrace) -> dict:
            if len(trace.data) == 0:
                raise ValueError("No data")
            return {"count": len(trace.data)}

        data = np.array([])
        trace = make_waveform_trace(data)

        result = graceful_degradation(analysis_needs_data, trace)

        assert result.quality_level in ["failed", "minimal", "degraded"]
        assert len(result.warnings) > 0


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestPartialDecode:
    """Test partial decode support (ERROR-003)."""

    def test_successful_full_decode(self) -> None:
        """Test partial decode when full decode succeeds."""

        def mock_decode(trace: WaveformTrace) -> list[dict]:
            # Return mock packets
            n_packets = len(trace.data) // 100
            return [{"data": i, "timestamp": i * 0.001} for i in range(n_packets)]

        data = np.random.randn(1000)
        trace = make_waveform_trace(data)

        result = partial_decode(trace, mock_decode)

        assert len(result.complete_packets) > 0
        assert len(result.error_regions) == 0
        assert result.decode_rate > 0.9
        assert result.confidence > 0.5

    def test_segment_by_segment_decode(self) -> None:
        """Test fallback to segment-by-segment decode."""

        def failing_full_decode(trace: WaveformTrace) -> list[dict]:
            # Fail on full trace but succeed on segments
            if len(trace.data) > 500:
                raise ValueError("Trace too large")
            return [{"data": len(trace.data)}]

        data = np.random.randn(1500)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = partial_decode(trace, failing_full_decode, segment_size=400)

        # Should have some partial packets from segments (since full decode failed)
        assert len(result.partial_packets) > 0
        assert result.decode_rate < 1.0  # Not perfect due to failures

    def test_partial_and_error_regions(self) -> None:
        """Test mix of complete, partial, and error regions."""

        def selective_decode(trace: WaveformTrace) -> list[dict]:
            # Succeed for data starting with positive mean, fail otherwise
            if np.mean(trace.data[:10]) > 0:
                return [{"timestamp": 0}]
            raise RuntimeError("Decode failed")

        # Create data with mixed regions
        data = np.concatenate([np.ones(300), -np.ones(300), np.ones(300)])
        trace = make_waveform_trace(data)

        result = partial_decode(trace, selective_decode, segment_size=300)

        # Should have some successes (decode_rate > 0) and possibly some failures
        assert result.decode_rate > 0.0
        assert (
            len(result.error_regions) > 0
            or len(result.complete_packets) > 0
            or len(result.partial_packets) > 0
        )

    def test_timestamp_adjustment(self) -> None:
        """Test that timestamps are adjusted for segments."""

        def mock_decode(trace: WaveformTrace) -> list[dict]:
            return [{"timestamp": 0.0, "sample": 0}]

        data = np.random.randn(2000)
        trace = make_waveform_trace(data, sample_rate=1e6)

        # Force segment-by-segment by making full decode fail
        def failing_decode(t: WaveformTrace) -> list[dict]:
            if len(t.data) > 500:
                raise ValueError("Too large")
            return mock_decode(t)

        result = partial_decode(trace, failing_decode, segment_size=500)

        # Timestamps should be different for different segments
        if len(result.complete_packets) > 1:
            timestamps = [p["timestamp"] for p in result.complete_packets if "timestamp" in p]
            if len(timestamps) > 1:
                assert not all(t == timestamps[0] for t in timestamps)

    def test_empty_trace_decode(self) -> None:
        """Test partial decode with empty trace."""

        def mock_decode(trace: WaveformTrace) -> list[dict]:
            return []

        data = np.array([])
        trace = make_waveform_trace(data)

        result = partial_decode(trace, mock_decode)

        assert len(result.complete_packets) == 0
        assert result.decode_rate >= 0.0


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestErrorContext:
    """Test error context preservation (ERROR-006)."""

    def test_capture_basic_exception(self) -> None:
        """Test capturing basic exception information."""
        error = ValueError("Test error message")

        context = ErrorContext.capture(error)

        assert context.error_type == "ValueError"
        assert context.error_message == "Test error message"
        assert len(context.suggestions) > 0

    def test_capture_with_trace_context(self) -> None:
        """Test capturing error with signal context."""
        data = np.sin(np.linspace(0, 4 * np.pi, 1000))
        trace = make_waveform_trace(data)
        error = RuntimeError("Threshold error")
        location = 500

        context = ErrorContext.capture(error, trace=trace, location=location, context_samples=50)

        assert context.location == 500
        assert context.context_before is not None
        assert len(context.context_before) == 50
        assert context.context_after is not None

    def test_context_suggestions_threshold(self) -> None:
        """Test that threshold errors get appropriate suggestions."""
        error = ValueError("threshold too high")

        context = ErrorContext.capture(error)

        assert any("threshold" in s.lower() for s in context.suggestions)

    def test_context_suggestions_memory(self) -> None:
        """Test that memory errors get appropriate suggestions."""
        error = MemoryError("Out of memory")

        context = ErrorContext.capture(error)

        assert any("memory" in s.lower() or "chunk" in s.lower() for s in context.suggestions)

    def test_context_with_parameters(self) -> None:
        """Test capturing analysis parameters."""
        error = RuntimeError("Analysis failed")
        params = {"threshold": 0.5, "window_size": 100}

        context = ErrorContext.capture(error, parameters=params)

        assert context.parameters == params

    def test_context_edge_location(self) -> None:
        """Test context capture at trace edges."""
        data = np.ones(100)
        trace = make_waveform_trace(data)
        error = ValueError("Test")

        # At beginning
        context1 = ErrorContext.capture(error, trace=trace, location=0, context_samples=50)
        assert context1.context_before is not None
        assert len(context1.context_before) == 0

        # At end
        context2 = ErrorContext.capture(error, trace=trace, location=99, context_samples=50)
        assert context2.context_after is not None


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestRetryWithAdjustment:
    """Test retry with parameter adjustment (ERROR-005)."""

    def test_success_on_first_try(self) -> None:
        """Test successful analysis on first attempt."""

        def always_succeed(trace: WaveformTrace, threshold: float = 0.5) -> dict:
            return {"result": "success", "threshold": threshold}

        data = np.random.randn(100)
        trace = make_waveform_trace(data)
        params = {"threshold": 0.5}

        result = retry_with_adjustment(always_succeed, trace, params)

        assert result.success is True
        assert result.attempts == 1
        assert len(result.adjustments_made) == 0

    def test_success_after_retry(self) -> None:
        """Test success after parameter adjustment."""
        call_count = {"count": 0}

        def fail_first_succeed_second(trace: WaveformTrace, threshold: float = 1.0) -> dict:
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise ValueError("Threshold too high")
            return {"result": "success"}

        data = np.random.randn(100)
        trace = make_waveform_trace(data)
        params = {"threshold": 1.0}

        result = retry_with_adjustment(fail_first_succeed_second, trace, params, max_retries=3)

        assert result.success is True
        assert result.attempts == 2
        assert len(result.adjustments_made) > 0

    def test_all_retries_exhausted(self) -> None:
        """Test when all retries are exhausted."""

        def always_fail(trace: WaveformTrace, threshold: float = 0.5) -> dict:
            raise RuntimeError("Always fails")

        data = np.random.randn(100)
        trace = make_waveform_trace(data)
        params = {"threshold": 0.5}

        result = retry_with_adjustment(always_fail, trace, params, max_retries=2)

        assert result.success is False
        assert result.attempts == 3  # Initial + 2 retries
        assert result.result is None

    def test_custom_adjustment_rules(self) -> None:
        """Test custom parameter adjustment rules."""

        def needs_larger_window(trace: WaveformTrace, window_size: int = 10) -> dict:
            if window_size < 50:
                raise ValueError("Window too small")
            return {"window_size": window_size}

        data = np.random.randn(1000)
        trace = make_waveform_trace(data)
        params = {"window_size": 10}

        # Custom rule: double window size each retry
        rules = {"window_size": lambda v, n: v * 2}

        result = retry_with_adjustment(
            needs_larger_window, trace, params, max_retries=5, adjustment_rules=rules
        )

        assert result.success is True
        assert result.final_parameters["window_size"] >= 50

    def test_parameter_adjustment_tracking(self) -> None:
        """Test that parameter adjustments are tracked."""

        def always_fail(
            trace: WaveformTrace, threshold: float = 1.0, tolerance: float = 0.1
        ) -> dict:
            raise ValueError("Fail")

        data = np.random.randn(100)
        trace = make_waveform_trace(data)
        params = {"threshold": 1.0, "tolerance": 0.1}

        result = retry_with_adjustment(always_fail, trace, params, max_retries=2)

        # Should have adjustments for both parameters
        assert len(result.adjustments_made) > 0
        assert any("threshold" in adj for adj in result.adjustments_made)

    def test_default_adjustment_rules(self) -> None:
        """Test that default adjustment rules work."""

        def check_threshold(trace: WaveformTrace, threshold: float = 1.0) -> dict:
            # Succeeds when threshold is reduced below 0.7
            if threshold > 0.7:
                raise ValueError("Threshold too high")
            return {"threshold": threshold}

        data = np.random.randn(100)
        trace = make_waveform_trace(data)
        params = {"threshold": 1.0}

        # Use default rules (threshold reduced by 0.9^n)
        result = retry_with_adjustment(check_threshold, trace, params, max_retries=5)

        assert result.success is True
        assert result.final_parameters["threshold"] < 0.7
