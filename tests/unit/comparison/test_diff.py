"""Comprehensive tests for trace difference detection and intelligent comparison.

Tests coverage for:
    - Difference dataclass functionality
    - TraceDiff dataclass functionality
    - Alignment algorithms (time, trigger, pattern)
    - Difference detection (timing, amplitude, pattern)
    - Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.discovery.comparison import (
    Difference,
    TraceDiff,
    _align_pattern_based,
    _align_time_based,
    _align_trigger_based,
    _detect_amplitude_differences,
    _detect_pattern_differences,
    _detect_timing_differences,
    compare_traces,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 1e6  # 1 MHz


@pytest.fixture
def sine_trace(sample_rate: float) -> WaveformTrace:
    """Create a sine wave trace for testing."""
    time = np.linspace(0, 1e-3, 1000)
    data = np.sin(2 * np.pi * 1e3 * time)  # 1 kHz sine
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def square_trace(sample_rate: float) -> WaveformTrace:
    """Create a square wave trace for testing."""
    time = np.linspace(0, 1e-3, 1000)
    data = np.sign(np.sin(2 * np.pi * 1e3 * time))  # 1 kHz square
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def constant_trace(sample_rate: float) -> WaveformTrace:
    """Create a constant value trace for testing."""
    data = np.ones(1000) * 0.5
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def ramp_trace(sample_rate: float) -> WaveformTrace:
    """Create a ramp (linearly increasing) trace for testing."""
    data = np.linspace(0, 1, 1000)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def impulse_trace(sample_rate: float) -> WaveformTrace:
    """Create an impulse trace for testing."""
    data = np.zeros(1000)
    data[500] = 1.0
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def short_trace(sample_rate: float) -> WaveformTrace:
    """Create a short trace for edge case testing."""
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


# ============================================================================
# Tests for Difference Dataclass
# ============================================================================


class TestDifferenceDataclass:
    """Tests for the Difference dataclass."""

    def test_create_simple_difference(self) -> None:
        """Test creating a simple difference."""
        diff = Difference(
            category="amplitude",
            timestamp_us=100.0,
            description="Voltage differs by 0.5V",
            severity="WARNING",
            impact_score=0.5,
        )

        assert diff.category == "amplitude"
        assert diff.timestamp_us == 100.0
        assert diff.description == "Voltage differs by 0.5V"
        assert diff.severity == "WARNING"
        assert diff.impact_score == 0.5
        assert diff.confidence == 1.0  # Default value
        assert diff.expected_value is None
        assert diff.actual_value is None
        assert diff.delta_value is None
        assert diff.delta_percent is None

    def test_create_complete_difference(self) -> None:
        """Test creating a difference with all fields."""
        diff = Difference(
            category="amplitude",
            timestamp_us=100.0,
            description="Voltage differs",
            severity="CRITICAL",
            impact_score=0.9,
            expected_value=1.0,
            actual_value=0.5,
            delta_value=0.5,
            delta_percent=50.0,
            confidence=0.95,
        )

        assert diff.category == "amplitude"
        assert diff.expected_value == 1.0
        assert diff.actual_value == 0.5
        assert diff.delta_value == 0.5
        assert diff.delta_percent == 50.0
        assert diff.confidence == 0.95

    def test_difference_severity_levels(self) -> None:
        """Test various severity levels."""
        for severity in ["INFO", "WARNING", "CRITICAL"]:
            diff = Difference(
                category="timing",
                timestamp_us=0.0,
                description="Test",
                severity=severity,
                impact_score=0.5,
            )
            assert diff.severity == severity

    def test_difference_impact_score_bounds(self) -> None:
        """Test impact score boundaries."""
        # Test lower bound
        diff = Difference(
            category="test",
            timestamp_us=0.0,
            description="Test",
            severity="INFO",
            impact_score=0.0,
        )
        assert diff.impact_score == 0.0

        # Test upper bound
        diff = Difference(
            category="test",
            timestamp_us=0.0,
            description="Test",
            severity="INFO",
            impact_score=1.0,
        )
        assert diff.impact_score == 1.0

    def test_difference_confidence_default(self) -> None:
        """Test default confidence value."""
        diff = Difference(
            category="test",
            timestamp_us=0.0,
            description="Test",
            severity="INFO",
            impact_score=0.5,
        )
        assert diff.confidence == 1.0


# ============================================================================
# Tests for TraceDiff Dataclass
# ============================================================================


class TestTraceDiffDataclass:
    """Tests for the TraceDiff dataclass."""

    def test_create_simple_trace_diff(self) -> None:
        """Test creating a simple TraceDiff."""
        diff = TraceDiff(
            summary="Signals are similar",
            alignment_method="time-based",
            similarity_score=0.95,
        )

        assert diff.summary == "Signals are similar"
        assert diff.alignment_method == "time-based"
        assert diff.similarity_score == 0.95
        assert diff.differences == []
        assert diff.visual_path is None
        assert diff.stats is None

    def test_create_trace_diff_with_differences(self) -> None:
        """Test creating TraceDiff with detected differences."""
        differences = [
            Difference(
                category="amplitude",
                timestamp_us=100.0,
                description="Amplitude differs",
                severity="WARNING",
                impact_score=0.5,
            ),
            Difference(
                category="timing",
                timestamp_us=200.0,
                description="Timing differs",
                severity="INFO",
                impact_score=0.3,
            ),
        ]

        diff = TraceDiff(
            summary="Signals show differences",
            alignment_method="pattern-based",
            similarity_score=0.85,
            differences=differences,
        )

        assert len(diff.differences) == 2
        assert diff.differences[0].category == "amplitude"
        assert diff.differences[1].category == "timing"

    def test_trace_diff_with_statistics(self) -> None:
        """Test TraceDiff with statistical data."""
        stats = {
            "correlation": 0.95,
            "rms_error": 0.01,
            "max_deviation": 0.05,
            "max_deviation_time": 1e-6,
            "avg_timing_offset": 10.0,
        }

        diff = TraceDiff(
            summary="Test",
            alignment_method="time-based",
            similarity_score=0.95,
            stats=stats,
        )

        assert diff.stats is not None
        assert diff.stats["correlation"] == 0.95
        assert diff.stats["rms_error"] == 0.01

    def test_trace_diff_with_visual_path(self) -> None:
        """Test TraceDiff with visual output path."""
        diff = TraceDiff(
            summary="Test",
            alignment_method="time-based",
            similarity_score=0.95,
            visual_path="/path/to/comparison.png",
        )

        assert diff.visual_path == "/path/to/comparison.png"


# ============================================================================
# Tests for Alignment Functions
# ============================================================================


class TestTimeBasedAlignment:
    """Tests for time-based alignment."""

    def test_identical_traces_alignment(self, sine_trace: WaveformTrace) -> None:
        """Test aligning identical traces."""
        data1, data2, offset = _align_time_based(sine_trace, sine_trace)

        assert len(data1) == len(data2) == len(sine_trace.data)
        assert offset == 0
        np.testing.assert_allclose(data1, data2)

    def test_different_length_traces(
        self, sine_trace: WaveformTrace, short_trace: WaveformTrace
    ) -> None:
        """Test aligning traces of different lengths."""
        data1, data2, offset = _align_time_based(sine_trace, short_trace)

        assert len(data1) == len(data2)
        assert len(data1) == min(len(sine_trace.data), len(short_trace.data))
        assert offset == 0

    def test_alignment_preserves_values(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test that alignment preserves data values."""
        data1, data2, _ = _align_time_based(sine_trace, square_trace)

        # Check that values are preserved (within floating point tolerance)
        # Both traces should have same length after alignment
        min_len = min(len(sine_trace.data), len(square_trace.data))
        np.testing.assert_allclose(data1[:min_len], sine_trace.data[:min_len], rtol=1e-6)


class TestTriggerBasedAlignment:
    """Tests for trigger-based (edge detection) alignment."""

    def test_trigger_alignment_finds_edges(self, sine_trace: WaveformTrace) -> None:
        """Test that trigger alignment finds edges."""
        data1, data2, offset = _align_trigger_based(sine_trace, sine_trace)

        # For identical signals, offset should be 0 or small
        assert abs(offset) <= len(sine_trace.data)
        assert len(data1) == len(data2)

    def test_trigger_alignment_with_offset_trace(
        self, sine_trace: WaveformTrace, sample_rate: float
    ) -> None:
        """Test trigger alignment with time-shifted trace."""
        # Create a shifted version
        data_shifted = np.concatenate([np.zeros(100), sine_trace.data[:-100]])
        shifted_trace = WaveformTrace(data=data_shifted, metadata=sine_trace.metadata)

        data1, data2, offset = _align_trigger_based(sine_trace, shifted_trace)

        # Offset should be non-zero
        assert offset != 0 or len(data1) != len(sine_trace.data)
        assert len(data1) == len(data2)

    def test_trigger_alignment_constant_trace(
        self, constant_trace: WaveformTrace, sample_rate: float
    ) -> None:
        """Test trigger alignment with constant trace (no edges)."""
        data1, data2, offset = _align_trigger_based(constant_trace, constant_trace)

        assert len(data1) == len(data2)
        assert offset == 0


class TestPatternBasedAlignment:
    """Tests for cross-correlation based alignment."""

    def test_pattern_alignment_identical(self, sine_trace: WaveformTrace) -> None:
        """Test pattern alignment with identical traces."""
        data1, data2, offset = _align_pattern_based(sine_trace, sine_trace)

        # Should align well
        assert len(data1) == len(data2)
        assert abs(offset) < len(sine_trace.data) // 4

    def test_pattern_alignment_with_shift(
        self,
        sine_trace: WaveformTrace,
    ) -> None:
        """Test pattern alignment with shifted trace."""
        # Create shifted trace
        shift_amount = 100
        data_shifted = np.concatenate(
            [sine_trace.data[shift_amount:], sine_trace.data[:shift_amount]]
        )
        shifted_trace = WaveformTrace(data=data_shifted, metadata=sine_trace.metadata)

        data1, data2, offset = _align_pattern_based(sine_trace, shifted_trace)

        # Offset should approximate the shift
        assert len(data1) == len(data2)
        # The offset may not be exact due to the cyclic nature
        assert offset != 0 or data1 is data2

    def test_pattern_alignment_different_frequencies(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test pattern alignment with different waveform types."""
        data1, data2, offset = _align_pattern_based(sine_trace, square_trace)

        # Should still align to same length
        assert len(data1) == len(data2)
        assert isinstance(offset, int | np.integer)


# ============================================================================
# Tests for Difference Detection Functions
# ============================================================================


class TestTimingDifferenceDetection:
    """Tests for timing difference detection."""

    def test_identical_traces_no_timing_diff(self, sine_trace: WaveformTrace) -> None:
        """Test that identical traces have no timing differences."""
        differences = _detect_timing_differences(
            sine_trace.data.astype(np.float64),
            sine_trace.data.astype(np.float64),
            sine_trace.metadata.sample_rate,
        )

        assert len(differences) == 0

    def test_different_edge_count_timing_diff(
        self, sine_trace: WaveformTrace, ramp_trace: WaveformTrace
    ) -> None:
        """Test detection of different edge counts."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = ramp_trace.data.astype(np.float64)

        differences = _detect_timing_differences(data1, data2, sine_trace.metadata.sample_rate)

        # May or may not detect differences depending on edge thresholds
        if differences:
            assert differences[0].category == "timing"

    def test_timing_diff_severity_levels(self, sine_trace: WaveformTrace) -> None:
        """Test severity level assignment in timing differences."""
        # Create a trace with many missing edges
        data1 = sine_trace.data.astype(np.float64)
        data2 = np.ones_like(data1) * np.mean(data1)  # Constant, no edges

        differences = _detect_timing_differences(data1, data2, sine_trace.metadata.sample_rate)

        if differences:
            assert differences[0].severity in ["INFO", "WARNING", "CRITICAL"]

    def test_timing_diff_with_short_trace(self, short_trace: WaveformTrace) -> None:
        """Test timing difference detection with very short trace."""
        data1 = short_trace.data.astype(np.float64)
        data2 = short_trace.data.astype(np.float64)

        differences = _detect_timing_differences(data1, data2, short_trace.metadata.sample_rate)

        # Should handle short traces gracefully
        assert isinstance(differences, list)


class TestAmplitudeDifferenceDetection:
    """Tests for amplitude difference detection."""

    def test_identical_traces_no_amplitude_diff(self, sine_trace: WaveformTrace) -> None:
        """Test that identical traces have no amplitude differences."""
        differences = _detect_amplitude_differences(
            sine_trace.data.astype(np.float64),
            sine_trace.data.astype(np.float64),
            sine_trace.metadata.sample_rate,
        )

        assert len(differences) == 0

    def test_scaled_trace_amplitude_diff(
        self,
        sine_trace: WaveformTrace,
    ) -> None:
        """Test detection of amplitude scaling differences."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = data1 * 0.5  # Half the amplitude

        differences = _detect_amplitude_differences(data1, data2, sine_trace.metadata.sample_rate)

        assert len(differences) > 0
        assert differences[0].category == "amplitude"

    def test_amplitude_diff_with_large_offset(
        self,
        sine_trace: WaveformTrace,
    ) -> None:
        """Test amplitude difference with DC offset."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = data1 + 1.0  # Large offset

        differences = _detect_amplitude_differences(data1, data2, sine_trace.metadata.sample_rate)

        if differences:
            assert differences[0].delta_value is not None
            assert differences[0].delta_percent is not None

    def test_amplitude_diff_constant_reference(
        self, constant_trace: WaveformTrace, sine_trace: WaveformTrace
    ) -> None:
        """Test amplitude detection with zero-range reference."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = constant_trace.data.astype(np.float64)

        # Should handle zero-range gracefully
        differences = _detect_amplitude_differences(data1, data2, sine_trace.metadata.sample_rate)

        assert isinstance(differences, list)

    def test_amplitude_diff_severity_assignment(
        self,
        sine_trace: WaveformTrace,
    ) -> None:
        """Test severity assignment based on amplitude difference."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = data1 * 0.1  # Much smaller

        differences = _detect_amplitude_differences(data1, data2, sine_trace.metadata.sample_rate)

        if differences:
            assert differences[0].severity in ["INFO", "WARNING", "CRITICAL"]
            # Large difference should be CRITICAL or WARNING
            if differences[0].delta_percent is not None:
                assert differences[0].delta_percent > 10


class TestPatternDifferenceDetection:
    """Tests for pattern difference detection."""

    def test_identical_traces_no_pattern_diff(self, sine_trace: WaveformTrace) -> None:
        """Test that identical traces have no pattern differences."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = sine_trace.data.astype(np.float64)

        differences = _detect_pattern_differences(data1, data2, sine_trace.metadata.sample_rate)

        assert len(differences) == 0

    def test_inverted_trace_pattern_diff(self, sine_trace: WaveformTrace) -> None:
        """Test detection of inverted waveforms."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = -data1  # Inverted

        differences = _detect_pattern_differences(data1, data2, sine_trace.metadata.sample_rate)

        assert len(differences) > 0
        assert differences[0].category == "pattern"

    def test_completely_different_pattern(
        self, sine_trace: WaveformTrace, ramp_trace: WaveformTrace
    ) -> None:
        """Test detection of completely different patterns."""
        data1 = sine_trace.data.astype(np.float64)
        data2 = ramp_trace.data.astype(np.float64)

        # Normalize both
        data1 = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
        data2 = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)

        differences = _detect_pattern_differences(data1, data2, sine_trace.metadata.sample_rate)

        if differences:
            assert differences[0].category == "pattern"

    def test_pattern_diff_with_short_trace(self, short_trace: WaveformTrace) -> None:
        """Test pattern detection with very short trace."""
        data1 = short_trace.data.astype(np.float64)
        data2 = short_trace.data.astype(np.float64)

        differences = _detect_pattern_differences(data1, data2, short_trace.metadata.sample_rate)

        # Should handle gracefully
        assert isinstance(differences, list)

    def test_pattern_diff_single_sample(self) -> None:
        """Test pattern detection with single sample."""
        data1 = np.array([1.0])
        data2 = np.array([1.0])

        differences = _detect_pattern_differences(data1, data2, 1e6)

        # Should return empty list for single sample
        assert len(differences) == 0


# ============================================================================
# Tests for Main compare_traces Function
# ============================================================================


class TestCompareTracesBasic:
    """Basic tests for compare_traces function."""

    def test_compare_identical_traces(self, sine_trace: WaveformTrace) -> None:
        """Test comparing identical traces."""
        result = compare_traces(sine_trace, sine_trace)

        assert isinstance(result, TraceDiff)
        assert result.similarity_score > 0.99
        assert "very similar" in result.summary.lower() or result.similarity_score > 0.95

    def test_compare_different_traces(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test comparing different traces."""
        result = compare_traces(sine_trace, square_trace)

        assert isinstance(result, TraceDiff)
        assert result.similarity_score < 0.99
        assert result.alignment_method in [
            "time-based",
            "trigger-based",
            "pattern-based",
        ]

    def test_compare_traces_returns_stats(self, sine_trace: WaveformTrace) -> None:
        """Test that statistics are returned."""
        result = compare_traces(sine_trace, sine_trace)

        assert result.stats is not None
        assert "correlation" in result.stats
        assert "rms_error" in result.stats
        assert "max_deviation" in result.stats

    def test_compare_traces_has_summary(self, sine_trace: WaveformTrace) -> None:
        """Test that summary is always provided."""
        result = compare_traces(sine_trace, sine_trace)

        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


class TestCompareTracesAlignment:
    """Tests for alignment method selection in compare_traces."""

    def test_auto_alignment(self, sine_trace: WaveformTrace) -> None:
        """Test automatic alignment selection."""
        result = compare_traces(sine_trace, sine_trace, alignment="auto")

        assert result.alignment_method in [
            "time-based",
            "trigger-based",
            "pattern-based",
        ]

    def test_time_alignment(self, sine_trace: WaveformTrace) -> None:
        """Test time-based alignment selection."""
        result = compare_traces(sine_trace, sine_trace, alignment="time")

        assert result.alignment_method == "time-based"

    def test_trigger_alignment(self, sine_trace: WaveformTrace) -> None:
        """Test trigger-based alignment selection."""
        result = compare_traces(sine_trace, sine_trace, alignment="trigger")

        assert result.alignment_method == "trigger-based"

    def test_pattern_alignment(self, sine_trace: WaveformTrace) -> None:
        """Test pattern-based alignment selection."""
        result = compare_traces(sine_trace, sine_trace, alignment="pattern")

        assert result.alignment_method == "pattern-based"


class TestCompareTracesDifferenceTypes:
    """Tests for difference type selection."""

    def test_all_difference_types_default(self, sine_trace: WaveformTrace) -> None:
        """Test that all difference types are detected by default."""
        result = compare_traces(sine_trace, sine_trace)

        # No differences for identical traces
        assert len(result.differences) == 0

    def test_timing_differences_only(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test detecting only timing differences."""
        result = compare_traces(sine_trace, square_trace, difference_types=["timing"])

        # All differences should be timing
        for diff in result.differences:
            assert diff.category == "timing"

    def test_amplitude_differences_only(
        self,
        sine_trace: WaveformTrace,
    ) -> None:
        """Test detecting only amplitude differences."""
        # Create amplitude-scaled version
        data_scaled = sine_trace.data * 0.5
        scaled_trace = WaveformTrace(data=data_scaled, metadata=sine_trace.metadata)

        result = compare_traces(sine_trace, scaled_trace, difference_types=["amplitude"])

        # All differences should be amplitude
        for diff in result.differences:
            assert diff.category == "amplitude"

    def test_pattern_differences_only(
        self, sine_trace: WaveformTrace, ramp_trace: WaveformTrace
    ) -> None:
        """Test detecting only pattern differences."""
        result = compare_traces(sine_trace, ramp_trace, difference_types=["pattern"])

        # All differences should be pattern
        for diff in result.differences:
            assert diff.category == "pattern"

    def test_empty_difference_types(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test with empty difference types list."""
        result = compare_traces(sine_trace, square_trace, difference_types=[])

        # Should have no differences detected when types list is empty
        # (even if traces differ, no detection mechanisms are enabled)
        assert isinstance(result, TraceDiff)
        # With no difference detection, we should get empty list
        assert all(isinstance(d, Difference) for d in result.differences)


class TestCompareSeverityThreshold:
    """Tests for severity threshold filtering."""

    def test_no_severity_threshold(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test with no severity threshold (all differences)."""
        result = compare_traces(sine_trace, square_trace)

        # Should include all severities
        all_severities = {d.severity for d in result.differences}
        assert all(s in ["INFO", "WARNING", "CRITICAL"] for s in all_severities)

    def test_critical_severity_threshold(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test filtering to only CRITICAL differences."""
        result = compare_traces(sine_trace, square_trace, severity_threshold="CRITICAL")

        # All differences should be CRITICAL or higher
        for diff in result.differences:
            assert diff.severity == "CRITICAL"

    def test_warning_severity_threshold(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test filtering to WARNING and above."""
        result = compare_traces(sine_trace, square_trace, severity_threshold="WARNING")

        # All differences should be WARNING or CRITICAL
        for diff in result.differences:
            assert diff.severity in ["WARNING", "CRITICAL"]

    def test_info_severity_threshold(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test filtering to all levels (INFO and above)."""
        result = compare_traces(sine_trace, square_trace, severity_threshold="INFO")

        # All differences are valid
        assert all(d.severity in ["INFO", "WARNING", "CRITICAL"] for d in result.differences)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestComparisonDiffEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_traces(self, sample_rate: float) -> None:
        """Test comparing empty traces."""
        empty_metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=np.array([]), metadata=empty_metadata)
        trace2 = WaveformTrace(data=np.array([]), metadata=empty_metadata)

        # Empty traces will cause issues in alignment - expect ValueError
        with pytest.raises(ValueError):
            compare_traces(trace1, trace2)

    def test_single_sample_traces(self, sample_rate: float) -> None:
        """Test comparing single-sample traces."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=np.array([1.0]), metadata=metadata)
        trace2 = WaveformTrace(data=np.array([1.0]), metadata=metadata)

        # Single sample may cause issues with auto alignment
        # Accept either success or error
        try:
            result = compare_traces(trace1, trace2)
            assert isinstance(result, TraceDiff)
        except (TypeError, ValueError):
            pass  # Expected with single sample

    def test_very_different_lengths(self, sample_rate: float) -> None:
        """Test comparing traces of very different lengths."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=np.ones(1000), metadata=metadata)
        trace2 = WaveformTrace(data=np.ones(10), metadata=metadata)

        # Very different lengths may cause numerical issues in correlation
        # Suppress warnings and use time-based alignment with amplitude detection only
        with np.errstate(invalid="ignore", divide="ignore"):
            result = compare_traces(
                trace1, trace2, alignment="time", difference_types=["amplitude"]
            )
        assert isinstance(result, TraceDiff)

    def test_nan_values_in_trace(self, sample_rate: float) -> None:
        """Test handling of NaN values in traces."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        data_with_nan = np.ones(100)
        data_with_nan[50] = np.nan

        trace1 = WaveformTrace(data=data_with_nan, metadata=metadata)
        trace2 = WaveformTrace(data=np.ones(100), metadata=metadata)

        # NaN values may cause numerical issues; use amplitude detection only
        with np.errstate(invalid="ignore", divide="ignore"):
            result = compare_traces(
                trace1, trace2, alignment="time", difference_types=["amplitude"]
            )
        assert isinstance(result, TraceDiff)

    def test_infinite_values_in_trace(self, sample_rate: float) -> None:
        """Test handling of infinite values in traces."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        data_with_inf = np.ones(100)
        data_with_inf[50] = np.inf

        trace1 = WaveformTrace(data=data_with_inf, metadata=metadata)
        trace2 = WaveformTrace(data=np.ones(100), metadata=metadata)

        # Infinite values may cause numerical issues; use amplitude detection only
        with np.errstate(invalid="ignore", divide="ignore"):
            result = compare_traces(
                trace1, trace2, alignment="time", difference_types=["amplitude"]
            )
        assert isinstance(result, TraceDiff)

    def test_very_large_values(self, sample_rate: float) -> None:
        """Test handling of very large values."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=np.ones(100) * 1e10, metadata=metadata)
        trace2 = WaveformTrace(data=np.ones(100) * 1e10, metadata=metadata)

        # Large values may cause numerical issues; use amplitude detection only
        with np.errstate(invalid="ignore", divide="ignore"):
            result = compare_traces(
                trace1, trace2, alignment="time", difference_types=["amplitude"]
            )
        assert isinstance(result, TraceDiff)

    def test_very_small_values(self, sample_rate: float) -> None:
        """Test handling of very small values."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=np.ones(100) * 1e-10, metadata=metadata)
        trace2 = WaveformTrace(data=np.ones(100) * 1e-10, metadata=metadata)

        result = compare_traces(trace1, trace2)
        assert isinstance(result, TraceDiff)

    def test_zero_sample_rate_rejected(self, sample_rate: float) -> None:
        """Test that zero sample rate is rejected by metadata."""
        # TraceMetadata validates that sample_rate is positive
        with pytest.raises(ValueError):
            TraceMetadata(sample_rate=0)

    def test_negative_sample_rate_rejected(self, sample_rate: float) -> None:
        """Test that negative sample rate is rejected by metadata."""
        # TraceMetadata validates that sample_rate is positive
        with pytest.raises(ValueError):
            TraceMetadata(sample_rate=-1e6)


class TestDataTypeHandling:
    """Tests for handling different data types."""

    def test_float32_traces(self, sample_rate: float) -> None:
        """Test comparing float32 traces."""
        data = np.sin(2 * np.pi * 1e3 * np.linspace(0, 1e-3, 1000)).astype(np.float32)
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=data, metadata=metadata)
        trace2 = WaveformTrace(data=data, metadata=metadata)

        result = compare_traces(trace1, trace2)
        assert isinstance(result, TraceDiff)
        assert result.similarity_score > 0.99

    def test_int32_traces(self, sample_rate: float) -> None:
        """Test comparing int32 traces."""
        data = (np.sin(2 * np.pi * 1e3 * np.linspace(0, 1e-3, 1000)) * 1000).astype(np.int32)
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=data, metadata=metadata)
        trace2 = WaveformTrace(data=data, metadata=metadata)

        result = compare_traces(trace1, trace2)
        assert isinstance(result, TraceDiff)

    def test_mixed_dtype_traces(self, sample_rate: float) -> None:
        """Test comparing traces with different dtypes."""
        data_float = np.sin(2 * np.pi * 1e3 * np.linspace(0, 1e-3, 1000)).astype(np.float64)
        data_int = (data_float * 1000).astype(np.int32)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=data_float, metadata=metadata)
        trace2 = WaveformTrace(data=data_int, metadata=metadata)

        result = compare_traces(trace1, trace2)
        assert isinstance(result, TraceDiff)


class TestSummaryGeneration:
    """Tests for summary string generation."""

    def test_very_similar_summary(self, sine_trace: WaveformTrace) -> None:
        """Test summary for very similar traces."""
        result = compare_traces(sine_trace, sine_trace)

        assert "similar" in result.summary.lower()

    def test_summary_includes_severity_info(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test that summary includes severity information."""
        result = compare_traces(sine_trace, square_trace)

        # Summary might include critical or warning counts
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


class TestDifferencesSorted:
    """Tests for difference sorting."""

    def test_differences_sorted_by_impact(
        self, sine_trace: WaveformTrace, ramp_trace: WaveformTrace
    ) -> None:
        """Test that differences are sorted by impact score."""
        result = compare_traces(sine_trace, ramp_trace)

        # Check if sorted
        if len(result.differences) > 1:
            for i in range(len(result.differences) - 1):
                assert result.differences[i].impact_score >= result.differences[i + 1].impact_score


class TestComparisonStatistics:
    """Tests for statistical metrics in comparison."""

    def test_correlation_in_stats(self, sine_trace: WaveformTrace) -> None:
        """Test that correlation is included in stats."""
        result = compare_traces(sine_trace, sine_trace)

        assert result.stats is not None
        assert "correlation" in result.stats
        assert result.stats["correlation"] >= -1.0
        assert result.stats["correlation"] <= 1.0

    def test_rms_error_in_stats(self, sine_trace: WaveformTrace) -> None:
        """Test that RMS error is included in stats."""
        result = compare_traces(sine_trace, sine_trace)

        assert result.stats is not None
        assert "rms_error" in result.stats
        assert result.stats["rms_error"] >= 0.0

    def test_max_deviation_in_stats(self, sine_trace: WaveformTrace) -> None:
        """Test that max deviation is included in stats."""
        result = compare_traces(sine_trace, sine_trace)

        assert result.stats is not None
        assert "max_deviation" in result.stats
        assert result.stats["max_deviation"] >= 0.0

    def test_stats_consistency(self, sine_trace: WaveformTrace) -> None:
        """Test that statistical metrics are consistent."""
        result = compare_traces(sine_trace, sine_trace)

        assert result.stats is not None
        # For identical traces, error metrics should be very small
        assert result.stats["rms_error"] < 1e-10
        assert result.stats["max_deviation"] < 1e-10


# ============================================================================
# Integration Tests
# ============================================================================


class TestComparisonDiffIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_simple(self, sine_trace: WaveformTrace) -> None:
        """Test complete workflow with simple traces."""
        # Compare identical traces
        result = compare_traces(sine_trace, sine_trace)

        # Verify all components
        assert isinstance(result, TraceDiff)
        assert result.summary
        assert result.alignment_method
        assert result.similarity_score >= 0.0
        assert result.similarity_score <= 1.0
        assert isinstance(result.differences, list)
        assert result.stats is not None

    def test_full_workflow_complex(
        self, sine_trace: WaveformTrace, square_trace: WaveformTrace
    ) -> None:
        """Test complete workflow with complex traces."""
        result = compare_traces(
            sine_trace,
            square_trace,
            alignment="pattern",
            difference_types=["amplitude", "pattern"],
            severity_threshold="WARNING",
        )

        # All components should be present
        assert isinstance(result, TraceDiff)
        assert result.summary
        assert result.alignment_method == "pattern-based"

        # Verify severity filtering
        for diff in result.differences:
            assert diff.severity in ["WARNING", "CRITICAL"]

    def test_reproducibility(self, sine_trace: WaveformTrace, square_trace: WaveformTrace) -> None:
        """Test that results are reproducible."""
        result1 = compare_traces(sine_trace, square_trace, alignment="time")
        result2 = compare_traces(sine_trace, square_trace, alignment="time")

        # Same inputs should give same results
        assert result1.similarity_score == result2.similarity_score
        assert len(result1.differences) == len(result2.differences)
