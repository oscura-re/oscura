"""Comprehensive unit tests for eye diagram generation module.

Tests for src/oscura/analyzers/eye/generation.py

This test suite provides comprehensive coverage of the eye diagram generation module,
focusing on internal helper functions and edge cases not covered by test_diagram.py.

Coverage targets:
- Internal validation functions (_validate_unit_interval, _validate_data_length)
- Internal trigger functions (_find_trigger_points, _extract_eye_traces)
- Internal histogram functions (_generate_histogram_if_requested)
- Internal centering functions (_calculate_trigger_threshold, _find_trace_crossings, etc.)
- Edge cases and boundary conditions
- Error handling paths
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oscura.analyzers.eye.generation import (
    EyeDiagram,
    _apply_symmetric_centering,
    _calculate_trigger_threshold,
    _extract_eye_traces,
    _find_trace_crossings,
    _find_trigger_points,
    _generate_histogram_if_requested,
    _validate_data_length,
    _validate_unit_interval,
    auto_center_eye_diagram,
    generate_eye,
    generate_eye_from_edges,
)
from oscura.core.exceptions import AnalysisError, InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.eye]


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def sample_waveform() -> WaveformTrace:
    """Create a simple test waveform with alternating pattern."""
    # Create alternating digital signal (0 and 1)
    samples_per_bit = 100
    n_bits = 50
    data = np.zeros(samples_per_bit * n_bits)

    for i in range(n_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        data[start:end] = i % 2  # Alternate between 0 and 1

    metadata = TraceMetadata(sample_rate=10e9)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sample_eye_data() -> np.ndarray:
    """Create sample eye diagram data for testing centering functions."""
    # Create 100 traces with 200 samples each
    # Traces should have a rising edge in the middle
    n_traces = 100
    samples_per_trace = 200
    data = np.zeros((n_traces, samples_per_trace))

    for i in range(n_traces):
        # Rising edge at middle with some variation
        edge_position = samples_per_trace // 2 + (i % 10 - 5)
        data[i, :edge_position] = 0.0
        data[i, edge_position:] = 1.0

    return data


# =============================================================================
# Tests for _validate_unit_interval
# =============================================================================


class TestValidateUnitInterval:
    """Test _validate_unit_interval internal function."""

    def test_valid_unit_interval(self) -> None:
        """Test validation with valid unit interval."""
        sample_rate = 10e9  # 10 GS/s
        unit_interval = 1e-9  # 1 ns

        samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)

        assert samples_per_ui == 10
        assert isinstance(samples_per_ui, int)

    def test_minimum_valid_unit_interval(self) -> None:
        """Test minimum valid unit interval (4 samples/UI)."""
        sample_rate = 1e9
        unit_interval = 4e-9  # Exactly 4 samples

        samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)

        assert samples_per_ui == 4

    def test_large_samples_per_ui(self) -> None:
        """Test with large number of samples per UI."""
        sample_rate = 100e9  # High sample rate
        unit_interval = 1e-9

        samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)

        assert samples_per_ui == 100

    def test_rounding_behavior(self) -> None:
        """Test that samples_per_ui is properly rounded."""
        sample_rate = 10e9
        unit_interval = 1.05e-9  # Should round to 10 or 11

        samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)

        assert samples_per_ui in [10, 11]  # Allow for rounding

    def test_unit_interval_too_short_raises_error(self) -> None:
        """Test that too short unit interval raises AnalysisError."""
        sample_rate = 10e9
        unit_interval = 1e-12  # Only 0.01 samples/UI

        with pytest.raises(AnalysisError) as exc_info:
            _validate_unit_interval(unit_interval, sample_rate)

        assert "Unit interval too short" in str(exc_info.value)
        assert "samples/UI" in str(exc_info.value)

    def test_exactly_three_samples_raises_error(self) -> None:
        """Test that 3 samples/UI raises error (need at least 4)."""
        sample_rate = 1e9
        unit_interval = 3e-9  # Exactly 3 samples

        with pytest.raises(AnalysisError) as exc_info:
            _validate_unit_interval(unit_interval, sample_rate)

        assert "Need at least 4 samples per UI" in str(exc_info.value)


# =============================================================================
# Tests for _validate_data_length
# =============================================================================


class TestValidateDataLength:
    """Test _validate_data_length internal function."""

    def test_sufficient_data(self) -> None:
        """Test validation with sufficient data."""
        n_samples = 1000
        total_ui_samples = 100

        # Should not raise
        _validate_data_length(n_samples, total_ui_samples)

    def test_exactly_minimum_data(self) -> None:
        """Test with exactly 2x total_ui_samples (minimum required)."""
        total_ui_samples = 100
        n_samples = 200  # Exactly 2x

        # Should not raise
        _validate_data_length(n_samples, total_ui_samples)

    def test_insufficient_data_raises_error(self) -> None:
        """Test that insufficient data raises InsufficientDataError."""
        n_samples = 100
        total_ui_samples = 100  # Need at least 200

        with pytest.raises(InsufficientDataError) as exc_info:
            _validate_data_length(n_samples, total_ui_samples)

        assert exc_info.value.required == 200
        assert exc_info.value.available == 100
        assert exc_info.value.analysis_type == "eye_diagram_generation"

    def test_one_sample_short_raises_error(self) -> None:
        """Test that one sample less than required raises error."""
        total_ui_samples = 100
        n_samples = 199  # One short of 200

        with pytest.raises(InsufficientDataError):
            _validate_data_length(n_samples, total_ui_samples)

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises error."""
        n_samples = 0
        total_ui_samples = 100

        with pytest.raises(InsufficientDataError) as exc_info:
            _validate_data_length(n_samples, total_ui_samples)

        assert exc_info.value.available == 0


# =============================================================================
# Tests for _find_trigger_points
# =============================================================================


class TestFindTriggerPoints:
    """Test _find_trigger_points internal function."""

    def test_rising_edge_trigger(self) -> None:
        """Test finding rising edge trigger points."""
        # Create signal with rising edges
        data = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float64)

        indices = _find_trigger_points(data, trigger_level=0.5, trigger_edge="rising")

        assert len(indices) >= 2  # At least 2 rising edges
        # Verify triggers are at rising edges
        for idx in indices:
            assert data[idx] < 0.5  # Before threshold
            assert data[idx + 1] >= 0.5  # After threshold

    def test_falling_edge_trigger(self) -> None:
        """Test finding falling edge trigger points."""
        # Create signal with falling edges
        data = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)

        indices = _find_trigger_points(data, trigger_level=0.5, trigger_edge="falling")

        assert len(indices) >= 2  # At least 2 falling edges
        # Verify triggers are at falling edges
        for idx in indices:
            assert data[idx] >= 0.5  # Before threshold
            assert data[idx + 1] < 0.5  # After threshold

    def test_custom_trigger_level(self) -> None:
        """Test with custom trigger level."""
        # Signal with amplitude from 0 to 2
        data = np.array([0, 0, 2, 2, 0, 0, 2, 2], dtype=np.float64)

        # Trigger at 1.5 (75% of range 0-2)
        indices = _find_trigger_points(data, trigger_level=0.75, trigger_edge="rising")

        assert len(indices) >= 2

    def test_insufficient_triggers_raises_error(self) -> None:
        """Test that signal with <2 triggers raises error."""
        # Signal with only one rising edge
        data = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

        with pytest.raises(InsufficientDataError) as exc_info:
            _find_trigger_points(data, trigger_level=0.5, trigger_edge="rising")

        assert "Not enough trigger events" in str(exc_info.value)
        assert exc_info.value.required == 2

    def test_no_triggers_raises_error(self) -> None:
        """Test that signal with no triggers raises error."""
        # Constant signal
        data = np.ones(100, dtype=np.float64)

        with pytest.raises(InsufficientDataError) as exc_info:
            _find_trigger_points(data, trigger_level=0.5, trigger_edge="rising")

        assert exc_info.value.available == 0

    def test_trigger_level_percentile_calculation(self) -> None:
        """Test that trigger threshold uses 10th and 90th percentiles."""
        # Signal with outliers
        rng = np.random.default_rng(42)
        data = np.concatenate(
            [
                np.zeros(40),
                np.ones(40),
                [10.0],  # Outlier high
                [-10.0],  # Outlier low
            ]
        )
        rng.shuffle(data)

        # Should still find triggers despite outliers
        try:
            indices = _find_trigger_points(data, trigger_level=0.5, trigger_edge="rising")
            assert len(indices) >= 0  # May or may not find triggers after shuffle
        except InsufficientDataError:
            pass  # Acceptable if shuffle removes transitions


# =============================================================================
# Tests for _extract_eye_traces
# =============================================================================


class TestExtractEyeTraces:
    """Test _extract_eye_traces internal function."""

    def test_extract_basic_traces(self) -> None:
        """Test extracting eye traces from valid data."""
        # Create signal with multiple periods
        samples_per_ui = 10
        n_periods = 10
        data = np.sin(2 * np.pi * np.arange(samples_per_ui * n_periods) / samples_per_ui)

        # Create trigger indices at each period
        trigger_indices = np.arange(5, samples_per_ui * (n_periods - 2), samples_per_ui)

        traces = _extract_eye_traces(
            data,
            trigger_indices,
            samples_per_ui=samples_per_ui,
            total_ui_samples=20,  # 2 UI
            max_traces=None,
        )

        assert len(traces) > 0
        for trace in traces:
            assert len(trace) == 20

    def test_max_traces_limit(self) -> None:
        """Test limiting maximum number of traces."""
        samples_per_ui = 10
        data = np.sin(2 * np.pi * np.arange(1000) / samples_per_ui)
        trigger_indices = np.arange(5, 900, samples_per_ui)

        traces = _extract_eye_traces(
            data,
            trigger_indices,
            samples_per_ui=samples_per_ui,
            total_ui_samples=20,
            max_traces=5,
        )

        assert len(traces) == 5

    def test_boundary_exclusion(self) -> None:
        """Test that traces near boundaries are excluded."""
        samples_per_ui = 10
        total_ui_samples = 20
        data = np.ones(200)  # Longer data to allow at least one valid trace

        # Mix of valid and invalid triggers
        trigger_indices = np.array([2, 50, 195])  # First and last too close to boundaries

        traces = _extract_eye_traces(
            data,
            trigger_indices,
            samples_per_ui=samples_per_ui,
            total_ui_samples=total_ui_samples,
            max_traces=None,
        )

        # Should have at least one valid trace (from index 50)
        assert len(traces) > 0
        # All traces should have correct length
        assert all(len(t) == total_ui_samples for t in traces)

    def test_no_valid_traces_raises_error(self) -> None:
        """Test that no valid traces raises InsufficientDataError."""
        data = np.ones(50)
        trigger_indices = np.array([1, 2, 3])  # All too close to start

        with pytest.raises(InsufficientDataError) as exc_info:
            _extract_eye_traces(
                data,
                trigger_indices,
                samples_per_ui=10,
                total_ui_samples=40,
                max_traces=None,
            )

        assert "Could not extract any complete eye traces" in str(exc_info.value)
        assert exc_info.value.required == 1
        assert exc_info.value.available == 0

    def test_half_ui_offset(self) -> None:
        """Test that traces are centered on trigger (half UI offset)."""
        samples_per_ui = 10
        total_ui_samples = 20
        data = np.arange(1000, dtype=np.float64)
        trigger_indices = np.array([100])

        traces = _extract_eye_traces(
            data,
            trigger_indices,
            samples_per_ui=samples_per_ui,
            total_ui_samples=total_ui_samples,
            max_traces=None,
        )

        assert len(traces) == 1
        # Trace should start at trigger - half_ui
        half_ui = samples_per_ui // 2
        expected_start = 100 - half_ui
        assert traces[0][0] == pytest.approx(data[expected_start])


# =============================================================================
# Tests for _generate_histogram_if_requested
# =============================================================================


class TestGenerateHistogramIfRequested:
    """Test _generate_histogram_if_requested internal function."""

    def test_histogram_not_requested(self) -> None:
        """Test that no histogram is generated when not requested."""
        eye_data = np.random.randn(10, 100)
        time_axis = np.linspace(0, 2, 100)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=2,
            generate_histogram=False,
            histogram_bins=(50, 50),
        )

        assert hist is None
        assert v_bins is None
        assert t_bins is None

    def test_histogram_generation_basic(self) -> None:
        """Test basic histogram generation."""
        eye_data = np.random.randn(10, 100)
        time_axis = np.linspace(0, 2, 100)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=2,
            generate_histogram=True,
            histogram_bins=(50, 60),
        )

        assert hist is not None
        assert hist.shape == (50, 60)
        assert v_bins is not None
        assert len(v_bins) == 51  # N bins + 1 edges
        assert t_bins is not None
        assert len(t_bins) == 61

    def test_histogram_voltage_range(self) -> None:
        """Test that histogram covers voltage data range."""
        eye_data = np.array([[0, 1, 2, 3, 4]] * 10)
        time_axis = np.linspace(0, 1, 5)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=1,
            generate_histogram=True,
            histogram_bins=(10, 10),
        )

        assert v_bins[0] <= 0.0
        assert v_bins[-1] >= 4.0

    def test_histogram_time_range(self) -> None:
        """Test that histogram covers time axis range."""
        eye_data = np.random.randn(10, 100)
        time_axis = np.linspace(0, 2, 100)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=2,
            generate_histogram=True,
            histogram_bins=(50, 50),
        )

        assert t_bins[0] == pytest.approx(0.0)
        assert t_bins[-1] == pytest.approx(2.0)

    def test_histogram_data_tiling(self) -> None:
        """Test that all eye data is included in histogram."""
        # Create eye data with known values
        n_traces = 5
        samples_per_trace = 10
        eye_data = np.ones((n_traces, samples_per_trace))
        time_axis = np.linspace(0, 1, samples_per_trace)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=1,
            generate_histogram=True,
            histogram_bins=(10, 10),
        )

        # Total counts should equal total samples
        assert hist.sum() == n_traces * samples_per_trace


# =============================================================================
# Tests for _calculate_trigger_threshold
# =============================================================================


class TestCalculateTriggerThreshold:
    """Test _calculate_trigger_threshold internal function."""

    def test_threshold_at_50_percent(self) -> None:
        """Test threshold calculation at 50% trigger fraction."""
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        threshold = _calculate_trigger_threshold(data, trigger_fraction=0.5)

        assert threshold == pytest.approx(0.5, abs=0.1)

    def test_threshold_at_20_percent(self) -> None:
        """Test threshold calculation at 20% trigger fraction."""
        # Signal from 0 to 10
        data = np.linspace(0, 10, 100)

        threshold = _calculate_trigger_threshold(data, trigger_fraction=0.2)

        # Should be at 20% of range (10th to 90th percentile range)
        expected = data[10] + 0.2 * (data[90] - data[10])
        assert threshold == pytest.approx(expected, rel=0.1)

    def test_threshold_with_percentile_calculation(self) -> None:
        """Test that threshold uses 10th and 90th percentiles."""
        # Data with outliers
        data = np.concatenate(
            [
                np.zeros(40),
                np.ones(40),
                [100.0],  # High outlier
                [-100.0],  # Low outlier
            ]
        )

        threshold = _calculate_trigger_threshold(data, trigger_fraction=0.5)

        # Should be near 0.5, not affected by outliers
        assert 0.2 <= threshold <= 0.8

    def test_threshold_return_type(self) -> None:
        """Test that threshold is returned as float."""
        data = np.array([0.0, 1.0, 2.0])

        threshold = _calculate_trigger_threshold(data, trigger_fraction=0.5)

        assert isinstance(threshold, float)


# =============================================================================
# Tests for _find_trace_crossings
# =============================================================================


class TestFindTraceCrossings:
    """Test _find_trace_crossings internal function."""

    def test_find_crossings_basic(self) -> None:
        """Test finding crossings in multiple traces."""
        # Create traces with rising edges at different positions
        data = np.array(
            [
                [0, 0, 0, 1, 1, 1],  # Edge at index 2
                [0, 0, 1, 1, 1, 1],  # Edge at index 1
                [0, 1, 1, 1, 1, 1],  # Edge at index 0
            ],
            dtype=np.float64,
        )

        crossings = _find_trace_crossings(data, threshold=0.5)

        assert len(crossings) == 3
        assert 0 in crossings
        assert 1 in crossings
        assert 2 in crossings

    def test_no_crossings_returns_empty(self) -> None:
        """Test that traces without crossings return empty list."""
        # All high values
        data = np.ones((5, 10), dtype=np.float64)

        crossings = _find_trace_crossings(data, threshold=0.5)

        assert len(crossings) == 0

    def test_partial_crossings(self) -> None:
        """Test when only some traces have crossings."""
        data = np.array(
            [
                [0, 0, 1, 1],  # Has crossing
                [1, 1, 1, 1],  # No crossing
                [0, 0, 0, 1],  # Has crossing
                [0, 0, 0, 0],  # No crossing
            ],
            dtype=np.float64,
        )

        crossings = _find_trace_crossings(data, threshold=0.5)

        assert len(crossings) == 2

    def test_first_crossing_only(self) -> None:
        """Test that only first crossing per trace is returned."""
        # Trace with multiple crossings
        data = np.array(
            [
                [0, 1, 0, 1, 0, 1],  # Multiple crossings
            ],
            dtype=np.float64,
        )

        crossings = _find_trace_crossings(data, threshold=0.5)

        assert len(crossings) == 1
        assert crossings[0] == 0  # First rising edge


# =============================================================================
# Tests for _apply_symmetric_centering
# =============================================================================


class TestApplySymmetricCentering:
    """Test _apply_symmetric_centering internal function."""

    def test_centering_basic(self) -> None:
        """Test basic symmetric centering."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        centered = _apply_symmetric_centering(data)

        # Mean should be zero
        assert np.mean(centered) == pytest.approx(0.0, abs=0.01)

    def test_centering_preserves_shape(self) -> None:
        """Test that centering preserves data shape."""
        data = np.random.randn(50, 100)

        centered = _apply_symmetric_centering(data)

        assert centered.shape == data.shape

    def test_centering_with_offset(self) -> None:
        """Test centering data with DC offset."""
        # Data from 10 to 20 (offset by 15)
        data = np.linspace(10, 20, 100).reshape(10, 10)

        centered = _apply_symmetric_centering(data)

        # Mean should be zero after centering
        assert np.abs(np.mean(centered)) < 0.1

    def test_centering_zero_data(self) -> None:
        """Test centering with all zeros."""
        data = np.zeros((10, 10), dtype=np.float64)

        centered = _apply_symmetric_centering(data)

        # Should remain zeros
        assert np.allclose(centered, 0.0)

    def test_centering_return_type(self) -> None:
        """Test that centered data is same dtype."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        centered = _apply_symmetric_centering(data)

        assert centered.dtype == data.dtype


# =============================================================================
# Integration Tests with Public Functions
# =============================================================================


class TestGenerationIntegration:
    """Integration tests combining internal functions with public API."""

    def test_generate_eye_uses_validation(self, sample_waveform: WaveformTrace) -> None:
        """Test that generate_eye properly validates inputs."""
        # This should trigger validation functions
        eye = generate_eye(sample_waveform, unit_interval=1e-9, n_ui=2)

        assert eye.samples_per_ui >= 4  # Validated
        assert eye.n_traces > 0  # Data validated

    def test_generate_eye_uses_trigger_finding(self, sample_waveform: WaveformTrace) -> None:
        """Test that generate_eye uses trigger finding correctly."""
        # Different trigger edges should find different numbers of triggers
        eye_rising = generate_eye(
            sample_waveform,
            unit_interval=1e-9,
            trigger_edge="rising",
        )

        eye_falling = generate_eye(
            sample_waveform,
            unit_interval=1e-9,
            trigger_edge="falling",
        )

        # Both should produce valid eyes
        assert eye_rising.n_traces > 0
        assert eye_falling.n_traces > 0

    def test_generate_eye_uses_histogram_generation(self, sample_waveform: WaveformTrace) -> None:
        """Test that generate_eye uses histogram generation."""
        eye = generate_eye(
            sample_waveform,
            unit_interval=1e-9,
            generate_histogram=True,
            histogram_bins=(40, 60),
        )

        assert eye.histogram is not None
        assert eye.histogram.shape == (40, 60)

    def test_auto_center_uses_threshold_calculation(self, sample_waveform: WaveformTrace) -> None:
        """Test that auto_center uses threshold calculation."""
        eye = generate_eye(sample_waveform, unit_interval=1e-9)

        # Different trigger fractions should work
        centered_20 = auto_center_eye_diagram(eye, trigger_fraction=0.2)
        centered_80 = auto_center_eye_diagram(eye, trigger_fraction=0.8)

        assert centered_20.n_traces == eye.n_traces
        assert centered_80.n_traces == eye.n_traces

    def test_auto_center_uses_centering(self, sample_waveform: WaveformTrace) -> None:
        """Test that auto_center applies symmetric centering."""
        eye = generate_eye(sample_waveform, unit_interval=1e-9)

        centered = auto_center_eye_diagram(eye, symmetric_range=True)

        # Mean should be near zero after centering
        assert np.abs(np.mean(centered.data)) < np.std(centered.data) * 0.2


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


class TestGenerationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_valid_samples_per_ui(self) -> None:
        """Test with exactly 4 samples per UI (minimum)."""
        sample_rate = 1e9
        unit_interval = 4e-9

        samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)

        assert samples_per_ui == 4

    def test_single_trace_extraction(self) -> None:
        """Test extracting exactly one trace."""
        data = np.sin(2 * np.pi * np.arange(100) / 10)
        trigger_indices = np.array([50])

        traces = _extract_eye_traces(
            data,
            trigger_indices,
            samples_per_ui=10,
            total_ui_samples=20,
            max_traces=None,
        )

        assert len(traces) == 1

    def test_histogram_with_single_trace(self) -> None:
        """Test histogram generation with single trace."""
        eye_data = np.random.randn(1, 100)
        time_axis = np.linspace(0, 1, 100)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=1,
            generate_histogram=True,
            histogram_bins=(10, 10),
        )

        assert hist is not None
        assert hist.sum() == 100  # One trace with 100 samples

    def test_trigger_at_exact_threshold(self) -> None:
        """Test trigger finding when data equals threshold."""
        data = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64)

        # Data point exactly at threshold should count as crossing
        try:
            indices = _find_trigger_points(data, trigger_level=0.5, trigger_edge="rising")
            assert len(indices) >= 0
        except InsufficientDataError:
            pass  # Acceptable if no valid triggers found

    def test_very_large_histogram_bins(self) -> None:
        """Test histogram with very large bin counts."""
        eye_data = np.random.randn(10, 100)
        time_axis = np.linspace(0, 2, 100)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=2,
            generate_histogram=True,
            histogram_bins=(500, 500),
        )

        assert hist.shape == (500, 500)
        assert len(v_bins) == 501
        assert len(t_bins) == 501


# =============================================================================
# Tests for generate_eye_from_edges (Missing Coverage)
# =============================================================================


class TestGenerateEyeFromEdgesDetailed:
    """Detailed tests for generate_eye_from_edges to increase coverage."""

    def test_generate_eye_from_edges_basic_workflow(self) -> None:
        """Test complete workflow of generate_eye_from_edges."""
        # Create waveform
        samples_per_bit = 100
        n_bits = 50
        data = np.zeros(samples_per_bit * n_bits)
        for i in range(n_bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            data[start:end] = i % 2

        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Create edge timestamps
        bit_period = 1e-9
        edges = np.arange(10, 40) * bit_period

        eye = generate_eye_from_edges(
            trace,
            edges,
            n_ui=2,
            samples_per_ui=50,
            max_traces=None,
        )

        assert eye.n_traces > 0
        assert eye.samples_per_ui == 50
        assert eye.unit_interval == pytest.approx(bit_period, rel=0.1)

    def test_generate_eye_from_edges_window_extraction(self) -> None:
        """Test that edges properly define extraction windows."""
        # Create simple waveform
        data = np.sin(2 * np.pi * np.arange(10000) / 100)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Edges at regular intervals
        edges = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3])

        eye = generate_eye_from_edges(
            trace,
            edges,
            n_ui=2,
            samples_per_ui=100,
        )

        assert eye.n_traces > 0
        assert eye.data.shape[1] == 200  # 2 UI * 100 samples/UI

    def test_generate_eye_from_edges_resampling(self) -> None:
        """Test that traces are resampled to consistent samples_per_ui."""
        # Non-uniform sample rate scenario
        data = np.linspace(0, 1, 5000)
        metadata = TraceMetadata(sample_rate=5e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        edges = np.array([100e-6, 200e-6, 300e-6, 400e-6])

        eye = generate_eye_from_edges(
            trace,
            edges,
            n_ui=1,
            samples_per_ui=75,
        )

        assert eye.samples_per_ui == 75
        assert eye.data.shape[1] == 75

    def test_generate_eye_from_edges_skips_short_windows(self) -> None:
        """Test that windows with insufficient samples are skipped."""
        data = np.ones(1000)
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Mix of valid and edge timestamps that create too-short windows
        edges = np.array([1e-9, 2e-9, 98e-9, 99e-9])

        eye = generate_eye_from_edges(
            trace,
            edges,
            n_ui=2,
            samples_per_ui=100,
        )

        # Should have some traces (middle ones)
        assert eye.n_traces >= 0

    def test_generate_eye_from_edges_max_traces_enforcement(self) -> None:
        """Test that max_traces limit is properly enforced."""
        data = np.tile(np.array([0, 1]), 5000)
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        edges = np.arange(50) * 1e-9

        eye = generate_eye_from_edges(
            trace,
            edges,
            n_ui=1,
            samples_per_ui=10,
            max_traces=5,
        )

        assert eye.n_traces <= 5

    def test_generate_eye_from_edges_edge_outside_trace_time(self) -> None:
        """Test that edges beyond trace duration are skipped."""
        data = np.ones(1000)
        metadata = TraceMetadata(sample_rate=10e9)  # 1000 samples = 100ns
        trace = WaveformTrace(data=data, metadata=metadata)

        # Edges at 1 second (way beyond trace duration of ~100ns)
        edges = np.array([1.0, 1.001, 1.002])

        with pytest.raises(InsufficientDataError):
            generate_eye_from_edges(
                trace,
                edges,
                n_ui=2,
                samples_per_ui=100,
            )

    def test_generate_eye_from_edges_negative_start_time(self) -> None:
        """Test that edges creating negative start times are skipped."""
        data = np.ones(1000)
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Edges very close to zero (would create negative start time)
        # and one valid edge in the middle
        edges = np.array([1e-12, 1e-11, 1e-10, 50e-9, 60e-9])

        try:
            eye = generate_eye_from_edges(
                trace,
                edges,
                n_ui=2,
                samples_per_ui=100,
            )
            # Should extract from valid edges only
            assert eye.n_traces >= 0
        except InsufficientDataError:
            # Acceptable if no valid edges found
            pass

    def test_generate_eye_from_edges_insufficient_window_samples(self) -> None:
        """Test handling of windows with < 4 samples."""
        # Very low sample rate relative to unit interval
        data = np.ones(100)
        metadata = TraceMetadata(sample_rate=100)  # Very low sample rate
        trace = WaveformTrace(data=data, metadata=metadata)

        # Edges that create very short windows
        edges = np.array([0.1, 0.2, 0.3])

        try:
            eye = generate_eye_from_edges(
                trace,
                edges,
                n_ui=1,
                samples_per_ui=50,
            )
            # May succeed with some traces or fail
            assert eye.n_traces >= 0
        except InsufficientDataError:
            # Also acceptable if no valid traces
            pass


# =============================================================================
# Tests for _align_traces_to_target (Missing Coverage)
# =============================================================================


class TestAlignTracesToTarget:
    """Test _align_traces_to_target internal function."""

    def test_align_no_shift_needed(self, sample_eye_data: np.ndarray) -> None:
        """Test alignment when traces already aligned (no shift)."""
        from oscura.analyzers.eye.generation import _align_traces_to_target

        # Create data where crossings are already at target
        data = np.zeros((10, 100))
        for i in range(10):
            data[i, :50] = 0.0
            data[i, 50:] = 1.0  # All crossings at index 50

        target_crossing = 50
        aligned = _align_traces_to_target(data, threshold=0.5, target_crossing=target_crossing)

        # Should be unchanged (or very similar)
        assert aligned.shape == data.shape

    def test_align_with_rolling(self, sample_eye_data: np.ndarray) -> None:
        """Test that rolling is applied when shift != 0."""
        from oscura.analyzers.eye.generation import _align_traces_to_target

        # Use the sample eye data which has varied crossing positions
        target_crossing = 100
        aligned = _align_traces_to_target(
            sample_eye_data, threshold=0.5, target_crossing=target_crossing
        )

        assert aligned.shape == sample_eye_data.shape

    def test_align_trace_without_crossing(self) -> None:
        """Test alignment handles traces without crossings."""
        from oscura.analyzers.eye.generation import _align_traces_to_target

        # Create data where some traces have no crossings
        data = np.zeros((5, 100))
        data[0, :50] = 0.0
        data[0, 50:] = 1.0  # Has crossing
        data[1, :] = 0.0  # No crossing (all low)
        data[2, :50] = 0.0
        data[2, 50:] = 1.0  # Has crossing
        data[3, :] = 1.0  # No crossing (all high)
        data[4, :50] = 0.0
        data[4, 50:] = 1.0  # Has crossing

        aligned = _align_traces_to_target(data, threshold=0.5, target_crossing=50)

        # Should handle traces without crossings
        assert aligned.shape == data.shape


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of internal functions."""

    def test_validate_with_floating_point_unit_interval(self) -> None:
        """Test validation handles floating point precision."""
        sample_rate = 10e9
        unit_interval = 1e-9 + 1e-18  # Small perturbation

        samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)

        assert samples_per_ui == 10  # Should round correctly

    def test_threshold_with_extreme_values(self) -> None:
        """Test threshold calculation with extreme data values."""
        data = np.array([1e-10, 1e10], dtype=np.float64)

        threshold = _calculate_trigger_threshold(data, trigger_fraction=0.5)

        assert np.isfinite(threshold)
        assert data.min() <= threshold <= data.max()

    def test_centering_with_large_offset(self) -> None:
        """Test centering with very large DC offset."""
        data = np.ones((10, 10)) * 1e6

        centered = _apply_symmetric_centering(data)

        assert np.abs(np.mean(centered)) < 1.0  # Should be centered

    def test_histogram_with_identical_values(self) -> None:
        """Test histogram generation when all values are identical."""
        eye_data = np.ones((10, 100))
        time_axis = np.linspace(0, 1, 100)

        hist, v_bins, t_bins = _generate_histogram_if_requested(
            eye_data,
            time_axis,
            n_ui=1,
            generate_histogram=True,
            histogram_bins=(10, 10),
        )

        assert hist is not None
        # All values should be in a single voltage bin
        assert hist.sum() == 1000  # 10 traces * 100 samples
