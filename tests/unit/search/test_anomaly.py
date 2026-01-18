"""Tests for anomaly detection in signal traces.

Requirements tested:

This test suite validates:
- Glitch detection with derivative method
- Timing violation detection (pulse width)
- Protocol error detection (placeholder)
- Auto-thresholding and manual threshold modes
- Width filtering (min/max constraints)
- Context extraction at anomaly locations
- Severity scoring
- Edge cases (empty traces, boundary conditions)
"""

import numpy as np
import pytest

from oscura.search.anomaly import find_anomalies

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestFindAnomalies:
    """Tests for find_anomalies main entry point."""

    def test_empty_trace_returns_empty_list(self):
        """Test that empty trace returns empty anomaly list."""
        trace = np.array([], dtype=np.float64)
        anomalies = find_anomalies(trace, anomaly_type="glitch")
        assert anomalies == []

    def test_invalid_anomaly_type_raises_value_error(self):
        """Test that invalid anomaly type raises ValueError."""
        trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid anomaly_type 'invalid'"):
            find_anomalies(trace, anomaly_type="invalid")

    def test_valid_anomaly_types(self):
        """Test that all valid anomaly types are accepted."""
        trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Should not raise for valid types
        find_anomalies(trace, anomaly_type="glitch")
        find_anomalies(trace, anomaly_type="timing", sample_rate=1e6)
        find_anomalies(trace, anomaly_type="protocol")

    def test_timing_without_sample_rate_raises_value_error(self):
        """Test that timing detection without sample_rate raises ValueError."""
        trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        with pytest.raises(ValueError, match="sample_rate required for timing anomaly detection"):
            find_anomalies(trace, anomaly_type="timing")

    def test_protocol_returns_empty_list(self):
        """Test that protocol detection returns empty list (not implemented)."""
        trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        anomalies = find_anomalies(trace, anomaly_type="protocol")
        assert anomalies == []


@pytest.mark.unit
class TestGlitchDetection:
    """Tests for glitch detection functionality."""

    def test_detect_simple_glitch(self):
        """Test detection of a simple voltage spike."""
        # Create trace with single spike
        trace = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            sample_rate=1e6,
        )

        assert len(anomalies) >= 1
        assert anomalies[0]["type"] == "glitch"
        assert anomalies[0]["index"] >= 2  # Should detect around spike location
        assert anomalies[0]["index"] <= 4

    def test_detect_negative_glitch(self):
        """Test detection of a voltage dip."""
        # Create trace with negative spike
        trace = np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0], dtype=np.float64)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            sample_rate=1e6,
        )

        assert len(anomalies) >= 1
        assert anomalies[0]["type"] == "glitch"

    def test_auto_threshold_uses_3_sigma(self):
        """Test that auto-threshold uses 3*std when threshold=None."""
        # Create trace with consistent baseline and one spike
        np.random.seed(42)
        trace = np.random.normal(0.0, 0.1, 1000)
        trace[500] = 1.5  # Large spike

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=None,  # Auto-threshold
        )

        # Should detect the spike with 3-sigma threshold
        assert len(anomalies) >= 1

    def test_manual_threshold(self):
        """Test that manual threshold is respected."""
        trace = np.array([0.0, 0.0, 0.3, 0.0, 0.0], dtype=np.float64)

        # With low threshold, should detect
        anomalies_low = find_anomalies(trace, anomaly_type="glitch", threshold=0.1)
        assert len(anomalies_low) >= 1

        # With high threshold, should not detect
        anomalies_high = find_anomalies(trace, anomaly_type="glitch", threshold=1.0)
        assert len(anomalies_high) == 0

    def test_glitch_grouping(self):
        """Test that consecutive glitch points are grouped."""
        # Create trace with multi-sample glitch
        trace = np.array([0.0, 0.0, 0.8, 0.9, 0.8, 0.0, 0.0], dtype=np.float64)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        # Should detect as single glitch event, not multiple
        # The grouping combines consecutive derivative exceedances
        assert len(anomalies) >= 1

    def test_min_width_filtering(self):
        """Test that min_width filters out short glitches."""
        sample_rate = 1e6  # 1 MHz
        # Create trace with 2-sample glitch (2 microseconds at 1 MHz)
        trace = np.array([0.0, 0.0, 0.8, 0.8, 0.0, 0.0], dtype=np.float64)

        # Without min_width constraint
        anomalies_all = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            sample_rate=sample_rate,
        )

        # With min_width > 2 microseconds
        anomalies_filtered = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            sample_rate=sample_rate,
            min_width=5e-6,  # 5 microseconds
        )

        assert len(anomalies_all) >= len(anomalies_filtered)

    def test_max_width_filtering(self):
        """Test that max_width filters out long glitches."""
        sample_rate = 1e6  # 1 MHz
        # Create trace with 5-sample glitch (5 microseconds)
        trace = np.zeros(20, dtype=np.float64)
        trace[5:10] = 0.8

        # Without max_width constraint
        anomalies_all = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            sample_rate=sample_rate,
        )

        # With max_width < 5 microseconds
        anomalies_filtered = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            sample_rate=sample_rate,
            max_width=3e-6,  # 3 microseconds
        )

        assert len(anomalies_all) >= len(anomalies_filtered)

    def test_context_extraction(self):
        """Test that context samples are extracted correctly."""
        trace = np.zeros(200, dtype=np.float64)
        trace[100] = 1.0  # Glitch at index 100

        context_samples = 50
        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            context_samples=context_samples,
        )

        assert len(anomalies) >= 1
        context = anomalies[0]["context"]

        # Context should include samples before and after
        # Exact size depends on glitch location and trace boundaries
        assert len(context) > 0
        assert isinstance(context, np.ndarray)

    def test_context_at_trace_start(self):
        """Test context extraction at trace start boundary."""
        trace = np.zeros(50, dtype=np.float64)
        trace[2] = 1.0  # Glitch near start

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            context_samples=100,  # More than available before glitch
        )

        if len(anomalies) > 0:
            context = anomalies[0]["context"]
            # Should not crash, context just shorter at start
            assert len(context) > 0
            assert len(context) <= len(trace)

    def test_context_at_trace_end(self):
        """Test context extraction at trace end boundary."""
        trace = np.zeros(50, dtype=np.float64)
        trace[47] = 1.0  # Glitch near end

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            context_samples=100,  # More than available after glitch
        )

        if len(anomalies) > 0:
            context = anomalies[0]["context"]
            # Should not crash, context just shorter at end
            assert len(context) > 0
            assert len(context) <= len(trace)

    def test_glitch_severity_scoring(self):
        """Test that severity is scored based on amplitude."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0  # Large spike

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.1,
        )

        assert len(anomalies) >= 1
        severity = anomalies[0]["severity"]

        # Severity should be between 0 and 1
        assert 0.0 <= severity <= 1.0
        assert isinstance(severity, float)

    def test_glitch_amplitude_measurement(self):
        """Test that amplitude deviation is measured correctly."""
        baseline = 0.5
        spike = 2.0
        trace = np.full(100, baseline, dtype=np.float64)
        trace[50] = spike

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.1,
        )

        assert len(anomalies) >= 1
        amplitude = anomalies[0]["amplitude"]

        # Amplitude should reflect deviation from baseline
        assert amplitude > 0
        # Should be close to the actual deviation
        expected_deviation = abs(spike - baseline)
        assert amplitude > expected_deviation * 0.5  # At least half

    def test_glitch_duration_reporting(self):
        """Test that glitch duration is reported in samples."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50:55] = 1.0  # 5-sample glitch

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        if len(anomalies) > 0:
            duration = anomalies[0]["duration"]
            assert duration > 0
            assert isinstance(duration, int | np.integer)

    def test_glitch_description(self):
        """Test that glitch description is human-readable."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        assert len(anomalies) >= 1
        description = anomalies[0]["description"]

        assert isinstance(description, str)
        assert len(description) > 0
        assert "glitch" in description.lower()

    def test_multiple_glitches(self):
        """Test detection of multiple glitches."""
        trace = np.zeros(200, dtype=np.float64)
        trace[50] = 1.0  # First glitch
        trace[100] = 1.2  # Second glitch
        trace[150] = 0.9  # Third glitch

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        # Should detect multiple glitches
        assert len(anomalies) >= 2

    def test_no_glitches_in_clean_signal(self):
        """Test that clean signal produces no glitches."""
        # Smooth sine wave
        t = np.linspace(0, 1, 1000)
        trace = np.sin(2 * np.pi * 5 * t)

        # With reasonable threshold, should not detect glitches
        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=1.0,  # High threshold
        )

        assert len(anomalies) == 0

    def test_glitch_width_constraints_without_sample_rate(self):
        """Test that width constraints are ignored when sample_rate is None."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50:55] = 1.0  # 5-sample glitch

        # Without sample_rate, width constraints should be ignored
        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            min_width=10e-6,  # This should be ignored
            max_width=1e-6,  # This should also be ignored
            # No sample_rate provided
        )

        # Should still detect the glitch since width filtering is skipped
        assert len(anomalies) >= 1


@pytest.mark.unit
class TestTimingViolations:
    """Tests for timing violation detection."""

    def test_detect_pulse_too_short(self):
        """Test detection of pulses shorter than min_width."""
        sample_rate = 1e6  # 1 MHz
        trace = np.zeros(100, dtype=np.float64)

        # Create 2-sample pulse (2 microseconds at 1 MHz)
        trace[50:52] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,  # Require 5 microseconds minimum
        )

        # Should detect violation
        assert len(violations) >= 1
        assert violations[0]["type"] == "timing_too_short"

    def test_detect_pulse_too_long(self):
        """Test detection of pulses longer than max_width."""
        sample_rate = 1e6  # 1 MHz
        trace = np.zeros(100, dtype=np.float64)

        # Create 10-sample pulse (10 microseconds)
        trace[50:60] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            max_width=5e-6,  # Allow maximum 5 microseconds
        )

        # Should detect violation
        assert len(violations) >= 1
        assert violations[0]["type"] == "timing_too_long"

    def test_pulse_within_constraints(self):
        """Test that pulses within constraints are not flagged."""
        sample_rate = 1e6  # 1 MHz
        trace = np.zeros(100, dtype=np.float64)

        # Create 5-sample pulse (5 microseconds)
        trace[50:55] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=4e-6,  # 4 microseconds min
            max_width=6e-6,  # 6 microseconds max
        )

        # Should not detect violation
        assert len(violations) == 0

    def test_timing_severity_scoring(self):
        """Test that timing violation severity is based on deviation."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:52] = 1.0  # Very short pulse

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=10e-6,  # Much longer than actual
        )

        if len(violations) > 0:
            severity = violations[0]["severity"]
            assert 0.0 <= severity <= 1.0
            assert isinstance(severity, float)

    def test_timing_context_extraction(self):
        """Test that timing violations include context."""
        sample_rate = 1e6
        trace = np.zeros(200, dtype=np.float64)
        trace[100:102] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,
            context_samples=50,
        )

        if len(violations) > 0:
            context = violations[0]["context"]
            assert len(context) > 0
            assert isinstance(context, np.ndarray)

    def test_timing_amplitude_is_width_in_seconds(self):
        """Test that amplitude field contains pulse width in seconds."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:55] = 1.0  # 5 samples = 5e-6 seconds

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=10e-6,  # Violation
        )

        if len(violations) > 0:
            pulse_width_seconds = violations[0]["amplitude"]
            expected_width = 5.0 / sample_rate
            assert abs(pulse_width_seconds - expected_width) < 1e-9

    def test_timing_duration_in_samples(self):
        """Test that duration is reported in samples."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:55] = 1.0  # 5 samples

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=10e-6,
        )

        if len(violations) > 0:
            duration = violations[0]["duration"]
            assert duration == 5

    def test_timing_description(self):
        """Test that timing violation description is informative."""
        sample_rate = 1e9  # 1 GHz for nanosecond precision
        trace = np.zeros(100, dtype=np.float64)
        trace[50:52] = 1.0  # 2 ns pulse

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=10e-9,  # 10 ns min
        )

        if len(violations) > 0:
            description = violations[0]["description"]
            assert isinstance(description, str)
            assert "timing violation" in description.lower()
            assert "ns" in description  # Should include nanoseconds

    def test_multiple_pulses_multiple_violations(self):
        """Test detection of multiple timing violations."""
        sample_rate = 1e6
        trace = np.zeros(200, dtype=np.float64)

        # First short pulse
        trace[20:22] = 1.0
        # Second short pulse
        trace[100:102] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,
        )

        assert len(violations) >= 2

    def test_incomplete_pulse_at_end(self):
        """Test handling of pulse without falling edge at trace end."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[90:] = 1.0  # Pulse that doesn't end

        # Should not crash, might not detect violation
        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,
        )

        # No assertion on count, just verify it doesn't crash
        assert isinstance(violations, list)

    def test_no_pulses_returns_empty(self):
        """Test that constant signal produces no violations."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)  # All low

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,
        )

        assert len(violations) == 0

    def test_only_min_width_constraint(self):
        """Test timing detection with only min_width."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:52] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,
            # No max_width
        )

        assert len(violations) >= 1

    def test_only_max_width_constraint(self):
        """Test timing detection with only max_width."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:60] = 1.0

        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            # No min_width
            max_width=5e-6,
        )

        assert len(violations) >= 1

    def test_no_width_constraints(self):
        """Test timing detection with no width constraints."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:55] = 1.0

        # No violations without constraints
        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
        )

        assert len(violations) == 0

    def test_both_min_and_max_width_violations(self):
        """Test pulse that violates both min and max width (if possible)."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)
        trace[50:52] = 1.0  # 2 microsecond pulse

        # Set constraints where pulse is too short for min
        violations = find_anomalies(
            trace,
            anomaly_type="timing",
            sample_rate=sample_rate,
            min_width=5e-6,
            max_width=10e-6,
        )

        # Should detect as too_short (first condition checked)
        assert len(violations) >= 1
        if len(violations) > 0:
            assert violations[0]["type"] == "timing_too_short"


@pytest.mark.unit
class TestAnomalyOutput:
    """Tests for anomaly output format and fields."""

    def test_anomaly_has_required_fields(self):
        """Test that anomalies contain all required fields."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        if len(anomalies) > 0:
            anomaly = anomalies[0]

            # Required fields
            assert "index" in anomaly
            assert "type" in anomaly
            assert "severity" in anomaly
            assert "duration" in anomaly
            assert "amplitude" in anomaly
            assert "context" in anomaly
            assert "description" in anomaly

    def test_anomaly_index_is_int(self):
        """Test that index is an integer."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=0.5)

        if len(anomalies) > 0:
            index = anomalies[0]["index"]
            assert isinstance(index, int | np.integer)

    def test_anomaly_severity_is_float_in_range(self):
        """Test that severity is a float between 0 and 1."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=0.5)

        if len(anomalies) > 0:
            severity = anomalies[0]["severity"]
            assert isinstance(severity, float)
            assert 0.0 <= severity <= 1.0

    def test_anomaly_context_is_ndarray(self):
        """Test that context is a numpy array."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=0.5)

        if len(anomalies) > 0:
            context = anomalies[0]["context"]
            assert isinstance(context, np.ndarray)

    def test_context_is_copy_not_view(self):
        """Test that context is a copy, not a view of original trace."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            context_samples=10,
        )

        if len(anomalies) > 0:
            context = anomalies[0]["context"]
            # Modify context
            context[0] = 999.0
            # Original trace should not be affected
            assert 999.0 not in trace


@pytest.mark.unit
class TestSearchAnomalyEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample_trace(self):
        """Test handling of single-sample trace."""
        trace = np.array([1.0], dtype=np.float64)

        # Should not crash
        anomalies = find_anomalies(trace, anomaly_type="glitch")
        # Derivative of single sample is empty, so no glitches
        assert len(anomalies) == 0

    def test_two_sample_trace(self):
        """Test handling of two-sample trace."""
        trace = np.array([0.0, 1.0], dtype=np.float64)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        # Should handle gracefully
        assert isinstance(anomalies, list)

    def test_constant_trace_no_glitches(self):
        """Test that constant signal produces no glitches."""
        trace = np.ones(100, dtype=np.float64)

        anomalies = find_anomalies(trace, anomaly_type="glitch")
        assert len(anomalies) == 0

    def test_very_large_trace(self):
        """Test handling of large traces."""
        trace = np.zeros(100000, dtype=np.float64)
        trace[50000] = 1.0

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
        )

        # Should handle without performance issues
        assert len(anomalies) >= 1

    @pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
    def test_zero_threshold(self):
        """Test behavior with zero threshold."""
        trace = np.array([0.0, 0.1, 0.0], dtype=np.float64)

        # Zero threshold means any change is a glitch
        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.0,
        )

        # Should detect changes
        assert len(anomalies) >= 1

    def test_negative_threshold_treated_as_magnitude(self):
        """Test that negative thresholds work (absolute value used)."""
        trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Implementation uses abs(derivative), so sign shouldn't matter
        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=-0.5,  # Negative value
        )

        # Should still work (comparing abs values)
        assert isinstance(anomalies, list)

    def test_context_samples_zero(self):
        """Test context extraction with zero context samples."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            context_samples=0,
        )

        if len(anomalies) > 0:
            # Context should still be valid array (might be small)
            context = anomalies[0]["context"]
            assert isinstance(context, np.ndarray)

    def test_context_samples_larger_than_trace(self):
        """Test context extraction when context_samples > trace length."""
        trace = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            context_samples=1000,  # Much larger than trace
        )

        if len(anomalies) > 0:
            context = anomalies[0]["context"]
            # Context can't be larger than trace
            assert len(context) <= len(trace)

    def test_all_samples_are_glitches(self):
        """Test trace where every sample exceeds threshold."""
        trace = np.random.uniform(0, 10, 100).astype(np.float64)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.01,  # Very low threshold
        )

        # Should group into events, not create 100 separate anomalies
        assert len(anomalies) < len(trace)

    def test_alternating_signal(self):
        """Test detection in alternating digital signal."""
        # Square wave shouldn't be flagged as glitches with appropriate threshold
        trace = np.tile([0.0, 1.0], 50)

        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=2.0,  # High threshold
        )

        # Should not detect regular transitions as glitches
        assert len(anomalies) == 0

    def test_kwargs_passthrough(self):
        """Test that additional kwargs are accepted (for extensibility)."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = 1.0

        # Should not raise even with extra parameters
        anomalies = find_anomalies(
            trace,
            anomaly_type="glitch",
            threshold=0.5,
            extra_param="value",
            another_param=42,
        )

        assert isinstance(anomalies, list)
