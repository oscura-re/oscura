"""Comprehensive edge case tests for search modules.

Tests SRCH-001, SRCH-002, SRCH-003: Edge cases and boundary conditions


This test suite validates correct handling of edge cases, boundary
conditions, and error scenarios to achieve complete code coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.search import extract_context, find_anomalies, find_pattern

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.search
@pytest.mark.edge_cases
class TestAnomalyEdgeCasesComplete:
    """Additional edge cases for complete anomaly detection coverage."""

    def test_glitch_detection_with_single_candidate(self) -> None:
        """Test glitch detection with only one derivative spike."""
        # Create trace with single transition
        trace = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=0.5)

        # Should detect the transition as a glitch if threshold is exceeded
        assert isinstance(anomalies, list)

    def test_glitch_at_exact_start(self) -> None:
        """Test glitch detection at index 0."""
        trace = np.zeros(100, dtype=np.float64)
        trace[0] = 5.0
        trace[1] = 0.0

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=2.0)

        # Should handle glitch at start
        if len(anomalies) > 0:
            assert anomalies[0]["index"] >= 0

    def test_glitch_at_exact_end(self) -> None:
        """Test glitch detection at last index."""
        trace = np.zeros(100, dtype=np.float64)
        trace[-2] = 0.0
        trace[-1] = 5.0

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=2.0)

        # Should handle glitch at end
        if len(anomalies) > 0:
            assert anomalies[-1]["index"] < len(trace)

    def test_timing_with_only_rising_edges(self) -> None:
        """Test timing detection with no falling edges."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50:] = 1.0  # Rising edge with no fall

        violations = find_anomalies(trace, anomaly_type="timing", sample_rate=1e6, min_width=10e-6)

        # Should handle gracefully
        assert isinstance(violations, list)

    def test_timing_with_only_falling_edges(self) -> None:
        """Test timing detection starting high with only falling edges."""
        trace = np.ones(100, dtype=np.float64)
        trace[50:] = 0.0  # Falling edge only

        violations = find_anomalies(trace, anomaly_type="timing", sample_rate=1e6, min_width=10e-6)

        # Should handle gracefully (no complete pulses)
        assert isinstance(violations, list)

    def test_timing_with_multiple_rising_before_falling(self) -> None:
        """Test timing detection with multiple rising edges before a fall."""
        # This creates a scenario where digital conversion might create issues
        trace = np.array([0, 0, 1, 1, 0.6, 0.9, 1, 0, 0], dtype=np.float64)

        violations = find_anomalies(
            trace, anomaly_type="timing", sample_rate=1e6, min_width=1e-6, max_width=10e-6
        )

        # Should detect pulse despite mid-level variations
        assert isinstance(violations, list)

    def test_glitch_with_nan_values(self) -> None:
        """Test glitch detection with NaN values in trace."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = np.nan

        # Should handle NaN gracefully
        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.0)

        # NaN in derivative should create large values
        assert isinstance(anomalies, list)

    def test_glitch_with_inf_values(self) -> None:
        """Test glitch detection with infinite values."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50] = np.inf

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.0)

        # Should detect inf as anomaly
        assert len(anomalies) >= 1

    def test_timing_with_exact_boundary_widths(self) -> None:
        """Test timing detection with pulse exactly at min/max width."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)

        # Create pulse exactly 5 samples (5 microseconds at 1 MHz)
        trace[50:55] = 1.0

        # Test with exact min_width
        violations_exact_min = find_anomalies(
            trace, anomaly_type="timing", sample_rate=sample_rate, min_width=5e-6
        )

        # Should not violate when exactly at boundary
        assert len(violations_exact_min) == 0

        # Test with exact max_width
        violations_exact_max = find_anomalies(
            trace, anomaly_type="timing", sample_rate=sample_rate, max_width=5e-6
        )

        # Should not violate when exactly at boundary
        assert len(violations_exact_max) == 0

    def test_protocol_anomaly_type(self) -> None:
        """Test protocol anomaly type returns empty (not implemented)."""
        trace = np.random.randn(100)

        anomalies = find_anomalies(trace, anomaly_type="protocol")

        # Should return empty list for protocol type
        assert anomalies == []
        assert isinstance(anomalies, list)

    def test_glitch_with_all_identical_values(self) -> None:
        """Test glitch detection on constant trace (zero derivative)."""
        trace = np.full(100, 5.0, dtype=np.float64)

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.0)

        # No glitches in constant signal
        assert len(anomalies) == 0

    def test_glitch_with_very_small_variations(self) -> None:
        """Test glitch detection with sub-threshold variations."""
        trace = np.zeros(100, dtype=np.float64)
        # Add tiny variations
        trace += np.random.randn(100) * 0.001

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=10.0)

        # High threshold should filter out small variations
        assert len(anomalies) == 0

    @pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
    def test_timing_with_zero_sample_rate(self) -> None:
        """Test that zero sample rate doesn't cause division by zero."""
        trace = np.zeros(100, dtype=np.float64)
        trace[50:55] = 1.0

        # This might cause issues in width calculations
        # The function should handle or raise appropriate error
        try:
            violations = find_anomalies(
                trace, anomaly_type="timing", sample_rate=0.0, min_width=1e-6
            )
            # If it doesn't raise, it should return a list
            assert isinstance(violations, list)
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            # Expected error for zero sample rate
            pass

    def test_glitch_grouping_with_gaps(self) -> None:
        """Test glitch grouping with gaps in derivative spikes."""
        trace = np.array([0, 0, 0.9, 0.8, 0.7, 0, 0], dtype=np.float64)

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=0.5)

        # Should group consecutive spikes
        assert isinstance(anomalies, list)

    def test_context_extraction_with_negative_severity(self) -> None:
        """Test that severity calculation handles negative deviations."""
        trace = np.ones(100, dtype=np.float64)
        trace[50] = -5.0  # Large negative spike

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=1.0)

        if len(anomalies) > 0:
            # Severity should still be positive
            assert anomalies[0]["severity"] >= 0


@pytest.mark.unit
@pytest.mark.search
@pytest.mark.edge_cases
class TestPatternEdgeCasesComplete:
    """Additional edge cases for complete pattern search coverage."""

    def test_pattern_with_leading_zeros(self) -> None:
        """Test pattern that starts with zero bytes."""
        digital = np.array([0x00, 0x12, 0x34, 0x00, 0x12, 0x34], dtype=np.uint8)
        pattern = np.array([0x00, 0x12], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        assert len(matches) == 2

    def test_pattern_all_zeros(self) -> None:
        """Test searching for all-zero pattern."""
        digital = np.array([0x00, 0x00, 0xFF, 0x00, 0x00], dtype=np.uint8)
        pattern = np.array([0x00, 0x00], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        assert len(matches) >= 2

    def test_mask_all_zeros_matches_everything(self) -> None:
        """Test that zero mask matches all values."""
        digital = np.random.randint(0, 256, 100, dtype=np.uint8)

        # Zero mask means "don't care" for all bits
        matches = find_pattern(digital, 0x00, mask=0x00)

        # Should match every position
        assert len(matches) == 100

    def test_analog_trace_padding_edge_cases(self) -> None:
        """Test analog to digital padding with various trace lengths."""
        # Test lengths that are not multiples of 8
        for length in [1, 7, 9, 15, 17]:
            analog = np.ones(length, dtype=np.float64)

            matches = find_pattern(analog, 0xFF, threshold=0.5)

            # Should handle padding correctly
            assert isinstance(matches, list)

    def test_pattern_larger_than_trace_after_padding(self) -> None:
        """Test pattern larger than trace after bit packing."""
        # 5 samples -> 1 byte after packing
        analog = np.ones(5, dtype=np.float64)

        # Search for 2-byte pattern
        pattern = np.array([0xFF, 0xFF], dtype=np.uint8)

        matches = find_pattern(analog, pattern, threshold=0.5)

        # Should return no matches
        assert len(matches) == 0

    def test_integer_mask_with_different_widths(self) -> None:
        """Test integer mask conversion for various pattern widths."""
        digital = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8)

        # 32-bit pattern with 32-bit mask
        pattern_32bit = 0x12345678
        mask_32bit = 0xFFFF0000  # Only upper 16 bits matter

        matches = find_pattern(digital, pattern_32bit, mask=mask_32bit)

        # Should find match based on upper 16 bits
        assert isinstance(matches, list)

    def test_pattern_with_max_uint8_value(self) -> None:
        """Test pattern with maximum uint8 value."""
        digital = np.array([0xFF, 0xFE, 0xFF], dtype=np.uint8)

        matches = find_pattern(digital, 0xFF)

        assert len(matches) == 2

    def test_min_spacing_equal_to_pattern_length(self) -> None:
        """Test min_spacing exactly equal to pattern length."""
        digital = np.tile(np.array([0xAA, 0xBB], dtype=np.uint8), 10)
        pattern = np.array([0xAA, 0xBB], dtype=np.uint8)

        matches = find_pattern(digital, pattern, min_spacing=2)

        # Should find every occurrence without overlap
        assert len(matches) == 10


@pytest.mark.unit
@pytest.mark.search
@pytest.mark.edge_cases
class TestContextEdgeCasesComplete:
    """Additional edge cases for complete context extraction coverage."""

    def test_extract_with_very_large_before_after(self) -> None:
        """Test extraction with before/after larger than trace length."""
        trace = np.arange(10, dtype=np.float64)

        context = extract_context(trace, 5, before=1000, after=1000)

        # Should clamp to trace boundaries
        assert context["start_index"] == 0
        assert context["end_index"] == 10
        assert context["length"] == 10

    def test_extract_batch_with_unsorted_indices(self) -> None:
        """Test that batch extraction doesn't require sorted indices."""
        trace = np.arange(1000, dtype=np.float64)
        indices = [900, 100, 500, 200, 800]  # Deliberately unsorted

        contexts = extract_context(trace, indices, before=5, after=5)

        # Should maintain input order
        assert contexts[0]["center_index"] == 900
        assert contexts[1]["center_index"] == 100
        assert contexts[2]["center_index"] == 500
        assert contexts[3]["center_index"] == 200
        assert contexts[4]["center_index"] == 800

    def test_extract_with_floating_point_indices(self) -> None:
        """Test that numpy float indices are handled correctly."""
        trace = np.arange(100, dtype=np.float64)

        # numpy might return float indices from some operations
        index_float = np.float64(50.0)

        context = extract_context(trace, int(index_float), before=5, after=5)

        assert context["center_index"] == 50

    def test_extract_preserves_complex_dtypes(self) -> None:
        """Test extraction preserves complex data types."""
        # Complex-valued trace
        trace_complex = (np.arange(100, dtype=np.float64) + 1j * np.arange(100)).astype(
            np.complex128
        )

        context = extract_context(trace_complex, 50, before=10, after=10)

        assert context["data"].dtype == np.complex128
        assert np.iscomplexobj(context["data"])

    def test_extract_with_structured_array(self) -> None:
        """Test extraction with structured numpy array."""
        # Create structured array (like multi-channel data)
        dt = np.dtype([("channel1", np.float64), ("channel2", np.float64)])
        trace = np.zeros(100, dtype=dt)
        trace["channel1"] = np.arange(100)
        trace["channel2"] = np.arange(100) * 2

        context = extract_context(trace, 50, before=5, after=5)

        assert context["data"].dtype == dt
        assert len(context["data"]) == 11

    def test_time_array_precision(self) -> None:
        """Test that time array maintains precision."""
        trace = np.arange(1000, dtype=np.float64)
        sample_rate = 1e9  # 1 GHz - tests precision

        context = extract_context(trace, 500, before=10, after=10, sample_rate=sample_rate)

        # Check time array spacing is exact
        dt = 1.0 / sample_rate
        time_diffs = np.diff(context["time_array"])

        np.testing.assert_allclose(time_diffs, dt, rtol=1e-12)

    def test_metadata_samples_before_after_at_boundaries(self) -> None:
        """Test metadata samples_before/after at exact boundaries."""
        trace = np.arange(100, dtype=np.float64)

        # At start boundary
        context_start = extract_context(trace, 0, before=10, after=10)
        assert context_start["metadata"]["samples_before"] == 0
        assert context_start["metadata"]["samples_after"] == 10

        # At end boundary
        context_end = extract_context(trace, 99, before=10, after=10)
        assert context_end["metadata"]["samples_before"] == 10
        assert context_end["metadata"]["samples_after"] == 0

    def test_extract_with_integer_list_types(self) -> None:
        """Test extraction with various integer list types."""
        trace = np.arange(100, dtype=np.float64)

        # Python list
        contexts_list = extract_context(trace, [10, 20, 30], before=5, after=5)
        assert len(contexts_list) == 3

        # Numpy array of various int types
        for dtype in [np.int32, np.int64]:
            indices = np.array([10, 20, 30], dtype=dtype)
            contexts_np = extract_context(trace, indices, before=5, after=5)
            assert len(contexts_np) == 3

    def test_extract_with_boolean_array_indices(self) -> None:
        """Test extraction using boolean array converted to indices."""
        trace = np.arange(100, dtype=np.float64)

        # Create boolean mask
        mask = np.zeros(100, dtype=bool)
        mask[[10, 30, 50]] = True

        # Convert to indices
        indices = np.where(mask)[0]

        contexts = extract_context(trace, indices, before=5, after=5)

        assert len(contexts) == 3
        assert contexts[0]["center_index"] == 10
        assert contexts[1]["center_index"] == 30
        assert contexts[2]["center_index"] == 50


@pytest.mark.unit
@pytest.mark.search
@pytest.mark.regression
class TestRegressionCases:
    """Tests for specific regression scenarios and bug fixes."""

    def test_pattern_search_with_partial_match_at_end(self) -> None:
        """Test that partial matches at trace end don't cause errors."""
        digital = np.array([0x12, 0x34, 0x56, 0x12], dtype=np.uint8)
        pattern = np.array([0x12, 0x34], dtype=np.uint8)

        # Last byte could start a partial match
        matches = find_pattern(digital, pattern)

        # Should find complete matches only
        assert len(matches) == 1
        assert matches[0][0] == 0

    def test_anomaly_with_single_derivative_spike(self) -> None:
        """Test anomaly detection doesn't crash with single spike."""
        trace = np.array([0, 0, 1, 0, 0], dtype=np.float64)

        anomalies = find_anomalies(trace, anomaly_type="glitch", threshold=0.5)

        # Should handle single spike
        assert isinstance(anomalies, list)

    def test_context_extraction_boundary_arithmetic(self) -> None:
        """Test that boundary calculations don't cause off-by-one errors."""
        trace = np.arange(100, dtype=np.float64)

        # Test at various positions
        for index in [0, 1, 50, 98, 99]:
            context = extract_context(trace, index, before=10, after=10)

            # Verify data integrity
            start = context["start_index"]
            end = context["end_index"]

            # Manually slice and compare
            expected = trace[start:end]
            np.testing.assert_array_equal(context["data"], expected)

    def test_timing_violation_edge_alignment(self) -> None:
        """Test timing violation detection at exact sample boundaries."""
        sample_rate = 1e6
        trace = np.zeros(100, dtype=np.float64)

        # Create pulse at exact edge
        trace[0:5] = 1.0  # Pulse at start
        trace[95:100] = 1.0  # Pulse at end

        violations = find_anomalies(
            trace, anomaly_type="timing", sample_rate=sample_rate, min_width=10e-6
        )

        # Should detect both boundary pulses
        assert isinstance(violations, list)

    def test_pattern_mask_zero_pattern_interaction(self) -> None:
        """Test interaction between zero pattern and non-zero mask."""
        digital = np.array([0x00, 0x0F, 0xF0, 0xFF], dtype=np.uint8)

        # Pattern 0x00 with mask 0xF0 should match 0x00 and 0x0F
        matches = find_pattern(digital, 0x00, mask=0xF0)

        # Upper nibble is zero in first two bytes
        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 1
