"""Unit tests for context extraction functionality.

Tests SRCH-003: Context Extraction
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.search.context import extract_context

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextBasic:
    """Test basic context extraction functionality."""

    def test_extract_single_index_center(self) -> None:
        """Test extracting context around a single index in the middle."""
        trace = np.arange(1000, dtype=np.float64)
        index = 500

        context = extract_context(trace, index, before=10, after=10)

        assert isinstance(context, dict)
        assert context["center_index"] == 500
        assert context["start_index"] == 490
        assert context["end_index"] == 511
        assert context["length"] == 21
        assert len(context["data"]) == 21
        np.testing.assert_array_equal(context["data"], trace[490:511])

    def test_extract_at_start_boundary(self) -> None:
        """Test context extraction at the start of trace."""
        trace = np.arange(100, dtype=np.float64)
        index = 5

        context = extract_context(trace, index, before=20, after=10)

        # Should start at 0, not at -15
        assert context["start_index"] == 0
        assert context["end_index"] == 16
        assert context["center_index"] == 5
        assert context["length"] == 16
        assert len(context["data"]) == 16
        np.testing.assert_array_equal(context["data"], trace[0:16])

    def test_extract_at_end_boundary(self) -> None:
        """Test context extraction at the end of trace."""
        trace = np.arange(100, dtype=np.float64)
        index = 95

        context = extract_context(trace, index, before=10, after=20)

        # Should end at 100, not at 116
        assert context["start_index"] == 85
        assert context["end_index"] == 100
        assert context["center_index"] == 95
        assert context["length"] == 15
        assert len(context["data"]) == 15
        np.testing.assert_array_equal(context["data"], trace[85:100])

    def test_extract_at_exact_start(self) -> None:
        """Test context extraction at index 0."""
        trace = np.arange(100, dtype=np.float64)
        index = 0

        context = extract_context(trace, index, before=10, after=10)

        assert context["start_index"] == 0
        assert context["end_index"] == 11
        assert context["center_index"] == 0
        assert context["length"] == 11

    def test_extract_at_exact_end(self) -> None:
        """Test context extraction at last index."""
        trace = np.arange(100, dtype=np.float64)
        index = 99

        context = extract_context(trace, index, before=10, after=10)

        assert context["start_index"] == 89
        assert context["end_index"] == 100
        assert context["center_index"] == 99
        assert context["length"] == 11

    def test_default_window_size(self) -> None:
        """Test default before and after parameters (100 each)."""
        trace = np.arange(1000, dtype=np.float64)
        index = 500

        context = extract_context(trace, index)

        # Default is 100 before and 100 after
        assert context["start_index"] == 400
        assert context["end_index"] == 601
        assert context["length"] == 201

    def test_asymmetric_window(self) -> None:
        """Test different before and after window sizes."""
        trace = np.arange(1000, dtype=np.float64)
        index = 500

        context = extract_context(trace, index, before=50, after=150)

        assert context["start_index"] == 450
        assert context["end_index"] == 651
        assert context["length"] == 201

    def test_data_is_copy_not_view(self) -> None:
        """Test that extracted data is a copy, not a view."""
        trace = np.arange(100, dtype=np.float64)
        context = extract_context(trace, 50, before=5, after=5)

        # Modify the context data
        context["data"][0] = 9999

        # Original trace should be unchanged
        assert trace[45] != 9999
        assert trace[45] == 45


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextBatch:
    """Test batch context extraction for multiple indices."""

    def test_extract_multiple_indices_list(self) -> None:
        """Test extracting contexts for a list of indices."""
        trace = np.arange(1000, dtype=np.float64)
        indices = [100, 200, 300]

        contexts = extract_context(trace, indices, before=10, after=10)

        assert isinstance(contexts, list)
        assert len(contexts) == 3

        for i, ctx in enumerate(contexts):
            expected_center = indices[i]
            assert ctx["center_index"] == expected_center
            assert ctx["start_index"] == expected_center - 10
            assert ctx["end_index"] == expected_center + 11
            assert ctx["length"] == 21

    def test_extract_multiple_indices_array(self) -> None:
        """Test extracting contexts for a numpy array of indices."""
        trace = np.arange(1000, dtype=np.float64)
        indices = np.array([250, 500, 750], dtype=np.int_)

        contexts = extract_context(trace, indices, before=5, after=5)

        assert isinstance(contexts, list)
        assert len(contexts) == 3

        for i, ctx in enumerate(contexts):
            expected_center = indices[i]
            assert ctx["center_index"] == expected_center
            assert ctx["length"] == 11

    def test_extract_single_element_list(self) -> None:
        """Test that single element list returns a list, not a dict."""
        trace = np.arange(100, dtype=np.float64)

        # Single element list should return list
        contexts = extract_context(trace, [50], before=5, after=5)

        assert isinstance(contexts, list)
        assert len(contexts) == 1
        assert contexts[0]["center_index"] == 50

    def test_extract_empty_list(self) -> None:
        """Test extracting with empty list of indices."""
        trace = np.arange(100, dtype=np.float64)

        contexts = extract_context(trace, [], before=5, after=5)

        assert isinstance(contexts, list)
        assert len(contexts) == 0

    def test_batch_with_boundary_indices(self) -> None:
        """Test batch extraction with indices at trace boundaries."""
        trace = np.arange(100, dtype=np.float64)
        indices = [0, 50, 99]

        contexts = extract_context(trace, indices, before=10, after=10)

        assert len(contexts) == 3

        # First index at start boundary
        assert contexts[0]["start_index"] == 0
        assert contexts[0]["metadata"]["at_start_boundary"] is True

        # Middle index no boundary issues
        assert contexts[1]["start_index"] == 40
        assert contexts[1]["end_index"] == 61

        # Last index at end boundary
        assert contexts[2]["end_index"] == 100
        assert contexts[2]["metadata"]["at_end_boundary"] is True


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextTimeReference:
    """Test time reference and sample rate functionality."""

    def test_time_reference_calculation(self) -> None:
        """Test time reference calculation with sample rate."""
        trace = np.arange(1000, dtype=np.float64)
        index = 500
        sample_rate = 1e6  # 1 MHz

        context = extract_context(trace, index, before=10, after=10, sample_rate=sample_rate)

        # Time reference should be start_index / sample_rate
        expected_time = 490 / 1e6
        assert "time_reference" in context
        assert context["time_reference"] == pytest.approx(expected_time)
        assert context["sample_rate"] == sample_rate

    def test_time_array_generation(self) -> None:
        """Test that time_array is correctly generated."""
        trace = np.arange(1000, dtype=np.float64)
        index = 500
        sample_rate = 1e6

        context = extract_context(trace, index, before=10, after=10, sample_rate=sample_rate)

        assert "time_array" in context
        assert len(context["time_array"]) == context["length"]

        # First time should be time_reference
        assert context["time_array"][0] == pytest.approx(context["time_reference"])

        # Time spacing should be 1/sample_rate
        dt = 1.0 / sample_rate
        time_diffs = np.diff(context["time_array"])
        np.testing.assert_allclose(time_diffs, dt)

    def test_time_reference_at_boundary(self) -> None:
        """Test time reference when extraction starts at index 0."""
        trace = np.arange(100, dtype=np.float64)
        index = 5
        sample_rate = 1e6

        context = extract_context(trace, index, before=20, after=10, sample_rate=sample_rate)

        # Start index is 0 (clamped), so time reference should be 0
        assert context["time_reference"] == 0.0
        assert context["time_array"][0] == 0.0

    def test_no_time_reference_without_sample_rate(self) -> None:
        """Test that time reference is not included when sample_rate is None."""
        trace = np.arange(100, dtype=np.float64)

        context = extract_context(trace, 50, before=10, after=10)

        assert "time_reference" not in context
        assert "sample_rate" not in context
        assert "time_array" not in context

    def test_different_sample_rates(self) -> None:
        """Test time reference with different sample rates."""
        trace = np.arange(1000, dtype=np.float64)
        index = 100

        # Test with 1 kHz
        context_1k = extract_context(trace, index, before=10, after=10, sample_rate=1e3)
        expected_1k = 90 / 1e3  # 0.09 seconds
        assert context_1k["time_reference"] == pytest.approx(expected_1k)

        # Test with 10 MHz
        context_10m = extract_context(trace, index, before=10, after=10, sample_rate=1e7)
        expected_10m = 90 / 1e7  # 9 microseconds
        assert context_10m["time_reference"] == pytest.approx(expected_10m)

    def test_batch_with_time_reference(self) -> None:
        """Test that each context in batch has correct time reference."""
        trace = np.arange(1000, dtype=np.float64)
        indices = [100, 500, 900]
        sample_rate = 1e6

        contexts = extract_context(trace, indices, before=10, after=10, sample_rate=sample_rate)

        for i, ctx in enumerate(contexts):
            expected_start = indices[i] - 10
            expected_time = expected_start / sample_rate
            assert ctx["time_reference"] == pytest.approx(expected_time)
            assert len(ctx["time_array"]) == ctx["length"]


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextMetadata:
    """Test metadata generation."""

    def test_metadata_included_by_default(self) -> None:
        """Test that metadata is included by default."""
        trace = np.arange(100, dtype=np.float64)

        context = extract_context(trace, 50, before=10, after=10)

        assert "metadata" in context
        assert isinstance(context["metadata"], dict)

    def test_metadata_content(self) -> None:
        """Test metadata contains expected fields."""
        trace = np.arange(100, dtype=np.float64)

        context = extract_context(trace, 50, before=10, after=10)

        metadata = context["metadata"]
        assert "samples_before" in metadata
        assert "samples_after" in metadata
        assert "at_start_boundary" in metadata
        assert "at_end_boundary" in metadata

    def test_metadata_samples_calculation(self) -> None:
        """Test that samples_before and samples_after are correctly calculated."""
        trace = np.arange(1000, dtype=np.float64)
        index = 500

        context = extract_context(trace, index, before=10, after=10)

        # Center is at 500, window is 490-511
        # samples_before = 500 - 490 = 10
        # samples_after = 511 - 500 - 1 = 10
        assert context["metadata"]["samples_before"] == 10
        assert context["metadata"]["samples_after"] == 10

    def test_metadata_boundary_flags(self) -> None:
        """Test boundary detection in metadata."""
        trace = np.arange(100, dtype=np.float64)

        # Test start boundary
        context_start = extract_context(trace, 5, before=20, after=10)
        assert context_start["metadata"]["at_start_boundary"] is True
        assert context_start["metadata"]["at_end_boundary"] is False

        # Test end boundary
        context_end = extract_context(trace, 95, before=10, after=20)
        assert context_end["metadata"]["at_start_boundary"] is False
        assert context_end["metadata"]["at_end_boundary"] is True

        # Test no boundary
        context_middle = extract_context(trace, 50, before=10, after=10)
        assert context_middle["metadata"]["at_start_boundary"] is False
        assert context_middle["metadata"]["at_end_boundary"] is False

    def test_metadata_at_exact_boundaries(self) -> None:
        """Test boundary flags when index is at exact start or end."""
        trace = np.arange(100, dtype=np.float64)

        # Index 0 with window should trigger start boundary
        context_0 = extract_context(trace, 0, before=10, after=10)
        assert context_0["metadata"]["at_start_boundary"] is True

        # Index 99 with window should trigger end boundary
        context_99 = extract_context(trace, 99, before=10, after=10)
        assert context_99["metadata"]["at_end_boundary"] is True

    def test_metadata_excluded_when_disabled(self) -> None:
        """Test that metadata is excluded when include_metadata=False."""
        trace = np.arange(100, dtype=np.float64)

        context = extract_context(trace, 50, before=10, after=10, include_metadata=False)

        assert "metadata" not in context

    def test_batch_metadata(self) -> None:
        """Test that each context in batch has correct metadata."""
        trace = np.arange(100, dtype=np.float64)
        indices = [10, 50, 90]

        contexts = extract_context(trace, indices, before=10, after=10)

        # First context at boundary
        assert contexts[0]["metadata"]["at_start_boundary"] is True

        # Middle context not at boundary
        assert contexts[1]["metadata"]["at_start_boundary"] is False
        assert contexts[1]["metadata"]["at_end_boundary"] is False

        # Last context at boundary
        assert contexts[2]["metadata"]["at_end_boundary"] is True


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextValidation:
    """Test input validation and error handling."""

    def test_negative_before_raises_error(self) -> None:
        """Test that negative before value raises ValueError."""
        trace = np.arange(100, dtype=np.float64)

        with pytest.raises(ValueError, match="before and after must be non-negative"):
            extract_context(trace, 50, before=-10, after=10)

    def test_negative_after_raises_error(self) -> None:
        """Test that negative after value raises ValueError."""
        trace = np.arange(100, dtype=np.float64)

        with pytest.raises(ValueError, match="before and after must be non-negative"):
            extract_context(trace, 50, before=10, after=-10)

    def test_index_out_of_bounds_raises_error(self) -> None:
        """Test that out of bounds index raises ValueError."""
        trace = np.arange(100, dtype=np.float64)

        with pytest.raises(ValueError, match="Index 150 out of bounds"):
            extract_context(trace, 150, before=10, after=10)

    def test_negative_index_raises_error(self) -> None:
        """Test that negative index raises ValueError."""
        trace = np.arange(100, dtype=np.float64)

        with pytest.raises(ValueError, match="Index -1 out of bounds"):
            extract_context(trace, -1, before=10, after=10)

    def test_batch_with_out_of_bounds_index(self) -> None:
        """Test that any out of bounds index in batch raises error."""
        trace = np.arange(100, dtype=np.float64)
        indices = [10, 50, 150]  # 150 is out of bounds

        with pytest.raises(ValueError, match="Index 150 out of bounds"):
            extract_context(trace, indices, before=10, after=10)

    def test_empty_trace_raises_error(self) -> None:
        """Test that empty trace raises ValueError."""
        trace = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="Trace cannot be empty"):
            extract_context(trace, 0, before=10, after=10)

    def test_zero_window_size(self) -> None:
        """Test extraction with zero window size."""
        trace = np.arange(100, dtype=np.float64)

        context = extract_context(trace, 50, before=0, after=0)

        # Should extract only the single sample at index 50
        assert context["length"] == 1
        assert context["start_index"] == 50
        assert context["end_index"] == 51
        assert context["data"][0] == 50


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_small_trace_large_window(self) -> None:
        """Test extraction with window larger than trace."""
        trace = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        context = extract_context(trace, 1, before=100, after=100)

        # Should extract entire trace
        assert context["start_index"] == 0
        assert context["end_index"] == 3
        assert context["length"] == 3
        np.testing.assert_array_equal(context["data"], trace)

    def test_single_sample_trace(self) -> None:
        """Test extraction from single sample trace."""
        trace = np.array([42.0], dtype=np.float64)

        context = extract_context(trace, 0, before=10, after=10)

        assert context["start_index"] == 0
        assert context["end_index"] == 1
        assert context["length"] == 1
        assert context["data"][0] == 42.0

    def test_numpy_integer_index(self) -> None:
        """Test that numpy integer types work as index."""
        trace = np.arange(100, dtype=np.float64)

        # Test various numpy integer types
        for dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            index = dtype(50)
            context = extract_context(trace, index, before=5, after=5)
            assert context["center_index"] == 50

    def test_preserve_trace_dtype(self) -> None:
        """Test that extracted data preserves trace dtype."""
        trace_f32 = np.arange(100, dtype=np.float32)
        trace_f64 = np.arange(100, dtype=np.float64)

        context_f32 = extract_context(trace_f32, 50, before=5, after=5)
        context_f64 = extract_context(trace_f64, 50, before=5, after=5)

        assert context_f32["data"].dtype == np.float32
        assert context_f64["data"].dtype == np.float64

    def test_complex_valued_trace(self) -> None:
        """Test extraction from complex-valued trace."""
        trace = np.arange(100, dtype=np.complex128) + 1j * np.arange(100)

        context = extract_context(trace, 50, before=5, after=5)

        assert context["data"].dtype == np.complex128
        assert len(context["data"]) == 11
        assert context["data"][5] == (50 + 50j)

    def test_multidimensional_not_supported(self) -> None:
        """Test that function works with 1D trace only."""
        # 2D trace - should work if treated as flat samples
        trace_2d = np.arange(100, dtype=np.float64).reshape(10, 10)

        # This will work because numpy indexing flattens
        context = extract_context(trace_2d, 5, before=2, after=2)
        assert len(context["data"]) == 5

    def test_batch_with_duplicates(self) -> None:
        """Test batch extraction with duplicate indices."""
        trace = np.arange(100, dtype=np.float64)
        indices = [50, 50, 50]  # Same index three times

        contexts = extract_context(trace, indices, before=5, after=5)

        assert len(contexts) == 3
        # All should be identical
        for ctx in contexts:
            assert ctx["center_index"] == 50
            np.testing.assert_array_equal(ctx["data"], contexts[0]["data"])

    def test_very_large_trace_performance(self) -> None:
        """Test extraction from large trace completes efficiently."""
        # Create 10 million sample trace
        trace = np.arange(10_000_000, dtype=np.float64)
        index = 5_000_000

        # Should complete quickly
        context = extract_context(trace, index, before=1000, after=1000)

        assert context["length"] == 2001
        assert context["center_index"] == 5_000_000

    def test_batch_extraction_maintains_order(self) -> None:
        """Test that batch extraction maintains input order."""
        trace = np.arange(1000, dtype=np.float64)
        indices = [900, 100, 500, 200]  # Not sorted

        contexts = extract_context(trace, indices, before=5, after=5)

        # Order should match input indices
        assert contexts[0]["center_index"] == 900
        assert contexts[1]["center_index"] == 100
        assert contexts[2]["center_index"] == 500
        assert contexts[3]["center_index"] == 200


@pytest.mark.unit
@pytest.mark.requirement("SRCH-003")
class TestExtractContextIntegration:
    """Test integration scenarios mimicking real usage."""

    def test_debug_workflow_glitch_detection(self) -> None:
        """Test typical debug workflow: finding and extracting glitch context."""
        # Simulate a trace with a glitch
        trace = np.ones(10000, dtype=np.float64)
        trace[5000] = 10.0  # Glitch spike

        # Find glitch
        glitch_idx = np.argmax(trace)

        # Extract context
        context = extract_context(
            trace, glitch_idx, before=100, after=100, sample_rate=1e6, include_metadata=True
        )

        assert context["center_index"] == 5000
        assert context["data"][100] == 10.0  # Glitch is at center
        assert "time_reference" in context
        assert context["metadata"]["at_start_boundary"] is False
        assert context["metadata"]["at_end_boundary"] is False

    def test_multiple_event_extraction(self) -> None:
        """Test extracting contexts around multiple detected events."""
        # Create trace with multiple events
        trace = np.zeros(10000, dtype=np.float64)
        event_indices = [1000, 3000, 5000, 7000]
        for idx in event_indices:
            trace[idx] = 1.0

        # Extract context around all events
        contexts = extract_context(trace, event_indices, before=50, after=50, sample_rate=1e6)

        assert len(contexts) == 4
        for i, ctx in enumerate(contexts):
            assert ctx["center_index"] == event_indices[i]
            assert ctx["data"][50] == 1.0  # Event at center
            assert ctx["length"] == 101

    def test_boundary_event_handling(self) -> None:
        """Test extraction of events near trace boundaries."""
        trace = np.zeros(1000, dtype=np.float64)
        # Events at start, middle, and end
        event_indices = [10, 500, 990]
        for idx in event_indices:
            trace[idx] = 1.0

        contexts = extract_context(
            trace, event_indices, before=50, after=50, sample_rate=1e6, include_metadata=True
        )

        # First event near start
        assert contexts[0]["metadata"]["at_start_boundary"] is True
        assert contexts[0]["length"] < 101  # Truncated

        # Middle event
        assert contexts[1]["metadata"]["at_start_boundary"] is False
        assert contexts[1]["metadata"]["at_end_boundary"] is False
        assert contexts[1]["length"] == 101

        # Last event near end
        assert contexts[2]["metadata"]["at_end_boundary"] is True
        assert contexts[2]["length"] < 101  # Truncated

    def test_time_correlation_workflow(self) -> None:
        """Test workflow requiring time correlation between events."""
        trace = np.sin(2 * np.pi * 1000 * np.arange(10000) / 1e6)  # 1 kHz sine at 1 MHz
        sample_rate = 1e6

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(trace)))[0][:5]

        contexts = extract_context(
            trace, list(zero_crossings), before=10, after=10, sample_rate=sample_rate
        )

        # Verify time references are consistent
        for ctx in contexts:
            assert "time_array" in ctx
            assert len(ctx["time_array"]) == ctx["length"]
            # Time array should be monotonically increasing
            assert np.all(np.diff(ctx["time_array"]) > 0)

    def test_protocol_analysis_workflow(self) -> None:
        """Test workflow for protocol reverse engineering."""
        # Simulate packet starts in a trace
        trace = np.random.randn(100000)
        packet_starts = [1000, 5000, 10000, 20000, 50000]

        # Extract packet contexts
        contexts = extract_context(
            trace,
            packet_starts,
            before=0,  # Just after packet start
            after=500,  # 500 samples per packet
            sample_rate=10e6,
            include_metadata=True,
        )

        assert len(contexts) == 5
        for i, ctx in enumerate(contexts):
            assert ctx["center_index"] == packet_starts[i]
            assert ctx["start_index"] == packet_starts[i]
            # Verify we can analyze packet timing
            assert "time_reference" in ctx
            assert "time_array" in ctx

    def test_comparative_analysis_workflow(self) -> None:
        """Test workflow comparing contexts from different events."""
        trace = np.random.randn(10000)
        indices = [1000, 2000, 3000]

        contexts = extract_context(trace, indices, before=50, after=50)

        # Verify all contexts have same shape for comparison
        shapes = [ctx["data"].shape for ctx in contexts]
        assert all(s == shapes[0] for s in shapes)

        # Can compute correlation between contexts
        corr_01 = np.corrcoef(contexts[0]["data"], contexts[1]["data"])[0, 1]
        corr_02 = np.corrcoef(contexts[0]["data"], contexts[2]["data"])[0, 1]

        # Correlations should be finite numbers
        assert np.isfinite(corr_01)
        assert np.isfinite(corr_02)
