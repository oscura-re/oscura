"""Comprehensive unit tests for multi-trace comparison.

This module tests comparing multiple traces simultaneously including
batch comparison and multi-golden comparison scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.utils.comparison.compare import compare_traces, similarity_score
from oscura.utils.comparison.golden import (
    batch_compare_to_golden,
    create_golden,
    golden_from_average,
)
from oscura.utils.comparison.limits import LimitSpec, check_limits

pytestmark = pytest.mark.unit


@pytest.fixture
def trace_list() -> list[WaveformTrace]:
    """Create a list of similar traces for batch testing."""
    np.random.seed(42)
    traces = []
    base_data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))

    for i in range(10):
        noise = np.random.normal(0, 0.01, 1000)
        data = base_data + noise
        metadata = TraceMetadata(sample_rate=1e6, channel_name=f"CH{i}")
        traces.append(WaveformTrace(data=data, metadata=metadata))

    return traces


@pytest.fixture
def varied_trace_list() -> list[WaveformTrace]:
    """Create traces with varying characteristics."""
    traces = []

    # Sine wave
    t = np.linspace(0, 1e-3, 1000)
    data1 = np.sin(2 * np.pi * 1e6 * t)
    traces.append(WaveformTrace(data=data1, metadata=TraceMetadata(sample_rate=1e6)))

    # Cosine wave
    data2 = np.cos(2 * np.pi * 1e6 * t)
    traces.append(WaveformTrace(data=data2, metadata=TraceMetadata(sample_rate=1e6)))

    # Square wave (digital)
    data3 = np.where(np.sin(2 * np.pi * 1e6 * t) > 0, 1.0, -1.0)
    traces.append(WaveformTrace(data=data3, metadata=TraceMetadata(sample_rate=1e6)))

    # Noise
    np.random.seed(123)
    data4 = np.random.normal(0, 0.5, 1000)
    traces.append(WaveformTrace(data=data4, metadata=TraceMetadata(sample_rate=1e6)))

    return traces


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestBatchComparison:
    """Test batch comparison of multiple traces."""

    def test_all_to_all_similarity(self, trace_list: list[WaveformTrace]) -> None:
        """Test computing similarity matrix for all traces."""
        n = len(trace_list)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = similarity_score(trace_list[i], trace_list[j])

        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_allclose(np.diag(similarity_matrix), 1.0, rtol=0.01)

        # Matrix should be symmetric
        np.testing.assert_allclose(similarity_matrix, similarity_matrix.T, rtol=0.01)

    def test_pairwise_comparison(self, trace_list: list[WaveformTrace]) -> None:
        """Test pairwise comparison of consecutive traces."""
        results = []
        for i in range(len(trace_list) - 1):
            result = compare_traces(trace_list[i], trace_list[i + 1])
            results.append(result)

        # All should have high similarity (same base signal + noise)
        assert all(r.similarity > 0.9 for r in results)

    def test_reference_comparison(self, trace_list: list[WaveformTrace]) -> None:
        """Test comparing all traces to a single reference."""
        reference = trace_list[0]
        results = [compare_traces(trace, reference) for trace in trace_list[1:]]

        # All should match the reference with small tolerance
        assert all(r.max_difference < 0.1 for r in results)

    def test_batch_golden_comparison(self, trace_list: list[WaveformTrace]) -> None:
        """Test batch comparison to golden reference."""
        # Create golden from first trace
        golden = create_golden(trace_list[0], tolerance_pct=5.0)

        # Compare all traces
        results = batch_compare_to_golden(trace_list, golden)

        assert len(results) == len(trace_list)
        # Most should pass (similar signals)
        pass_rate = sum(r.passed for r in results) / len(results)
        assert pass_rate > 0.8

    def test_averaged_golden(self, trace_list: list[WaveformTrace]) -> None:
        """Test creating golden from averaged traces."""
        # Create golden from average
        golden = golden_from_average(trace_list, tolerance_sigma=3.0)

        # Each trace should pass when compared to averaged golden
        results = batch_compare_to_golden(trace_list, golden)

        # Most should pass
        pass_rate = sum(r.passed for r in results) / len(results)
        assert pass_rate > 0.7


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestVariedTraceComparison:
    """Test comparison of traces with different characteristics."""

    def test_dissimilar_traces(self, varied_trace_list: list[WaveformTrace]) -> None:
        """Test that dissimilar traces have low similarity."""
        # Compare sine to noise
        score = similarity_score(varied_trace_list[0], varied_trace_list[3])
        assert score < 0.7  # Should have low similarity

    def test_similar_pattern_different_phase(self, varied_trace_list: list[WaveformTrace]) -> None:
        """Test sine vs cosine (90 degree phase shift)."""
        # Sine vs Cosine
        score = similarity_score(varied_trace_list[0], varied_trace_list[1])
        # Orthogonal signals should have low correlation
        assert score < 0.6

    def test_analog_vs_digital(self, varied_trace_list: list[WaveformTrace]) -> None:
        """Test comparison of analog sine to digital square wave."""
        score = similarity_score(varied_trace_list[0], varied_trace_list[2])
        # Square wave and sine at same frequency are highly correlated
        # (square wave is essentially sign(sine wave))
        # Expect high similarity score
        assert 0.8 < score < 1.0

    def test_batch_limit_checking(self, trace_list: list[WaveformTrace]) -> None:
        """Test checking limits on multiple traces."""
        spec = LimitSpec(upper=1.5, lower=-1.5)

        results = [check_limits(trace, limits=spec) for trace in trace_list]

        # All should pass (sine wave is between -1 and 1)
        assert all(r.passed for r in results)


@pytest.mark.unit
@pytest.mark.comparison
class TestMultiTraceStatistics:
    """Test statistical analysis of multiple traces."""

    def test_mean_trace(self, trace_list: list[WaveformTrace]) -> None:
        """Test computing mean across multiple traces."""
        # Get common length
        min_len = min(len(t.data) for t in trace_list)
        stacked = np.array([t.data[:min_len] for t in trace_list])

        mean_data = np.mean(stacked, axis=0)
        mean_trace = WaveformTrace(data=mean_data, metadata=trace_list[0].metadata)

        # Mean should have high similarity to each trace
        scores = [similarity_score(mean_trace, trace) for trace in trace_list]
        assert all(s > 0.9 for s in scores)

    def test_std_trace(self, trace_list: list[WaveformTrace]) -> None:
        """Test computing standard deviation across traces."""
        min_len = min(len(t.data) for t in trace_list)
        stacked = np.array([t.data[:min_len] for t in trace_list])

        std_data = np.std(stacked, axis=0)

        # Std should be small for similar traces
        assert np.mean(std_data) < 0.1

    def test_min_max_envelope(self, trace_list: list[WaveformTrace]) -> None:
        """Test computing min/max envelope across traces."""
        min_len = min(len(t.data) for t in trace_list)
        stacked = np.array([t.data[:min_len] for t in trace_list])

        min_envelope = np.min(stacked, axis=0)
        max_envelope = np.max(stacked, axis=0)

        # All traces should be within envelope
        for trace in trace_list:
            data = trace.data[:min_len]
            assert np.all(data >= min_envelope)
            assert np.all(data <= max_envelope)


@pytest.mark.unit
@pytest.mark.comparison
class TestComparisonMultiTraceEdgeCases:
    """Test edge cases for multi-trace comparison."""

    def test_empty_trace_list(self) -> None:
        """Test with empty trace list."""
        from oscura.core.exceptions import AnalysisError

        with pytest.raises(AnalysisError):
            golden_from_average([])

    def test_single_trace_list(self) -> None:
        """Test with single trace."""
        trace = WaveformTrace(data=np.ones(100), metadata=TraceMetadata(sample_rate=1e6))

        golden = golden_from_average([trace])
        assert golden.num_samples == 100

    def test_different_lengths(self) -> None:
        """Test traces of different lengths."""
        trace1 = WaveformTrace(data=np.ones(1000), metadata=TraceMetadata(sample_rate=1e6))
        trace2 = WaveformTrace(data=np.ones(500), metadata=TraceMetadata(sample_rate=1e6))

        golden = golden_from_average([trace1, trace2])
        # Should use minimum length
        assert golden.num_samples == 500

    def test_large_batch(self) -> None:
        """Test with large number of traces."""
        np.random.seed(42)
        traces = []
        for _i in range(100):
            data = np.random.normal(0, 1, 100)
            trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))
            traces.append(trace)

        golden = golden_from_average(traces)
        results = batch_compare_to_golden(traces[:10], golden)

        assert len(results) == 10
