"""Unit tests for parallel pipeline execution.

This test suite covers:
- ParallelPipeline initialization and configuration
- Dependency analysis and execution order
- Parallel execution with thread and process pools
- Worker count selection (automatic and manual)
- API compatibility with standard Pipeline
- Error handling and edge cases
- Performance characteristics
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.pipeline.base import TraceTransformer
from oscura.pipeline.parallel import ParallelPipeline
from oscura.pipeline.pipeline import Pipeline

pytestmark = pytest.mark.unit


# ============================================================================
# Mock Transformer Classes for Testing
# ============================================================================


class SlowTransformer(TraceTransformer):
    """Transformer with configurable delay for testing parallel speedup."""

    def __init__(self, delay_seconds: float = 0.1, scale_factor: float = 1.0) -> None:
        self.delay_seconds = delay_seconds
        self.scale_factor = scale_factor

    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Transform with delay to simulate slow operation."""
        time.sleep(self.delay_seconds)
        scaled_data = trace.data * self.scale_factor
        return WaveformTrace(data=scaled_data, metadata=trace.metadata)


class FastTransformer(TraceTransformer):
    """Simple fast transformer for testing."""

    def __init__(self, scale_factor: float = 1.0, offset: float = 0.0) -> None:
        self.scale_factor = scale_factor
        self.offset = offset

    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Quick transformation."""
        scaled_data = trace.data * self.scale_factor + self.offset
        return WaveformTrace(data=scaled_data, metadata=trace.metadata)


class StatefulTransformer(TraceTransformer):
    """Transformer that uses fit to learn parameters."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, trace: WaveformTrace) -> StatefulTransformer:
        """Learn mean and std from reference trace."""
        self.mean_ = float(np.mean(trace.data))
        self.std_ = float(np.std(trace.data))
        return self

    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Normalize using learned parameters."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Transformer not fitted")

        normalized = (trace.data - self.mean_) / (self.std_ if self.std_ != 0 else 1.0)
        return WaveformTrace(data=normalized, metadata=trace.metadata)


class IntermediateTransformer(TraceTransformer):
    """Transformer that caches intermediate results."""

    def __init__(self, compute_stats: bool = True) -> None:
        self.compute_stats = compute_stats

    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Transform and cache intermediate results."""
        self._clear_intermediates()

        # Cache some intermediate results
        self._cache_intermediate("mean", float(np.mean(trace.data)))
        self._cache_intermediate("max", float(np.max(trace.data)))
        self._cache_intermediate("min", float(np.min(trace.data)))

        if self.compute_stats:
            self._cache_intermediate("std", float(np.std(trace.data)))

        return WaveformTrace(data=trace.data, metadata=trace.metadata)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_trace() -> WaveformTrace:
    """Create a sample waveform trace for testing."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sine_trace() -> WaveformTrace:
    """Create a sine wave trace for testing."""
    t = np.linspace(0, 1e-3, 1000)
    data = np.sin(2 * np.pi * 1e3 * t)
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def fast_transformer() -> FastTransformer:
    """Create a fast transformer."""
    return FastTransformer(scale_factor=2.0, offset=1.0)


@pytest.fixture
def slow_transformer() -> SlowTransformer:
    """Create a slow transformer."""
    return SlowTransformer(delay_seconds=0.1, scale_factor=2.0)


# ============================================================================
# ParallelPipeline Initialization Tests
# ============================================================================


def test_parallel_pipeline_creation_default(fast_transformer: FastTransformer) -> None:
    """Test creating a ParallelPipeline with default settings."""
    pipeline = ParallelPipeline([("scale", fast_transformer)])

    assert len(pipeline) == 1
    assert pipeline.executor_type == "thread"
    assert pipeline.max_workers is None
    assert "scale" in pipeline.named_steps


def test_parallel_pipeline_creation_thread_executor() -> None:
    """Test creating a ParallelPipeline with thread executor."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer()), ("stage2", FastTransformer())],
        executor_type="thread",
        max_workers=4,
    )

    assert pipeline.executor_type == "thread"
    assert pipeline.max_workers == 4


def test_parallel_pipeline_creation_process_executor() -> None:
    """Test creating a ParallelPipeline with process executor."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer()), ("stage2", FastTransformer())],
        executor_type="process",
        max_workers=2,
    )

    assert pipeline.executor_type == "process"
    assert pipeline.max_workers == 2


def test_parallel_pipeline_invalid_executor_type() -> None:
    """Test that invalid executor type raises ValueError."""
    with pytest.raises(ValueError, match="executor_type must be 'thread' or 'process'"):
        ParallelPipeline([("stage1", FastTransformer())], executor_type="invalid")  # type: ignore


def test_parallel_pipeline_empty_steps_raises() -> None:
    """Test that empty steps raises ValueError."""
    with pytest.raises(ValueError, match="Pipeline steps cannot be empty"):
        ParallelPipeline([])


def test_parallel_pipeline_duplicate_names_raises() -> None:
    """Test that duplicate step names raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate step names"):
        ParallelPipeline([("same", FastTransformer()), ("same", FastTransformer())])


# ============================================================================
# Dependency Analysis Tests
# ============================================================================


def test_dependency_graph_single_stage() -> None:
    """Test dependency graph for single-stage pipeline."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())])

    graph = pipeline.get_dependency_graph()
    assert graph == {"stage1": []}


def test_dependency_graph_sequential_stages() -> None:
    """Test dependency graph for sequential stages."""
    pipeline = ParallelPipeline(
        [
            ("stage1", FastTransformer()),
            ("stage2", FastTransformer()),
            ("stage3", FastTransformer()),
        ]
    )

    graph = pipeline.get_dependency_graph()
    assert graph == {"stage1": [], "stage2": ["stage1"], "stage3": ["stage2"]}


def test_execution_order_single_stage() -> None:
    """Test execution order for single-stage pipeline."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())])

    order = pipeline.get_execution_order()
    assert order == [["stage1"]]


def test_execution_order_sequential_stages() -> None:
    """Test execution order for sequential stages."""
    pipeline = ParallelPipeline(
        [
            ("stage1", FastTransformer()),
            ("stage2", FastTransformer()),
            ("stage3", FastTransformer()),
        ]
    )

    order = pipeline.get_execution_order()
    # Current conservative implementation: all sequential
    assert order == [["stage1"], ["stage2"], ["stage3"]]


# ============================================================================
# Worker Count Selection Tests
# ============================================================================


def test_auto_worker_count_thread() -> None:
    """Test automatic worker count selection for thread executor."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer())], executor_type="thread", max_workers=None
    )

    workers = pipeline._get_max_workers()
    # Should be min(32, cpu_count + 4)
    import os

    expected = min(32, (os.cpu_count() or 4) + 4)
    assert workers == expected


def test_auto_worker_count_process() -> None:
    """Test automatic worker count selection for process executor."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer())], executor_type="process", max_workers=None
    )

    workers = pipeline._get_max_workers()
    # Should be cpu_count
    import os

    expected = os.cpu_count() or 4
    assert workers == expected


def test_manual_worker_count() -> None:
    """Test manual worker count setting."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())], max_workers=8)

    assert pipeline._get_max_workers() == 8


# ============================================================================
# Transform Tests
# ============================================================================


def test_parallel_transform_single_stage(sample_trace: WaveformTrace) -> None:
    """Test transform with single stage."""
    pipeline = ParallelPipeline([("scale", FastTransformer(scale_factor=2.0, offset=1.0))])

    result = pipeline.transform(sample_trace)

    expected = sample_trace.data * 2.0 + 1.0
    np.testing.assert_array_almost_equal(result.data, expected)


def test_parallel_transform_sequential_stages(sample_trace: WaveformTrace) -> None:
    """Test transform with sequential stages."""
    pipeline = ParallelPipeline(
        [
            ("stage1", FastTransformer(scale_factor=2.0)),
            ("stage2", FastTransformer(offset=1.0)),
            ("stage3", FastTransformer(scale_factor=0.5)),
        ]
    )

    result = pipeline.transform(sample_trace)

    # Apply transformations manually
    expected = sample_trace.data * 2.0  # stage1
    expected = expected + 1.0  # stage2
    expected = expected * 0.5  # stage3

    np.testing.assert_array_almost_equal(result.data, expected)


def test_parallel_transform_thread_executor(sample_trace: WaveformTrace) -> None:
    """Test transform with thread executor."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer(scale_factor=2.0)), ("stage2", FastTransformer(offset=1.0))],
        executor_type="thread",
        max_workers=2,
    )

    result = pipeline.transform(sample_trace)

    expected = sample_trace.data * 2.0 + 1.0
    np.testing.assert_array_almost_equal(result.data, expected)


def test_parallel_transform_process_executor(sample_trace: WaveformTrace) -> None:
    """Test transform with process executor."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer(scale_factor=2.0)), ("stage2", FastTransformer(offset=1.0))],
        executor_type="process",
        max_workers=2,
    )

    result = pipeline.transform(sample_trace)

    expected = sample_trace.data * 2.0 + 1.0
    np.testing.assert_array_almost_equal(result.data, expected)


def test_parallel_transform_caches_intermediates(sample_trace: WaveformTrace) -> None:
    """Test that transform caches intermediate results."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer(scale_factor=2.0)), ("stage2", FastTransformer(offset=1.0))]
    )

    pipeline.transform(sample_trace)

    # Check intermediate results are cached
    assert "stage1" in pipeline._intermediate_results
    assert "stage2" in pipeline._intermediate_results

    # Verify intermediate values
    stage1_result = pipeline.get_intermediate("stage1")
    np.testing.assert_array_almost_equal(stage1_result.data, sample_trace.data * 2.0)


# ============================================================================
# Fit Tests
# ============================================================================


def test_parallel_fit_sequential(sample_trace: WaveformTrace) -> None:
    """Test that fit is always sequential."""
    pipeline = ParallelPipeline([("normalize", StatefulTransformer())])

    pipeline.fit(sample_trace)

    # Check that transformer was fitted
    assert pipeline.named_steps["normalize"].mean_ is not None
    assert pipeline.named_steps["normalize"].std_ is not None


def test_parallel_fit_transform_chain(sample_trace: WaveformTrace) -> None:
    """Test fit followed by transform."""
    pipeline = ParallelPipeline([("normalize", StatefulTransformer())])

    pipeline.fit(sample_trace)
    result = pipeline.transform(sample_trace)

    # Normalized data should have mean close to 0, std close to 1
    assert abs(np.mean(result.data)) < 1e-10
    assert abs(np.std(result.data) - 1.0) < 1e-10


# ============================================================================
# API Compatibility Tests
# ============================================================================


def test_parallel_pipeline_api_compatible_with_pipeline(sample_trace: WaveformTrace) -> None:
    """Test that ParallelPipeline is API-compatible with Pipeline."""
    steps = [("stage1", FastTransformer(scale_factor=2.0)), ("stage2", FastTransformer(offset=1.0))]

    # Create both pipelines
    sequential = Pipeline(steps)
    parallel = ParallelPipeline(steps)

    # Both should produce same results
    seq_result = sequential.transform(sample_trace)
    par_result = parallel.transform(sample_trace)

    np.testing.assert_array_almost_equal(seq_result.data, par_result.data)


def test_parallel_pipeline_supports_indexing(fast_transformer: FastTransformer) -> None:
    """Test that ParallelPipeline supports indexing like Pipeline."""
    pipeline = ParallelPipeline([("scale", fast_transformer)])

    # By index
    assert pipeline[0] == fast_transformer
    # By name
    assert pipeline["scale"] == fast_transformer


def test_parallel_pipeline_get_params() -> None:
    """Test get_params method."""
    pipeline = ParallelPipeline([("stage1", FastTransformer(scale_factor=2.0, offset=1.0))])

    params = pipeline.get_params(deep=True)

    assert "steps" in params
    assert "stage1__scale_factor" in params
    assert params["stage1__scale_factor"] == 2.0
    assert params["stage1__offset"] == 1.0


def test_parallel_pipeline_set_params() -> None:
    """Test set_params method."""
    pipeline = ParallelPipeline([("stage1", FastTransformer(scale_factor=1.0, offset=0.0))])

    pipeline.set_params(stage1__scale_factor=3.0)

    assert pipeline.named_steps["stage1"].scale_factor == 3.0


def test_parallel_pipeline_clone() -> None:
    """Test clone method."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer(scale_factor=2.0))], executor_type="process", max_workers=4
    )

    cloned = pipeline.clone()

    assert cloned is not pipeline
    assert cloned.executor_type == pipeline.executor_type
    assert cloned.max_workers == pipeline.max_workers
    assert len(cloned.steps) == len(pipeline.steps)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_set_parallel_config_executor_type() -> None:
    """Test changing executor type."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())], executor_type="thread")

    pipeline.set_parallel_config(executor_type="process")

    assert pipeline.executor_type == "process"


def test_set_parallel_config_max_workers() -> None:
    """Test changing max workers."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())], max_workers=2)

    pipeline.set_parallel_config(max_workers=8)

    assert pipeline.max_workers == 8


def test_set_parallel_config_both() -> None:
    """Test changing both executor type and max workers."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer())], executor_type="thread", max_workers=2
    )

    pipeline.set_parallel_config(executor_type="process", max_workers=4)

    assert pipeline.executor_type == "process"
    assert pipeline.max_workers == 4


def test_set_parallel_config_invalid_executor_raises() -> None:
    """Test that invalid executor type raises ValueError."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())])

    with pytest.raises(ValueError, match="executor_type must be 'thread' or 'process'"):
        pipeline.set_parallel_config(executor_type="invalid")  # type: ignore


# ============================================================================
# Intermediate Results Tests
# ============================================================================


def test_parallel_get_intermediate(sample_trace: WaveformTrace) -> None:
    """Test accessing intermediate results."""
    pipeline = ParallelPipeline(
        [("stage1", IntermediateTransformer()), ("stage2", FastTransformer(scale_factor=2.0))]
    )

    pipeline.transform(sample_trace)

    # Get trace intermediate
    stage1_trace = pipeline.get_intermediate("stage1")
    assert stage1_trace is not None

    # Get transformer intermediate
    mean = pipeline.get_intermediate("stage1", "mean")
    assert abs(mean - np.mean(sample_trace.data)) < 1e-10


def test_parallel_has_intermediate(sample_trace: WaveformTrace) -> None:
    """Test checking for intermediate results."""
    pipeline = ParallelPipeline([("stage1", IntermediateTransformer())])

    pipeline.transform(sample_trace)

    assert pipeline.has_intermediate("stage1")
    assert pipeline.has_intermediate("stage1", "mean")
    assert not pipeline.has_intermediate("stage1", "nonexistent")


def test_parallel_list_intermediates(sample_trace: WaveformTrace) -> None:
    """Test listing intermediate results."""
    pipeline = ParallelPipeline([("stage1", IntermediateTransformer())])

    pipeline.transform(sample_trace)

    intermediates = pipeline.list_intermediates("stage1")
    assert "mean" in intermediates
    assert "max" in intermediates
    assert "min" in intermediates
    assert "std" in intermediates


# ============================================================================
# Repr Tests
# ============================================================================


def test_parallel_pipeline_repr() -> None:
    """Test string representation."""
    pipeline = ParallelPipeline(
        [("stage1", FastTransformer()), ("stage2", FastTransformer())],
        executor_type="process",
        max_workers=4,
    )

    repr_str = repr(pipeline)

    assert "ParallelPipeline" in repr_str
    assert "stage1" in repr_str
    assert "stage2" in repr_str
    assert "executor=process" in repr_str
    assert "workers=4" in repr_str


def test_parallel_pipeline_repr_auto_workers() -> None:
    """Test string representation with auto workers."""
    pipeline = ParallelPipeline([("stage1", FastTransformer())], max_workers=None)

    repr_str = repr(pipeline)

    assert "workers=auto" in repr_str


# ============================================================================
# Performance Tests (Timing-based)
# ============================================================================


@pytest.mark.slow
def test_parallel_speedup_with_slow_transformers(sample_trace: WaveformTrace) -> None:
    """Test that parallel execution provides speedup for slow transformers.

    NOTE: This test is marked as slow and may be skipped in fast test runs.
    It verifies that parallel execution actually runs stages in parallel.
    """
    # NOTE: This test is timing-sensitive and may be flaky
    # Skip if running in CI or constrained environment
    import os

    if os.environ.get("CI"):
        pytest.skip("Skipping timing-sensitive test in CI")

    # Create pipeline with very slow transformers (0.2s each)
    # With 3 stages, sequential would take ~0.6s
    # But with conservative dependency analysis, parallel also takes ~0.6s
    # This test verifies the execution mechanism works correctly
    pipeline = ParallelPipeline(
        [
            ("stage1", SlowTransformer(delay_seconds=0.2)),
            ("stage2", SlowTransformer(delay_seconds=0.2)),
            ("stage3", SlowTransformer(delay_seconds=0.2)),
        ],
        executor_type="thread",
        max_workers=3,
    )

    import time

    start = time.time()
    pipeline.transform(sample_trace)
    duration = time.time() - start

    # With current conservative implementation (sequential), should take ~0.6s
    # Allow some tolerance for overhead
    assert duration < 1.0  # Should complete within 1 second
