"""Comprehensive unit tests for pipeline.py module.

Requirements tested:

This test suite covers:
- Pipeline creation and initialization
- Step management and validation
- fit/transform pattern
- Intermediate result caching and retrieval
- Parameter get/set operations
- Pipeline cloning
- Error handling and edge cases
- Integration with TraceTransformer
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.pipeline.base import TraceTransformer
from oscura.pipeline.pipeline import Pipeline

pytestmark = pytest.mark.unit


# ============================================================================
# Mock Transformer Classes for Testing
# ============================================================================


class MockTransformer(TraceTransformer):
    """Simple mock transformer for testing."""

    def __init__(self, scale_factor: float = 1.0, offset: float = 0.0) -> None:
        self.scale_factor = scale_factor
        self.offset = offset

    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Scale and offset the trace data."""
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
def simple_transformer() -> MockTransformer:
    """Create a simple mock transformer."""
    return MockTransformer(scale_factor=2.0, offset=1.0)


@pytest.fixture
def stateful_transformer() -> StatefulTransformer:
    """Create a stateful transformer."""
    return StatefulTransformer()


@pytest.fixture
def intermediate_transformer() -> IntermediateTransformer:
    """Create a transformer with intermediates."""
    return IntermediateTransformer()


@pytest.fixture
def simple_pipeline(simple_transformer: MockTransformer) -> Pipeline:
    """Create a simple pipeline with one step."""
    return Pipeline([("scale", simple_transformer)])


@pytest.fixture
def multi_stage_pipeline() -> Pipeline:
    """Create a pipeline with multiple stages."""
    return Pipeline(
        [
            ("stage1", MockTransformer(scale_factor=2.0)),
            ("stage2", MockTransformer(scale_factor=0.5, offset=1.0)),
            ("stage3", MockTransformer(offset=-0.5)),
        ]
    )


# ============================================================================
# Pipeline Creation and Initialization Tests
# ============================================================================


class TestPipelineCreation:
    """Tests for Pipeline creation and initialization."""

    def test_create_empty_pipeline_raises_error(self) -> None:
        """Test that creating empty pipeline raises ValueError."""
        with pytest.raises(ValueError, match="Pipeline steps cannot be empty"):
            Pipeline([])

    def test_create_single_step_pipeline(self, simple_transformer: MockTransformer) -> None:
        """Test creating pipeline with single step."""
        pipeline = Pipeline([("transform", simple_transformer)])

        assert len(pipeline) == 1
        assert len(pipeline.steps) == 1
        assert "transform" in pipeline.named_steps
        assert pipeline.named_steps["transform"] is simple_transformer

    def test_create_multi_step_pipeline(self, multi_stage_pipeline: Pipeline) -> None:
        """Test creating pipeline with multiple steps."""
        assert len(multi_stage_pipeline) == 3
        assert len(multi_stage_pipeline.steps) == 3
        assert set(multi_stage_pipeline.named_steps.keys()) == {"stage1", "stage2", "stage3"}

    def test_steps_stored_as_list(self, multi_stage_pipeline: Pipeline) -> None:
        """Test that steps are stored as list of tuples."""
        assert isinstance(multi_stage_pipeline.steps, list)
        for name, transformer in multi_stage_pipeline.steps:
            assert isinstance(name, str)
            assert isinstance(transformer, TraceTransformer)

    def test_named_steps_dict_mapping(self, multi_stage_pipeline: Pipeline) -> None:
        """Test that named_steps provides dict mapping."""
        assert isinstance(multi_stage_pipeline.named_steps, dict)
        assert multi_stage_pipeline.named_steps["stage1"] is multi_stage_pipeline.steps[0][1]
        assert multi_stage_pipeline.named_steps["stage2"] is multi_stage_pipeline.steps[1][1]


# ============================================================================
# Step Validation Tests
# ============================================================================


class TestStepValidation:
    """Tests for pipeline step validation."""

    def test_empty_step_name_raises_error(self, simple_transformer: MockTransformer) -> None:
        """Test that empty step name raises ValueError."""
        with pytest.raises(ValueError, match="Step name cannot be empty"):
            Pipeline([("", simple_transformer)])

    def test_duplicate_step_names_raise_error(self, simple_transformer: MockTransformer) -> None:
        """Test that duplicate step names raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate step names"):
            Pipeline(
                [
                    ("step1", simple_transformer),
                    ("step1", MockTransformer()),
                ]
            )

    def test_non_transformer_step_raises_error(self) -> None:
        """Test that non-TraceTransformer step raises TypeError."""
        with pytest.raises(TypeError, match="must be TraceTransformer instances"):
            Pipeline([("invalid", "not a transformer")])  # type: ignore

    def test_none_transformer_raises_error(self) -> None:
        """Test that None transformer raises TypeError."""
        with pytest.raises(TypeError, match="must be TraceTransformer instances"):
            Pipeline([("invalid", None)])  # type: ignore

    def test_multiple_duplicate_names_identified(self) -> None:
        """Test that all duplicate names are identified."""
        with pytest.raises(ValueError, match="Duplicate step names"):
            Pipeline(
                [
                    ("dup", MockTransformer()),
                    ("dup", MockTransformer()),
                    ("dup", MockTransformer()),
                ]
            )


# ============================================================================
# Transform Tests
# ============================================================================


class TestPipelineTransform:
    """Tests for pipeline transform operation."""

    def test_single_step_transform(
        self, simple_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test transform with single step."""
        result = simple_pipeline.transform(sample_trace)

        # MockTransformer scales by 2.0 and adds 1.0
        expected = sample_trace.data * 2.0 + 1.0
        np.testing.assert_array_equal(result.data, expected)

    def test_multi_step_transform_chaining(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test that multi-step pipeline chains transformations."""
        result = multi_stage_pipeline.transform(sample_trace)

        # stage1: data * 2.0
        # stage2: data * 0.5 + 1.0
        # stage3: data - 0.5
        expected = ((sample_trace.data * 2.0) * 0.5 + 1.0) - 0.5
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_transform_preserves_metadata(
        self, simple_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test that transform preserves trace metadata."""
        result = simple_pipeline.transform(sample_trace)

        assert result.metadata is sample_trace.metadata
        assert result.metadata.sample_rate == sample_trace.metadata.sample_rate

    def test_transform_updates_intermediate_results(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test that transform caches intermediate results."""
        result = multi_stage_pipeline.transform(sample_trace)

        # Should have intermediate result for each stage
        assert "stage1" in multi_stage_pipeline._intermediate_results
        assert "stage2" in multi_stage_pipeline._intermediate_results
        assert "stage3" in multi_stage_pipeline._intermediate_results

    def test_transform_clears_previous_intermediates(
        self, simple_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test that transform clears previous intermediate results."""
        # First transform
        simple_pipeline.transform(sample_trace)
        first_intermediates = dict(simple_pipeline._intermediate_results)

        # Second transform should clear and rebuild
        simple_trace_2 = WaveformTrace(
            data=np.array([10.0, 20.0, 30.0]), metadata=TraceMetadata(sample_rate=1e6)
        )
        simple_pipeline.transform(simple_trace_2)

        # Intermediates should exist but be different
        assert simple_pipeline._intermediate_results.keys() == first_intermediates.keys()


# ============================================================================
# Fit Tests
# ============================================================================


class TestPipelineFit:
    """Tests for pipeline fit operation."""

    def test_fit_calls_fit_on_all_steps(self, sample_trace: WaveformTrace) -> None:
        """Test that fit calls fit on all transformers."""
        mock1 = Mock(spec=TraceTransformer)
        mock1.fit.return_value = mock1
        mock1.transform.return_value = sample_trace

        mock2 = Mock(spec=TraceTransformer)
        mock2.fit.return_value = mock2
        mock2.transform.return_value = sample_trace

        pipeline = Pipeline([("step1", mock1), ("step2", mock2)])
        result = pipeline.fit(sample_trace)

        assert result is pipeline
        assert mock1.fit.call_count == 1
        assert mock2.fit.call_count == 1

    def test_fit_chains_transformations(self, sample_trace: WaveformTrace) -> None:
        """Test that fit uses transformed output for next stage."""
        transformer1 = MockTransformer(scale_factor=2.0)
        transformer2 = StatefulTransformer()

        pipeline = Pipeline([("scale", transformer1), ("normalize", transformer2)])
        pipeline.fit(sample_trace)

        # transformer2 should have learned from scaled data
        expected_mean = np.mean(sample_trace.data * 2.0)
        assert transformer2.mean_ == pytest.approx(expected_mean)

    def test_fit_returns_self(self, simple_pipeline: Pipeline, sample_trace: WaveformTrace) -> None:
        """Test that fit returns self for method chaining."""
        result = simple_pipeline.fit(sample_trace)
        assert result is simple_pipeline

    def test_fit_with_stateful_transformer(self, sample_trace: WaveformTrace) -> None:
        """Test fit with stateful transformer."""
        stateful = StatefulTransformer()
        pipeline = Pipeline([("normalize", stateful)])

        pipeline.fit(sample_trace)

        assert stateful.mean_ is not None
        assert stateful.std_ is not None
        assert stateful.mean_ == pytest.approx(np.mean(sample_trace.data))


# ============================================================================
# Fit-Transform Tests
# ============================================================================


class TestPipelineFitTransform:
    """Tests for fit_transform pattern."""

    def test_fit_transform_inherited_from_base(
        self, simple_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test that fit_transform is inherited from TraceTransformer."""
        result = simple_pipeline.fit_transform(sample_trace)
        assert isinstance(result, WaveformTrace)

    def test_fit_transform_equals_fit_then_transform(self, sample_trace: WaveformTrace) -> None:
        """Test that fit_transform gives same result as fit then transform."""
        stateful = StatefulTransformer()
        pipeline1 = Pipeline([("normalize", stateful)])
        result1 = pipeline1.fit_transform(sample_trace)

        stateful2 = StatefulTransformer()
        pipeline2 = Pipeline([("normalize", stateful2)])
        pipeline2.fit(sample_trace)
        result2 = pipeline2.transform(sample_trace)

        np.testing.assert_array_almost_equal(result1.data, result2.data)


# ============================================================================
# Intermediate Result Access Tests
# ============================================================================


class TestIntermediateResults:
    """Tests for get_intermediate and intermediate result management."""

    def test_get_intermediate_returns_cached_trace(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test get_intermediate returns cached trace output."""
        multi_stage_pipeline.transform(sample_trace)

        stage1_output = multi_stage_pipeline.get_intermediate("stage1")
        assert isinstance(stage1_output, WaveformTrace)

        # stage1 scales by 2.0
        expected = sample_trace.data * 2.0
        np.testing.assert_array_equal(stage1_output.data, expected)

    def test_get_intermediate_before_transform_raises_error(
        self, multi_stage_pipeline: Pipeline
    ) -> None:
        """Test get_intermediate before transform raises KeyError."""
        with pytest.raises(KeyError, match="Call transform\\(\\) first"):
            multi_stage_pipeline.get_intermediate("stage1")

    def test_get_intermediate_invalid_step_raises_error(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test get_intermediate with invalid step name raises KeyError."""
        multi_stage_pipeline.transform(sample_trace)

        with pytest.raises(KeyError, match="not found in pipeline"):
            multi_stage_pipeline.get_intermediate("nonexistent")

    def test_get_intermediate_with_key_accesses_transformer(
        self, sample_trace: WaveformTrace
    ) -> None:
        """Test get_intermediate with key accesses transformer's intermediate."""
        intermediate_transformer = IntermediateTransformer()
        pipeline = Pipeline([("stats", intermediate_transformer)])

        pipeline.transform(sample_trace)

        mean = pipeline.get_intermediate("stats", "mean")
        assert mean == pytest.approx(np.mean(sample_trace.data))

        max_val = pipeline.get_intermediate("stats", "max")
        assert max_val == pytest.approx(np.max(sample_trace.data))

    def test_get_intermediate_all_stages(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test getting intermediates from all stages."""
        multi_stage_pipeline.transform(sample_trace)

        stage1 = multi_stage_pipeline.get_intermediate("stage1")
        stage2 = multi_stage_pipeline.get_intermediate("stage2")
        stage3 = multi_stage_pipeline.get_intermediate("stage3")

        assert isinstance(stage1, WaveformTrace)
        assert isinstance(stage2, WaveformTrace)
        assert isinstance(stage3, WaveformTrace)


class TestHasIntermediate:
    """Tests for has_intermediate method."""

    def test_has_intermediate_before_transform(self, simple_pipeline: Pipeline) -> None:
        """Test has_intermediate returns False before transform."""
        assert simple_pipeline.has_intermediate("scale") is False

    def test_has_intermediate_after_transform(
        self, simple_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test has_intermediate returns True after transform."""
        simple_pipeline.transform(sample_trace)
        assert simple_pipeline.has_intermediate("scale") is True

    def test_has_intermediate_with_key(self, sample_trace: WaveformTrace) -> None:
        """Test has_intermediate with key checks transformer intermediates."""
        intermediate_transformer = IntermediateTransformer()
        pipeline = Pipeline([("stats", intermediate_transformer)])

        pipeline.transform(sample_trace)

        assert pipeline.has_intermediate("stats", "mean") is True
        assert pipeline.has_intermediate("stats", "std") is True
        assert pipeline.has_intermediate("stats", "nonexistent") is False

    def test_has_intermediate_invalid_step(self, simple_pipeline: Pipeline) -> None:
        """Test has_intermediate with invalid step returns False."""
        assert simple_pipeline.has_intermediate("nonexistent") is False


class TestListIntermediates:
    """Tests for list_intermediates method."""

    def test_list_intermediates_for_step(self, sample_trace: WaveformTrace) -> None:
        """Test listing intermediates for specific step."""
        intermediate_transformer = IntermediateTransformer()
        pipeline = Pipeline([("stats", intermediate_transformer)])

        pipeline.transform(sample_trace)

        intermediates = pipeline.list_intermediates("stats")
        assert isinstance(intermediates, list)
        assert "mean" in intermediates
        assert "max" in intermediates
        assert "min" in intermediates
        assert "std" in intermediates

    def test_list_intermediates_all_steps(self, sample_trace: WaveformTrace) -> None:
        """Test listing intermediates for all steps."""
        pipeline = Pipeline(
            [
                ("stats1", IntermediateTransformer()),
                ("stats2", IntermediateTransformer()),
            ]
        )

        pipeline.transform(sample_trace)

        all_intermediates = pipeline.list_intermediates()
        assert isinstance(all_intermediates, dict)
        assert "stats1" in all_intermediates
        assert "stats2" in all_intermediates
        assert isinstance(all_intermediates["stats1"], list)

    def test_list_intermediates_excludes_steps_without_intermediates(
        self, sample_trace: WaveformTrace
    ) -> None:
        """Test that steps without intermediates are excluded."""
        pipeline = Pipeline(
            [
                ("transform", MockTransformer()),
                ("stats", IntermediateTransformer()),
            ]
        )

        pipeline.transform(sample_trace)

        all_intermediates = pipeline.list_intermediates()
        # MockTransformer doesn't cache intermediates
        assert "transform" not in all_intermediates
        assert "stats" in all_intermediates

    def test_list_intermediates_invalid_step_raises_error(self, simple_pipeline: Pipeline) -> None:
        """Test list_intermediates with invalid step raises KeyError."""
        with pytest.raises(KeyError, match="not found in pipeline"):
            simple_pipeline.list_intermediates("nonexistent")


# ============================================================================
# Parameter Get/Set Tests
# ============================================================================


class TestGetParams:
    """Tests for get_params method."""

    def test_get_params_shallow(self, simple_pipeline: Pipeline) -> None:
        """Test get_params with deep=False."""
        params = simple_pipeline.get_params(deep=False)

        assert "steps" in params
        assert params["steps"] == simple_pipeline.steps

    def test_get_params_deep(self, simple_pipeline: Pipeline) -> None:
        """Test get_params with deep=True includes transformer params."""
        params = simple_pipeline.get_params(deep=True)

        assert "steps" in params
        assert "scale__scale_factor" in params
        assert "scale__offset" in params
        assert params["scale__scale_factor"] == 2.0
        assert params["scale__offset"] == 1.0

    def test_get_params_multi_stage(self, multi_stage_pipeline: Pipeline) -> None:
        """Test get_params with multiple stages."""
        params = multi_stage_pipeline.get_params(deep=True)

        assert "stage1__scale_factor" in params
        assert "stage2__scale_factor" in params
        assert "stage2__offset" in params
        assert "stage3__offset" in params


class TestSetParams:
    """Tests for set_params method."""

    def test_set_params_updates_transformer(self, simple_pipeline: Pipeline) -> None:
        """Test set_params updates transformer parameters."""
        simple_pipeline.set_params(scale__scale_factor=5.0)

        transformer = simple_pipeline.named_steps["scale"]
        assert transformer.scale_factor == 5.0

    def test_set_params_returns_self(self, simple_pipeline: Pipeline) -> None:
        """Test set_params returns self for method chaining."""
        result = simple_pipeline.set_params(scale__scale_factor=3.0)
        assert result is simple_pipeline

    def test_set_params_multiple_parameters(self, simple_pipeline: Pipeline) -> None:
        """Test set_params with multiple parameters."""
        simple_pipeline.set_params(scale__scale_factor=3.0, scale__offset=2.0)

        transformer = simple_pipeline.named_steps["scale"]
        assert transformer.scale_factor == 3.0
        assert transformer.offset == 2.0

    def test_set_params_invalid_step_raises_error(self, simple_pipeline: Pipeline) -> None:
        """Test set_params with invalid step raises ValueError."""
        with pytest.raises(ValueError, match="Step 'invalid' not found"):
            simple_pipeline.set_params(invalid__param=1.0)

    def test_set_params_missing_delimiter_raises_error(self, simple_pipeline: Pipeline) -> None:
        """Test set_params without __ delimiter raises ValueError."""
        with pytest.raises(ValueError, match="must use 'step__param' syntax"):
            simple_pipeline.set_params(invalid_syntax=1.0)

    def test_set_params_steps_directly(self) -> None:
        """Test set_params can update steps directly."""
        pipeline = Pipeline([("step1", MockTransformer())])

        new_steps = [("new_step", MockTransformer(scale_factor=5.0))]
        pipeline.set_params(steps=new_steps)

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "new_step"
        assert "new_step" in pipeline.named_steps


# ============================================================================
# Clone Tests
# ============================================================================


class TestClone:
    """Tests for clone method."""

    def test_clone_creates_new_instance(self, simple_pipeline: Pipeline) -> None:
        """Test clone creates new pipeline instance."""
        cloned = simple_pipeline.clone()

        assert cloned is not simple_pipeline
        assert isinstance(cloned, Pipeline)

    def test_clone_preserves_structure(self, multi_stage_pipeline: Pipeline) -> None:
        """Test clone preserves pipeline structure."""
        cloned = multi_stage_pipeline.clone()

        assert len(cloned.steps) == len(multi_stage_pipeline.steps)
        assert list(cloned.named_steps.keys()) == list(multi_stage_pipeline.named_steps.keys())

    def test_clone_creates_new_transformers(self, simple_pipeline: Pipeline) -> None:
        """Test clone creates new transformer instances."""
        cloned = simple_pipeline.clone()

        original_transformer = simple_pipeline.named_steps["scale"]
        cloned_transformer = cloned.named_steps["scale"]

        assert cloned_transformer is not original_transformer
        assert cloned_transformer.scale_factor == original_transformer.scale_factor

    def test_clone_preserves_parameters(self, multi_stage_pipeline: Pipeline) -> None:
        """Test clone preserves transformer parameters."""
        cloned = multi_stage_pipeline.clone()

        for name in multi_stage_pipeline.named_steps:
            original = multi_stage_pipeline.named_steps[name]
            cloned_t = cloned.named_steps[name]

            assert cloned_t.scale_factor == original.scale_factor
            assert cloned_t.offset == original.offset

    def test_clone_independent_modification(self, simple_pipeline: Pipeline) -> None:
        """Test that modifying clone doesn't affect original."""
        cloned = simple_pipeline.clone()

        cloned.set_params(scale__scale_factor=10.0)

        assert simple_pipeline.named_steps["scale"].scale_factor == 2.0
        assert cloned.named_steps["scale"].scale_factor == 10.0


# ============================================================================
# Indexing and Access Tests
# ============================================================================


class TestIndexing:
    """Tests for __getitem__ and __len__."""

    def test_len_returns_step_count(self, multi_stage_pipeline: Pipeline) -> None:
        """Test len returns number of steps."""
        assert len(multi_stage_pipeline) == 3

    def test_getitem_by_index(self, multi_stage_pipeline: Pipeline) -> None:
        """Test accessing step by integer index."""
        first = multi_stage_pipeline[0]
        second = multi_stage_pipeline[1]

        assert isinstance(first, MockTransformer)
        assert isinstance(second, MockTransformer)
        assert first.scale_factor == 2.0
        assert second.scale_factor == 0.5

    def test_getitem_by_name(self, multi_stage_pipeline: Pipeline) -> None:
        """Test accessing step by string name."""
        stage1 = multi_stage_pipeline["stage1"]
        stage2 = multi_stage_pipeline["stage2"]

        assert isinstance(stage1, MockTransformer)
        assert stage1.scale_factor == 2.0
        assert stage2.scale_factor == 0.5

    def test_getitem_invalid_index_raises_error(self, simple_pipeline: Pipeline) -> None:
        """Test accessing invalid index raises IndexError."""
        with pytest.raises(IndexError):
            _ = simple_pipeline[10]

    def test_getitem_invalid_name_raises_error(self, simple_pipeline: Pipeline) -> None:
        """Test accessing invalid name raises KeyError."""
        with pytest.raises(KeyError):
            _ = simple_pipeline["nonexistent"]


# ============================================================================
# String Representation Tests
# ============================================================================


class TestStringRepresentation:
    """Tests for __repr__."""

    def test_repr_single_step(self, simple_pipeline: Pipeline) -> None:
        """Test repr with single step."""
        repr_str = repr(simple_pipeline)

        assert "Pipeline([" in repr_str
        assert "'scale'" in repr_str
        assert "MockTransformer" in repr_str

    def test_repr_multi_step(self, multi_stage_pipeline: Pipeline) -> None:
        """Test repr with multiple steps."""
        repr_str = repr(multi_stage_pipeline)

        assert "Pipeline([" in repr_str
        assert "'stage1'" in repr_str
        assert "'stage2'" in repr_str
        assert "'stage3'" in repr_str
        assert "MockTransformer" in repr_str

    def test_repr_format(self, simple_pipeline: Pipeline) -> None:
        """Test repr format is correct."""
        repr_str = repr(simple_pipeline)

        assert repr_str.startswith("Pipeline([")
        assert repr_str.endswith("])")
        assert "('scale', MockTransformer)" in repr_str


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestPipelinePipelineEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_pipeline_with_single_complex_trace(self, sine_trace: WaveformTrace) -> None:
        """Test pipeline handles complex sine wave correctly."""
        pipeline = Pipeline(
            [
                ("scale", MockTransformer(scale_factor=2.0)),
                ("offset", MockTransformer(offset=-1.0)),
            ]
        )

        result = pipeline.transform(sine_trace)

        expected = sine_trace.data * 2.0 - 1.0
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_nested_pipeline_as_step(self, sample_trace: WaveformTrace) -> None:
        """Test pipeline can contain another pipeline as a step."""
        inner_pipeline = Pipeline([("inner", MockTransformer(scale_factor=2.0))])
        outer_pipeline = Pipeline([("outer", inner_pipeline)])

        result = outer_pipeline.transform(sample_trace)

        expected = sample_trace.data * 2.0
        np.testing.assert_array_equal(result.data, expected)

    def test_transform_with_empty_trace(self, simple_pipeline: Pipeline) -> None:
        """Test transform handles empty trace data."""
        empty_trace = WaveformTrace(data=np.array([]), metadata=TraceMetadata(sample_rate=1e6))

        result = simple_pipeline.transform(empty_trace)

        assert len(result.data) == 0

    def test_transform_with_large_trace(self, simple_pipeline: Pipeline) -> None:
        """Test transform handles large trace efficiently."""
        large_data = np.random.randn(1_000_000)
        large_trace = WaveformTrace(data=large_data, metadata=TraceMetadata(sample_rate=1e6))

        result = simple_pipeline.transform(large_trace)

        assert len(result.data) == len(large_data)

    def test_multiple_transform_calls(
        self, simple_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test multiple transform calls work correctly."""
        result1 = simple_pipeline.transform(sample_trace)
        result2 = simple_pipeline.transform(sample_trace)

        np.testing.assert_array_equal(result1.data, result2.data)

    def test_fit_after_transform(
        self, sample_trace: WaveformTrace, sine_trace: WaveformTrace
    ) -> None:
        """Test fit can be called after transform."""
        stateful = StatefulTransformer()
        pipeline = Pipeline([("normalize", stateful)])

        # Transform first (should fail)
        with pytest.raises(ValueError, match="not fitted"):
            pipeline.transform(sample_trace)

        # Fit then transform
        pipeline.fit(sample_trace)
        result = pipeline.transform(sine_trace)

        assert isinstance(result, WaveformTrace)

    def test_intermediate_cleared_between_transforms(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test intermediates are cleared between transforms."""
        multi_stage_pipeline.transform(sample_trace)
        first_intermediate = multi_stage_pipeline._intermediate_results["stage1"].data.copy()

        # Transform with different data
        new_trace = WaveformTrace(
            data=np.array([10.0, 20.0, 30.0]), metadata=TraceMetadata(sample_rate=1e6)
        )
        multi_stage_pipeline.transform(new_trace)
        second_intermediate = multi_stage_pipeline._intermediate_results["stage1"].data

        # Intermediates should be different
        assert not np.array_equal(first_intermediate, second_intermediate)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for realistic pipeline usage."""

    def test_data_processing_pipeline(self, sample_trace: WaveformTrace) -> None:
        """Test realistic data processing pipeline."""
        pipeline = Pipeline(
            [
                ("scale", MockTransformer(scale_factor=0.5)),
                ("offset", MockTransformer(offset=2.0)),
                ("final_scale", MockTransformer(scale_factor=2.0)),
            ]
        )

        result = pipeline.transform(sample_trace)

        # (data * 0.5 + 2.0) * 2.0
        expected = (sample_trace.data * 0.5 + 2.0) * 2.0
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_fit_transform_workflow(self, sample_trace: WaveformTrace) -> None:
        """Test fit-transform workflow with stateful transformers."""
        # Create pipeline with stateful normalizer
        pipeline = Pipeline([("normalize", StatefulTransformer())])

        # Fit on reference trace
        pipeline.fit(sample_trace)

        # Transform new trace
        new_trace = WaveformTrace(
            data=np.array([6.0, 7.0, 8.0, 9.0, 10.0]),
            metadata=TraceMetadata(sample_rate=1e6),
        )
        result = pipeline.transform(new_trace)

        # Should be normalized using sample_trace statistics
        assert isinstance(result, WaveformTrace)

    def test_intermediate_extraction_workflow(self, sample_trace: WaveformTrace) -> None:
        """Test extracting intermediate results at each stage."""
        pipeline = Pipeline(
            [
                ("stage1", IntermediateTransformer()),
                ("stage2", MockTransformer(scale_factor=2.0)),
                ("stage3", IntermediateTransformer()),
            ]
        )

        pipeline.transform(sample_trace)

        # Extract intermediates from different stages
        stage1_mean = pipeline.get_intermediate("stage1", "mean")
        stage3_mean = pipeline.get_intermediate("stage3", "mean")

        # stage3 operates on scaled data
        assert stage3_mean == pytest.approx(stage1_mean * 2.0)

    def test_parameter_tuning_workflow(self, sample_trace: WaveformTrace) -> None:
        """Test parameter tuning workflow."""
        pipeline = Pipeline([("scale", MockTransformer(scale_factor=1.0))])

        # Get initial result
        result1 = pipeline.transform(sample_trace)

        # Tune parameters
        pipeline.set_params(scale__scale_factor=3.0)

        # Get new result
        result2 = pipeline.transform(sample_trace)

        # Results should differ
        assert not np.array_equal(result1.data, result2.data)
        np.testing.assert_array_almost_equal(result2.data, sample_trace.data * 3.0)

    def test_pipeline_cloning_for_experiments(
        self, multi_stage_pipeline: Pipeline, sample_trace: WaveformTrace
    ) -> None:
        """Test cloning pipeline for experimentation."""
        # Clone pipeline
        experiment = multi_stage_pipeline.clone()

        # Modify experiment parameters
        experiment.set_params(stage1__scale_factor=5.0)

        # Transform with both
        original_result = multi_stage_pipeline.transform(sample_trace)
        experiment_result = experiment.transform(sample_trace)

        # Results should differ
        assert not np.array_equal(original_result.data, experiment_result.data)

    def test_pipeline_serialization_compatibility(self, simple_pipeline: Pipeline) -> None:
        """Test pipeline is compatible with serialization."""
        import pickle

        # Serialize
        serialized = pickle.dumps(simple_pipeline)

        # Deserialize
        deserialized = pickle.loads(serialized)

        assert isinstance(deserialized, Pipeline)
        assert len(deserialized) == len(simple_pipeline)

    def test_mixed_stateful_stateless_pipeline(
        self, sample_trace: WaveformTrace, sine_trace: WaveformTrace
    ) -> None:
        """Test pipeline with mix of stateful and stateless transformers."""
        pipeline = Pipeline(
            [
                ("scale", MockTransformer(scale_factor=2.0)),
                ("normalize", StatefulTransformer()),
                ("offset", MockTransformer(offset=1.0)),
            ]
        )

        # Fit on sample trace
        pipeline.fit(sample_trace)

        # Transform sine trace
        result = pipeline.transform(sine_trace)

        assert isinstance(result, WaveformTrace)
        # Verify stateful step was fitted
        assert pipeline.named_steps["normalize"].mean_ is not None


# ============================================================================
# Performance and Stress Tests
# ============================================================================


class TestPerformance:
    """Performance and stress tests."""

    def test_many_pipeline_steps(self, sample_trace: WaveformTrace) -> None:
        """Test pipeline with many steps."""
        steps = [(f"step{i}", MockTransformer(scale_factor=1.0)) for i in range(50)]
        pipeline = Pipeline(steps)

        result = pipeline.transform(sample_trace)

        # After 50 identity transforms, should be unchanged
        np.testing.assert_array_almost_equal(result.data, sample_trace.data)

    def test_deep_nested_pipelines(self, sample_trace: WaveformTrace) -> None:
        """Test deeply nested pipelines."""
        inner = Pipeline([("inner", MockTransformer(scale_factor=2.0))])
        middle = Pipeline([("middle", inner)])
        outer = Pipeline([("outer", middle)])

        result = outer.transform(sample_trace)

        expected = sample_trace.data * 2.0
        np.testing.assert_array_equal(result.data, expected)

    def test_repeated_fit_calls(self, sample_trace: WaveformTrace) -> None:
        """Test repeated fit calls update state correctly."""
        stateful = StatefulTransformer()
        pipeline = Pipeline([("normalize", stateful)])

        # First fit
        pipeline.fit(sample_trace)
        first_mean = stateful.mean_

        # Second fit with different data
        new_trace = WaveformTrace(
            data=np.array([10.0, 20.0, 30.0]), metadata=TraceMetadata(sample_rate=1e6)
        )
        pipeline.fit(new_trace)
        second_mean = stateful.mean_

        assert first_mean != second_mean
