"""Comprehensive tests for DAG workflow composition and chaining.

Requirements tested:

This test suite covers:
- Composing multiple DAGs together
- Chaining workflow outputs
- Nested workflow execution
- Workflow reuse patterns
- State sharing between composed workflows
"""

from __future__ import annotations

from typing import Any

import pytest

from oscura.core.exceptions import AnalysisError
from oscura.workflow.dag import WorkflowDAG

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def preprocessing_dag() -> WorkflowDAG:
    """DAG for data preprocessing workflow."""
    dag = WorkflowDAG()

    def load_data(state: dict[str, Any]) -> dict[str, Any]:
        raw_data = state.get("input", [1, 2, 3, 4, 5])
        return {"raw_data": raw_data, "loaded": True}

    def normalize(state: dict[str, Any]) -> dict[str, Any]:
        data = state["raw_data"]
        max_val = max(data)
        normalized = [x / max_val for x in data]
        return {"normalized_data": normalized}

    def filter_noise(state: dict[str, Any]) -> dict[str, Any]:
        data = state["normalized_data"]
        # Simple threshold filter
        filtered = [x if x > 0.2 else 0.0 for x in data]
        return {"filtered_data": filtered, "preprocessing_complete": True}

    dag.add_task("load", load_data)
    dag.add_task("normalize", normalize, depends_on=["load"])
    dag.add_task("filter", filter_noise, depends_on=["normalize"])

    return dag


@pytest.fixture
def analysis_dag() -> WorkflowDAG:
    """DAG for data analysis workflow."""
    dag = WorkflowDAG()

    def compute_statistics(state: dict[str, Any]) -> dict[str, Any]:
        data = state.get("filtered_data", state.get("normalized_data", []))
        return {
            "mean": sum(data) / len(data) if data else 0,
            "max": max(data) if data else 0,
            "min": min(data) if data else 0,
        }

    def detect_peaks(state: dict[str, Any]) -> dict[str, Any]:
        # Simple peak detection
        return {"peaks": [1, 3], "peak_count": 2}

    def generate_summary(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": {
                "statistics": {
                    "mean": state.get("mean", 0),
                    "max": state.get("max", 0),
                },
                "peaks": state.get("peak_count", 0),
                "analysis_complete": True,
            }
        }

    dag.add_task("statistics", compute_statistics)
    dag.add_task("peaks", detect_peaks)
    dag.add_task("summary", generate_summary, depends_on=["statistics", "peaks"])

    return dag


@pytest.fixture
def export_dag() -> WorkflowDAG:
    """DAG for data export workflow."""
    dag = WorkflowDAG()

    def format_results(state: dict[str, Any]) -> dict[str, Any]:
        summary = state.get("summary", {})
        return {"formatted_results": {"data": summary, "format": "json"}}

    def validate_output(state: dict[str, Any]) -> dict[str, Any]:
        results = state.get("formatted_results", {})
        is_valid = "data" in results and "format" in results
        return {"validation_passed": is_valid}

    def export_data(state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("validation_passed", False):
            raise ValueError("Validation failed")
        return {"exported": True, "export_complete": True}

    dag.add_task("format", format_results)
    dag.add_task("validate", validate_output, depends_on=["format"])
    dag.add_task("export", export_data, depends_on=["validate"])

    return dag


# ============================================================================
# Workflow Composition Tests
# ============================================================================


class TestWorkflowComposition:
    """Tests for composing multiple workflows together."""

    def test_sequential_workflow_composition(
        self,
        preprocessing_dag: WorkflowDAG,
        analysis_dag: WorkflowDAG,
    ) -> None:
        """Test executing workflows in sequence with state passing."""
        # Execute preprocessing
        state = preprocessing_dag.execute(initial_state={"input": [10, 20, 30, 40, 50]})
        assert state["preprocessing_complete"] is True
        assert "filtered_data" in state

        # Execute analysis with preprocessing output
        final_state = analysis_dag.execute(initial_state=state)
        assert "summary" in final_state
        assert final_state["summary"]["analysis_complete"] is True

    def test_three_stage_pipeline(
        self,
        preprocessing_dag: WorkflowDAG,
        analysis_dag: WorkflowDAG,
        export_dag: WorkflowDAG,
    ) -> None:
        """Test three-stage pipeline composition."""
        # Stage 1: Preprocessing
        state = preprocessing_dag.execute(initial_state={"input": [5, 10, 15, 20, 25]})

        # Stage 2: Analysis
        state = analysis_dag.execute(initial_state=state)

        # Stage 3: Export
        final_state = export_dag.execute(initial_state=state)

        assert final_state["export_complete"] is True
        assert final_state["exported"] is True

    def test_composed_workflow_state_isolation(
        self,
        preprocessing_dag: WorkflowDAG,
        analysis_dag: WorkflowDAG,
    ) -> None:
        """Test that each workflow maintains state isolation."""
        # Execute preprocessing
        state1 = preprocessing_dag.execute(initial_state={"input": [1, 2, 3]})

        # Execute analysis separately
        state2 = analysis_dag.execute(initial_state={"filtered_data": [0.5, 1.0, 0.5]})

        # States should be independent
        assert "preprocessing_complete" in state1
        assert "preprocessing_complete" not in state2
        assert "summary" in state2
        assert "summary" not in state1

    def test_workflow_reuse_with_reset(self, preprocessing_dag: WorkflowDAG) -> None:
        """Test reusing a workflow with different inputs after reset."""
        # First execution
        state1 = preprocessing_dag.execute(initial_state={"input": [10, 20, 30]})
        result1 = state1["raw_data"]

        # Reset and re-execute
        preprocessing_dag.reset()
        state2 = preprocessing_dag.execute(initial_state={"input": [100, 200, 300]})
        result2 = state2["raw_data"]

        # Results should be different (raw data)
        assert result1 != result2
        assert result1 == [10, 20, 30]
        assert result2 == [100, 200, 300]

    def test_parallel_workflow_branches(self) -> None:
        """Test executing multiple independent workflow branches."""
        # Create two independent processing branches
        branch_a = WorkflowDAG()
        branch_b = WorkflowDAG()

        def process_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"result_a": state.get("value", 0) * 2}

        def process_b(state: dict[str, Any]) -> dict[str, Any]:
            return {"result_b": state.get("value", 0) + 10}

        branch_a.add_task("process_a", process_a)
        branch_b.add_task("process_b", process_b)

        # Execute both branches
        input_state = {"value": 5}
        state_a = branch_a.execute(initial_state=input_state)
        state_b = branch_b.execute(initial_state=input_state)

        # Combine results
        combined = {**state_a, **state_b}
        assert combined["result_a"] == 10
        assert combined["result_b"] == 15


# ============================================================================
# Workflow Chaining Tests
# ============================================================================


class TestWorkflowChaining:
    """Tests for chaining workflow outputs."""

    def test_chain_with_data_transformation(self) -> None:
        """Test chaining workflows with data transformation between stages."""
        # Stage 1: Generate data
        stage1 = WorkflowDAG()

        def generate(state: dict[str, Any]) -> dict[str, Any]:
            return {"numbers": [1, 2, 3, 4, 5]}

        stage1.add_task("generate", generate)

        # Stage 2: Transform data
        stage2 = WorkflowDAG()

        def square(state: dict[str, Any]) -> dict[str, Any]:
            nums = state["numbers"]
            return {"squares": [x**2 for x in nums]}

        def sum_squares(state: dict[str, Any]) -> dict[str, Any]:
            return {"sum": sum(state["squares"])}

        stage2.add_task("square", square)
        stage2.add_task("sum", sum_squares, depends_on=["square"])

        # Execute chain
        state = stage1.execute()
        final_state = stage2.execute(initial_state=state)

        assert final_state["sum"] == 55  # 1 + 4 + 9 + 16 + 25

    def test_conditional_workflow_chain(self) -> None:
        """Test chaining workflows with conditional execution."""
        # Preprocessing workflow
        preprocess = WorkflowDAG()

        def check_quality(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data", [])
            quality_ok = len(data) >= 3
            return {"quality_ok": quality_ok, "data": data}

        preprocess.add_task("quality_check", check_quality)

        # Main processing (only if quality OK)
        process = WorkflowDAG()

        def analyze(state: dict[str, Any]) -> dict[str, Any]:
            if not state.get("quality_ok", False):
                raise AnalysisError("Quality check failed")
            return {"analysis_result": "processed"}

        process.add_task("analyze", analyze)

        # Test with good data
        state1 = preprocess.execute(initial_state={"data": [1, 2, 3, 4]})
        result1 = process.execute(initial_state=state1)
        assert result1["analysis_result"] == "processed"

        # Test with bad data
        preprocess.reset()
        state2 = preprocess.execute(initial_state={"data": [1]})
        with pytest.raises(AnalysisError, match="Quality check failed"):
            process.execute(initial_state=state2)

    def test_workflow_chain_with_accumulation(self) -> None:
        """Test chaining workflows that accumulate state."""
        workflow = WorkflowDAG()

        def stage1(state: dict[str, Any]) -> dict[str, Any]:
            history = state.get("history", [])
            history.append("stage1")
            return {"history": history, "count": 1}

        def stage2(state: dict[str, Any]) -> dict[str, Any]:
            history = state["history"]
            history.append("stage2")
            count = state.get("count", 0) + 1
            return {"history": history, "count": count}

        def stage3(state: dict[str, Any]) -> dict[str, Any]:
            history = state["history"]
            history.append("stage3")
            count = state.get("count", 0) + 1
            return {"history": history, "count": count}

        workflow.add_task("s1", stage1)
        workflow.add_task("s2", stage2, depends_on=["s1"])
        workflow.add_task("s3", stage3, depends_on=["s2"])

        state = workflow.execute(initial_state={"history": []})

        assert state["history"] == ["stage1", "stage2", "stage3"]
        assert state["count"] == 3


# ============================================================================
# Nested Workflow Tests
# ============================================================================


class TestNestedWorkflows:
    """Tests for nested workflow execution patterns."""

    def test_workflow_as_task(self) -> None:
        """Test using a workflow as a task in another workflow."""
        # Inner workflow
        inner_dag = WorkflowDAG()

        def inner_task(state: dict[str, Any]) -> dict[str, Any]:
            return {"inner_result": state.get("value", 0) * 2}

        inner_dag.add_task("inner", inner_task)

        # Outer workflow that executes inner workflow
        outer_dag = WorkflowDAG()

        def setup(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": 10}

        def execute_inner(state: dict[str, Any]) -> dict[str, Any]:
            # Execute inner workflow
            inner_state = inner_dag.execute(initial_state=state)
            return {"nested_result": inner_state["inner_result"]}

        def finalize(state: dict[str, Any]) -> dict[str, Any]:
            return {"final": state["nested_result"] + 5}

        outer_dag.add_task("setup", setup)
        outer_dag.add_task("inner_workflow", execute_inner, depends_on=["setup"])
        outer_dag.add_task("finalize", finalize, depends_on=["inner_workflow"])

        result = outer_dag.execute()
        assert result["final"] == 25  # (10 * 2) + 5

    def test_multi_level_nesting(self) -> None:
        """Test multiple levels of workflow nesting."""
        # Level 3: Innermost
        level3 = WorkflowDAG()

        def l3_task(state: dict[str, Any]) -> dict[str, Any]:
            return {"l3": state.get("x", 0) + 1}

        level3.add_task("l3", l3_task)

        # Level 2: Middle
        level2 = WorkflowDAG()

        def l2_task(state: dict[str, Any]) -> dict[str, Any]:
            inner_state = level3.execute(initial_state=state)
            return {"l2": inner_state["l3"] + 1}

        level2.add_task("l2", l2_task)

        # Level 1: Outer
        level1 = WorkflowDAG()

        def l1_task(state: dict[str, Any]) -> dict[str, Any]:
            inner_state = level2.execute(initial_state=state)
            return {"l1": inner_state["l2"] + 1}

        level1.add_task("l1", l1_task)

        result = level1.execute(initial_state={"x": 0})
        assert result["l1"] == 3  # 0 + 1 + 1 + 1

    def test_nested_workflow_error_propagation(self) -> None:
        """Test error propagation from nested workflows."""
        # Inner workflow that fails
        inner = WorkflowDAG()

        def failing_task(state: dict[str, Any]) -> None:
            raise ValueError("Inner workflow error")

        inner.add_task("fail", failing_task)

        # Outer workflow
        outer = WorkflowDAG()

        def call_inner(state: dict[str, Any]) -> dict[str, Any]:
            return inner.execute(initial_state=state)

        outer.add_task("call_inner", call_inner)

        # Error should propagate
        with pytest.raises(AnalysisError, match="Inner workflow error"):
            outer.execute()


# ============================================================================
# State Sharing Tests
# ============================================================================


class TestWorkflowStateSharing:
    """Tests for state sharing patterns between workflows."""

    def test_shared_state_object(self) -> None:
        """Test multiple workflows operating on shared state."""
        shared_state: dict[str, Any] = {"counter": 0, "items": []}

        # Workflow 1: Increment counter
        wf1 = WorkflowDAG()

        def increment(state: dict[str, Any]) -> dict[str, Any]:
            state["counter"] += 1
            return state

        wf1.add_task("inc", increment)

        # Workflow 2: Add item
        wf2 = WorkflowDAG()

        def add_item(state: dict[str, Any]) -> dict[str, Any]:
            state["items"].append(f"item_{state['counter']}")
            return state

        wf2.add_task("add", add_item)

        # Execute workflows alternately
        for _ in range(3):
            wf1.execute(initial_state=shared_state)
            wf2.execute(initial_state=shared_state)
            wf1.reset()
            wf2.reset()

        assert shared_state["counter"] == 3
        assert shared_state["items"] == ["item_1", "item_2", "item_3"]

    def test_state_merging_from_multiple_workflows(self) -> None:
        """Test merging state from multiple independent workflows."""
        # Workflow A: Compute statistics
        wf_stats = WorkflowDAG()

        def compute_stats(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data", [])
            return {"mean": sum(data) / len(data) if data else 0}

        wf_stats.add_task("stats", compute_stats)

        # Workflow B: Detect anomalies
        wf_anomaly = WorkflowDAG()

        def detect_anomaly(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data", [])
            anomalies = [x for x in data if x > 100]
            return {"anomaly_count": len(anomalies)}

        wf_anomaly.add_task("anomaly", detect_anomaly)

        # Execute both with same input
        input_data = {"data": [10, 20, 150, 30, 200]}
        state_a = wf_stats.execute(initial_state=input_data)
        state_b = wf_anomaly.execute(initial_state=input_data)

        # Merge results
        merged = {**input_data, **state_a, **state_b}
        assert merged["mean"] == 82.0
        assert merged["anomaly_count"] == 2

    def test_state_transformation_chain(self) -> None:
        """Test chaining workflows that transform state format."""
        # Stage 1: Raw to structured
        stage1 = WorkflowDAG()

        def structurize(state: dict[str, Any]) -> dict[str, Any]:
            raw = state.get("raw", "")
            return {"structured": {"type": "data", "content": raw}}

        stage1.add_task("struct", structurize)

        # Stage 2: Structured to enriched
        stage2 = WorkflowDAG()

        def enrich(state: dict[str, Any]) -> dict[str, Any]:
            struct = state.get("structured", {})
            return {
                "enriched": {
                    **struct,
                    "metadata": {"processed": True},
                }
            }

        stage2.add_task("enrich", enrich)

        # Stage 3: Enriched to final
        stage3 = WorkflowDAG()

        def finalize(state: dict[str, Any]) -> dict[str, Any]:
            enriched = state.get("enriched", {})
            return {"final": {"data": enriched, "version": 1}}

        stage3.add_task("final", finalize)

        # Execute chain
        state = stage1.execute(initial_state={"raw": "test_data"})
        state = stage2.execute(initial_state=state)
        state = stage3.execute(initial_state=state)

        assert state["final"]["version"] == 1
        assert state["final"]["data"]["type"] == "data"
        assert state["final"]["data"]["metadata"]["processed"] is True


# ============================================================================
# Workflow Reuse Patterns Tests
# ============================================================================


@pytest.mark.workflow
class TestWorkflowReusePatterns:
    """Tests for workflow reuse patterns."""

    def test_workflow_template_pattern(self) -> None:
        """Test using workflows as reusable templates."""

        def create_processing_workflow(multiplier: int) -> WorkflowDAG:
            """Factory function to create parameterized workflows."""
            dag = WorkflowDAG()

            def process(state: dict[str, Any]) -> dict[str, Any]:
                value = state.get("value", 0)
                return {"result": value * multiplier}

            dag.add_task("process", process)
            return dag

        # Create multiple instances with different parameters
        wf_double = create_processing_workflow(2)
        wf_triple = create_processing_workflow(3)

        # Use different input states to avoid shared state
        result_double = wf_double.execute(initial_state={"value": 10})
        result_triple = wf_triple.execute(initial_state={"value": 10})

        assert result_double["result"] == 20
        assert result_triple["result"] == 30

    def test_workflow_library_pattern(self) -> None:
        """Test maintaining a library of reusable workflow components."""
        # Library of common workflows
        workflow_library: dict[str, WorkflowDAG] = {}

        # Register common workflows
        def register_validation_workflow() -> None:
            dag = WorkflowDAG()

            def validate(state: dict[str, Any]) -> dict[str, Any]:
                data = state.get("data", [])
                is_valid = len(data) > 0 and all(isinstance(x, int | float) for x in data)
                return {"valid": is_valid}

            dag.add_task("validate", validate)
            workflow_library["validation"] = dag

        def register_normalization_workflow() -> None:
            dag = WorkflowDAG()

            def normalize(state: dict[str, Any]) -> dict[str, Any]:
                data = state.get("data", [])
                if not data:
                    return {"normalized": []}
                max_val = max(data)
                return {"normalized": [x / max_val for x in data]}

            dag.add_task("normalize", normalize)
            workflow_library["normalization"] = dag

        # Build library
        register_validation_workflow()
        register_normalization_workflow()

        # Use library workflows
        test_data = {"data": [10, 20, 30, 40]}

        validation_result = workflow_library["validation"].execute(initial_state=test_data)
        assert validation_result["valid"] is True

        workflow_library["normalization"].reset()
        normalization_result = workflow_library["normalization"].execute(initial_state=test_data)
        assert normalization_result["normalized"] == [0.25, 0.5, 0.75, 1.0]

    def test_configurable_workflow_pattern(self) -> None:
        """Test workflows that adapt based on configuration."""

        def create_configurable_workflow(config: dict[str, Any]) -> WorkflowDAG:
            dag = WorkflowDAG()

            def load(state: dict[str, Any]) -> dict[str, Any]:
                return {"data": state.get("input", [])}

            dag.add_task("load", load)

            if config.get("enable_filtering", False):

                def filter_data(state: dict[str, Any]) -> dict[str, Any]:
                    threshold = config.get("filter_threshold", 0.5)
                    data = state["data"]
                    filtered = [x for x in data if x > threshold]
                    return {"data": filtered}

                dag.add_task("filter", filter_data, depends_on=["load"])

            if config.get("enable_transform", False):

                def transform(state: dict[str, Any]) -> dict[str, Any]:
                    scale = config.get("transform_scale", 1.0)
                    data = state["data"]
                    transformed = [x * scale for x in data]
                    return {"data": transformed}

                deps = ["filter"] if config.get("enable_filtering") else ["load"]
                dag.add_task("transform", transform, depends_on=deps)

            return dag

        # Configuration 1: No filtering, with transform
        config1 = {"enable_filtering": False, "enable_transform": True, "transform_scale": 2.0}
        wf1 = create_configurable_workflow(config1)
        result1 = wf1.execute(initial_state={"input": [1, 2, 3]})
        assert result1["data"] == [2.0, 4.0, 6.0]

        # Configuration 2: With filtering, no transform
        config2 = {"enable_filtering": True, "filter_threshold": 1.5}
        wf2 = create_configurable_workflow(config2)
        result2 = wf2.execute(initial_state={"input": [1, 2, 3]})
        assert result2["data"] == [2, 3]

        # Configuration 3: Both filtering and transform
        config3 = {
            "enable_filtering": True,
            "enable_transform": True,
            "filter_threshold": 1.5,
            "transform_scale": 10.0,
        }
        wf3 = create_configurable_workflow(config3)
        result3 = wf3.execute(initial_state={"input": [1, 2, 3]})
        assert result3["data"] == [20.0, 30.0]


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.workflow
class TestWorkflowCompositionIntegration:
    """Integration tests for realistic workflow composition scenarios."""

    def test_complete_data_pipeline(self) -> None:
        """Test complete end-to-end data processing pipeline."""
        # Stage 1: Data acquisition
        acquisition = WorkflowDAG()

        def acquire(state: dict[str, Any]) -> dict[str, Any]:
            # Simulate data acquisition
            return {"raw_signal": [10, 15, 20, 100, 25, 30, 35]}

        acquisition.add_task("acquire", acquire)

        # Stage 2: Preprocessing
        preprocessing = WorkflowDAG()

        def remove_outliers(state: dict[str, Any]) -> dict[str, Any]:
            data = state["raw_signal"]
            # Simple outlier removal (> 3x median)
            median = sorted(data)[len(data) // 2]
            clean = [x for x in data if x <= median * 3]
            return {"clean_signal": clean}

        def downsample(state: dict[str, Any]) -> dict[str, Any]:
            data = state["clean_signal"]
            # Take every other sample
            downsampled = data[::2]
            return {"processed_signal": downsampled}

        preprocessing.add_task("outliers", remove_outliers)
        preprocessing.add_task("downsample", downsample, depends_on=["outliers"])

        # Stage 3: Analysis
        analysis = WorkflowDAG()

        def compute_metrics(state: dict[str, Any]) -> dict[str, Any]:
            data = state["processed_signal"]
            return {
                "metrics": {
                    "mean": sum(data) / len(data),
                    "samples": len(data),
                }
            }

        analysis.add_task("metrics", compute_metrics)

        # Execute complete pipeline
        state = acquisition.execute()
        state = preprocessing.execute(initial_state=state)
        final_state = analysis.execute(initial_state=state)

        assert "metrics" in final_state
        assert final_state["metrics"]["samples"] == 3
        # Downsampled [10, 20, 30] -> mean = 20
        assert final_state["metrics"]["mean"] == pytest.approx(20.0, rel=0.01)

    def test_branching_convergent_pipeline(self) -> None:
        """Test pipeline with branching and convergence."""
        # Input stage
        input_wf = WorkflowDAG()

        def load_input(state: dict[str, Any]) -> dict[str, Any]:
            return {"signal": [1.0, 2.0, 3.0, 4.0, 5.0]}

        input_wf.add_task("load", load_input)

        # Branch A: Time domain analysis
        time_domain = WorkflowDAG()

        def time_analysis(state: dict[str, Any]) -> dict[str, Any]:
            signal = state["signal"]
            return {"time_metrics": {"peak": max(signal), "samples": len(signal)}}

        time_domain.add_task("time", time_analysis)

        # Branch B: Frequency domain analysis
        freq_domain = WorkflowDAG()

        def freq_analysis(state: dict[str, Any]) -> dict[str, Any]:
            signal = state["signal"]
            # Simplified frequency analysis
            return {"freq_metrics": {"dc_component": sum(signal) / len(signal)}}

        freq_domain.add_task("freq", freq_analysis)

        # Convergence: Combine results
        combine_wf = WorkflowDAG()

        def combine_results(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "complete_analysis": {
                    "time": state.get("time_metrics", {}),
                    "frequency": state.get("freq_metrics", {}),
                }
            }

        combine_wf.add_task("combine", combine_results)

        # Execute pipeline
        state = input_wf.execute()
        time_state = time_domain.execute(initial_state=state)
        freq_state = freq_domain.execute(initial_state=state)

        # Merge branch results
        merged_state = {**state, **time_state, **freq_state}
        final_state = combine_wf.execute(initial_state=merged_state)

        assert final_state["complete_analysis"]["time"]["peak"] == 5.0
        assert final_state["complete_analysis"]["frequency"]["dc_component"] == 3.0
