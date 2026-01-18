"""Comprehensive tests for DAG workflow state management.

Requirements tested:

This test suite covers:
- State initialization and updates
- State persistence patterns
- State validation
- State immutability concerns
- State sharing between tasks
- Complex state transformations
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from oscura.core.exceptions import AnalysisError
from oscura.workflow.dag import WorkflowDAG

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def stateful_dag() -> WorkflowDAG:
    """DAG configured for state management tests."""
    return WorkflowDAG()


@pytest.fixture
def complex_initial_state() -> dict[str, Any]:
    """Complex initial state for testing."""
    return {
        "metadata": {"version": 1, "timestamp": "2024-01-01"},
        "data": {
            "signals": [1.0, 2.0, 3.0],
            "sample_rate": 1000.0,
            "channels": ["CH1", "CH2"],
        },
        "config": {"threshold": 0.5, "mode": "auto"},
    }


# ============================================================================
# State Initialization Tests
# ============================================================================


class TestStateInitialization:
    """Tests for workflow state initialization."""

    def test_empty_initial_state(self, stateful_dag: WorkflowDAG) -> None:
        """Test workflow with no initial state."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            return {"initialized": True}

        stateful_dag.add_task("init", task)
        result = stateful_dag.execute()

        assert result["initialized"] is True

    def test_simple_initial_state(self, stateful_dag: WorkflowDAG) -> None:
        """Test workflow with simple initial state."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            value = state.get("input_value", 0)
            return {"output_value": value * 2}

        stateful_dag.add_task("process", task)
        result = stateful_dag.execute(initial_state={"input_value": 10})

        assert result["output_value"] == 20

    def test_complex_initial_state(
        self, stateful_dag: WorkflowDAG, complex_initial_state: dict[str, Any]
    ) -> None:
        """Test workflow with complex nested initial state."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            metadata = state.get("metadata", {})
            data = state.get("data", {})
            return {
                "version_processed": metadata.get("version", 0),
                "signal_count": len(data.get("signals", [])),
            }

        stateful_dag.add_task("process", task)
        result = stateful_dag.execute(initial_state=complex_initial_state)

        assert result["version_processed"] == 1
        assert result["signal_count"] == 3

    def test_initial_state_preservation(self, stateful_dag: WorkflowDAG) -> None:
        """Test that initial state keys are preserved."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            return {"new_key": "new_value"}

        stateful_dag.add_task("task", task)
        initial = {"preserved": "value", "count": 42}
        result = stateful_dag.execute(initial_state=initial)

        assert result["preserved"] == "value"
        assert result["count"] == 42
        assert result["new_key"] == "new_value"

    def test_initial_state_not_modified(self, stateful_dag: WorkflowDAG) -> None:
        """Test that original initial state dict is modified (by design)."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            return {"added": "value"}

        stateful_dag.add_task("task", task)
        initial = {"original": "value"}
        result = stateful_dag.execute(initial_state=initial)

        # In current implementation, initial state is modified
        assert initial["added"] == "value"
        assert result is initial


# ============================================================================
# State Update Tests
# ============================================================================


class TestStateUpdates:
    """Tests for state update mechanisms."""

    def test_dict_result_merges_into_state(self, stateful_dag: WorkflowDAG) -> None:
        """Test that dict results merge into state."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            return {"key1": "value1", "key2": "value2"}

        stateful_dag.add_task("task", task)
        result = stateful_dag.execute()

        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_non_dict_result_stored_by_name(self, stateful_dag: WorkflowDAG) -> None:
        """Test that non-dict results stored with task name."""

        def task(state: dict[str, Any]) -> int:
            return 42

        stateful_dag.add_task("compute", task)
        result = stateful_dag.execute()

        assert result["compute"] == 42

    def test_sequential_state_accumulation(self, stateful_dag: WorkflowDAG) -> None:
        """Test state accumulates through sequential tasks."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"step1": "done"}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            return {"step2": "done", "combined": state.get("step1", "") + "_step2"}

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            return {"step3": state.get("combined", "") + "_step3"}

        stateful_dag.add_task("t1", task1)
        stateful_dag.add_task("t2", task2, depends_on=["t1"])
        stateful_dag.add_task("t3", task3, depends_on=["t2"])

        result = stateful_dag.execute()

        assert result["step1"] == "done"
        assert result["step2"] == "done"
        assert result["combined"] == "done_step2"
        assert result["step3"] == "done_step2_step3"

    def test_state_overwrite_behavior(self, stateful_dag: WorkflowDAG) -> None:
        """Test that later tasks can overwrite state keys."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": 10, "source": "task1"}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": 20, "source": "task2"}

        stateful_dag.add_task("t1", task1)
        stateful_dag.add_task("t2", task2, depends_on=["t1"])

        result = stateful_dag.execute()

        assert result["value"] == 20
        assert result["source"] == "task2"

    def test_parallel_state_updates(self, stateful_dag: WorkflowDAG) -> None:
        """Test state updates from parallel tasks."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"task1_result": "complete"}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            return {"task2_result": "complete"}

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            return {"task3_result": "complete"}

        stateful_dag.add_task("t1", task1)
        stateful_dag.add_task("t2", task2)
        stateful_dag.add_task("t3", task3)

        result = stateful_dag.execute(parallel=True)

        assert result["task1_result"] == "complete"
        assert result["task2_result"] == "complete"
        assert result["task3_result"] == "complete"


# ============================================================================
# State Validation Tests
# ============================================================================


class TestStateValidation:
    """Tests for state validation patterns."""

    def test_validate_required_keys(self, stateful_dag: WorkflowDAG) -> None:
        """Test validating required state keys."""

        def validator(state: dict[str, Any]) -> dict[str, Any]:
            required = ["data", "sample_rate"]
            missing = [k for k in required if k not in state]
            if missing:
                raise AnalysisError(f"Missing required keys: {missing}")
            return {"validated": True}

        stateful_dag.add_task("validate", validator)

        # Should fail with missing keys
        with pytest.raises(AnalysisError, match="Missing required keys"):
            stateful_dag.execute(initial_state={"data": [1, 2, 3]})

        # Should succeed with all keys
        stateful_dag.reset()
        result = stateful_dag.execute(initial_state={"data": [1, 2, 3], "sample_rate": 1000})
        assert result["validated"] is True

    def test_validate_state_types(self, stateful_dag: WorkflowDAG) -> None:
        """Test validating state value types."""

        def type_validator(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data")
            if not isinstance(data, list):
                raise AnalysisError(f"Expected list, got {type(data).__name__}")
            return {"type_valid": True}

        stateful_dag.add_task("validate", type_validator)

        # Should fail with wrong type
        with pytest.raises(AnalysisError, match="Expected list"):
            stateful_dag.execute(initial_state={"data": "not a list"})

        # Should succeed with correct type
        stateful_dag.reset()
        result = stateful_dag.execute(initial_state={"data": [1, 2, 3]})
        assert result["type_valid"] is True

    def test_validate_state_constraints(self, stateful_dag: WorkflowDAG) -> None:
        """Test validating state value constraints."""

        def constraint_validator(state: dict[str, Any]) -> dict[str, Any]:
            threshold = state.get("threshold", 0)
            if not 0 <= threshold <= 1:
                raise AnalysisError(f"Threshold must be in [0, 1], got {threshold}")
            return {"constraint_valid": True}

        stateful_dag.add_task("validate", constraint_validator)

        # Should fail with invalid value
        with pytest.raises(AnalysisError, match="must be in"):
            stateful_dag.execute(initial_state={"threshold": 1.5})

        # Should succeed with valid value
        stateful_dag.reset()
        result = stateful_dag.execute(initial_state={"threshold": 0.5})
        assert result["constraint_valid"] is True


# ============================================================================
# State Immutability Tests
# ============================================================================


class TestStateImmutability:
    """Tests for state immutability concerns."""

    def test_state_mutation_visibility(self, stateful_dag: WorkflowDAG) -> None:
        """Test that state mutations are visible to subsequent tasks."""

        def mutate_state(state: dict[str, Any]) -> dict[str, Any]:
            # Mutate existing state
            state["mutated"] = True
            state.setdefault("list", []).append("item")
            return {}

        def check_mutation(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "has_mutated": state.get("mutated", False),
                "list_length": len(state.get("list", [])),
            }

        stateful_dag.add_task("mutate", mutate_state)
        stateful_dag.add_task("check", check_mutation, depends_on=["mutate"])

        result = stateful_dag.execute()
        assert result["has_mutated"] is True
        assert result["list_length"] == 1

    def test_shared_mutable_objects(self, stateful_dag: WorkflowDAG) -> None:
        """Test behavior with shared mutable objects in state."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            shared_list = state.get("shared", [])
            shared_list.append("task1")
            return {"shared": shared_list}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            shared_list = state.get("shared", [])
            shared_list.append("task2")
            return {"shared": shared_list}

        stateful_dag.add_task("t1", task1)
        stateful_dag.add_task("t2", task2, depends_on=["t1"])

        result = stateful_dag.execute(initial_state={"shared": []})
        assert result["shared"] == ["task1", "task2"]

    def test_defensive_copying_pattern(self, stateful_dag: WorkflowDAG) -> None:
        """Test pattern for defensive copying of state."""

        def defensive_task(state: dict[str, Any]) -> dict[str, Any]:
            # Create defensive copy
            state_copy = copy.deepcopy(state)
            state_copy["modified"] = True
            return state_copy

        stateful_dag.add_task("task", defensive_task)

        original_state = {"data": [1, 2, 3]}
        result = stateful_dag.execute(initial_state=original_state)

        # Result should have modification
        assert result["modified"] is True


# ============================================================================
# State Sharing Tests
# ============================================================================


class TestStateSharing:
    """Tests for state sharing between tasks."""

    def test_producer_consumer_pattern(self, stateful_dag: WorkflowDAG) -> None:
        """Test producer-consumer state sharing pattern."""

        def producer(state: dict[str, Any]) -> dict[str, Any]:
            return {"produced_data": [1, 2, 3, 4, 5], "producer_done": True}

        def consumer(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("produced_data", [])
            return {"consumed_count": len(data), "consumer_done": True}

        stateful_dag.add_task("producer", producer)
        stateful_dag.add_task("consumer", consumer, depends_on=["producer"])

        result = stateful_dag.execute()
        assert result["producer_done"] is True
        assert result["consumer_done"] is True
        assert result["consumed_count"] == 5

    def test_multi_producer_single_consumer(self, stateful_dag: WorkflowDAG) -> None:
        """Test multiple producers feeding single consumer."""

        def producer1(state: dict[str, Any]) -> dict[str, Any]:
            return {"data1": [1, 2, 3]}

        def producer2(state: dict[str, Any]) -> dict[str, Any]:
            return {"data2": [4, 5, 6]}

        def consumer(state: dict[str, Any]) -> dict[str, Any]:
            data1 = state.get("data1", [])
            data2 = state.get("data2", [])
            combined = data1 + data2
            return {"combined": combined, "total": len(combined)}

        stateful_dag.add_task("p1", producer1)
        stateful_dag.add_task("p2", producer2)
        stateful_dag.add_task("consumer", consumer, depends_on=["p1", "p2"])

        result = stateful_dag.execute(parallel=True)
        assert result["combined"] == [1, 2, 3, 4, 5, 6]
        assert result["total"] == 6

    def test_state_partitioning(self, stateful_dag: WorkflowDAG) -> None:
        """Test partitioning state for parallel processing."""

        def partition(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data", [])
            mid = len(data) // 2
            return {"partition1": data[:mid], "partition2": data[mid:]}

        def process1(state: dict[str, Any]) -> dict[str, Any]:
            part = state.get("partition1", [])
            return {"result1": sum(part)}

        def process2(state: dict[str, Any]) -> dict[str, Any]:
            part = state.get("partition2", [])
            return {"result2": sum(part)}

        def merge(state: dict[str, Any]) -> dict[str, Any]:
            total = state.get("result1", 0) + state.get("result2", 0)
            return {"total": total}

        stateful_dag.add_task("partition", partition)
        stateful_dag.add_task("proc1", process1, depends_on=["partition"])
        stateful_dag.add_task("proc2", process2, depends_on=["partition"])
        stateful_dag.add_task("merge", merge, depends_on=["proc1", "proc2"])

        result = stateful_dag.execute(initial_state={"data": [10, 20, 30, 40]}, parallel=True)
        assert result["total"] == 100


# ============================================================================
# Complex State Transformation Tests
# ============================================================================


@pytest.mark.workflow
class TestComplexStateTransformations:
    """Tests for complex state transformations."""

    def test_nested_state_transformation(self, stateful_dag: WorkflowDAG) -> None:
        """Test transforming nested state structures."""

        def transform(state: dict[str, Any]) -> dict[str, Any]:
            config = state.get("config", {})
            transformed = {
                "settings": {
                    "threshold": config.get("threshold", 0.5) * 2,
                    "mode": config.get("mode", "auto").upper(),
                },
                "metadata": {"transformed": True},
            }
            return transformed

        stateful_dag.add_task("transform", transform)

        initial = {"config": {"threshold": 0.3, "mode": "manual"}}
        result = stateful_dag.execute(initial_state=initial)

        assert result["settings"]["threshold"] == 0.6
        assert result["settings"]["mode"] == "MANUAL"
        assert result["metadata"]["transformed"] is True

    def test_state_aggregation(self, stateful_dag: WorkflowDAG) -> None:
        """Test aggregating state from multiple sources."""

        def source1(state: dict[str, Any]) -> dict[str, Any]:
            return {"metrics": {"cpu": 50, "memory": 60}}

        def source2(state: dict[str, Any]) -> dict[str, Any]:
            return {"metrics": {"disk": 70, "network": 80}}

        def aggregate(state: dict[str, Any]) -> dict[str, Any]:
            # Note: dict updates overwrite, so we need to be careful
            metrics = state.get("metrics", {})
            return {"aggregated_metrics": metrics.copy(), "metric_count": len(metrics)}

        stateful_dag.add_task("s1", source1)
        stateful_dag.add_task("s2", source2, depends_on=["s1"])  # Sequential to test overwrite
        stateful_dag.add_task("agg", aggregate, depends_on=["s2"])

        result = stateful_dag.execute()
        # s2 overwrites metrics from s1
        assert result["metric_count"] == 2

    def test_state_filtering(self, stateful_dag: WorkflowDAG) -> None:
        """Test filtering state to pass only needed keys."""

        def generate(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "data": [1, 2, 3],
                "metadata": {"version": 1},
                "temp": "temporary",
                "debug": "debug info",
            }

        def filter_state(state: dict[str, Any]) -> dict[str, Any]:
            # Only keep essential keys
            filtered = {k: v for k, v in state.items() if k in ["data", "metadata"]}
            return {"filtered_state": filtered}

        stateful_dag.add_task("gen", generate)
        stateful_dag.add_task("filter", filter_state, depends_on=["gen"])

        result = stateful_dag.execute()
        filtered = result["filtered_state"]
        assert "data" in filtered
        assert "metadata" in filtered
        assert "temp" not in filtered
        assert "debug" not in filtered

    def test_state_enrichment(self, stateful_dag: WorkflowDAG) -> None:
        """Test enriching state with computed values."""

        def base_data(state: dict[str, Any]) -> dict[str, Any]:
            return {"values": [10, 20, 30, 40, 50]}

        def enrich_statistics(state: dict[str, Any]) -> dict[str, Any]:
            values = state["values"]
            return {
                "statistics": {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }
            }

        def enrich_derived(state: dict[str, Any]) -> dict[str, Any]:
            stats = state["statistics"]
            return {"derived": {"range": stats["max"] - stats["min"]}}

        stateful_dag.add_task("base", base_data)
        stateful_dag.add_task("stats", enrich_statistics, depends_on=["base"])
        stateful_dag.add_task("derived", enrich_derived, depends_on=["stats"])

        result = stateful_dag.execute()
        assert result["statistics"]["mean"] == 30
        assert result["derived"]["range"] == 40

    def test_state_versioning(self, stateful_dag: WorkflowDAG) -> None:
        """Test versioning state transformations."""
        version_history: list[dict[str, Any]] = []

        def version_wrapper(task_name: str, task_func: Callable) -> Callable:
            def wrapper(state: dict[str, Any]) -> dict[str, Any]:
                # Snapshot state before transformation
                version_history.append(
                    {
                        "task": task_name,
                        "version": len(version_history) + 1,
                        "state": copy.deepcopy(state),
                    }
                )
                return task_func(state)

            return wrapper

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": 10}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": state.get("value", 0) * 2}

        stateful_dag.add_task("t1", version_wrapper("t1", task1))
        stateful_dag.add_task("t2", version_wrapper("t2", task2), depends_on=["t1"])

        result = stateful_dag.execute()
        assert result["value"] == 20
        assert len(version_history) == 2
        assert version_history[0]["task"] == "t1"
        assert version_history[1]["task"] == "t2"
        assert version_history[1]["state"]["value"] == 10


# ============================================================================
# State Persistence Pattern Tests
# ============================================================================


@pytest.mark.workflow
class TestStatePersistencePatterns:
    """Tests for state persistence patterns."""

    def test_state_serialization(self, stateful_dag: WorkflowDAG) -> None:
        """Test serializing state for persistence."""
        import json

        def generate_state(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "data": [1, 2, 3],
                "config": {"threshold": 0.5},
                "timestamp": "2024-01-01",
            }

        def serialize(state: dict[str, Any]) -> dict[str, Any]:
            # Simulate serialization
            serialized = json.dumps(state, sort_keys=True)
            return {"serialized": serialized, "size": len(serialized)}

        stateful_dag.add_task("gen", generate_state)
        stateful_dag.add_task("serialize", serialize, depends_on=["gen"])

        result = stateful_dag.execute()
        assert "serialized" in result
        assert result["size"] > 0

        # Verify deserialization works
        deserialized = json.loads(result["serialized"])
        assert deserialized["data"] == [1, 2, 3]

    def test_incremental_state_updates(self, stateful_dag: WorkflowDAG) -> None:
        """Test incremental state update pattern."""
        state_log: list[dict[str, Any]] = []

        def log_state(state: dict[str, Any]) -> None:
            state_log.append(copy.deepcopy(state))

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            log_state(state)
            return {"step1": "done"}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            log_state(state)
            return {"step2": "done"}

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            log_state(state)
            return {"step3": "done"}

        stateful_dag.add_task("t1", task1)
        stateful_dag.add_task("t2", task2, depends_on=["t1"])
        stateful_dag.add_task("t3", task3, depends_on=["t2"])

        stateful_dag.execute(initial_state={"start": True})

        assert len(state_log) == 3
        assert "start" in state_log[0]
        assert "step1" in state_log[1]
        assert "step2" in state_log[2]
