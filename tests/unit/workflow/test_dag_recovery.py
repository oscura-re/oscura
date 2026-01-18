"""Comprehensive tests for DAG workflow error handling and recovery.

Requirements tested:

This test suite covers:
- Error detection and reporting
- Graceful degradation
- Partial execution recovery
- Rollback mechanisms
- Error propagation patterns
- Retry strategies
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

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
def error_tracking_dag() -> WorkflowDAG:
    """DAG with error tracking capability."""
    dag = WorkflowDAG()
    # Store errors in state for testing
    dag._error_log: list[str] = []  # type: ignore
    return dag


@pytest.fixture
def flaky_task() -> tuple[Callable, Mock]:
    """Task that fails intermittently."""
    mock = Mock()
    attempt_count = {"count": 0}

    def task(state: dict[str, Any]) -> dict[str, Any]:
        attempt_count["count"] += 1
        mock(attempt_count["count"])
        if attempt_count["count"] < 3:
            raise RuntimeError(f"Attempt {attempt_count['count']} failed")
        return {"success": True, "attempts": attempt_count["count"]}

    return task, mock


# ============================================================================
# Error Detection Tests
# ============================================================================


class TestErrorDetection:
    """Tests for error detection in workflows."""

    def test_task_exception_detection(self, empty_dag: WorkflowDAG) -> None:
        """Test detection of exception in task execution."""

        def failing_task(state: dict[str, Any]) -> None:
            raise ValueError("Task error")

        empty_dag.add_task("fail", failing_task)

        with pytest.raises(AnalysisError, match="Task 'fail' failed"):
            empty_dag.execute()

    def test_exception_with_context(self, empty_dag: WorkflowDAG) -> None:
        """Test exception includes task context."""

        def contextual_failure(state: dict[str, Any]) -> None:
            value = state.get("value", 0)
            if value == 0:
                raise ValueError(f"Invalid value: {value}")

        empty_dag.add_task("check", contextual_failure)

        with pytest.raises(AnalysisError) as exc_info:
            empty_dag.execute(initial_state={"value": 0})

        assert "Invalid value: 0" in str(exc_info.value)
        assert "Task 'check' failed" in str(exc_info.value)

    def test_detect_failure_in_chain(self, empty_dag: WorkflowDAG) -> None:
        """Test detecting which task failed in a chain."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"step1": "ok"}

        def task2(state: dict[str, Any]) -> None:
            raise RuntimeError("Task 2 failed")

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            return {"step3": "ok"}

        empty_dag.add_task("t1", task1)
        empty_dag.add_task("t2", task2, depends_on=["t1"])
        empty_dag.add_task("t3", task3, depends_on=["t2"])

        with pytest.raises(AnalysisError, match="Task 't2' failed"):
            empty_dag.execute()

        # Verify t1 completed but t3 didn't
        assert empty_dag.tasks["t1"].completed is True
        assert empty_dag.tasks["t2"].completed is False
        assert empty_dag.tasks["t3"].completed is False

    def test_detect_parallel_task_failure(self, empty_dag: WorkflowDAG) -> None:
        """Test detecting failure in parallel task execution."""

        def good_task(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "ok"}

        def bad_task(state: dict[str, Any]) -> None:
            raise RuntimeError("Parallel task failed")

        empty_dag.add_task("good1", good_task)
        empty_dag.add_task("good2", good_task)
        empty_dag.add_task("bad", bad_task)

        with pytest.raises(AnalysisError, match="Task 'bad' failed"):
            empty_dag.execute(parallel=True)


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation patterns."""

    def test_optional_task_with_fallback(self) -> None:
        """Test handling optional task failure with fallback."""
        dag = WorkflowDAG()

        def required_task(state: dict[str, Any]) -> dict[str, Any]:
            return {"required_data": [1, 2, 3]}

        def optional_task_wrapper(state: dict[str, Any]) -> dict[str, Any]:
            """Wrapper that provides fallback for optional task."""
            try:
                # Simulate optional processing
                raise RuntimeError("Optional feature unavailable")
            except RuntimeError:
                # Fallback to default
                return {"optional_data": None, "used_fallback": True}

        def consumer_task(state: dict[str, Any]) -> dict[str, Any]:
            optional = state.get("optional_data")
            if optional is None:
                # Use alternative processing
                return {"result": "processed_without_optional"}
            return {"result": "processed_with_optional"}

        dag.add_task("required", required_task)
        dag.add_task("optional", optional_task_wrapper, depends_on=["required"])
        dag.add_task("consumer", consumer_task, depends_on=["required", "optional"])

        result = dag.execute()
        assert result["used_fallback"] is True
        assert result["result"] == "processed_without_optional"

    def test_partial_result_preservation(self) -> None:
        """Test preserving partial results when later stages fail."""
        dag = WorkflowDAG()
        partial_results: dict[str, Any] = {}

        def stage1(state: dict[str, Any]) -> dict[str, Any]:
            result = {"stage1_result": "completed"}
            partial_results.update(result)
            return result

        def stage2(state: dict[str, Any]) -> dict[str, Any]:
            result = {"stage2_result": "completed"}
            partial_results.update(result)
            return result

        def stage3(state: dict[str, Any]) -> None:
            raise ValueError("Stage 3 failed")

        dag.add_task("s1", stage1)
        dag.add_task("s2", stage2, depends_on=["s1"])
        dag.add_task("s3", stage3, depends_on=["s2"])

        with pytest.raises(AnalysisError):
            dag.execute()

        # Partial results should be preserved
        assert partial_results["stage1_result"] == "completed"
        assert partial_results["stage2_result"] == "completed"


# ============================================================================
# Partial Execution Recovery Tests
# ============================================================================


class TestPartialExecutionRecovery:
    """Tests for recovering from partial execution."""

    def test_resume_after_failure(self) -> None:
        """Test resuming workflow after fixing error."""
        dag = WorkflowDAG()
        should_fail = {"value": True}

        def stage1(state: dict[str, Any]) -> dict[str, Any]:
            return {"stage1": "done"}

        def stage2(state: dict[str, Any]) -> dict[str, Any]:
            if should_fail["value"]:
                raise RuntimeError("Stage 2 temporary failure")
            return {"stage2": "done"}

        def stage3(state: dict[str, Any]) -> dict[str, Any]:
            return {"stage3": "done"}

        dag.add_task("s1", stage1)
        dag.add_task("s2", stage2, depends_on=["s1"])
        dag.add_task("s3", stage3, depends_on=["s2"])

        # First attempt fails
        with pytest.raises(AnalysisError):
            dag.execute()

        # Fix the issue
        should_fail["value"] = False

        # Create new DAG instance for retry (simulates fixing and re-running)
        dag2 = WorkflowDAG()
        dag2.add_task("s1", stage1)
        dag2.add_task("s2", stage2, depends_on=["s1"])
        dag2.add_task("s3", stage3, depends_on=["s2"])

        # Second attempt succeeds
        result = dag2.execute()
        assert result["stage1"] == "done"
        assert result["stage2"] == "done"
        assert result["stage3"] == "done"

    def test_skip_completed_tasks(self) -> None:
        """Test pattern for skipping already-completed tasks."""
        dag = WorkflowDAG()
        completed_cache: set[str] = set()

        def cacheable_task(task_name: str) -> Callable:
            def task(state: dict[str, Any]) -> dict[str, Any]:
                if task_name in completed_cache:
                    # Return cached result
                    return state.get(f"{task_name}_cached", {})
                # Execute task
                result = {f"{task_name}_result": "computed"}
                completed_cache.add(task_name)
                # Cache result
                state[f"{task_name}_cached"] = result
                return result

            return task

        dag.add_task("t1", cacheable_task("t1"))
        dag.add_task("t2", cacheable_task("t2"), depends_on=["t1"])
        dag.add_task("t3", cacheable_task("t3"), depends_on=["t2"])

        # First execution
        state1 = dag.execute()
        assert "t1_result" in state1
        assert len(completed_cache) == 3

        # Reset and re-execute (tasks should use cache)
        dag.reset()
        state2 = dag.execute()
        assert state2 == {}  # All tasks returned empty (from cache)

    def test_checkpoint_and_restore(self) -> None:
        """Test checkpointing workflow state for recovery."""
        checkpoint: dict[str, Any] = {}

        def create_checkpoint_dag() -> WorkflowDAG:
            dag = WorkflowDAG()

            def stage1(state: dict[str, Any]) -> dict[str, Any]:
                result = {"stage1": "complete", "data": [1, 2, 3]}
                checkpoint.update(result)
                return result

            def stage2(state: dict[str, Any]) -> dict[str, Any]:
                result = {"stage2": "complete"}
                checkpoint.update(result)
                return result

            def stage3(state: dict[str, Any]) -> None:
                raise RuntimeError("Stage 3 failed")

            dag.add_task("s1", stage1)
            dag.add_task("s2", stage2, depends_on=["s1"])
            dag.add_task("s3", stage3, depends_on=["s2"])
            return dag

        # First attempt with failure
        dag1 = create_checkpoint_dag()
        with pytest.raises(AnalysisError):
            dag1.execute()

        # Restore from checkpoint
        assert checkpoint["stage1"] == "complete"
        assert checkpoint["stage2"] == "complete"
        assert checkpoint["data"] == [1, 2, 3]


# ============================================================================
# Rollback Tests
# ============================================================================


class TestRollback:
    """Tests for rollback mechanisms."""

    def test_rollback_on_cycle_detection(self) -> None:
        """Test DAG rolls back when cycle is detected."""

        def simple_task(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "success"}

        dag = WorkflowDAG()
        dag.add_task("A", simple_task)
        dag.add_task("B", simple_task, depends_on=["A"])

        initial_count = len(dag.tasks)

        # Force cycle detection
        original_has_cycle = dag._has_cycle

        def mock_has_cycle() -> bool:
            return True

        dag._has_cycle = mock_has_cycle  # type: ignore

        with pytest.raises(AnalysisError, match="would create a cycle"):
            dag.add_task("C", simple_task, depends_on=["B"])

        # Verify rollback occurred
        assert len(dag.tasks) == initial_count
        assert "C" not in dag.tasks

    def test_state_isolation_on_error(self) -> None:
        """Test that failed execution doesn't corrupt state."""
        dag = WorkflowDAG()

        def modify_state(state: dict[str, Any]) -> dict[str, Any]:
            state["modified"] = True
            raise RuntimeError("Fail after modification")

        dag.add_task("modify", modify_state)

        initial_state = {"value": 10}
        with pytest.raises(AnalysisError):
            dag.execute(initial_state=initial_state)

        # Original state should still be modified (passed by reference)
        # but task should not be marked complete
        assert dag.tasks["modify"].completed is False
        assert dag.tasks["modify"].result is None


# ============================================================================
# Error Propagation Tests
# ============================================================================


class TestErrorPropagation:
    """Tests for error propagation patterns."""

    def test_error_stops_dependent_tasks(self, empty_dag: WorkflowDAG) -> None:
        """Test that error prevents dependent tasks from executing."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"t1": "done"}

        def task2(state: dict[str, Any]) -> None:
            raise RuntimeError("Task 2 failed")

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            return {"t3": "done"}

        empty_dag.add_task("t1", task1)
        empty_dag.add_task("t2", task2, depends_on=["t1"])
        empty_dag.add_task("t3", task3, depends_on=["t2"])

        with pytest.raises(AnalysisError):
            empty_dag.execute()

        # t3 should not have executed
        assert empty_dag.tasks["t3"].completed is False
        assert empty_dag.tasks["t3"].result is None

    def test_parallel_error_propagation(self, empty_dag: WorkflowDAG) -> None:
        """Test error propagation in parallel execution."""

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"t1": "done"}

        def task2(state: dict[str, Any]) -> None:
            raise RuntimeError("Task 2 failed")

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            return {"t3": "done"}

        def task4(state: dict[str, Any]) -> dict[str, Any]:
            # Depends on both t2 and t3
            return {"t4": "done"}

        empty_dag.add_task("t1", task1)
        empty_dag.add_task("t2", task2, depends_on=["t1"])
        empty_dag.add_task("t3", task3, depends_on=["t1"])
        empty_dag.add_task("t4", task4, depends_on=["t2", "t3"])

        with pytest.raises(AnalysisError):
            empty_dag.execute(parallel=True)

        # t4 should not execute because t2 failed
        assert empty_dag.tasks["t4"].completed is False

    def test_exception_chain_preservation(self, empty_dag: WorkflowDAG) -> None:
        """Test that exception chain is preserved."""

        def task(state: dict[str, Any]) -> None:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Wrapper error") from e

        empty_dag.add_task("task", task)

        with pytest.raises(AnalysisError) as exc_info:
            empty_dag.execute()

        # Check that original exception is in the chain
        assert exc_info.value.__cause__ is not None
        assert "Wrapper error" in str(exc_info.value)


# ============================================================================
# Retry Strategy Tests
# ============================================================================


@pytest.mark.workflow
class TestRetryStrategies:
    """Tests for implementing retry strategies."""

    def test_simple_retry_wrapper(self, flaky_task: tuple[Callable, Mock]) -> None:
        """Test simple retry wrapper for tasks."""
        task_func, mock = flaky_task

        def retry_wrapper(func: Callable, max_attempts: int = 3) -> Callable:
            def wrapper(state: dict[str, Any]) -> dict[str, Any]:
                last_error = None
                for attempt in range(max_attempts):
                    try:
                        return func(state)
                    except Exception as e:
                        last_error = e
                        if attempt == max_attempts - 1:
                            raise
                        continue
                raise last_error  # type: ignore

            return wrapper

        dag = WorkflowDAG()
        dag.add_task("flaky", retry_wrapper(task_func, max_attempts=5))

        result = dag.execute()
        assert result["success"] is True
        assert result["attempts"] == 3
        assert mock.call_count == 3

    def test_exponential_backoff_retry(self) -> None:
        """Test retry with exponential backoff."""
        attempt_times: list[float] = []

        def flaky_with_timing(state: dict[str, Any]) -> dict[str, Any]:
            import time

            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise RuntimeError("Temporary failure")
            return {"success": True}

        def retry_with_backoff(
            func: Callable, max_attempts: int = 5, base_delay: float = 0.01
        ) -> Callable:
            def wrapper(state: dict[str, Any]) -> dict[str, Any]:
                import time

                for attempt in range(max_attempts):
                    try:
                        return func(state)
                    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
                        if attempt == max_attempts - 1:
                            raise
                        delay = base_delay * (2**attempt)
                        time.sleep(delay)
                        continue
                return {}

            return wrapper

        dag = WorkflowDAG()
        dag.add_task("flaky", retry_with_backoff(flaky_with_timing))

        result = dag.execute()
        assert result["success"] is True
        assert len(attempt_times) == 3

        # Verify exponential backoff occurred
        if len(attempt_times) >= 3:
            delay1 = attempt_times[1] - attempt_times[0]
            delay2 = attempt_times[2] - attempt_times[1]
            # Second delay should be roughly 2x first delay
            assert delay2 > delay1

    def test_conditional_retry(self) -> None:
        """Test retry based on exception type."""
        attempts = {"count": 0}

        def task_with_different_errors(state: dict[str, Any]) -> dict[str, Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise OSError("Temporary IO error")
            if attempts["count"] == 2:
                raise OSError("Another IO error")
            if attempts["count"] == 3:
                # This should not be retried
                raise ValueError("Logic error")
            return {"success": True}

        def selective_retry(func: Callable, retryable_exceptions: tuple) -> Callable:
            def wrapper(state: dict[str, Any]) -> dict[str, Any]:
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        return func(state)
                    except retryable_exceptions:
                        if attempt == max_attempts - 1:
                            raise
                        continue
                    except (ValueError, TypeError, RuntimeError, KeyError) as e:
                        # Non-retryable exception - re-raise immediately
                        raise
                return {}

            return wrapper

        dag = WorkflowDAG()
        dag.add_task("task", selective_retry(task_with_different_errors, (IOError,)))

        # Should fail on ValueError after retrying IOErrors
        with pytest.raises(AnalysisError, match="Logic error"):
            dag.execute()

        assert attempts["count"] == 3


# ============================================================================
# Error Recovery Integration Tests
# ============================================================================


@pytest.mark.workflow
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery scenarios."""

    def test_multi_stage_recovery(self) -> None:
        """Test recovery across multiple workflow stages."""
        execution_log: list[str] = []
        failure_mode = {"stage2": True}

        def stage1(state: dict[str, Any]) -> dict[str, Any]:
            execution_log.append("stage1")
            return {"stage1_done": True}

        def stage2(state: dict[str, Any]) -> dict[str, Any]:
            execution_log.append("stage2_attempt")
            if failure_mode["stage2"]:
                raise RuntimeError("Stage 2 failure")
            return {"stage2_done": True}

        def stage3(state: dict[str, Any]) -> dict[str, Any]:
            execution_log.append("stage3")
            return {"stage3_done": True}

        # First attempt
        dag1 = WorkflowDAG()
        dag1.add_task("s1", stage1)
        dag1.add_task("s2", stage2, depends_on=["s1"])
        dag1.add_task("s3", stage3, depends_on=["s2"])

        with pytest.raises(AnalysisError):
            dag1.execute()

        assert execution_log == ["stage1", "stage2_attempt"]

        # Fix and retry
        failure_mode["stage2"] = False
        dag2 = WorkflowDAG()
        dag2.add_task("s1", stage1)
        dag2.add_task("s2", stage2, depends_on=["s1"])
        dag2.add_task("s3", stage3, depends_on=["s2"])

        result = dag2.execute()
        assert result["stage3_done"] is True
        assert execution_log == ["stage1", "stage2_attempt", "stage1", "stage2_attempt", "stage3"]

    def test_cascading_failure_handling(self) -> None:
        """Test handling cascading failures gracefully."""
        dag = WorkflowDAG()

        def task1(state: dict[str, Any]) -> dict[str, Any]:
            return {"data": [1, 2, 3]}

        def task2(state: dict[str, Any]) -> dict[str, Any]:
            data = state.get("data", [])
            if not data:
                raise ValueError("No data available")
            return {"processed": len(data)}

        def task3(state: dict[str, Any]) -> dict[str, Any]:
            processed = state.get("processed", 0)
            if processed == 0:
                raise ValueError("Nothing to export")
            return {"exported": True}

        dag.add_task("t1", task1)
        dag.add_task("t2", task2, depends_on=["t1"])
        dag.add_task("t3", task3, depends_on=["t2"])

        # Should complete successfully
        result = dag.execute()
        assert result["exported"] is True

        # Now test with missing data
        dag.reset()

        def task1_empty(state: dict[str, Any]) -> dict[str, Any]:
            return {"data": []}

        dag2 = WorkflowDAG()
        dag2.add_task("t1", task1_empty)
        dag2.add_task("t2", task2, depends_on=["t1"])
        dag2.add_task("t3", task3, depends_on=["t2"])

        with pytest.raises(AnalysisError, match="No data available"):
            dag2.execute()
