"""Comprehensive tests for DAG workflow execution.

Requirements tested:

This test suite covers:
- DAG creation and node management
- Edge connections and dependencies
- Cycle detection
- Topological sorting
- Execution ordering (sequential and parallel)
- Error handling and edge cases
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from oscura.core.exceptions import AnalysisError
from oscura.workflow.dag import TaskNode, WorkflowDAG

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_task() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Simple task function for testing."""

    def task(state: dict[str, Any]) -> dict[str, Any]:
        return {"result": "success"}

    return task


@pytest.fixture
def stateful_task() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Task that reads and updates state."""

    def task(state: dict[str, Any]) -> dict[str, Any]:
        value = state.get("value", 0)
        return {"value": value + 1}

    return task


@pytest.fixture
def failing_task() -> Callable[[dict[str, Any]], None]:
    """Task that raises an exception."""

    def task(state: dict[str, Any]) -> None:
        raise ValueError("Task failed")

    return task


@pytest.fixture
def empty_dag() -> WorkflowDAG:
    """Empty DAG for testing."""
    return WorkflowDAG()


@pytest.fixture
def simple_dag(simple_task: Callable) -> WorkflowDAG:
    """DAG with a single task."""
    dag = WorkflowDAG()
    dag.add_task("task1", simple_task)
    return dag


@pytest.fixture
def linear_dag(simple_task: Callable) -> WorkflowDAG:
    """DAG with linear dependencies: task1 -> task2 -> task3."""
    dag = WorkflowDAG()

    def task1(state: dict[str, Any]) -> dict[str, Any]:
        return {"step1": "complete"}

    def task2(state: dict[str, Any]) -> dict[str, Any]:
        return {"step2": state.get("step1", "") + "_step2"}

    def task3(state: dict[str, Any]) -> dict[str, Any]:
        return {"step3": state.get("step2", "") + "_step3"}

    dag.add_task("task1", task1)
    dag.add_task("task2", task2, depends_on=["task1"])
    dag.add_task("task3", task3, depends_on=["task2"])
    return dag


@pytest.fixture
def diamond_dag() -> WorkflowDAG:
    """DAG with diamond dependency structure.

         task1
        /     \\
    task2    task3
        \\     /
         task4
    """
    dag = WorkflowDAG()

    def task1(state: dict[str, Any]) -> dict[str, Any]:
        return {"data": [1, 2, 3]}

    def task2(state: dict[str, Any]) -> dict[str, Any]:
        data = state.get("data", [])
        return {"sum": sum(data)}

    def task3(state: dict[str, Any]) -> dict[str, Any]:
        data = state.get("data", [])
        return {"product": 1 if not data else data[0] * data[1] * data[2]}

    def task4(state: dict[str, Any]) -> dict[str, Any]:
        return {"combined": state.get("sum", 0) + state.get("product", 0)}

    dag.add_task("task1", task1)
    dag.add_task("task2", task2, depends_on=["task1"])
    dag.add_task("task3", task3, depends_on=["task1"])
    dag.add_task("task4", task4, depends_on=["task2", "task3"])
    return dag


# ============================================================================
# TaskNode Tests
# ============================================================================


class TestTaskNode:
    """Tests for TaskNode dataclass."""

    def test_task_node_creation(self, simple_task: Callable) -> None:
        """Test creating a TaskNode with minimal parameters."""
        node = TaskNode(name="test_task", func=simple_task)

        assert node.name == "test_task"
        assert node.func == simple_task
        assert node.depends_on == []
        assert node.result is None
        assert node.completed is False

    def test_task_node_with_dependencies(self, simple_task: Callable) -> None:
        """Test creating a TaskNode with dependencies."""
        node = TaskNode(name="dependent_task", func=simple_task, depends_on=["task1", "task2"])

        assert node.depends_on == ["task1", "task2"]
        assert len(node.depends_on) == 2

    def test_task_node_state_tracking(self, simple_task: Callable) -> None:
        """Test TaskNode tracks execution state."""
        node = TaskNode(name="test_task", func=simple_task)

        # Initial state
        assert node.completed is False
        assert node.result is None

        # After execution
        node.result = {"data": 42}
        node.completed = True

        assert node.completed is True
        assert node.result == {"data": 42}


# ============================================================================
# WorkflowDAG Creation and Node Management Tests
# ============================================================================


class TestDAGCreation:
    """Tests for DAG creation and initialization."""

    def test_empty_dag_creation(self, empty_dag: WorkflowDAG) -> None:
        """Test creating an empty DAG."""
        assert len(empty_dag.tasks) == 0
        assert isinstance(empty_dag.tasks, dict)
        assert str(empty_dag) == "WorkflowDAG with 0 tasks:"
        assert repr(empty_dag) == "WorkflowDAG(tasks=0)"

    def test_add_single_task(self, empty_dag: WorkflowDAG, simple_task: Callable) -> None:
        """Test adding a single task to DAG."""
        empty_dag.add_task("task1", simple_task)

        assert len(empty_dag.tasks) == 1
        assert "task1" in empty_dag.tasks
        assert empty_dag.tasks["task1"].name == "task1"
        assert empty_dag.tasks["task1"].func == simple_task

    def test_add_multiple_tasks(self, empty_dag: WorkflowDAG, simple_task: Callable) -> None:
        """Test adding multiple independent tasks."""
        empty_dag.add_task("task1", simple_task)
        empty_dag.add_task("task2", simple_task)
        empty_dag.add_task("task3", simple_task)

        assert len(empty_dag.tasks) == 3
        assert all(name in empty_dag.tasks for name in ["task1", "task2", "task3"])

    def test_add_duplicate_task_raises_error(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test adding a task with duplicate name raises error."""
        empty_dag.add_task("task1", simple_task)

        with pytest.raises(AnalysisError, match="Task 'task1' already exists"):
            empty_dag.add_task("task1", simple_task)

    def test_add_task_with_none_depends_on(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test adding task with None depends_on uses empty list."""
        empty_dag.add_task("task1", simple_task, depends_on=None)

        assert empty_dag.tasks["task1"].depends_on == []


# ============================================================================
# Edge Connections and Dependencies Tests
# ============================================================================


class TestDAGDependencies:
    """Tests for managing task dependencies."""

    def test_add_task_with_dependencies(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test adding tasks with dependencies."""
        empty_dag.add_task("task1", simple_task)
        empty_dag.add_task("task2", simple_task, depends_on=["task1"])

        assert empty_dag.tasks["task2"].depends_on == ["task1"]
        assert "task2" in empty_dag._adjacency["task1"]

    def test_add_task_with_missing_dependency_raises_error(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test adding task with non-existent dependency raises error."""
        with pytest.raises(AnalysisError, match="Dependency 'nonexistent' not found"):
            empty_dag.add_task("task1", simple_task, depends_on=["nonexistent"])

    def test_add_task_with_multiple_dependencies(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test adding task with multiple dependencies."""
        empty_dag.add_task("task1", simple_task)
        empty_dag.add_task("task2", simple_task)
        empty_dag.add_task("task3", simple_task, depends_on=["task1", "task2"])

        assert empty_dag.tasks["task3"].depends_on == ["task1", "task2"]
        assert "task3" in empty_dag._adjacency["task1"]
        assert "task3" in empty_dag._adjacency["task2"]

    def test_adjacency_list_structure(self, diamond_dag: WorkflowDAG) -> None:
        """Test adjacency list is correctly built."""
        # task1 should have task2 and task3 as children
        assert set(diamond_dag._adjacency["task1"]) == {"task2", "task3"}

        # task2 and task3 should have task4 as child
        assert diamond_dag._adjacency["task2"] == ["task4"]
        assert diamond_dag._adjacency["task3"] == ["task4"]


# ============================================================================
# Cycle Detection Tests
# ============================================================================


class TestCycleDetection:
    """Tests for cycle detection in DAG."""

    def test_simple_cycle_detection(self, simple_task: Callable) -> None:
        """Test detecting a simple cycle: A -> B -> A."""
        # Test cycle detection by manually creating a cycle
        dag = WorkflowDAG()
        dag.add_task("A", simple_task)
        dag.add_task("B", simple_task, depends_on=["A"])

        # Manually create cycle A -> B -> A for testing _has_cycle
        dag.tasks["A"].depends_on.append("B")
        dag._adjacency["B"].append("A")

        assert dag._has_cycle() is True

    def test_no_cycle_in_linear_dag(self, linear_dag: WorkflowDAG) -> None:
        """Test linear DAG has no cycles."""
        assert linear_dag._has_cycle() is False

    def test_no_cycle_in_diamond_dag(self, diamond_dag: WorkflowDAG) -> None:
        """Test diamond DAG has no cycles."""
        assert diamond_dag._has_cycle() is False

    def test_self_cycle_detection(self, empty_dag: WorkflowDAG, simple_task: Callable) -> None:
        """Test detecting self-referential cycle."""
        empty_dag.add_task("A", simple_task)

        # Manually create self-cycle for testing
        empty_dag.tasks["A"].depends_on = ["A"]
        empty_dag._adjacency["A"].append("A")

        assert empty_dag._has_cycle() is True

    def test_cycle_rollback_on_detection(self, simple_task: Callable) -> None:
        """Test that DAG state is rolled back when cycle is detected."""
        # Create a scenario where add_task would create a cycle
        # We need to test the rollback mechanism in add_task when _has_cycle returns True

        # Create a custom DAG that we can manipulate
        dag = WorkflowDAG()
        dag.add_task("A", simple_task)
        dag.add_task("B", simple_task, depends_on=["A"])
        dag.add_task("C", simple_task, depends_on=["B"])

        # Monkey-patch _has_cycle to return True only for the next call
        original_has_cycle = dag._has_cycle
        call_count = [0]

        def mock_has_cycle() -> bool:
            call_count[0] += 1
            # Return True on second call (first is during add_task for D)
            if call_count[0] == 1:
                return True
            return original_has_cycle()

        dag._has_cycle = mock_has_cycle  # type: ignore

        # Try to add task D which will trigger cycle detection
        initial_task_count = len(dag.tasks)

        with pytest.raises(AnalysisError, match="would create a cycle"):
            dag.add_task("D", simple_task, depends_on=["C"])

        # Verify rollback occurred - task count should be unchanged
        assert len(dag.tasks) == initial_task_count
        assert "D" not in dag.tasks


# ============================================================================
# Topological Sorting Tests
# ============================================================================


class TestTopologicalSort:
    """Tests for topological sorting algorithm."""

    def test_topological_sort_single_task(self, simple_dag: WorkflowDAG) -> None:
        """Test topological sort with single task."""
        levels = simple_dag._topological_sort()

        assert len(levels) == 1
        assert levels[0] == ["task1"]

    def test_topological_sort_linear_dag(self, linear_dag: WorkflowDAG) -> None:
        """Test topological sort with linear dependencies."""
        levels = linear_dag._topological_sort()

        assert len(levels) == 3
        assert levels[0] == ["task1"]
        assert levels[1] == ["task2"]
        assert levels[2] == ["task3"]

    def test_topological_sort_diamond_dag(self, diamond_dag: WorkflowDAG) -> None:
        """Test topological sort with diamond structure."""
        levels = diamond_dag._topological_sort()

        assert len(levels) == 3
        assert levels[0] == ["task1"]
        assert set(levels[1]) == {"task2", "task3"}  # Can run in parallel
        assert levels[2] == ["task4"]

    def test_topological_sort_independent_tasks(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test topological sort with all independent tasks."""
        for i in range(5):
            empty_dag.add_task(f"task{i}", simple_task)

        levels = empty_dag._topological_sort()

        assert len(levels) == 1
        assert set(levels[0]) == {f"task{i}" for i in range(5)}

    def test_topological_sort_complex_dag(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test topological sort with complex dependency structure."""
        #     A
        #    / \\
        #   B   C
        #   |   |\\
        #   D   E F
        #    \\ |/
        #      G

        empty_dag.add_task("A", simple_task)
        empty_dag.add_task("B", simple_task, depends_on=["A"])
        empty_dag.add_task("C", simple_task, depends_on=["A"])
        empty_dag.add_task("D", simple_task, depends_on=["B"])
        empty_dag.add_task("E", simple_task, depends_on=["C"])
        empty_dag.add_task("F", simple_task, depends_on=["C"])
        empty_dag.add_task("G", simple_task, depends_on=["D", "E", "F"])

        levels = empty_dag._topological_sort()

        assert len(levels) == 4
        assert levels[0] == ["A"]
        assert set(levels[1]) == {"B", "C"}
        assert set(levels[2]) == {"D", "E", "F"}
        assert levels[3] == ["G"]

    def test_topological_sort_with_cycle_raises_error(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test topological sort raises error if cycle exists."""
        empty_dag.add_task("A", simple_task)
        empty_dag.add_task("B", simple_task, depends_on=["A"])

        # Manually create cycle
        empty_dag.tasks["A"].depends_on.append("B")
        empty_dag._adjacency["B"].append("A")

        with pytest.raises(AnalysisError, match="contains a cycle or unreachable tasks"):
            empty_dag._topological_sort()


# ============================================================================
# Execution Ordering Tests
# ============================================================================


class TestDAGExecution:
    """Tests for DAG execution."""

    def test_execute_empty_dag(self, empty_dag: WorkflowDAG) -> None:
        """Test executing empty DAG returns empty state."""
        result = empty_dag.execute()
        assert result == {}

    def test_execute_empty_dag_with_initial_state(self, empty_dag: WorkflowDAG) -> None:
        """Test executing empty DAG preserves initial state."""
        initial = {"key": "value"}
        result = empty_dag.execute(initial_state=initial)
        assert result == initial

    def test_execute_single_task(self, simple_dag: WorkflowDAG) -> None:
        """Test executing single task."""
        result = simple_dag.execute()

        assert "result" in result
        assert result["result"] == "success"
        assert simple_dag.tasks["task1"].completed is True

    def test_execute_linear_dag(self, linear_dag: WorkflowDAG) -> None:
        """Test executing linear DAG."""
        result = linear_dag.execute()

        assert result["step1"] == "complete"
        assert result["step2"] == "complete_step2"
        assert result["step3"] == "complete_step2_step3"

        # Verify all tasks completed
        assert all(task.completed for task in linear_dag.tasks.values())

    def test_execute_diamond_dag(self, diamond_dag: WorkflowDAG) -> None:
        """Test executing diamond DAG."""
        result = diamond_dag.execute()

        assert result["data"] == [1, 2, 3]
        assert result["sum"] == 6
        assert result["product"] == 6
        assert result["combined"] == 12

        # Verify all tasks completed
        assert all(task.completed for task in diamond_dag.tasks.values())

    def test_execute_with_initial_state(
        self, empty_dag: WorkflowDAG, stateful_task: Callable
    ) -> None:
        """Test execution with initial state."""
        empty_dag.add_task("increment", stateful_task)

        result = empty_dag.execute(initial_state={"value": 5})

        assert result["value"] == 6

    def test_execute_sequential_mode(self, diamond_dag: WorkflowDAG) -> None:
        """Test executing with parallel=False."""
        result = diamond_dag.execute(parallel=False)

        assert result["combined"] == 12
        assert all(task.completed for task in diamond_dag.tasks.values())

    def test_execute_parallel_mode(self, diamond_dag: WorkflowDAG) -> None:
        """Test executing with parallel=True."""
        result = diamond_dag.execute(parallel=True)

        assert result["combined"] == 12
        assert all(task.completed for task in diamond_dag.tasks.values())

    def test_execute_with_max_workers(self, diamond_dag: WorkflowDAG) -> None:
        """Test execution with custom max_workers."""
        result = diamond_dag.execute(parallel=True, max_workers=2)

        assert result["combined"] == 12

    def test_execute_task_failure_raises_error(
        self, empty_dag: WorkflowDAG, simple_task: Callable, failing_task: Callable
    ) -> None:
        """Test that task failure raises AnalysisError."""
        empty_dag.add_task("task1", simple_task)
        empty_dag.add_task("task2", failing_task, depends_on=["task1"])

        with pytest.raises(AnalysisError, match="Task 'task2' failed"):
            empty_dag.execute()

    def test_execute_updates_task_state(self, simple_dag: WorkflowDAG) -> None:
        """Test execution updates task completion state."""
        task = simple_dag.tasks["task1"]

        assert task.completed is False
        assert task.result is None

        simple_dag.execute()

        assert task.completed is True
        assert task.result is not None

    def test_execute_stores_dict_results_in_state(self, empty_dag: WorkflowDAG) -> None:
        """Test that dict results are merged into state."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            return {"key1": "value1", "key2": "value2"}

        empty_dag.add_task("task1", task)
        result = empty_dag.execute()

        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_execute_stores_non_dict_results_by_name(self, empty_dag: WorkflowDAG) -> None:
        """Test that non-dict results are stored with task name as key."""

        def task(state: dict[str, Any]) -> int:
            return 42

        empty_dag.add_task("task1", task)
        result = empty_dag.execute()

        assert result["task1"] == 42

    def test_execute_level_sequential(self, diamond_dag: WorkflowDAG) -> None:
        """Test _execute_level_sequential method."""
        state: dict[str, Any] = {}

        # Execute first level
        diamond_dag._execute_level_sequential(["task1"], state)

        assert state["data"] == [1, 2, 3]
        assert diamond_dag.tasks["task1"].completed is True

    def test_execute_level_parallel(self, diamond_dag: WorkflowDAG) -> None:
        """Test _execute_level_parallel method."""
        # First execute task1 to set up state
        state: dict[str, Any] = {}
        diamond_dag._execute_level_sequential(["task1"], state)

        # Then execute task2 and task3 in parallel
        diamond_dag._execute_level_parallel(["task2", "task3"], state, max_workers=2)

        assert state["sum"] == 6
        assert state["product"] == 6
        assert diamond_dag.tasks["task2"].completed is True
        assert diamond_dag.tasks["task3"].completed is True

    def test_execute_parallel_task_failure(
        self, empty_dag: WorkflowDAG, failing_task: Callable
    ) -> None:
        """Test parallel execution handles task failures."""
        empty_dag.add_task("task1", failing_task)
        empty_dag.add_task("task2", failing_task)

        with pytest.raises(AnalysisError, match=r"Task .* failed"):
            empty_dag.execute(parallel=True)

    def test_execute_parallel_with_non_dict_results(self, empty_dag: WorkflowDAG) -> None:
        """Test parallel execution with non-dict results."""

        def task1(state: dict[str, Any]) -> int:
            return 42

        def task2(state: dict[str, Any]) -> str:
            return "hello"

        empty_dag.add_task("task1", task1)
        empty_dag.add_task("task2", task2)

        result = empty_dag.execute(parallel=True)

        assert result["task1"] == 42
        assert result["task2"] == "hello"


# ============================================================================
# Result Retrieval Tests
# ============================================================================


class TestResultRetrieval:
    """Tests for retrieving task results."""

    def test_get_result_after_execution(self, simple_dag: WorkflowDAG) -> None:
        """Test getting result after task execution."""
        simple_dag.execute()

        result = simple_dag.get_result("task1")
        assert result == {"result": "success"}

    def test_get_result_before_execution_raises_error(self, simple_dag: WorkflowDAG) -> None:
        """Test getting result before execution raises error."""
        with pytest.raises(AnalysisError, match="has not been executed yet"):
            simple_dag.get_result("task1")

    def test_get_result_nonexistent_task_raises_error(self, simple_dag: WorkflowDAG) -> None:
        """Test getting result for non-existent task raises error."""
        with pytest.raises(AnalysisError, match="not found in DAG"):
            simple_dag.get_result("nonexistent")

    def test_get_result_from_diamond_dag(self, diamond_dag: WorkflowDAG) -> None:
        """Test getting results from multiple tasks."""
        diamond_dag.execute()

        assert diamond_dag.get_result("task1") == {"data": [1, 2, 3]}
        assert diamond_dag.get_result("task2") == {"sum": 6}
        assert diamond_dag.get_result("task3") == {"product": 6}
        assert diamond_dag.get_result("task4") == {"combined": 12}


# ============================================================================
# DAG Reset Tests
# ============================================================================


class TestDAGReset:
    """Tests for resetting DAG state."""

    def test_reset_clears_completion_state(self, simple_dag: WorkflowDAG) -> None:
        """Test reset clears task completion flags."""
        simple_dag.execute()

        assert simple_dag.tasks["task1"].completed is True
        assert simple_dag.tasks["task1"].result is not None

        simple_dag.reset()

        assert simple_dag.tasks["task1"].completed is False
        assert simple_dag.tasks["task1"].result is None

    def test_reset_all_tasks(self, diamond_dag: WorkflowDAG) -> None:
        """Test reset clears all tasks in DAG."""
        diamond_dag.execute()

        # All tasks should be completed
        assert all(task.completed for task in diamond_dag.tasks.values())

        diamond_dag.reset()

        # All tasks should be reset
        assert all(not task.completed for task in diamond_dag.tasks.values())
        assert all(task.result is None for task in diamond_dag.tasks.values())

    def test_reset_allows_reexecution(self, linear_dag: WorkflowDAG) -> None:
        """Test DAG can be re-executed after reset."""
        # First execution
        result1 = linear_dag.execute()
        assert result1["step3"] == "complete_step2_step3"

        # Reset
        linear_dag.reset()

        # Second execution with different initial state
        result2 = linear_dag.execute()
        assert result2["step3"] == "complete_step2_step3"

    def test_reset_preserves_dag_structure(self, diamond_dag: WorkflowDAG) -> None:
        """Test reset preserves DAG structure and dependencies."""
        diamond_dag.execute()

        task_count = len(diamond_dag.tasks)
        dependencies = {name: task.depends_on.copy() for name, task in diamond_dag.tasks.items()}

        diamond_dag.reset()

        # Structure should be unchanged
        assert len(diamond_dag.tasks) == task_count
        for name, task in diamond_dag.tasks.items():
            assert task.depends_on == dependencies[name]


# ============================================================================
# Visualization Tests
# ============================================================================


class TestDAGVisualization:
    """Tests for DAG visualization methods."""

    def test_to_graphviz_empty_dag(self, empty_dag: WorkflowDAG) -> None:
        """Test Graphviz output for empty DAG."""
        dot = empty_dag.to_graphviz()

        assert "digraph WorkflowDAG {" in dot
        assert "rankdir=LR;" in dot
        assert "}" in dot

    def test_to_graphviz_single_task(self, simple_dag: WorkflowDAG) -> None:
        """Test Graphviz output for single task."""
        dot = simple_dag.to_graphviz()

        assert "task1" in dot
        assert "lightblue" in dot  # Not completed
        assert "filled" in dot

    def test_to_graphviz_completed_task(self, simple_dag: WorkflowDAG) -> None:
        """Test Graphviz output shows completed tasks differently."""
        simple_dag.execute()
        dot = simple_dag.to_graphviz()

        assert "task1" in dot
        assert "lightgreen" in dot  # Completed
        assert "filled,bold" in dot

    def test_to_graphviz_with_dependencies(self, linear_dag: WorkflowDAG) -> None:
        """Test Graphviz output includes edges."""
        dot = linear_dag.to_graphviz()

        assert '"task1" -> "task2"' in dot
        assert '"task2" -> "task3"' in dot

    def test_to_graphviz_diamond_structure(self, diamond_dag: WorkflowDAG) -> None:
        """Test Graphviz output for diamond structure."""
        dot = diamond_dag.to_graphviz()

        assert '"task1" -> "task2"' in dot
        assert '"task1" -> "task3"' in dot
        assert '"task2" -> "task4"' in dot
        assert '"task3" -> "task4"' in dot


# ============================================================================
# String Representation Tests
# ============================================================================


class TestStringRepresentation:
    """Tests for __str__ and __repr__ methods."""

    def test_repr_empty_dag(self, empty_dag: WorkflowDAG) -> None:
        """Test repr of empty DAG."""
        assert repr(empty_dag) == "WorkflowDAG(tasks=0)"

    def test_repr_with_tasks(self, diamond_dag: WorkflowDAG) -> None:
        """Test repr with tasks."""
        assert repr(diamond_dag) == "WorkflowDAG(tasks=4)"

    def test_str_empty_dag(self, empty_dag: WorkflowDAG) -> None:
        """Test str of empty DAG."""
        s = str(empty_dag)
        assert "WorkflowDAG with 0 tasks:" in s

    def test_str_with_uncompleted_tasks(self, diamond_dag: WorkflowDAG) -> None:
        """Test str shows uncompleted tasks."""
        s = str(diamond_dag)

        assert "WorkflowDAG with 4 tasks:" in s
        assert "○ task1" in s  # Uncompleted marker
        assert "depends on: none" in s

    def test_str_with_completed_tasks(self, diamond_dag: WorkflowDAG) -> None:
        """Test str shows completed tasks."""
        diamond_dag.execute()
        s = str(diamond_dag)

        assert "✓ task1" in s  # Completed marker
        assert "✓ task2" in s
        assert "✓ task3" in s
        assert "✓ task4" in s

    def test_str_shows_dependencies(self, linear_dag: WorkflowDAG) -> None:
        """Test str shows task dependencies."""
        s = str(linear_dag)

        assert "task1 (depends on: none)" in s
        assert "task2 (depends on: task1)" in s
        assert "task3 (depends on: task2)" in s


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestWorkflowDagEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_task_with_empty_depends_on_list(
        self, empty_dag: WorkflowDAG, simple_task: Callable
    ) -> None:
        """Test task with empty depends_on list."""
        empty_dag.add_task("task1", simple_task, depends_on=[])

        assert empty_dag.tasks["task1"].depends_on == []

    def test_large_parallel_execution(self, empty_dag: WorkflowDAG) -> None:
        """Test executing many independent tasks in parallel."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        # Add 50 independent tasks
        for i in range(50):
            empty_dag.add_task(f"task{i}", task)

        result = empty_dag.execute(parallel=True, max_workers=4)

        assert all(empty_dag.tasks[f"task{i}"].completed for i in range(50))

    def test_deep_linear_dag(self, empty_dag: WorkflowDAG) -> None:
        """Test executing deep linear dependency chain."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            count = state.get("count", 0)
            return {"count": count + 1}

        # Create chain of 20 tasks
        empty_dag.add_task("task0", task)
        for i in range(1, 20):
            empty_dag.add_task(f"task{i}", task, depends_on=[f"task{i - 1}"])

        result = empty_dag.execute(initial_state={"count": 0})

        assert result["count"] == 20

    def test_task_returning_none(self, empty_dag: WorkflowDAG) -> None:
        """Test task that returns None."""

        def task(state: dict[str, Any]) -> None:
            return None

        empty_dag.add_task("task1", task)
        result = empty_dag.execute()

        assert result["task1"] is None

    def test_task_modifying_state_directly(self, empty_dag: WorkflowDAG) -> None:
        """Test task that modifies state dict directly."""

        def task(state: dict[str, Any]) -> dict[str, Any]:
            state["modified"] = True
            return {"result": "ok"}

        empty_dag.add_task("task1", task)
        result = empty_dag.execute()

        assert result["modified"] is True
        assert result["result"] == "ok"

    def test_exception_in_parallel_execution_stops_workflow(self, empty_dag: WorkflowDAG) -> None:
        """Test exception in one parallel task stops workflow."""

        def good_task(state: dict[str, Any]) -> dict[str, Any]:
            return {"status": "ok"}

        def bad_task(state: dict[str, Any]) -> None:
            raise RuntimeError("Task failed")

        empty_dag.add_task("good1", good_task)
        empty_dag.add_task("good2", good_task)
        empty_dag.add_task("bad", bad_task)

        with pytest.raises(AnalysisError, match="Task 'bad' failed"):
            empty_dag.execute(parallel=True)

    def test_sequential_execution_with_single_task_in_level(self, linear_dag: WorkflowDAG) -> None:
        """Test sequential execution handles single-task levels correctly."""
        # Linear DAG has one task per level
        result = linear_dag.execute(parallel=False)

        assert result["step3"] == "complete_step2_step3"

    def test_parallel_execution_disabled_for_single_task_levels(
        self, linear_dag: WorkflowDAG
    ) -> None:
        """Test parallel mode falls back to sequential for single-task levels."""
        # Even with parallel=True, single tasks execute sequentially
        result = linear_dag.execute(parallel=True)

        assert result["step3"] == "complete_step2_step3"


# ============================================================================
# Integration Tests
# ============================================================================


class TestDAGIntegration:
    """Integration tests for realistic workflows."""

    def test_data_processing_pipeline(self, empty_dag: WorkflowDAG) -> None:
        """Test realistic data processing pipeline."""

        def load_data(state: dict[str, Any]) -> dict[str, Any]:
            return {"data": [1, 2, 3, 4, 5]}

        def compute_mean(state: dict[str, Any]) -> dict[str, Any]:
            data = state["data"]
            return {"mean": sum(data) / len(data)}

        def compute_variance(state: dict[str, Any]) -> dict[str, Any]:
            data = state["data"]
            mean = state["mean"]
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return {"variance": variance}

        def compute_std(state: dict[str, Any]) -> dict[str, Any]:
            return {"std": state["variance"] ** 0.5}

        empty_dag.add_task("load", load_data)
        empty_dag.add_task("mean", compute_mean, depends_on=["load"])
        empty_dag.add_task("variance", compute_variance, depends_on=["load", "mean"])
        empty_dag.add_task("std", compute_std, depends_on=["variance"])

        result = empty_dag.execute()

        assert result["mean"] == 3.0
        assert result["variance"] == 2.0
        assert result["std"] == pytest.approx(1.414, rel=0.01)

    def test_multi_stage_analysis(self, empty_dag: WorkflowDAG) -> None:
        """Test multi-stage analysis workflow with parallel branches."""

        def preprocess(state: dict[str, Any]) -> dict[str, Any]:
            return {"signal": [1.0, 2.0, 3.0, 2.0, 1.0]}

        def fft_analysis(state: dict[str, Any]) -> dict[str, Any]:
            return {"fft_complete": True}

        def time_domain_analysis(state: dict[str, Any]) -> dict[str, Any]:
            return {"time_complete": True}

        def frequency_domain_analysis(state: dict[str, Any]) -> dict[str, Any]:
            return {"freq_complete": True}

        def generate_report(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "report": {
                    "fft": state.get("fft_complete", False),
                    "time": state.get("time_complete", False),
                    "freq": state.get("freq_complete", False),
                }
            }

        empty_dag.add_task("preprocess", preprocess)
        empty_dag.add_task("fft", fft_analysis, depends_on=["preprocess"])
        empty_dag.add_task("time", time_domain_analysis, depends_on=["preprocess"])
        empty_dag.add_task("freq", frequency_domain_analysis, depends_on=["fft"])
        empty_dag.add_task("report", generate_report, depends_on=["fft", "time", "freq"])

        result = empty_dag.execute(parallel=True)

        assert result["report"]["fft"] is True
        assert result["report"]["time"] is True
        assert result["report"]["freq"] is True
