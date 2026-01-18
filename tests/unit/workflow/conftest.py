"""Pytest configuration and fixtures for workflow tests.


NOTE: Markers are defined in pyproject.toml. Do NOT add pytest_configure
hooks here to avoid conflicts with the root conftest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from oscura.workflow.dag import WorkflowDAG

if TYPE_CHECKING:
    from collections.abc import Callable


# ============================================================================
# Basic Fixtures
# ============================================================================


@pytest.fixture
def empty_dag() -> WorkflowDAG:
    """Empty DAG for testing."""
    return WorkflowDAG()


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


# ============================================================================
# DAG Fixtures
# ============================================================================


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
