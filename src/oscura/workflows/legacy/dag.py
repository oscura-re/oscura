"""Directed Acyclic Graph (DAG) execution for complex workflows.

This module provides DAG-based workflow execution with automatic dependency
resolution and parallel execution of independent tasks.

NOTE: This is legacy API for backward compatibility.
For new code, use the modern workflow composition patterns.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TaskNode:
    """Node in a workflow DAG.

    Attributes:
        name: Unique task name.
        func: Callable function to execute.
        depends_on: List of task names this task depends on.
        result: Computed result (set after execution).
        completed: Whether task has been executed.

    Example:
        >>> def compute_fft(state):
        ...     return {'fft': np.fft.fft(state['trace'])}
        >>> node = TaskNode(name='fft', func=compute_fft, depends_on=['load'])

    References:
        API-013: DAG Execution
    """

    name: str
    func: Callable[[dict[str, Any]], Any]
    depends_on: list[str] = field(default_factory=list)
    result: Any = None
    completed: bool = False


class WorkflowDAG:
    """Directed Acyclic Graph for workflow execution.

    Manages task dependencies and executes tasks in topological order
    with automatic parallelization of independent tasks.

    Example:
        >>> from oscura.workflows.legacy.dag import WorkflowDAG
        >>> dag = WorkflowDAG()
        >>> dag.add_task('load', load_trace, depends_on=[])
        >>> dag.add_task('fft', compute_fft, depends_on=['load'])
        >>> dag.add_task('rise_time', compute_rise_time, depends_on=['load'])
        >>> dag.add_task('enob', compute_enob, depends_on=['fft', 'rise_time'])
        >>> results = dag.execute()

    References:
        API-013: DAG Execution for Complex Workflows
    """

    def __init__(self) -> None:
        """Initialize empty DAG."""
        self.tasks: dict[str, TaskNode] = {}
        self._adjacency: dict[str, list[str]] = defaultdict(list)

    def add_task(
        self,
        name: str,
        func: Callable[[dict[str, Any]], Any],
        depends_on: list[str] | None = None,
    ) -> None:
        """Add a task to the DAG.

        Args:
            name: Unique name for the task.
            func: Function to execute. Should accept state dict and return result.
            depends_on: List of task names this task depends on.

        Raises:
            AnalysisError: If task name already exists or creates a cycle.

        Example:
            >>> dag.add_task('fft', compute_fft, depends_on=['load'])

        References:
            API-013: DAG Execution
        """
        if name in self.tasks:
            raise AnalysisError(f"Task '{name}' already exists in DAG")

        depends_on = depends_on or []

        # Verify dependencies exist
        for dep in depends_on:
            if dep not in self.tasks:
                raise AnalysisError(f"Dependency '{dep}' not found for task '{name}'")

        # Create task node
        task = TaskNode(name=name, func=func, depends_on=depends_on)
        self.tasks[name] = task

        # Update adjacency list
        for dep in depends_on:
            self._adjacency[dep].append(name)

        # Check for cycles
        if self._has_cycle():
            # Rollback - remove task
            del self.tasks[name]
            for dep in depends_on:
                self._adjacency[dep].remove(name)
            raise AnalysisError(f"Adding task '{name}' would create a cycle in DAG")

    def _has_cycle(self) -> bool:
        """Check if DAG contains a cycle.

        Returns:
            True if cycle detected.
        """
        # Use DFS to detect cycles
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        return any(task_name not in visited and dfs(task_name) for task_name in self.tasks)

    def _topological_sort(self) -> list[list[str]]:
        """Compute topological sort grouped by execution level.

        Tasks at the same level can be executed in parallel.

        Returns:
            List of levels, where each level is a list of task names.

        Raises:
            AnalysisError: If DAG contains a cycle or unreachable tasks.

        Example:
            >>> levels = dag._topological_sort()
            >>> # [[load], [fft, rise_time], [enob]]
        """
        # Compute in-degree for each node
        in_degree = {name: len(task.depends_on) for name, task in self.tasks.items()}

        # Find nodes with no dependencies (level 0)
        levels: list[list[str]] = []
        queue = deque([name for name, degree in in_degree.items() if degree == 0])

        while queue:
            # All tasks at this level can run in parallel
            current_level = list(queue)
            levels.append(current_level)
            queue.clear()

            # Process current level
            for task_name in current_level:
                # Reduce in-degree of dependent tasks
                for dependent in self._adjacency.get(task_name, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Verify all tasks were included
        total_tasks = sum(len(level) for level in levels)
        if total_tasks != len(self.tasks):
            raise AnalysisError("DAG contains a cycle or unreachable tasks")

        return levels

    def execute(
        self,
        *,
        initial_state: dict[str, Any] | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Execute the workflow DAG.

        Tasks are executed in topological order with automatic parallelization
        of independent tasks at each level.

        Args:
            initial_state: Initial state dictionary passed to first tasks.
            parallel: Enable parallel execution of independent tasks.
            max_workers: Maximum number of parallel workers. None uses CPU count.

        Returns:
            Final state dictionary containing all task results.

        Example:
            >>> results = dag.execute(initial_state={'trace': trace_data})
            >>> print(results['enob'])

        References:
            API-013: DAG Execution
        """
        if not self.tasks:
            return initial_state or {}

        state = initial_state or {}
        levels = self._topological_sort()

        for level in levels:
            if parallel and len(level) > 1:
                # Execute level in parallel
                self._execute_level_parallel(level, state, max_workers)
            else:
                # Execute level sequentially
                self._execute_level_sequential(level, state)

        return state

    def _execute_level_sequential(self, level: list[str], state: dict[str, Any]) -> None:
        """Execute a level of tasks sequentially.

        Args:
            level: List of task names to execute.
            state: Shared state dictionary.

        Raises:
            AnalysisError: If task execution fails.
        """
        for task_name in level:
            task = self.tasks[task_name]
            try:
                result = task.func(state)
                task.result = result
                task.completed = True

                # Update state with result
                if isinstance(result, dict):
                    state.update(result)
                else:
                    state[task_name] = result

            except Exception as e:
                raise AnalysisError(f"Task '{task_name}' failed: {e}") from e

    def _execute_level_parallel(
        self, level: list[str], state: dict[str, Any], max_workers: int | None
    ) -> None:
        """Execute a level of tasks in parallel.

        Args:
            level: List of task names to execute in parallel.
            state: Shared state dictionary.
            max_workers: Maximum number of workers.

        Raises:
            AnalysisError: If task execution fails.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self.tasks[name].func, state): name for name in level}

            # Collect results
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                task = self.tasks[task_name]

                try:
                    result = future.result()
                    task.result = result
                    task.completed = True

                    # Update state with result
                    if isinstance(result, dict):
                        state.update(result)
                    else:
                        state[task_name] = result

                except Exception as e:
                    raise AnalysisError(f"Task '{task_name}' failed: {e}") from e

    def get_result(self, task_name: str) -> Any:
        """Get result from a completed task.

        Args:
            task_name: Name of the task.

        Returns:
            Task result.

        Raises:
            AnalysisError: If task doesn't exist or hasn't been executed.

        Example:
            >>> fft_result = dag.get_result('fft')
        """
        if task_name not in self.tasks:
            raise AnalysisError(f"Task '{task_name}' not found in DAG")

        task = self.tasks[task_name]
        if not task.completed:
            raise AnalysisError(f"Task '{task_name}' has not been executed yet")

        return task.result

    def reset(self) -> None:
        """Reset all task completion states.

        Allows re-execution of the DAG with different initial state.

        Example:
            >>> dag.reset()
            >>> results = dag.execute(initial_state={'trace': new_trace})
        """
        for task in self.tasks.values():
            task.completed = False
            task.result = None

    def to_graphviz(self) -> str:
        """Generate Graphviz DOT representation of the DAG.

        Returns:
            DOT format string for visualization.

        Example:
            >>> dot = dag.to_graphviz()
            >>> with open('workflow.dot', 'w') as f:
            ...     f.write(dot)
            >>> # Then: dot -Tpng workflow.dot -o workflow.png

        References:
            API-013: DAG Execution
        """
        lines = ["digraph WorkflowDAG {", "  rankdir=LR;", "  node [shape=box];", ""]

        # Add nodes
        for task_name, task in self.tasks.items():
            style = "filled,bold" if task.completed else "filled"
            color = "lightgreen" if task.completed else "lightblue"
            lines.append(f'  "{task_name}" [style="{style}", fillcolor="{color}"];')

        lines.append("")

        # Add edges
        for task_name, task in self.tasks.items():
            for dep in task.depends_on:
                lines.append(f'  "{dep}" -> "{task_name}";')

        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of DAG."""
        return f"WorkflowDAG(tasks={len(self.tasks)})"

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [f"WorkflowDAG with {len(self.tasks)} tasks:"]
        for task_name, task in self.tasks.items():
            deps = ", ".join(task.depends_on) if task.depends_on else "none"
            status = "✓" if task.completed else "○"
            lines.append(f"  {status} {task_name} (depends on: {deps})")
        return "\n".join(lines)


__all__ = ["TaskNode", "WorkflowDAG"]
