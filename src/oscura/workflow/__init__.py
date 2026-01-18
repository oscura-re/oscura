"""Workflow execution and DAG-based analysis.

This module provides directed acyclic graph (DAG) execution for complex
multi-step analysis workflows with automatic dependency resolution and
parallel execution.
"""

from oscura.workflow.dag import TaskNode, WorkflowDAG

__all__ = [
    "TaskNode",
    "WorkflowDAG",
]
