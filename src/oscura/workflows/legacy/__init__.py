"""Legacy workflow patterns for backward compatibility.

This module provides legacy workflow execution patterns that have been
superseded by modern composition patterns. Kept for backward compatibility
with existing code and tests.

For new code, use the modern workflow patterns in the parent module.
"""

from oscura.workflows.legacy.dag import TaskNode, WorkflowDAG

__all__ = ["TaskNode", "WorkflowDAG"]
