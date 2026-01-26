"""Intelligent trace comparison and difference detection.

This module provides automatic trace comparison with alignment, difference
detection, and plain-language explanations. It is a wrapper around the
discovery.comparison module to maintain API compatibility.


Example:
    >>> from oscura.utils.comparison import compare_traces
    >>> diff = compare_traces(trace1, trace2)
    >>> for d in diff.differences:
    ...     print(f"{d.category}: {d.description}")

References:
    Oscura Auto-Discovery Specification
    Phase 34 Task-245
"""

# Re-export everything from discovery.comparison
from oscura.discovery.comparison import (
    Difference,
    TraceDiff,
    compare_traces,
)

__all__ = [
    "Difference",
    "TraceDiff",
    "compare_traces",
]
