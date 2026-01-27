"""Pattern search and anomaly detection for Oscura.


This module enables efficient pattern matching, anomaly detection, and
context extraction for debugging and analysis workflows.
"""

from oscura.utils.search.anomaly import find_anomalies
from oscura.utils.search.context import extract_context
from oscura.utils.search.pattern import find_pattern

__all__ = [
    "extract_context",
    "find_anomalies",
    "find_pattern",
]
