"""Comparison and limit testing module for Oscura.

This module provides waveform comparison, limit testing, mask testing,
and golden waveform comparison functionality.
"""

from oscura.utils.comparison.compare import (
    compare_traces,
    correlation,
    difference,
    similarity_score,
)
from oscura.utils.comparison.golden import (
    GoldenReference,
    compare_to_golden,
    create_golden,
    tolerance_envelope,
)
from oscura.utils.comparison.limits import (
    LimitSpec,
    check_limits,
    create_limit_spec,
    margin_analysis,
)
from oscura.utils.comparison.mask import (
    Mask,
    create_mask,
    eye_mask,
    mask_test,
)
from oscura.utils.comparison.trace_diff import (
    Difference,
    TraceDiff,
)

# Note: compare_traces is imported from both compare.py and trace_diff.py
# The trace_diff version is from discovery.comparison (intelligent comparison)
# Import as compare_traces_intelligent to avoid conflict
from oscura.utils.comparison.trace_diff import compare_traces as compare_traces_intelligent

__all__ = [
    # Intelligent trace diff (DISC-004)
    "Difference",
    # Golden reference
    "GoldenReference",
    # Limits
    "LimitSpec",
    # Mask testing
    "Mask",
    "TraceDiff",
    "check_limits",
    "compare_to_golden",
    # Comparison
    "compare_traces",
    "compare_traces_intelligent",
    "correlation",
    "create_golden",
    "create_limit_spec",
    "create_mask",
    "difference",
    "eye_mask",
    "margin_analysis",
    "mask_test",
    "similarity_score",
    "tolerance_envelope",
]
