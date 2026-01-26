"""High-level workflow presets for Oscura.

This module provides one-call analysis workflows for common signal
characterization tasks.

Example:
    >>> import oscura as osc
    >>> # Reverse engineer unknown signal
    >>> result = osc.workflows.reverse_engineer_signal(trace)
    >>> print(result.protocol_spec)
    >>>
    >>> # EMC compliance testing
    >>> results = osc.workflows.emc_compliance_test(trace)
    >>>
    >>> # Multi-trace analysis
    >>> stats = osc.workflows.load_all(["trace1.wfm", "trace2.wfm"])
"""

from oscura.workflows.complete_re import CompleteREResult, full_protocol_re
from oscura.workflows.compliance import emc_compliance_test
from oscura.workflows.digital import characterize_buffer
from oscura.workflows.multi_trace import (
    AlignmentMethod,
    MultiTraceResults,
    MultiTraceWorkflow,
    TraceStatistics,
    load_all,
)
from oscura.workflows.power import power_analysis
from oscura.workflows.protocol import debug_protocol
from oscura.workflows.reverse_engineering import (
    FieldSpec,
    InferredFrame,
    ProtocolSpec,
    ReverseEngineeringResult,
    reverse_engineer_signal,
)
from oscura.workflows.signal_integrity import signal_integrity_audit

__all__ = [
    # Multi-trace
    "AlignmentMethod",
    # Reverse engineering
    "CompleteREResult",
    "FieldSpec",
    "InferredFrame",
    "MultiTraceResults",
    "MultiTraceWorkflow",
    "ProtocolSpec",
    "ReverseEngineeringResult",
    "TraceStatistics",
    # Domain workflows
    "characterize_buffer",
    "debug_protocol",
    "emc_compliance_test",
    "full_protocol_re",
    "load_all",
    "power_analysis",
    "reverse_engineer_signal",
    "signal_integrity_audit",
]
