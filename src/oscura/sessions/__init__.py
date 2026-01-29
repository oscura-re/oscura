"""Unified session management for Oscura.

This module provides the AnalysisSession hierarchy - a unified pattern for
interactive signal analysis across different domains.

All analysis sessions (CAN, Serial, BlackBox, RF, etc.) inherit from
AnalysisSession and provide consistent interfaces for:
- Recording management (add, list, compare)
- Differential analysis
- Result export
- Domain-specific analysis methods

Example - Generic Session:
    >>> import oscura as osc
    >>> from oscura.sessions import GenericSession
    >>>
    >>> session = GenericSession(name="Analysis")
    >>> trace = osc.load("capture.wfm")
    >>> session.add_recording("test", trace)
    >>> results = session.analyze()
    >>> print(results["summary"]["test"]["mean"])

Example - BlackBox Session (Protocol RE):
    >>> from oscura.sessions import BlackBoxSession
    >>>
    >>> session = BlackBoxSession(name="IoT Protocol RE")
    >>> session.add_recording("baseline", osc.load("baseline.wfm"))
    >>> session.add_recording("stimulus", osc.load("button.wfm"))
    >>> diff = session.compare("baseline", "stimulus")
    >>> print(f"Changed: {diff.changed_bytes} bytes")

Pattern Decision Table:
    - Use GenericSession for general waveform analysis
    - Use BlackBoxSession for unknown protocol reverse engineering
    - Extend AnalysisSession for custom domain-specific workflows

Architecture:
    Layer 3 (High-Level API) - User-Facing
    ├── AnalysisSession (ABC)
    │   ├── GenericSession
    │   ├── BlackBoxSession
    │   └── [Future domain sessions]
    └── [Workflows wrapping sessions]

References:
    Architecture Plan Phase 0.3: AnalysisSession Base Class
    docs/architecture/api-patterns.md: When to use Sessions vs Workflows
"""

from oscura.sessions.base import AnalysisSession, ComparisonResult
from oscura.sessions.blackbox import BlackBoxSession, FieldHypothesis, ProtocolSpec
from oscura.sessions.generic import GenericSession

__all__ = [
    "AnalysisSession",
    "BlackBoxSession",
    "ComparisonResult",
    "FieldHypothesis",
    "GenericSession",
    "ProtocolSpec",
]
