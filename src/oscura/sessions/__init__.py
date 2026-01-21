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
    >>> from oscura.sessions import GenericSession
    >>> from oscura.acquisition import FileSource
    >>>
    >>> session = GenericSession()
    >>> session.add_recording("test", FileSource("capture.wfm"))
    >>> results = session.analyze()
    >>> print(results["summary"]["test"]["mean"])

Example - Domain-Specific Session:
    >>> from oscura.sessions import AnalysisSession
    >>> from oscura.acquisition import FileSource
    >>>
    >>> class CANSession(AnalysisSession):
    ...     def analyze(self):
    ...         # CAN-specific signal discovery
    ...         return self.discover_signals()
    ...
    ...     def discover_signals(self):
    ...         # Extract CAN signals from recordings
    ...         pass
    >>>
    >>> session = CANSession()
    >>> session.add_recording("baseline", FileSource("idle.blf"))
    >>> signals = session.analyze()

Pattern Decision Table:
    - Use GenericSession for general waveform analysis
    - Extend AnalysisSession for domain-specific workflows
    - Use existing session.Session for backward compatibility

Architecture:
    Layer 3 (High-Level API) - User-Facing
    ├── AnalysisSession (ABC)
    │   ├── GenericSession
    │   ├── CANSession (Phase 1)
    │   ├── SerialSession (Phase 1)
    │   ├── BlackBoxSession (Phase 1)
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
