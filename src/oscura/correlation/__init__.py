"""Multi-protocol correlation and session analysis.

This module provides tools for correlating messages and sessions across
different protocols (CAN, Ethernet, Serial, etc.) to discover cross-protocol
communication patterns and dependencies.

Example:
    >>> from oscura.correlation import MultiProtocolCorrelator, ProtocolMessage
    >>> correlator = MultiProtocolCorrelator(time_window=0.1, min_confidence=0.5)
    >>>
    >>> # Add messages from different protocols
    >>> can_msg = ProtocolMessage(
    ...     protocol="can",
    ...     timestamp=1.234,
    ...     message_id=0x123,
    ...     payload=b"\\x01\\x02\\x03"
    ... )
    >>> eth_msg = ProtocolMessage(
    ...     protocol="ethernet",
    ...     timestamp=1.238,
    ...     payload=b"\\x01\\x02\\x03\\x04"
    ... )
    >>> correlator.add_message(can_msg)
    >>> correlator.add_message(eth_msg)
    >>>
    >>> # Find correlations
    >>> correlations = correlator.correlate_all()
    >>> print(f"Found {len(correlations)} correlations")
    >>>
    >>> # Extract sessions
    >>> sessions = correlator.extract_sessions()
    >>> print(f"Protocols used: {sessions[0].protocols}")

References:
    Network protocol analysis
    Graph theory for dependency analysis
    Session correlation algorithms
"""

from oscura.correlation.multi_protocol import (
    MessageCorrelation,
    MultiProtocolCorrelator,
    ProtocolMessage,
    SessionFlow,
)

__all__ = [
    "MessageCorrelation",
    "MultiProtocolCorrelator",
    "ProtocolMessage",
    "SessionFlow",
]
