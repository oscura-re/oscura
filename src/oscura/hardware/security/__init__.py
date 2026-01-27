"""Security analysis and vulnerability detection for hardware systems.

This module provides security analysis capabilities for hardware systems including
side-channel attack detection, timing analysis, power analysis, and vulnerability
assessment.

Example:
    >>> from oscura.hardware.security.side_channel_detector import SideChannelDetector
    >>> detector = SideChannelDetector()
    >>> report = detector.analyze_traces(traces)
"""

from __future__ import annotations

__all__ = [
    "SideChannelDetector",
    "SideChannelVulnerability",
    "VulnerabilityReport",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name: str) -> object:
    """Lazy import for security modules."""
    if name == "SideChannelDetector":
        from oscura.hardware.security.side_channel_detector import SideChannelDetector

        return SideChannelDetector
    if name == "SideChannelVulnerability":
        from oscura.hardware.security.side_channel_detector import SideChannelVulnerability

        return SideChannelVulnerability
    if name == "VulnerabilityReport":
        from oscura.hardware.security.side_channel_detector import VulnerabilityReport

        return VulnerabilityReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
