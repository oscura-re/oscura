"""Hardware abstraction layer analysis.

This module provides tools for analyzing hardware abstraction layers,
register access patterns, and peripheral drivers in firmware binaries.
"""

from oscura.hardware.hal_detector import (
    HALAnalysisResult,
    HALDetector,
    Peripheral,
    RegisterAccess,
)

__all__ = [
    "HALAnalysisResult",
    "HALDetector",
    "Peripheral",
    "RegisterAccess",
]
