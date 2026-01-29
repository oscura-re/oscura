"""Firmware analysis and pattern recognition.

This module provides firmware reverse engineering capabilities including:
- Pattern-based function detection
- Architecture fingerprinting
- String and data region identification
- Interrupt vector table extraction
- Compiler signature detection
"""

from oscura.hardware.firmware.pattern_recognition import (
    Architecture,
    CompilerSignature,
    FirmwareAnalysisResult,
    FirmwarePatternRecognizer,
    Function,
    InterruptVector,
    StringTable,
)

__all__ = [
    "Architecture",
    "CompilerSignature",
    "FirmwareAnalysisResult",
    "FirmwarePatternRecognizer",
    "Function",
    "InterruptVector",
    "StringTable",
]
