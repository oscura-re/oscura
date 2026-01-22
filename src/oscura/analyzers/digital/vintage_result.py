"""Vintage logic analysis result data structures.

This module defines dataclasses for aggregating vintage logic analysis results,
enabling comprehensive reporting and export capabilities.

Example:
    >>> from oscura.analyzers.digital.vintage_result import VintageLogicAnalysisResult
    >>> result = VintageLogicAnalysisResult(
    ...     timestamp=datetime.now(),
    ...     detected_family="TTL",
    ...     family_confidence=0.95,
    ...     # ... more fields ...
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.analyzers.digital.timing_paths import TimingPathResult


@dataclass
class ICIdentificationResult:
    """Single IC identification result.

    Attributes:
        ic_name: Identified IC part number (e.g., "74LS74").
        confidence: Confidence score (0.0-1.0).
        timing_params: Measured timing parameters in seconds.
        validation: Validation results from validate_ic_timing().
        family: Logic family of the IC.
    """

    ic_name: str
    confidence: float
    timing_params: dict[str, float]
    validation: dict[str, dict[str, Any]]
    family: str


@dataclass
class ModernReplacementIC:
    """Modern IC recommendation for vintage part.

    Attributes:
        original_ic: Original vintage IC part number.
        replacement_ic: Recommended modern replacement.
        family: Replacement logic family (e.g., "74HCTxx").
        benefits: List of benefits (speed, power, availability).
        notes: Optional additional notes.
    """

    original_ic: str
    replacement_ic: str
    family: str
    benefits: list[str]
    notes: str | None = None


@dataclass
class BOMEntry:
    """Bill of materials entry.

    Attributes:
        part_number: Component part number.
        description: Component description.
        quantity: Number of components needed.
        category: Component category ("IC", "Capacitor", "Buffer", etc.).
        notes: Optional notes (pinout, alternatives, etc.).
    """

    part_number: str
    description: str
    quantity: int
    category: str
    notes: str | None = None


@dataclass
class VintageLogicAnalysisResult:
    """Complete vintage logic analysis result.

    Aggregates all analysis outputs for comprehensive reporting and export.

    Attributes:
        timestamp: Analysis timestamp.
        source_file: Source file path if loaded from file.
        analysis_duration: Analysis execution time in seconds.
        detected_family: Detected logic family name.
        family_confidence: Logic family detection confidence (0.0-1.0).
        voltage_levels: Measured voltage levels (VCC, VIL, VIH, VOL, VOH).
        identified_ics: List of identified ICs.
        timing_measurements: Dictionary of timing measurements in seconds.
        timing_paths: Multi-IC timing path analysis results.
        decoded_protocols: Protocol decoder results if applicable.
        open_collector_detected: Whether open-collector output detected.
        asymmetry_ratio: Rise/fall time asymmetry ratio.
        modern_replacements: List of modern IC recommendations.
        bom: Bill of materials entries.
        warnings: List of warning messages.
        confidence_scores: Dictionary of confidence scores by analysis type.
    """

    # Analysis metadata
    timestamp: datetime
    source_file: str | None
    analysis_duration: float

    # Logic family detection results
    detected_family: str
    family_confidence: float
    voltage_levels: dict[str, float]  # VCC, VIL, VIH, VOL, VOH

    # IC identification results
    identified_ics: list[ICIdentificationResult] = field(default_factory=list)

    # Timing measurements
    timing_measurements: dict[str, float] = field(default_factory=dict)

    # Multi-IC path analysis
    timing_paths: list[TimingPathResult] | None = None

    # Protocol decoder results (if applicable)
    decoded_protocols: dict[str, list[Any]] | None = None

    # Open-collector detection
    open_collector_detected: bool = False
    asymmetry_ratio: float = 1.0

    # Recommendations
    modern_replacements: list[ModernReplacementIC] = field(default_factory=list)
    bom: list[BOMEntry] = field(default_factory=list)

    # Quality metrics
    warnings: list[str] = field(default_factory=list)
    confidence_scores: dict[str, float] = field(default_factory=dict)


__all__ = [
    "BOMEntry",
    "ICIdentificationResult",
    "ModernReplacementIC",
    "VintageLogicAnalysisResult",
]
