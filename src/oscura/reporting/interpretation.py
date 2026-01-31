"""Measurement interpretation and quality assessment for reports.

This module provides intelligent interpretation of measurement results,
generating findings, quality scores, and compliance checks against
IEEE standards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityLevel(Enum):
    """Signal or measurement quality level."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"
    FAILED = "failed"


class ComplianceStatus(Enum):
    """Compliance status against standards."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    MARGINAL = "marginal"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


@dataclass
class MeasurementInterpretation:
    """Interpretation of a measurement result.

    Attributes:
        measurement_name: Name of the measurement.
        value: Measured value.
        units: Measurement units.
        interpretation: Human-readable interpretation.
        quality: Quality assessment.
        recommendations: List of recommendations.
        standard_reference: IEEE standard reference if applicable.
    """

    measurement_name: str
    value: float | str
    units: str = ""
    interpretation: str = ""
    quality: QualityLevel = QualityLevel.ACCEPTABLE
    recommendations: list[str] = field(default_factory=list)
    standard_reference: str | None = None


@dataclass
class Finding:
    """A finding from measurement analysis.

    Attributes:
        title: Finding title.
        description: Detailed description.
        severity: Severity level (info, warning, critical).
        measurements: Related measurements.
        recommendation: Recommended action.
    """

    title: str
    description: str
    severity: str = "info"  # info, warning, critical
    measurements: list[str] = field(default_factory=list)
    recommendation: str = ""


def interpret_measurement(
    name: str,
    value: float | str,
    units: str = "",
    spec_min: float | None = None,
    spec_max: float | None = None,
    context: dict[str, Any] | None = None,
) -> MeasurementInterpretation:
    """Interpret a measurement and provide context.

    Args:
        name: Measurement name (e.g., "rise_time", "snr").
        value: Measured value.
        units: Measurement units.
        spec_min: Minimum specification (if applicable).
        spec_max: Maximum specification (if applicable).
        context: Additional context dictionary.

    Returns:
        MeasurementInterpretation object with analysis.

    Example:
        >>> interp = interpret_measurement("rise_time", 2.5e-9, "s", spec_max=5e-9)
        >>> interp.quality
        <QualityLevel.GOOD: 'good'>
        >>> "fast" in interp.interpretation.lower()
        True
    """
    if isinstance(value, str):
        return MeasurementInterpretation(
            measurement_name=name,
            value=value,
            units=units,
            interpretation="Non-numeric measurement",
        )

    interpretation = ""
    quality = QualityLevel.ACCEPTABLE
    recommendations: list[str] = []

    # Rise time interpretation
    if "rise" in name.lower() or "fall" in name.lower():
        if value < 1e-9:
            interpretation = "Very fast transition time, indicating high bandwidth signal path"
            quality = QualityLevel.EXCELLENT
        elif value < 10e-9:
            interpretation = "Fast transition time, suitable for high-speed digital signals"
            quality = QualityLevel.GOOD
        elif value < 100e-9:
            interpretation = "Moderate transition time, adequate for standard digital signals"
            quality = QualityLevel.ACCEPTABLE
        else:
            interpretation = "Slow transition time, may limit signal bandwidth"
            quality = QualityLevel.MARGINAL
            recommendations.append("Consider improving signal path bandwidth")

    # SNR interpretation
    elif "snr" in name.lower():
        if value > 60:
            interpretation = "Excellent signal-to-noise ratio, minimal noise impact"
            quality = QualityLevel.EXCELLENT
        elif value > 40:
            interpretation = "Good signal-to-noise ratio, acceptable for most applications"
            quality = QualityLevel.GOOD
        elif value > 20:
            interpretation = "Moderate SNR, noise may affect precision measurements"
            quality = QualityLevel.ACCEPTABLE
            recommendations.append("Consider noise reduction techniques")
        else:
            interpretation = "Poor SNR, signal quality compromised by noise"
            quality = QualityLevel.POOR
            recommendations.extend(
                ["Investigate noise sources", "Consider signal averaging or filtering"]
            )

    # Jitter interpretation
    elif "jitter" in name.lower():
        # Assuming jitter in seconds
        jitter_ps = value * 1e12  # Convert to picoseconds
        if jitter_ps < 10:
            interpretation = "Very low jitter, excellent timing stability"
            quality = QualityLevel.EXCELLENT
        elif jitter_ps < 50:
            interpretation = "Low jitter, good timing performance"
            quality = QualityLevel.GOOD
        elif jitter_ps < 200:
            interpretation = "Moderate jitter, acceptable for most applications"
            quality = QualityLevel.ACCEPTABLE
        else:
            interpretation = "High jitter, timing stability may be compromised"
            quality = QualityLevel.MARGINAL
            recommendations.append("Investigate jitter sources (clock, power, crosstalk)")

    # Bandwidth interpretation
    elif "bandwidth" in name.lower():
        if value > 1e9:
            interpretation = (
                f"Wide bandwidth ({value / 1e9:.1f} GHz), suitable for high-frequency signals"
            )
            quality = QualityLevel.EXCELLENT
        elif value > 100e6:
            interpretation = (
                f"Good bandwidth ({value / 1e6:.0f} MHz), adequate for most applications"
            )
            quality = QualityLevel.GOOD
        elif value > 10e6:
            interpretation = f"Moderate bandwidth ({value / 1e6:.1f} MHz)"
            quality = QualityLevel.ACCEPTABLE
        else:
            interpretation = f"Limited bandwidth ({value / 1e6:.2f} MHz)"
            quality = QualityLevel.MARGINAL

    # Generic interpretation based on specs
    elif spec_min is not None or spec_max is not None:
        if spec_min is not None and value < spec_min:
            interpretation = f"Below minimum specification ({spec_min} {units})"
            quality = QualityLevel.FAILED
            recommendations.append("Value does not meet specification")
        elif spec_max is not None and value > spec_max:
            interpretation = f"Above maximum specification ({spec_max} {units})"
            quality = QualityLevel.FAILED
            recommendations.append("Value exceeds specification limit")
        else:
            # Within spec - check margin
            if spec_min is not None and spec_max is not None:
                range_val = spec_max - spec_min
                margin_low = (value - spec_min) / range_val if range_val > 0 else 0
                margin_high = (spec_max - value) / range_val if range_val > 0 else 0
                min_margin = min(margin_low, margin_high)

                if min_margin > 0.3:
                    interpretation = "Well within specification with good margin"
                    quality = QualityLevel.GOOD
                elif min_margin > 0.1:
                    interpretation = "Within specification with adequate margin"
                    quality = QualityLevel.ACCEPTABLE
                else:
                    interpretation = "Within specification but marginal"
                    quality = QualityLevel.MARGINAL
                    recommendations.append("Low margin to specification limits")
            else:
                interpretation = "Within specification"
                quality = QualityLevel.ACCEPTABLE

    else:
        interpretation = f"Measured value: {value} {units}"
        quality = QualityLevel.ACCEPTABLE

    return MeasurementInterpretation(
        measurement_name=name,
        value=value,
        units=units,
        interpretation=interpretation,
        quality=quality,
        recommendations=recommendations,
    )


def generate_finding(
    title: str,
    measurements: dict[str, Any],
    threshold: float | None = None,
    severity: str = "info",
) -> Finding:
    """Generate a finding from measurement analysis.

    Args:
        title: Finding title.
        measurements: Dictionary of measurements.
        threshold: Threshold for determining severity.
        severity: Override severity level.

    Returns:
        Finding object.

    Example:
        >>> measurements = {"snr": 25.5, "thd": -60.2}
        >>> finding = generate_finding("Signal Quality Assessment", measurements)
        >>> finding.title
        'Signal Quality Assessment'
    """
    description_parts = []

    for name, value in measurements.items():
        if isinstance(value, float):
            description_parts.append(f"{name}: {value:.3f}")
        else:
            description_parts.append(f"{name}: {value}")

    description = "\n".join(description_parts)

    recommendation = ""
    if severity == "warning":
        recommendation = "Review measurements and verify against specifications"
    elif severity == "critical":
        recommendation = "Immediate attention required - critical parameter out of range"

    return Finding(
        title=title,
        description=description,
        severity=severity,
        measurements=list(measurements.keys()),
        recommendation=recommendation,
    )


def quality_score(
    measurements: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[float, QualityLevel]:
    """Calculate overall quality score from measurements.

    Args:
        measurements: Dictionary of measurement names to values (0-100 scale).
        weights: Optional weights for each measurement.

    Returns:
        Tuple of (score, quality_level) where score is 0-100.

    Example:
        >>> measurements = {"snr": 45, "bandwidth": 80, "jitter": 60}
        >>> score, level = quality_score(measurements)
        >>> 0 <= score <= 100
        True
        >>> level in QualityLevel
        True
    """
    if not measurements:
        return 0.0, QualityLevel.POOR

    if weights is None:
        weights = dict.fromkeys(measurements, 1.0)

    total_weight = sum(weights.get(name, 1.0) for name in measurements)
    weighted_sum = sum(value * weights.get(name, 1.0) for name, value in measurements.items())

    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Determine quality level
    if score >= 90:
        level = QualityLevel.EXCELLENT
    elif score >= 75:
        level = QualityLevel.GOOD
    elif score >= 60:
        level = QualityLevel.ACCEPTABLE
    elif score >= 40:
        level = QualityLevel.MARGINAL
    else:
        level = QualityLevel.POOR

    return score, level


def compliance_check(
    measurement_name: str,
    value: float,
    standard_id: str,
    limits: dict[str, float] | None = None,
) -> tuple[ComplianceStatus, str]:
    """Check measurement compliance against IEEE standard.

    Args:
        measurement_name: Name of measurement.
        value: Measured value.
        standard_id: IEEE standard ID (e.g., "181", "1241").
        limits: Optional dict with "min" and/or "max" keys.

    Returns:
        Tuple of (compliance_status, explanation).

    Example:
        >>> status, msg = compliance_check("rise_time", 2.5e-9, "181", {"max": 5e-9})
        >>> status
        <ComplianceStatus.COMPLIANT: 'compliant'>
        >>> "compliant" in msg.lower()
        True
    """
    if limits is None:
        return ComplianceStatus.NOT_APPLICABLE, f"No limits defined for {measurement_name}"

    min_limit = limits.get("min")
    max_limit = limits.get("max")

    # Check limits
    if min_limit is not None and value < min_limit:
        explanation = (
            f"{measurement_name} = {value:.3e} is below minimum limit {min_limit:.3e} "
            f"(IEEE {standard_id})"
        )
        return ComplianceStatus.NON_COMPLIANT, explanation

    if max_limit is not None and value > max_limit:
        explanation = (
            f"{measurement_name} = {value:.3e} exceeds maximum limit {max_limit:.3e} "
            f"(IEEE {standard_id})"
        )
        return ComplianceStatus.NON_COMPLIANT, explanation

    # Within limits - check margin
    margin_pct = 100.0
    if min_limit is not None and max_limit is not None:
        range_val = max_limit - min_limit
        margin_low = ((value - min_limit) / range_val * 100) if range_val > 0 else 100
        margin_high = ((max_limit - value) / range_val * 100) if range_val > 0 else 100
        margin_pct = min(margin_low, margin_high)

    if margin_pct < 10:
        explanation = (
            f"{measurement_name} = {value:.3e} is compliant but marginal "
            f"({margin_pct:.1f}% margin to IEEE {standard_id} limits)"
        )
        return ComplianceStatus.MARGINAL, explanation

    explanation = (
        f"{measurement_name} = {value:.3e} is compliant with IEEE {standard_id} "
        f"({margin_pct:.1f}% margin)"
    )
    return ComplianceStatus.COMPLIANT, explanation


def interpret_results_batch(
    results: dict[str, dict[str, Any]],
) -> dict[str, MeasurementInterpretation]:
    """Interpret multiple measurement results.

    Args:
        results: Dictionary mapping measurement names to result dicts.
                Each result dict should have "value", "units", and optionally
                "spec_min" and "spec_max" keys.

    Returns:
        Dictionary mapping measurement names to interpretations.

    Example:
        >>> results = {
        ...     "rise_time": {"value": 2.5e-9, "units": "s", "spec_max": 5e-9},
        ...     "snr": {"value": 45.2, "units": "dB"}
        ... }
        >>> interps = interpret_results_batch(results)
        >>> len(interps)
        2
    """
    interpretations = {}

    for name, result in results.items():
        value = result.get("value", 0.0)
        units = result.get("units", "")
        spec_min = result.get("spec_min")
        spec_max = result.get("spec_max")

        interp = interpret_measurement(name, value, units, spec_min, spec_max)
        interpretations[name] = interp

    return interpretations
