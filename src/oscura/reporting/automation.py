"""Automated analysis and anomaly detection for reports.

This module provides automated interpretation of results, anomaly flagging,
and intelligent recommendations generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from oscura.reporting.interpretation import MeasurementInterpretation, interpret_measurement
from oscura.reporting.summary import (
    identify_key_findings,
    recommendations_from_findings,
    summarize_measurements,
)


def auto_interpret_results(results: dict[str, Any]) -> dict[str, MeasurementInterpretation]:
    """Automatically interpret all results in a dictionary.

    Args:
        results: Dictionary of measurement results. Can be nested.

    Returns:
        Dictionary mapping measurement names to interpretations.

    Example:
        >>> results = {"snr": 45.2, "bandwidth": 1e9, "jitter": 50e-12}
        >>> interpretations = auto_interpret_results(results)
        >>> len(interpretations) >= 3
        True
    """
    interpretations = {}

    for key, value in results.items():
        if isinstance(value, dict):
            # Nested dictionary - recurse
            if "value" in value:
                # This is a measurement dict with value and metadata
                meas_value = value["value"]
                units = value.get("units", "")
                spec_min = value.get("spec_min")
                spec_max = value.get("spec_max")

                interp = interpret_measurement(key, meas_value, units, spec_min, spec_max)
                interpretations[key] = interp
            else:
                # Recurse into nested dict
                nested = auto_interpret_results(value)
                interpretations.update(nested)
        elif isinstance(value, int | float):
            # Simple numeric value
            interp = interpret_measurement(key, value)
            interpretations[key] = interp

    return interpretations


def generate_summary(
    results: dict[str, Any],
    include_recommendations: bool = True,
) -> dict[str, Any]:
    """Generate comprehensive automated summary of results.

    Args:
        results: Dictionary of measurement results.
        include_recommendations: Whether to include recommendations.

    Returns:
        Summary dictionary with statistics, findings, and recommendations.

    Example:
        >>> results = {"snr": 45.2, "thd": -60.5, "bandwidth": 1e9}
        >>> summary = generate_summary(results)
        >>> "statistics" in summary
        True
        >>> "key_findings" in summary
        True
    """
    # Flatten results to simple dict
    flat_results = _flatten_results(results)

    # Generate statistics
    statistics = summarize_measurements(flat_results)

    # Interpret results
    interpretations = auto_interpret_results(results)

    # Identify key findings
    key_findings = identify_key_findings(flat_results, interpretations, max_findings=10)

    # Generate recommendations
    recommendations = []
    if include_recommendations:
        recommendations = recommendations_from_findings(flat_results, interpretations)

    # Flag anomalies
    anomalies = flag_anomalies(flat_results)

    return {
        "statistics": statistics,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "anomalies": anomalies,
        "interpretations": {k: v.__dict__ for k, v in interpretations.items()},
    }


def flag_anomalies(
    measurements: dict[str, Any],
    threshold_std: float = 3.0,
) -> list[dict[str, Any]]:
    """Flag anomalous measurements using statistical analysis.

    Args:
        measurements: Dictionary of measurements.
        threshold_std: Number of standard deviations for outlier detection.

    Returns:
        List of anomaly dictionaries with name, value, and reason.

    Example:
        >>> measurements = {"m1": 10, "m2": 11, "m3": 12, "m4": 100}
        >>> anomalies = flag_anomalies(measurements, threshold_std=2.0)
        >>> len(anomalies) >= 1
        True
    """
    anomalies: list[dict[str, Any]] = []

    # Extract numeric values
    numeric_measurements = {k: v for k, v in measurements.items() if isinstance(v, int | float)}

    if len(numeric_measurements) < 3:
        # Not enough data for statistical analysis
        return anomalies

    values = np.array(list(numeric_measurements.values()))
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return anomalies

    # Check each measurement
    for name, value in numeric_measurements.items():
        z_score = abs((value - mean) / std)

        if z_score > threshold_std:
            anomalies.append(
                {
                    "name": name,
                    "value": value,
                    "z_score": float(z_score),
                    "reason": f"Value is {z_score:.1f} std devs from mean ({mean:.3e})",
                    "severity": "high" if z_score > 5 else "medium",
                }
            )

    # Domain-specific anomaly checks
    for name, value in measurements.items():
        if not isinstance(value, int | float):
            continue

        # Negative SNR
        if "snr" in name.lower() and value < 0:
            anomalies.append(
                {
                    "name": name,
                    "value": value,
                    "reason": "Negative SNR indicates signal weaker than noise",
                    "severity": "critical",
                }
            )

        # Negative bandwidth
        if "bandwidth" in name.lower() and value <= 0:
            anomalies.append(
                {
                    "name": name,
                    "value": value,
                    "reason": "Invalid negative or zero bandwidth",
                    "severity": "critical",
                }
            )

        # Extremely high jitter (>1 second)
        if "jitter" in name.lower() and value > 1.0:
            anomalies.append(
                {
                    "name": name,
                    "value": value,
                    "reason": "Unrealistically high jitter value (>1 second)",
                    "severity": "critical",
                }
            )

        # Invalid power factor (must be -1 to +1)
        if "power_factor" in name.lower() and abs(value) > 1.0:
            anomalies.append(
                {
                    "name": name,
                    "value": value,
                    "reason": "Power factor must be between -1 and +1",
                    "severity": "critical",
                }
            )

    return anomalies


def suggest_follow_up_analyses(
    measurements: dict[str, Any],
    interpretations: dict[str, MeasurementInterpretation] | None = None,
) -> list[str]:
    """Suggest follow-up analyses based on measurement results.

    Args:
        measurements: Dictionary of measurements.
        interpretations: Optional measurement interpretations.

    Returns:
        List of suggested analysis descriptions.

    Example:
        >>> measurements = {"snr": 15.5, "thd": -40}
        >>> suggestions = suggest_follow_up_analyses(measurements)
        >>> len(suggestions) > 0
        True
    """
    suggestions = []

    # Low SNR - suggest noise analysis
    if "snr" in measurements:
        snr = measurements["snr"]
        if isinstance(snr, int | float) and snr < 30:
            suggestions.append("Perform detailed noise analysis to identify noise sources")
            suggestions.append("Analyze frequency spectrum for interference peaks")

    # High THD - suggest harmonic analysis
    if "thd" in measurements:
        thd = measurements["thd"]
        if isinstance(thd, int | float) and abs(thd) < 40:  # THD typically negative dB
            suggestions.append("Perform harmonic analysis to identify distortion sources")
            suggestions.append("Check for clipping or saturation in signal path")

    # Jitter present - suggest jitter decomposition
    if any("jitter" in k.lower() for k in measurements):
        suggestions.append("Perform jitter decomposition (RJ, DJ, PJ) for root cause analysis")
        suggestions.append("Analyze clock quality and investigate timing sources")

    # Power measurements - suggest power quality
    if any("power" in k.lower() for k in measurements):
        suggestions.append("Perform power quality analysis (harmonics, flicker, sags/swells)")

    # Eye diagram quality issues
    if interpretations:
        poor_quality = [
            k for k, v in interpretations.items() if v.quality.value in ("marginal", "poor")
        ]
        if poor_quality:
            suggestions.append(
                f"Investigate {len(poor_quality)} marginal/poor measurements: "
                f"{', '.join(poor_quality[:3])}"
            )

    return suggestions


def identify_issues(
    measurements: dict[str, Any],
    anomalies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify issues from measurements and anomalies.

    Args:
        measurements: Dictionary of measurements.
        anomalies: List of detected anomalies.

    Returns:
        List of issue dictionaries with severity and description.

    Example:
        >>> measurements = {"snr": 10}
        >>> anomalies = [{"name": "snr", "value": 10, "severity": "high"}]
        >>> issues = identify_issues(measurements, anomalies)
        >>> len(issues) > 0
        True
    """
    issues = []

    # Add critical anomalies as issues
    for anomaly in anomalies:
        if anomaly.get("severity") == "critical":
            issues.append(
                {
                    "severity": "critical",
                    "measurement": anomaly["name"],
                    "description": anomaly["reason"],
                    "value": anomaly["value"],
                }
            )

    # Domain-specific issue detection
    if "snr" in measurements:
        snr = measurements["snr"]
        if isinstance(snr, int | float) and snr < 20:
            issues.append(
                {
                    "severity": "high",
                    "measurement": "snr",
                    "description": f"Low SNR ({snr:.1f} dB) may impact measurement accuracy",
                    "value": snr,
                }
            )

    if "bandwidth" in measurements:
        bw = measurements["bandwidth"]
        if isinstance(bw, int | float) and bw < 10e6:
            issues.append(
                {
                    "severity": "medium",
                    "measurement": "bandwidth",
                    "description": f"Limited bandwidth ({bw / 1e6:.1f} MHz) may restrict signal fidelity",
                    "value": bw,
                }
            )

    return issues


def _flatten_results(results: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested results dictionary."""
    flat = {}

    for key, value in results.items():
        if isinstance(value, dict):
            if "value" in value:
                flat[key] = value["value"]
            else:
                nested = _flatten_results(value)
                flat.update(nested)
        elif isinstance(value, int | float | str):
            flat[key] = value

    return flat
