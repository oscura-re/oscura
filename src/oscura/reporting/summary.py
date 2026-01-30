"""Executive summary and key findings generation.

This module provides automated generation of executive summaries,
measurement summaries, and key findings identification for reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oscura.reporting.interpretation import MeasurementInterpretation, QualityLevel


@dataclass
class ExecutiveSummarySection:
    """A section in an executive summary.

    Attributes:
        title: Section title.
        content: Section content.
        bullet_points: List of key bullet points.
        priority: Priority level (1=highest).
    """

    title: str
    content: str = ""
    bullet_points: list[str] = field(default_factory=list)
    priority: int = 1


def generate_executive_summary(
    measurements: dict[str, Any],
    interpretations: dict[str, MeasurementInterpretation] | None = None,
    max_findings: int = 5,
) -> str:
    """Generate executive summary from measurements.

    Args:
        measurements: Dictionary of measurement results.
        interpretations: Optional interpretations of measurements.
        max_findings: Maximum number of key findings to include.

    Returns:
        Executive summary text.

    Example:
        >>> measurements = {"snr": 45.2, "thd": -60.5, "bandwidth": 1e9}
        >>> summary = generate_executive_summary(measurements)
        >>> "Executive Summary" in summary or len(summary) > 0
        True
    """
    sections = []

    # Overall status
    status_section = _generate_status_section(measurements, interpretations)
    sections.append(status_section)

    # Key findings
    findings_section = _generate_findings_section(measurements, interpretations, max_findings)
    sections.append(findings_section)

    # Recommendations
    if interpretations:
        rec_section = _generate_recommendations_section(interpretations)
        if rec_section.bullet_points:
            sections.append(rec_section)

    # Format as text
    lines = ["# Executive Summary", ""]

    for section in sections:
        lines.append(f"## {section.title}")
        lines.append("")

        if section.content:
            lines.append(section.content)
            lines.append("")

        if section.bullet_points:
            for point in section.bullet_points:
                lines.append(f"- {point}")
            lines.append("")

    return "\n".join(lines)


def _generate_status_section(
    measurements: dict[str, Any],
    interpretations: dict[str, MeasurementInterpretation] | None,
) -> ExecutiveSummarySection:
    """Generate overall status section."""
    total = len(measurements)

    if interpretations:
        excellent = sum(1 for i in interpretations.values() if i.quality == QualityLevel.EXCELLENT)
        good = sum(1 for i in interpretations.values() if i.quality == QualityLevel.GOOD)
        acceptable = sum(
            1 for i in interpretations.values() if i.quality == QualityLevel.ACCEPTABLE
        )
        marginal = sum(1 for i in interpretations.values() if i.quality == QualityLevel.MARGINAL)
        poor = sum(1 for i in interpretations.values() if i.quality == QualityLevel.POOR)
        failed = sum(1 for i in interpretations.values() if i.quality == QualityLevel.FAILED)

        if failed > 0:
            overall = "CRITICAL"
            content = f"{failed} of {total} measurements failed critical requirements."
        elif marginal > total / 2:
            overall = "MARGINAL"
            content = f"{marginal} of {total} measurements are marginal."
        elif excellent + good > total * 0.7:
            overall = "GOOD"
            content = f"Signal quality is good: {excellent + good} of {total} measurements are excellent or good."
        else:
            overall = "ACCEPTABLE"
            content = f"{acceptable} of {total} measurements are acceptable."

        bullet_points = [
            f"Excellent: {excellent}",
            f"Good: {good}",
            f"Acceptable: {acceptable}",
            f"Marginal: {marginal}",
            f"Poor: {poor}",
            f"Failed: {failed}",
        ]
    else:
        overall = "COMPLETE"
        content = f"Analysis complete with {total} measurements."
        bullet_points = []

    return ExecutiveSummarySection(
        title=f"Overall Status: {overall}",
        content=content,
        bullet_points=bullet_points,
        priority=1,
    )


def _generate_findings_section(
    measurements: dict[str, Any],
    interpretations: dict[str, MeasurementInterpretation] | None,
    max_findings: int,
) -> ExecutiveSummarySection:
    """Generate key findings section."""
    findings = identify_key_findings(measurements, interpretations, max_findings)

    return ExecutiveSummarySection(
        title="Key Findings",
        content="",
        bullet_points=findings,
        priority=2,
    )


def _generate_recommendations_section(
    interpretations: dict[str, MeasurementInterpretation],
) -> ExecutiveSummarySection:
    """Generate recommendations section."""
    all_recommendations = []

    for interp in interpretations.values():
        all_recommendations.extend(interp.recommendations)

    # Deduplicate
    unique_recs = list(dict.fromkeys(all_recommendations))

    return ExecutiveSummarySection(
        title="Recommendations",
        content="",
        bullet_points=unique_recs[:5],  # Top 5
        priority=3,
    )


def summarize_measurements(
    measurements: dict[str, Any],
    group_by: str | None = None,
) -> dict[str, Any]:
    """Summarize measurements with statistics.

    Args:
        measurements: Dictionary of measurements.
        group_by: Optional grouping key (e.g., "domain", "type").

    Returns:
        Summary dictionary with statistics.

    Example:
        >>> measurements = {"snr": 45.2, "thd": -60.5, "bandwidth": 1e9}
        >>> summary = summarize_measurements(measurements)
        >>> summary["count"]
        3
    """
    numeric_values = [v for v in measurements.values() if isinstance(v, int | float)]

    summary: dict[str, Any] = {
        "count": len(measurements),
        "numeric_count": len(numeric_values),
    }

    if numeric_values:
        import numpy as np

        summary.update(
            {
                "mean": float(np.mean(numeric_values)),
                "median": float(np.median(numeric_values)),
                "std": float(np.std(numeric_values)),
                "min": float(np.min(numeric_values)),
                "max": float(np.max(numeric_values)),
            }
        )

    return summary


def identify_key_findings(
    measurements: dict[str, Any],
    interpretations: dict[str, MeasurementInterpretation] | None = None,
    max_findings: int = 5,
) -> list[str]:
    """Automatically identify key findings from measurements.

    Args:
        measurements: Dictionary of measurements.
        interpretations: Optional interpretations.
        max_findings: Maximum number of findings to return.

    Returns:
        List of key finding strings.

    Example:
        >>> measurements = {"snr": 45.2, "bandwidth": 1e9}
        >>> findings = identify_key_findings(measurements, max_findings=3)
        >>> len(findings) <= 3
        True
    """
    findings = []

    # Check for exceptional values
    if interpretations:
        # Failed or poor quality
        failed = [
            name
            for name, interp in interpretations.items()
            if interp.quality in (QualityLevel.FAILED, QualityLevel.POOR)
        ]
        if failed:
            findings.append(f"Critical: {len(failed)} measurements failed or poor quality")

        # Excellent quality
        excellent = [
            name
            for name, interp in interpretations.items()
            if interp.quality == QualityLevel.EXCELLENT
        ]
        if excellent:
            findings.append(f"{len(excellent)} measurements show excellent performance")

    # Domain-specific findings
    if "snr" in measurements:
        snr = measurements["snr"]
        if isinstance(snr, int | float):
            if snr > 60:
                findings.append(f"Excellent SNR: {snr:.1f} dB")
            elif snr < 20:
                findings.append(f"Low SNR: {snr:.1f} dB - noise mitigation recommended")

    if "bandwidth" in measurements:
        bw = measurements["bandwidth"]
        if isinstance(bw, int | float):
            if bw > 1e9:
                findings.append(f"Wide bandwidth: {bw / 1e9:.2f} GHz")

    if "jitter" in measurements or "rms_jitter" in measurements:
        jitter_key = "rms_jitter" if "rms_jitter" in measurements else "jitter"
        jitter = measurements[jitter_key]
        if isinstance(jitter, int | float):
            jitter_ps = jitter * 1e12
            if jitter_ps < 10:
                findings.append("Excellent timing: RMS jitter < 10 ps")
            elif jitter_ps > 200:
                findings.append(f"High jitter: {jitter_ps:.1f} ps - investigate timing issues")

    # Limit to max_findings
    return findings[:max_findings]


def recommendations_from_findings(
    measurements: dict[str, Any],
    interpretations: dict[str, MeasurementInterpretation] | None = None,
) -> list[str]:
    """Generate actionable recommendations from findings.

    Args:
        measurements: Dictionary of measurements.
        interpretations: Optional interpretations.

    Returns:
        List of recommendation strings.

    Example:
        >>> from oscura.reporting.interpretation import interpret_measurement
        >>> measurements = {"snr": 15.5}
        >>> interpretations = {"snr": interpret_measurement("snr", 15.5, "dB")}
        >>> recs = recommendations_from_findings(measurements, interpretations)
        >>> len(recs) > 0
        True
    """
    recommendations = []

    if interpretations:
        # Collect all recommendations from interpretations
        for interp in interpretations.values():
            recommendations.extend(interp.recommendations)

    # Add domain-specific recommendations
    if "snr" in measurements:
        snr = measurements["snr"]
        if isinstance(snr, int | float) and snr < 30:
            recommendations.append("Investigate noise sources and consider filtering")

    if "bandwidth" in measurements:
        bw = measurements["bandwidth"]
        if isinstance(bw, int | float) and bw < 100e6:
            recommendations.append("Verify signal path bandwidth requirements")

    # Deduplicate and return
    return list(dict.fromkeys(recommendations))
