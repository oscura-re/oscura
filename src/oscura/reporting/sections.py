"""Report section generation for Oscura.

This module provides utilities for creating standardized report sections
including title, summary, measurements, plots, and conclusions.


Example:
    >>> from oscura.reporting.sections import create_title_section
    >>> section = create_title_section("Signal Analysis Report", author="Engineer")
"""

from datetime import datetime
from typing import Any

from oscura.reporting.core import Section


def create_title_section(
    title: str,
    *,
    author: str | None = None,
    date: datetime | None = None,
    subtitle: str | None = None,
) -> Section:
    """Create title section for report.

    Args:
        title: Report title.
        author: Report author.
        date: Report date (defaults to now).
        subtitle: Optional subtitle.

    Returns:
        Title Section object.

    References:
        REPORT-006
    """
    content_parts = []

    if subtitle:
        content_parts.append(subtitle)

    if author:
        content_parts.append(f"Author: {author}")

    if date is None:
        date = datetime.now()
    content_parts.append(f"Date: {date.strftime('%Y-%m-%d %H:%M')}")

    content = "\n".join(content_parts)

    return Section(
        title=title,
        content=content,
        level=0,  # Top level
        visible=True,
    )


def create_executive_summary_section(
    results: dict[str, Any],
    *,
    key_findings: list[str] | None = None,
    length: str = "summary",
) -> Section:
    """Create executive summary section.

    Args:
        results: Analysis results dictionary.
        key_findings: List of key findings to highlight.
        length: Summary length (short, summary, detailed).

    Returns:
        Executive Summary Section object.

    References:
        REPORT-004, REPORT-006
    """
    content_parts: list[str] = []

    _add_test_status(content_parts, results)
    _add_key_findings(content_parts, key_findings)
    _add_margin_analysis(content_parts, results)
    _add_recommendations(content_parts, results, length)

    return Section(
        title="Executive Summary",
        content="\n".join(content_parts),
        level=1,
        visible=True,
    )


def _add_test_status(content_parts: list[str], results: dict[str, Any]) -> None:
    """Add test status to summary.

    Args:
        content_parts: List to append to.
        results: Results dictionary.
    """
    if "pass_count" not in results or "total_count" not in results:
        return

    pass_count = results["pass_count"]
    total = results["total_count"]

    if pass_count == total:
        content_parts.append(f"All {total} tests passed with satisfactory margins.")
    else:
        fail_count = total - pass_count
        fail_rate = fail_count / total * 100
        content_parts.append(
            f"{fail_count} of {total} tests failed ({fail_rate:.0f}% failure rate)."
        )


def _add_key_findings(content_parts: list[str], key_findings: list[str] | None) -> None:
    """Add key findings to summary.

    Args:
        content_parts: List to append to.
        key_findings: Findings to add.
    """
    if not key_findings:
        return

    content_parts.append("\n**Key Findings:**")
    for finding in key_findings[:5]:
        content_parts.append(f"- {finding}")


def _add_margin_analysis(content_parts: list[str], results: dict[str, Any]) -> None:
    """Add margin analysis to summary.

    Args:
        content_parts: List to append to.
        results: Results dictionary.
    """
    if "min_margin" not in results:
        return

    margin = results["min_margin"]
    content_parts.append("\n**Margin Analysis:**")

    if margin < 0:
        content_parts.append(f"Critical: Minimum margin is {margin:.1f}% (violation).")
    elif margin < 10:
        content_parts.append(f"Warning: Minimum margin is {margin:.1f}% (below recommended 10%).")
    elif margin < 20:
        content_parts.append(f"Acceptable: Minimum margin is {margin:.1f}% (below target 20%).")
    else:
        content_parts.append(f"Good: Minimum margin is {margin:.1f}% (exceeds target 20%).")


def _add_recommendations(content_parts: list[str], results: dict[str, Any], length: str) -> None:
    """Add recommendations to summary if detailed.

    Args:
        content_parts: List to append to.
        results: Results dictionary.
        length: Summary length mode.
    """
    if length != "detailed" or "violations" not in results:
        return

    violations = results["violations"]
    if not violations:
        return

    content_parts.append("\n**Recommendations:**")
    for violation in violations[:3]:
        param = violation.get("parameter", "measurement")
        content_parts.append(f"- Address {param} violation")


def create_measurement_results_section(
    measurements: dict[str, Any],
    *,
    include_plots: bool = False,
) -> Section:
    """Create measurement results section.

    Args:
        measurements: Dictionary of measurement results.
        include_plots: Include measurement plots.

    Returns:
        Measurement Results Section object.

    References:
        REPORT-006
    """
    from oscura.reporting.tables import create_measurement_table

    # Create measurement table
    table = create_measurement_table(measurements, format="dict")

    content = [table]

    # Add interpretation if any failures
    failed_count = sum(1 for m in measurements.values() if not m.get("passed", True))
    if failed_count > 0:
        interpretation = f"\n{failed_count} measurement(s) failed specification limits."
        content.insert(0, interpretation)

    return Section(
        title="Measurement Results",
        content=content,
        level=1,
        visible=True,
    )


def create_plots_section(
    figures: list[dict[str, Any]],
) -> Section:
    """Create plots section.

    Args:
        figures: List of figure dictionaries.

    Returns:
        Plots Section object.

    References:
        REPORT-006
    """
    content = []

    for fig in figures:
        content.append(fig)

    return Section(
        title="Waveform Plots",
        content=content,
        level=1,
        visible=True,
    )


def create_methodology_section(
    analysis_params: dict[str, Any],
    *,
    verbosity: str = "standard",
) -> Section:
    """Create methodology section.

    Args:
        analysis_params: Analysis parameters and settings.
        verbosity: Detail level (summary, standard, detailed).

    Returns:
        Methodology Section object.

    References:
        REPORT-006
    """
    content_parts = []

    # Test setup
    if "sample_rate" in analysis_params:
        content_parts.append(f"Sample rate: {analysis_params['sample_rate']:.3g} Hz")
    if "num_samples" in analysis_params:
        content_parts.append(f"Number of samples: {analysis_params['num_samples']:,}")
    if "duration" in analysis_params:
        content_parts.append(f"Capture duration: {analysis_params['duration']:.6g} s")

    # Analysis methods
    if verbosity in ("standard", "detailed"):
        content_parts.append("\n**Analysis Methods:**")
        methods = analysis_params.get("methods", [])
        if methods:
            for method in methods:
                content_parts.append(f"- {method}")
        else:
            content_parts.append("- Standard signal analysis algorithms")

    # Standards compliance
    if "standards" in analysis_params:
        content_parts.append("\n**Standards:**")
        for standard in analysis_params["standards"]:
            content_parts.append(f"- {standard}")

    # Detailed parameters
    if verbosity == "detailed":
        content_parts.append("\n**Detailed Parameters:**")
        for key, value in analysis_params.items():
            if key not in (
                "sample_rate",
                "num_samples",
                "duration",
                "methods",
                "standards",
            ):
                content_parts.append(f"- {key}: {value}")

    content = "\n".join(content_parts)

    return Section(
        title="Methodology",
        content=content,
        level=1,
        visible=True,
        collapsible=True,
    )


def create_conclusions_section(
    results: dict[str, Any],
    *,
    recommendations: list[str] | None = None,
) -> Section:
    """Create conclusions section.

    Args:
        results: Analysis results.
        recommendations: List of recommendations.

    Returns:
        Conclusions Section object.

    References:
        REPORT-006
    """
    content_parts = []

    # Overall conclusion
    if "pass_count" in results and "total_count" in results:
        pass_count = results["pass_count"]
        total = results["total_count"]

        if pass_count == total:
            content_parts.append(
                "The device under test meets all specifications and is ready for deployment."
            )
        else:
            fail_count = total - pass_count
            content_parts.append(
                f"The device under test has {fail_count} specification violation(s) "
                "that must be addressed before deployment."
            )

    # Risk assessment
    if "min_margin" in results:
        margin = results["min_margin"]
        content_parts.append("\n**Risk Assessment:**")
        if margin < 0:
            content_parts.append("HIGH RISK: Specification violations detected.")
        elif margin < 10:
            content_parts.append("MEDIUM RISK: Insufficient design margin.")
        elif margin < 20:
            content_parts.append("LOW RISK: Adequate margin but below target.")
        else:
            content_parts.append("ACCEPTABLE: Sufficient design margin.")

    # Recommendations
    if recommendations:
        content_parts.append("\n**Recommendations:**")
        for rec in recommendations:
            content_parts.append(f"- {rec}")

    content = "\n".join(content_parts)

    return Section(
        title="Conclusions",
        content=content,
        level=1,
        visible=True,
    )


def create_appendix_section(
    raw_data: dict[str, Any],
    *,
    include_provenance: bool = True,
) -> Section:
    """Create appendix section with raw data and provenance.

    Args:
        raw_data: Raw data to include.
        include_provenance: Include data provenance information.

    Returns:
        Appendix Section object.

    References:
        REPORT-006
    """
    content_parts = []

    # Provenance
    if include_provenance:
        content_parts.append("**Data Provenance:**")
        if "source_file" in raw_data:
            content_parts.append(f"Source: {raw_data['source_file']}")
        if "timestamp" in raw_data:
            content_parts.append(f"Timestamp: {raw_data['timestamp']}")
        if "tool_version" in raw_data:
            content_parts.append(f"Tool Version: {raw_data['tool_version']}")

    # Raw data (truncated)
    content_parts.append("\n**Raw Data:** (See detailed output for full data)")

    content = "\n".join(content_parts)

    return Section(
        title="Appendix",
        content=content,
        level=1,
        visible=True,
        collapsible=True,
    )


def create_violations_section(
    violations: list[dict[str, Any]],
) -> Section:
    """Create violations section highlighting failures.

    Args:
        violations: List of violation dictionaries.

    Returns:
        Violations Section object.

    References:
        REPORT-005 (Smart Content Filtering)
    """
    if not violations:
        return Section(
            title="Violations",
            content="No specification violations detected.",
            level=1,
            visible=False,  # Hide if no violations
        )

    content_parts = [f"**{len(violations)} specification violation(s) detected:**\n"]

    for v in violations:
        param = v.get("parameter", "Unknown")
        value = v.get("value", "N/A")
        spec = v.get("specification", "N/A")
        severity = v.get("severity", "WARNING")

        content_parts.append(f"- **{param}**: {value} (spec: {spec}) [{severity}]")

    content = "\n".join(content_parts)

    return Section(
        title="Violations",
        content=content,
        level=1,
        visible=True,
    )


def create_standard_report_sections(
    results: dict[str, Any],
    *,
    verbosity: str = "standard",
) -> list[Section]:
    """Create standard set of report sections.

    Args:
        results: Complete analysis results.
        verbosity: Report verbosity level.

    Returns:
        List of Section objects for a complete report.

    References:
        REPORT-006
    """
    sections = []

    # Executive summary (all verbosity levels)
    if "summary" in results or "pass_count" in results:
        sections.append(create_executive_summary_section(results))

    # Violations (if any)
    if results.get("violations"):
        sections.append(create_violations_section(results["violations"]))

    # Measurement results
    if "measurements" in results:
        sections.append(create_measurement_results_section(results["measurements"]))

    # Plots (if available)
    if results.get("figures"):
        sections.append(create_plots_section(results["figures"]))

    # Methodology (standard and above)
    if verbosity in ("standard", "detailed", "debug") and "analysis_params" in results:
        sections.append(create_methodology_section(results["analysis_params"], verbosity=verbosity))

    # Conclusions
    if "conclusions" in results or "recommendations" in results:
        sections.append(
            create_conclusions_section(
                results,
                recommendations=results.get("recommendations"),
            )
        )

    # Appendix (detailed and debug only)
    if verbosity in ("detailed", "debug"):
        sections.append(create_appendix_section(results))

    return sections
