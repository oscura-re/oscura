"""Multi-channel report generation for Oscura.

This module provides utilities for generating reports across multiple channels
with channel comparison and aggregation.


Example:
    >>> from oscura.reporting.multichannel import generate_multichannel_report
    >>> report = generate_multichannel_report(channel_results, "multi_report.pdf")
"""

from __future__ import annotations

from typing import Any

from oscura.reporting.core import Report, ReportConfig, Section
from oscura.reporting.tables import create_measurement_table


def generate_multichannel_report(
    channel_results: dict[str, dict[str, Any]],
    *,
    title: str = "Multi-Channel Analysis Report",
    compare_channels: bool = True,
    aggregate_statistics: bool = True,
    individual_sections: bool = True,
    **kwargs: Any,
) -> Report:
    """Generate report for multi-channel analysis.

    Args:
        channel_results: Dictionary mapping channel name to results.
        title: Report title.
        compare_channels: Include channel comparison section.
        aggregate_statistics: Include aggregate statistics across channels.
        individual_sections: Include individual channel sections.
        **kwargs: Additional report configuration options.

    Returns:
        Multi-channel Report object.

    References:
        REPORT-007
    """
    config = ReportConfig(title=title, **kwargs)
    report = Report(config=config)

    # Add executive summary
    summary_content = _generate_multichannel_summary(channel_results)
    report.add_section("Executive Summary", summary_content, level=1)

    # Add aggregate statistics
    if aggregate_statistics:
        stats_section = _create_aggregate_statistics_section(channel_results)
        report.sections.append(stats_section)

    # Add channel comparison
    if compare_channels and len(channel_results) > 1:
        comparison_section = _create_channel_comparison_section(channel_results)
        report.sections.append(comparison_section)

    # Add individual channel sections
    if individual_sections:
        for channel_name, results in channel_results.items():
            channel_section = _create_channel_section(channel_name, results)
            report.sections.append(channel_section)

    return report


def _generate_multichannel_summary(channel_results: dict[str, dict[str, Any]]) -> str:
    """Generate summary for multi-channel report."""
    summary_parts = []

    total_channels = len(channel_results)
    summary_parts.append(f"Analyzed {total_channels} channel(s).")

    # Aggregate pass/fail across channels
    total_tests = 0
    total_passed = 0

    for results in channel_results.values():
        total_tests += results.get("total_count", 0)
        total_passed += results.get("pass_count", 0)

    if total_tests > 0:
        total_failed = total_tests - total_passed
        summary_parts.append(
            f"\nOverall: {total_passed}/{total_tests} tests passed "
            f"({total_passed / total_tests * 100:.1f}% pass rate)."
        )

        if total_failed > 0:
            summary_parts.append(f"{total_failed} test(s) failed across all channels.")

    # Channel-specific summary
    failed_channels = []
    for channel_name, results in channel_results.items():
        pass_count = results.get("pass_count", 0)
        total_count = results.get("total_count", 0)
        if total_count > 0 and pass_count < total_count:
            failed_channels.append(channel_name)

    if failed_channels:
        summary_parts.append(f"\nChannels with failures: {', '.join(failed_channels)}")
    else:
        summary_parts.append("\nAll channels passed all tests.")

    return "\n".join(summary_parts)


def _create_aggregate_statistics_section(
    channel_results: dict[str, dict[str, Any]],
) -> Section:
    """Create aggregate statistics section across all channels."""
    # Collect all measurement parameters
    all_params = set()
    for results in channel_results.values():
        if "measurements" in results:
            all_params.update(results["measurements"].keys())

    # Build aggregate table
    import numpy as np

    headers = ["Parameter", "Min", "Mean", "Max", "Std Dev"]
    rows = []

    for param in sorted(all_params):
        values = []
        unit = ""

        for results in channel_results.values():
            if "measurements" in results and param in results["measurements"]:
                meas = results["measurements"][param]
                if "value" in meas and meas["value"] is not None:
                    values.append(meas["value"])
                    if not unit and "unit" in meas:
                        unit = meas["unit"]

        if values:
            from oscura.reporting.formatting import NumberFormatter

            formatter = NumberFormatter()
            rows.append(
                [
                    param,
                    formatter.format(np.min(values), unit),
                    formatter.format(np.mean(values), unit),
                    formatter.format(np.max(values), unit),
                    formatter.format(np.std(values), unit),
                ]
            )

    table = {"type": "table", "headers": headers, "data": rows}

    return Section(
        title="Aggregate Statistics",
        content=[table],
        level=1,
        visible=True,
    )


def _create_channel_comparison_section(
    channel_results: dict[str, dict[str, Any]],
) -> Section:
    """Create channel-to-channel comparison section."""
    from oscura.reporting.formatting import NumberFormatter

    formatter = NumberFormatter()

    # Build comparison table
    channel_names = list(channel_results.keys())
    headers = ["Parameter", *channel_names]

    # Collect all parameters
    all_params = set()
    for results in channel_results.values():
        if "measurements" in results:
            all_params.update(results["measurements"].keys())

    rows = []
    for param in sorted(all_params):
        row = [param]

        for channel_name in channel_names:
            results = channel_results[channel_name]
            if "measurements" in results and param in results["measurements"]:
                meas = results["measurements"][param]
                value = meas.get("value")
                unit = meas.get("unit", "")
                if value is not None:
                    row.append(formatter.format(value, unit))
                else:
                    row.append("-")
            else:
                row.append("-")

        rows.append(row)

    table = {"type": "table", "headers": headers, "data": rows}

    return Section(
        title="Channel Comparison",
        content=[table],
        level=1,
        visible=True,
    )


def _create_channel_section(
    channel_name: str,
    results: dict[str, Any],
) -> Section:
    """Create individual channel section."""
    subsections = []

    # Channel summary
    summary_parts = []
    if "pass_count" in results and "total_count" in results:
        pass_count = results["pass_count"]
        total = results["total_count"]
        summary_parts.append(
            f"{pass_count}/{total} tests passed ({pass_count / total * 100:.1f}% pass rate)."
        )

    # Measurements
    if "measurements" in results:
        table = create_measurement_table(results["measurements"], format="dict")
        subsections.append(
            Section(
                title="Measurements",
                content=[table],
                level=3,
                visible=True,
            )
        )

    return Section(
        title=f"Channel: {channel_name}",
        content="\n".join(summary_parts) if summary_parts else "",
        level=2,
        visible=True,
        subsections=subsections,
    )


def create_channel_crosstalk_section(
    crosstalk_results: dict[str, Any],
) -> Section:
    """Create channel crosstalk analysis section.

    Args:
        crosstalk_results: Crosstalk analysis results between channels.

    Returns:
        Crosstalk Section object.

    References:
        REPORT-007
    """
    from oscura.reporting.formatting import NumberFormatter

    formatter = NumberFormatter()

    if "crosstalk_matrix" in crosstalk_results:
        matrix = crosstalk_results["crosstalk_matrix"]
        channels = crosstalk_results.get("channels", [])

        headers = ["Aggressor → Victim", *channels]
        rows = []

        for i, aggressor in enumerate(channels):
            row = [aggressor]
            for j, _victim in enumerate(channels):
                if i == j:
                    row.append("-")
                else:
                    crosstalk_db = matrix[i][j]
                    row.append(formatter.format(crosstalk_db, "dB"))
            rows.append(row)

        table = {"type": "table", "headers": headers, "data": rows}
        content = [
            "Channel-to-channel crosstalk measurements:\n",
            table,
        ]
    else:
        content = "No crosstalk analysis available."  # type: ignore[assignment]

    return Section(
        title="Channel Crosstalk Analysis",
        content=content,
        level=2,
        visible=True,
    )
