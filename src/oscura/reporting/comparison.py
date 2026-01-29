"""Comparison report generation for Oscura.

This module provides utilities for comparing multiple traces or test runs
and generating comparison reports with diff visualization.


Example:
    >>> from oscura.reporting.comparison import generate_comparison_report
    >>> report = generate_comparison_report(baseline, current, "comparison.pdf")
"""

from typing import Any, Literal

from oscura.reporting.core import Report, ReportConfig, Section
from oscura.reporting.tables import create_comparison_table


def generate_comparison_report(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    title: str = "Comparison Report",
    mode: Literal["side_by_side", "inline"] = "side_by_side",
    show_only_changes: bool = False,
    highlight_changes: bool = False,
    **kwargs: Any,
) -> Report:
    """Generate comparison report between baseline and current results.

    Args:
        baseline: Baseline results dictionary.
        current: Current results dictionary.
        title: Report title.
        mode: Comparison mode (side_by_side or inline).
        show_only_changes: Only show changed measurements.
        highlight_changes: Highlight changes in output. Reserved for future use.
        **kwargs: Additional report configuration options.

    Returns:
        Comparison Report object.

    References:
        REPORT-008
    """
    config = ReportConfig(title=title, **kwargs)
    report = Report(config=config)

    # Add summary
    summary = _generate_comparison_summary(baseline, current)
    report.add_section("Comparison Summary", summary, level=1)

    # Add change details
    changes_section = _create_changes_section(
        baseline,
        current,
        show_only_changes=show_only_changes,
    )
    report.sections.append(changes_section)

    # Add violations comparison
    if "violations" in baseline or "violations" in current:
        violations_section = _create_violations_comparison_section(
            baseline,
            current,
        )
        report.sections.append(violations_section)

    # Add detailed comparison
    if "measurements" in baseline or "measurements" in current:
        detailed_section = _create_detailed_comparison_section(
            baseline.get("measurements", {}),
            current.get("measurements", {}),
            mode=mode,
        )
        report.sections.append(detailed_section)

    return report


def _generate_comparison_summary(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> str:
    """Generate comparison summary."""
    summary_parts: list[str] = []

    baseline_meas = baseline.get("measurements", {})
    current_meas = current.get("measurements", {})

    changed, improved, degraded = _categorize_parameter_changes(baseline_meas, current_meas)

    _add_parameter_change_summary(
        summary_parts, baseline_meas, current_meas, changed, improved, degraded
    )
    _add_pass_rate_comparison(summary_parts, baseline, current)

    return "\n".join(summary_parts)


def _categorize_parameter_changes(
    baseline_meas: dict[str, Any], current_meas: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    """Categorize parameter changes.

    Args:
        baseline_meas: Baseline measurements.
        current_meas: Current measurements.

    Returns:
        Tuple of (changed, improved, degraded) parameter lists.
    """
    all_params = set(baseline_meas.keys()) | set(current_meas.keys())
    changed_params = []
    improved_params = []
    degraded_params = []

    for param in all_params:
        base_val = baseline_meas.get(param, {}).get("value")
        curr_val = current_meas.get(param, {}).get("value")

        if base_val is not None and curr_val is not None:
            if abs(curr_val - base_val) / abs(base_val) > 0.05:
                changed_params.append(param)

                base_passed = baseline_meas.get(param, {}).get("passed", True)
                curr_passed = current_meas.get(param, {}).get("passed", True)

                if not base_passed and curr_passed:
                    improved_params.append(param)
                elif base_passed and not curr_passed:
                    degraded_params.append(param)

    return changed_params, improved_params, degraded_params


def _add_parameter_change_summary(
    summary_parts: list[str],
    baseline_meas: dict[str, Any],
    current_meas: dict[str, Any],
    changed: list[str],
    improved: list[str],
    degraded: list[str],
) -> None:
    """Add parameter change summary.

    Args:
        summary_parts: List to append to.
        baseline_meas: Baseline measurements.
        current_meas: Current measurements.
        changed: Changed parameters.
        improved: Improved parameters.
        degraded: Degraded parameters.
    """
    all_params = set(baseline_meas.keys()) | set(current_meas.keys())
    summary_parts.append(f"Comparing {len(all_params)} parameter(s) between baseline and current.")

    if changed:
        summary_parts.append(f"\n{len(changed)} measurement(s) changed significantly (>5%).")

    if improved:
        summary_parts.append(f"\n✓ {len(improved)} parameter(s) improved (failures → passes).")

    if degraded:
        summary_parts.append(f"\n✗ {len(degraded)} parameter(s) degraded (passes → failures).")


def _add_pass_rate_comparison(
    summary_parts: list[str], baseline: dict[str, Any], current: dict[str, Any]
) -> None:
    """Add pass rate comparison.

    Args:
        summary_parts: List to append to.
        baseline: Baseline results.
        current: Current results.
    """
    baseline_pass = baseline.get("pass_count", 0)
    baseline_total = baseline.get("total_count", 0)
    current_pass = current.get("pass_count", 0)
    current_total = current.get("total_count", 0)

    if baseline_total > 0 and current_total > 0:
        baseline_rate = baseline_pass / baseline_total * 100
        current_rate = current_pass / current_total * 100
        delta = current_rate - baseline_rate

        summary_parts.append(
            f"\nPass rate: {baseline_rate:.1f}% → {current_rate:.1f}% ({delta:+.1f}% change)"
        )


def _create_changes_section(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    show_only_changes: bool = False,
) -> Section:
    """Create section detailing changes."""
    baseline_meas = baseline.get("measurements", {})
    current_meas = current.get("measurements", {})

    # Create comparison table
    table = create_comparison_table(
        baseline_meas,
        current_meas,
        format="dict",
        show_delta=True,
        show_percent_change=True,
    )

    # Filter to only changes if requested
    if show_only_changes:
        filtered_rows = []
        for row in table["data"]:  # type: ignore[index]
            # Check if delta is significant
            if len(row) >= 4:  # Has delta column
                delta_str = str(row[3])
                if delta_str not in {"-", "0"}:
                    filtered_rows.append(row)

        table["data"] = filtered_rows  # type: ignore[index]

    return Section(
        title="Measurement Changes",
        content=[table],
        level=1,
        visible=True,
    )


def _create_violations_comparison_section(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> Section:
    """Create section comparing violations."""
    baseline_violations = {v.get("parameter") for v in baseline.get("violations", [])}
    current_violations = {v.get("parameter") for v in current.get("violations", [])}

    content_parts = []

    # New violations
    new_violations = current_violations - baseline_violations
    if new_violations:
        content_parts.append("**New Violations:**")
        for param in sorted(new_violations):
            content_parts.append(f"- {param}")

    # Resolved violations
    resolved_violations = baseline_violations - current_violations
    if resolved_violations:
        content_parts.append("\n**Resolved Violations:**")
        for param in sorted(resolved_violations):
            content_parts.append(f"- {param}")

    # Persistent violations
    persistent_violations = baseline_violations & current_violations
    if persistent_violations:
        content_parts.append("\n**Persistent Violations:**")
        for param in sorted(persistent_violations):
            content_parts.append(f"- {param}")

    if not content_parts:
        content_parts.append("No violations in either baseline or current.")

    content = "\n".join(content_parts)

    return Section(
        title="Violations Comparison",
        content=content,
        level=1,
        visible=True,
    )


def _create_detailed_comparison_section(
    baseline_meas: dict[str, Any],
    current_meas: dict[str, Any],
    *,
    mode: str = "side_by_side",
) -> Section:
    """Create detailed measurement comparison section."""
    from oscura.reporting.formatting import NumberFormatter

    formatter = NumberFormatter()

    content_parts = []

    all_params = sorted(set(baseline_meas.keys()) | set(current_meas.keys()))

    for param in all_params:
        base = baseline_meas.get(param, {})
        curr = current_meas.get(param, {})

        base_val = base.get("value")
        curr_val = curr.get("value")
        unit = base.get("unit", curr.get("unit", ""))

        if base_val is not None and curr_val is not None:
            delta = curr_val - base_val
            pct_change = (delta / base_val * 100) if base_val != 0 else 0

            base_str = formatter.format(base_val, unit)
            curr_str = formatter.format(curr_val, unit)
            delta_str = formatter.format(delta, unit)

            # Determine if improved/degraded
            base_passed = base.get("passed", True)
            curr_passed = curr.get("passed", True)

            status = ""
            if not base_passed and curr_passed:
                status = " ✓ IMPROVED"
            elif base_passed and not curr_passed:
                status = " ✗ DEGRADED"

            content_parts.append(
                f"**{param}:** {base_str} → {curr_str} (Δ {delta_str}, {pct_change:+.1f}%){status}"
            )

    content = "\n\n".join(content_parts) if content_parts else "No measurements to compare."

    return Section(
        title="Detailed Comparison",
        content=content,
        level=1,
        visible=True,
        collapsible=True,
    )


def compare_waveforms(
    baseline_signal: dict[str, Any],
    current_signal: dict[str, Any],
) -> dict[str, Any]:
    """Compare two waveforms and extract differences.

    Args:
        baseline_signal: Baseline waveform data.
        current_signal: Current waveform data.

    Returns:
        Dictionary with comparison metrics.

    References:
        REPORT-008
    """
    import numpy as np

    comparison = {
        "correlation": None,
        "rms_difference": None,
        "max_difference": None,
        "mean_difference": None,
    }

    base_data = baseline_signal.get("data")
    curr_data = current_signal.get("data")

    if base_data is not None and curr_data is not None:
        # Ensure same length
        min_len = min(len(base_data), len(curr_data))
        base_data = base_data[:min_len]
        curr_data = curr_data[:min_len]

        # Correlation
        comparison["correlation"] = np.corrcoef(base_data, curr_data)[0, 1]

        # Differences
        diff = curr_data - base_data
        comparison["rms_difference"] = np.sqrt(np.mean(diff**2))
        comparison["max_difference"] = np.max(np.abs(diff))
        comparison["mean_difference"] = np.mean(diff)

    return comparison
