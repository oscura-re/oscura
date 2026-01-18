"""Executive summary auto-generation.

Automatically generates concise executive summaries with key findings,
pass/fail status, and critical violations highlighted.


References:
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ExecutiveSummary:
    """Executive summary of analysis results.

    Attributes:
        overall_status: Overall pass/fail status.
        pass_count: Number of passing tests.
        total_count: Total number of tests.
        key_findings: List of 3-5 key findings.
        critical_violations: List of critical violations.
        min_margin_pct: Minimum margin percentage.
        summary_text: Natural language summary.

    References:
        REPORT-004: Executive Summary Auto-Generation
    """

    overall_status: bool
    pass_count: int
    total_count: int
    key_findings: list[str] = field(default_factory=list)
    critical_violations: list[str] = field(default_factory=list)
    min_margin_pct: float | None = None
    summary_text: str = ""


def generate_executive_summary(
    results: dict[str, Any],
    *,
    max_findings: int = 5,
    length: Literal["short", "detailed"] = "short",
) -> ExecutiveSummary:
    """Generate executive summary from analysis results.

    Automatically extracts top 3-5 key findings, pass/fail status in first
    sentence, and critical violations in bullet list.

    Args:
        results: Analysis results dictionary.
        max_findings: Maximum number of key findings (default 5).
        length: Summary length (short=1 paragraph, detailed=1 page).

    Returns:
        ExecutiveSummary with generated content.

    Example:
        >>> results = {"pass_count": 10, "total_count": 12}
        >>> summary = generate_executive_summary(results)
        >>> print(summary.summary_text)
        'All 10 tests passed with >25% margin. No violations.'

    References:
        REPORT-004: Executive Summary Auto-Generation
    """
    # Extract basic counts
    pass_count = results.get("pass_count", 0)
    total_count = results.get("total_count", 0)
    fail_count = total_count - pass_count if total_count else 0
    overall_status = fail_count == 0

    # Extract violations
    violations = results.get("violations", [])
    critical_violations = [v for v in violations if v.get("severity", "").lower() == "critical"]

    # Extract key findings
    key_findings: list[str] = []
    if critical_violations:
        key_findings.append(
            f"{len(critical_violations)} critical violation(s) require immediate attention"
        )
    elif violations:
        key_findings.append(f"{len(violations)} violation(s) detected")

    # Add margin information
    min_margin = results.get("min_margin")
    if min_margin is not None and min_margin < 20:
        status = "critical" if min_margin < 10 else "marginal"
        key_findings.append(f"Minimum margin is {min_margin:.1f}% ({status})")

    # Build summary text in natural language
    if overall_status and total_count > 0:
        summary_text = f"All {pass_count} tests passed."
        if min_margin is not None and min_margin > 20:
            summary_text += f" Minimum margin: {min_margin:.1f}%."
    elif total_count > 0:
        pct = fail_count / total_count * 100
        summary_text = f"{fail_count} of {total_count} tests failed ({pct:.0f}% failure rate)."
    else:
        summary_text = "Analysis completed successfully."

    if critical_violations:
        summary_text += (
            f" Critical: {len(critical_violations)} violation(s) require immediate action."
        )

    # Add detailed findings for detailed mode
    if length == "detailed" and key_findings:
        summary_text += "\n\nKey Findings:\n"
        summary_text += "\n".join(f"  - {finding}" for finding in key_findings[:max_findings])

    return ExecutiveSummary(
        overall_status=overall_status,
        pass_count=pass_count,
        total_count=total_count,
        key_findings=key_findings[:max_findings],
        critical_violations=[str(v) for v in critical_violations],
        min_margin_pct=min_margin,
        summary_text=summary_text,
    )


__all__ = ["ExecutiveSummary", "generate_executive_summary"]
