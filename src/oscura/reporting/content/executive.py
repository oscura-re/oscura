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
    # Extract counts and violations
    pass_count, total_count, fail_count, overall_status = _extract_counts(results)
    critical_violations = _extract_critical_violations(results)

    # Build key findings
    min_margin = results.get("min_margin")
    key_findings = _build_key_findings(
        critical_violations, results.get("violations", []), min_margin
    )

    # Build summary text
    summary_text = _build_summary_text(
        overall_status,
        pass_count,
        total_count,
        fail_count,
        min_margin,
        critical_violations,
        key_findings,
        length,
        max_findings,
    )

    return ExecutiveSummary(
        overall_status=overall_status,
        pass_count=pass_count,
        total_count=total_count,
        key_findings=key_findings[:max_findings],
        critical_violations=[str(v) for v in critical_violations],
        min_margin_pct=min_margin,
        summary_text=summary_text,
    )


def _extract_counts(results: dict[str, Any]) -> tuple[int, int, int, bool]:
    """Extract pass/fail counts from results."""
    pass_count = results.get("pass_count", 0)
    total_count = results.get("total_count", 0)
    fail_count = total_count - pass_count if total_count else 0
    overall_status = fail_count == 0
    return pass_count, total_count, fail_count, overall_status


def _extract_critical_violations(results: dict[str, Any]) -> list[Any]:
    """Extract critical violations from results."""
    violations = results.get("violations", [])
    return [v for v in violations if v.get("severity", "").lower() == "critical"]


def _build_key_findings(
    critical_violations: list[Any], violations: list[Any], min_margin: float | None
) -> list[str]:
    """Build key findings list."""
    key_findings: list[str] = []

    if critical_violations:
        key_findings.append(
            f"{len(critical_violations)} critical violation(s) require immediate attention"
        )
    elif violations:
        key_findings.append(f"{len(violations)} violation(s) detected")

    if min_margin is not None and min_margin < 20:
        status = "critical" if min_margin < 10 else "marginal"
        key_findings.append(f"Minimum margin is {min_margin:.1f}% ({status})")

    return key_findings


def _build_summary_text(
    overall_status: bool,
    pass_count: int,
    total_count: int,
    fail_count: int,
    min_margin: float | None,
    critical_violations: list[Any],
    key_findings: list[str],
    length: Literal["short", "detailed"],
    max_findings: int,
) -> str:
    """Build natural language summary text."""
    # Use list + join for O(n) string building instead of O(n²) +=
    text_parts: list[str] = []

    # Base status message
    if overall_status and total_count > 0:
        text_parts.append(f"All {pass_count} tests passed.")
        if min_margin is not None and min_margin > 20:
            text_parts.append(f" Minimum margin: {min_margin:.1f}%.")
    elif total_count > 0:
        pct = fail_count / total_count * 100
        text_parts.append(f"{fail_count} of {total_count} tests failed ({pct:.0f}% failure rate).")
    else:
        text_parts.append("Analysis completed successfully.")

    # Add critical violations warning
    if critical_violations:
        text_parts.append(
            f" Critical: {len(critical_violations)} violation(s) require immediate action."
        )

    # Add detailed findings if requested
    if length == "detailed" and key_findings:
        text_parts.append("\n\nKey Findings:\n")
        text_parts.append("\n".join(f"  - {finding}" for finding in key_findings[:max_findings]))

    return "".join(text_parts)


__all__ = ["ExecutiveSummary", "generate_executive_summary"]
