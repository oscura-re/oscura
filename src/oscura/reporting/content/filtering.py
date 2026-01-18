"""Smart content filtering for reports.

Shows only relevant information based on context, severity, and audience
with conditional sections and violation-only modes.


References:
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from oscura.reporting.formatting.standards import Severity


class AudienceType(Enum):
    """Audience types for content filtering."""

    EXECUTIVE = "executive"
    ENGINEERING = "engineering"
    DEBUG = "debug"
    REGULATORY = "regulatory"
    PRODUCTION = "production"


@dataclass
class ContentFilter:
    """Smart content filtering configuration.

    Attributes:
        severity_threshold: Minimum severity to show (critical/warning/info).
        audience: Target audience type.
        show_only: Show only specific content (violations, changes, all).
        hide_empty_sections: Hide sections with no data.
        relevance_threshold: Minimum relevance score (0.0-1.0).

    References:
        REPORT-005: Smart Content Filtering
    """

    severity_threshold: Severity = Severity.INFO
    audience: AudienceType = AudienceType.ENGINEERING
    show_only: Literal["all", "violations", "changes"] = "all"
    hide_empty_sections: bool = True
    relevance_threshold: float = 0.5


def filter_by_severity(
    items: list[dict[str, Any]],
    min_severity: Severity | str,
) -> list[dict[str, Any]]:
    """Filter items by severity level.

    Args:
        items: List of items with 'severity' field.
        min_severity: Minimum severity to include.

    Returns:
        Filtered list of items.

    Example:
        >>> items = [{"name": "test1", "severity": "critical"},
        ...          {"name": "test2", "severity": "info"}]
        >>> filtered = filter_by_severity(items, "warning")
        >>> len(filtered)  # Only critical items
        1

    References:
        REPORT-005: Smart Content Filtering
    """
    if isinstance(min_severity, str):
        min_severity = Severity(min_severity.lower())

    severity_order = {
        Severity.INFO: 0,
        Severity.WARNING: 1,
        Severity.ERROR: 2,
        Severity.CRITICAL: 3,
    }

    min_level = severity_order.get(min_severity, 0)

    filtered = []
    for item in items:
        item_severity_str = item.get("severity", "info").lower()
        try:
            item_severity = Severity(item_severity_str)
            item_level = severity_order.get(item_severity, 0)
            if item_level >= min_level:
                filtered.append(item)
        except (ValueError, KeyError):
            continue

    return filtered


def filter_by_audience(
    content: dict[str, Any],
    audience: AudienceType | str,
) -> dict[str, Any]:
    """Filter content for specific audience.

    Args:
        content: Content dictionary with various sections.
        audience: Target audience type.

    Returns:
        Filtered content dictionary.

    Example:
        >>> content = {"methodology": "...", "results": "...", "raw_data": "..."}
        >>> filtered = filter_by_audience(content, "executive")
        >>> "raw_data" in filtered  # False - executives don't need raw data
        False

    References:
        REPORT-005: Smart Content Filtering
    """
    if isinstance(audience, str):
        audience = AudienceType(audience.lower())

    # Define what each audience sees
    audience_sections = {
        AudienceType.EXECUTIVE: ["executive_summary", "key_findings", "recommendations"],
        AudienceType.ENGINEERING: ["summary", "results", "methodology", "plots"],
        AudienceType.DEBUG: ["summary", "results", "methodology", "plots", "raw_data", "logs"],
        AudienceType.REGULATORY: ["summary", "compliance", "test_procedures", "standards"],
        AudienceType.PRODUCTION: ["summary", "pass_fail", "margin", "yield"],
    }

    allowed_sections = audience_sections.get(audience, list(content.keys()))

    filtered = {}
    for key, value in content.items():
        if key in allowed_sections:
            filtered[key] = value

    return filtered


def calculate_relevance_score(
    item: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> float:
    """Calculate relevance score for content item.

    Args:
        item: Content item.
        context: Optional context for relevance scoring.

    Returns:
        Relevance score 0.0-1.0.

    References:
        REPORT-005: Smart Content Filtering
    """
    score = 0.5  # Base score

    # Increase score for violations
    if item.get("status") == "fail":
        score += 0.3

    # Increase score for critical severity
    severity = item.get("severity", "").lower()
    if severity == "critical":
        score += 0.3
    elif severity == "warning":
        score += 0.1

    # Increase score for outliers
    if item.get("is_outlier"):
        score += 0.2

    # Increase score for low margins
    margin = item.get("margin_pct")
    if margin is not None and margin < 20:
        score += 0.2

    return min(1.0, score)


__all__ = [
    "AudienceType",
    "ContentFilter",
    "calculate_relevance_score",
    "filter_by_audience",
    "filter_by_severity",
]
