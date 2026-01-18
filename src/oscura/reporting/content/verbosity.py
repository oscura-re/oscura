"""Verbosity level control for reports.

Configurable report detail level for different audiences with 5 distinct
verbosity levels from executive to debug.


References:
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.reporting.core import Report


class VerbosityLevel(Enum):
    """Verbosity levels for report detail.

    - EXECUTIVE: 1 page, pass/fail + key findings only
    - SUMMARY: 2-5 pages, results + brief context, no raw data
    - STANDARD: 5-20 pages, full results + methodology + plots
    - DETAILED: 20-50 pages, all measurements + intermediate results
    - DEBUG: 50+ pages, raw data + traces + full provenance

    References:
        REPORT-009: Verbosity Level Control
    """

    EXECUTIVE = "executive"
    SUMMARY = "summary"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class VerbosityController:
    """Verbosity level controller.

    Attributes:
        level: Current verbosity level.

    References:
        REPORT-009: Verbosity Level Control
    """

    level: VerbosityLevel = VerbosityLevel.STANDARD

    def should_include_section(self, section_name: str) -> bool:
        """Determine if section should be included at current verbosity.

        Args:
            section_name: Name of section to check.

        Returns:
            True if section should be included.

        References:
            REPORT-009: Verbosity Level Control
        """
        # Define sections for each level
        sections_by_level = {
            VerbosityLevel.EXECUTIVE: {"executive_summary", "key_findings"},
            VerbosityLevel.SUMMARY: {"summary", "results", "key_plots"},
            VerbosityLevel.STANDARD: {
                "summary",
                "results",
                "methodology",
                "plots",
                "tables",
            },
            VerbosityLevel.DETAILED: {
                "summary",
                "results",
                "methodology",
                "plots",
                "tables",
                "measurements",
                "intermediate_results",
            },
            VerbosityLevel.DEBUG: {
                "summary",
                "results",
                "methodology",
                "plots",
                "tables",
                "measurements",
                "intermediate_results",
                "raw_data",
                "logs",
                "provenance",
            },
        }

        allowed_sections = sections_by_level.get(self.level, set())
        return section_name in allowed_sections

    def get_max_pages(self) -> int:
        """Get maximum pages for current verbosity level.

        Returns:
            Maximum page count.

        References:
            REPORT-009: Verbosity Level Control
        """
        max_pages = {
            VerbosityLevel.EXECUTIVE: 1,
            VerbosityLevel.SUMMARY: 5,
            VerbosityLevel.STANDARD: 20,
            VerbosityLevel.DETAILED: 50,
            VerbosityLevel.DEBUG: 999,
        }
        return max_pages.get(self.level, 20)


def apply_verbosity_level(
    report: Report,
    level: VerbosityLevel | str,
) -> None:
    """Apply verbosity level to report.

    Args:
        report: Report object to modify.
        level: Verbosity level to apply.

    Example:
        >>> from oscura.reporting.core import Report, ReportConfig
        >>> report = Report(config=ReportConfig())
        >>> apply_verbosity_level(report, "summary")

    References:
        REPORT-009: Verbosity Level Control
    """
    if isinstance(level, str):
        level = VerbosityLevel(level.lower())

    controller = VerbosityController(level=level)

    # Filter sections based on verbosity
    if hasattr(report, "sections"):
        visible_sections = []
        for section in report.sections:
            if controller.should_include_section(section.title.lower()):
                visible_sections.append(section)

        report.sections = visible_sections

    # Update config
    if hasattr(report, "config"):
        report.config.verbosity = level.value


__all__ = [
    "VerbosityController",
    "VerbosityLevel",
    "apply_verbosity_level",
]
