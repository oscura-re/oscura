"""Content generation utilities for reports."""

from oscura.reporting.content.executive import (
    ExecutiveSummary,
    generate_executive_summary,
)
from oscura.reporting.content.filtering import (
    ContentFilter,
    filter_by_audience,
    filter_by_severity,
)
from oscura.reporting.content.minimal import (
    MinimalContent,
    auto_caption,
    generate_compact_text,
)
from oscura.reporting.content.verbosity import (
    VerbosityController,
    VerbosityLevel,
    apply_verbosity_level,
)

__all__ = [
    # Filtering
    "ContentFilter",
    # Executive Summary
    "ExecutiveSummary",
    # Minimal
    "MinimalContent",
    # Verbosity
    "VerbosityController",
    "VerbosityLevel",
    "apply_verbosity_level",
    "auto_caption",
    "filter_by_audience",
    "filter_by_severity",
    "generate_compact_text",
    "generate_executive_summary",
]
