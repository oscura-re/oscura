"""Core report generation functionality - Multi-format renderers.

Note: Renamed from 'core' to 'core_formats' to avoid naming conflict
with core.py module which contains Report, ReportConfig, Section.
"""

from oscura.reporting.core_formats.multi_format import (
    MultiFormatRenderer,
    detect_format_from_extension,
    render_all_formats,
)

__all__ = [
    "MultiFormatRenderer",
    "detect_format_from_extension",
    "render_all_formats",
]
