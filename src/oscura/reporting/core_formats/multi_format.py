"""Multi-format report output.

Generate PDF, HTML, Markdown, DOCX from single report definition with
format-specific customization hooks.


References:
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from oscura.reporting.core import Report

FormatType = Literal["pdf", "html", "markdown", "docx", "json"]


@dataclass
class MultiFormatRenderer:
    """Multi-format report renderer.

    One YAML template to multiple renderers (PDF/HTML/Markdown/DOCX).

    Attributes:
        formats: List of output formats to generate.
        format_options: Format-specific options.
        auto_detect_features: Auto-select features per format (e.g., interactive plots in HTML).
        consistent_content: Ensure same data across formats.

    References:
        REPORT-010: Multi-Format Output
    """

    formats: list[FormatType] = field(default_factory=lambda: ["pdf"])
    format_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    auto_detect_features: bool = True
    consistent_content: bool = True


def detect_format_from_extension(path: str | Path) -> FormatType:
    """Detect output format from file extension.

    Args:
        path: File path.

    Returns:
        Detected format type.

    Example:
        >>> detect_format_from_extension("report.pdf")
        'pdf'
        >>> detect_format_from_extension("report.html")
        'html'

    References:
        REPORT-010: Multi-Format Output
    """
    path = Path(path)
    ext = path.suffix.lower()

    format_map = {
        ".pdf": "pdf",
        ".html": "html",
        ".htm": "html",
        ".md": "markdown",
        ".markdown": "markdown",
        ".docx": "docx",
        ".json": "json",
    }

    return format_map.get(ext, "pdf")  # type: ignore[return-value]


def render_all_formats(
    report: Report,
    base_path: str | Path,
    formats: list[FormatType] | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Generate report in multiple formats.

    Batch generation of all formats in single call with consistent content.

    Args:
        report: Report object to render.
        base_path: Base output path (without extension).
        formats: List of formats to generate (default: ["pdf", "html"]).
        **kwargs: Additional rendering options.

    Returns:
        Dictionary mapping format to output path.

    Example:
        >>> from oscura.reporting.core import Report, ReportConfig
        >>> report = Report(config=ReportConfig(title="Test Report"))
        >>> paths = render_all_formats(report, "report", formats=["pdf", "html"])
        >>> print(paths)
        {'pdf': 'report.pdf', 'html': 'report.html'}

    References:
        REPORT-010: Multi-Format Output
    """
    if formats is None:
        formats = ["pdf", "html"]

    base_path = Path(base_path)
    output_paths: dict[str, str] = {}

    for fmt in formats:
        # Determine output path
        if fmt == "pdf":
            output_path = base_path.with_suffix(".pdf")
            _render_pdf(report, output_path, **kwargs)
        elif fmt == "html":
            output_path = base_path.with_suffix(".html")
            _render_html(report, output_path, **kwargs)
        elif fmt == "markdown":
            output_path = base_path.with_suffix(".md")
            _render_markdown(report, output_path, **kwargs)
        elif fmt == "docx":
            output_path = base_path.with_suffix(".docx")
            _render_docx(report, output_path, **kwargs)
        elif fmt == "json":
            output_path = base_path.with_suffix(".json")
            _render_json(report, output_path, **kwargs)
        else:
            continue  # type: ignore[unreachable]

        output_paths[fmt] = str(output_path)

    return output_paths


def _render_pdf(report: Report, path: Path, **kwargs: Any) -> None:
    """Render to PDF."""
    from oscura.reporting.renderers.pdf import render_to_pdf

    render_to_pdf(report, str(path), **kwargs)


def _render_html(report: Report, path: Path, **kwargs: Any) -> None:
    """Render to HTML."""
    from oscura.reporting.html import save_html_report

    save_html_report(report, str(path))


def _render_markdown(report: Report, path: Path, **kwargs: Any) -> None:
    """Render to Markdown."""
    # Use list + join for O(n) string building instead of O(n²) +=
    content_parts = [f"# {report.config.title}\n\n"]

    if hasattr(report.config, "author") and report.config.author:
        content_parts.append(f"**Author:** {report.config.author}  \n")

    content_parts.append(f"**Date:** {report.config.created.strftime('%Y-%m-%d')}\n\n")

    # Add sections with list.append instead of += in loops
    for section in report.sections:
        if not section.visible:
            continue

        content_parts.append(f"{'#' * (section.level + 1)} {section.title}\n\n")

        if isinstance(section.content, str):
            content_parts.append(section.content)
            content_parts.append("\n\n")
        elif isinstance(section.content, list):
            for item in section.content:
                content_parts.append(str(item))
                content_parts.append("\n\n")

    content = "".join(content_parts)
    path.write_text(content)


def _render_docx(report: Report, path: Path, **kwargs: Any) -> None:
    """Render to DOCX."""
    # Placeholder - would use python-docx
    path.write_text(f"DOCX rendering not yet implemented for {report.config.title}")


def _render_json(report: Report, path: Path, **kwargs: Any) -> None:
    """Render to JSON."""
    import json

    data = {
        "title": report.config.title,
        "author": report.config.author,
        "created": report.config.created.isoformat(),
        "sections": [
            {
                "title": s.title,
                "level": s.level,
                "content": str(s.content),
            }
            for s in report.sections
            if s.visible
        ],
    }

    path.write_text(json.dumps(data, indent=2))


__all__ = [
    "MultiFormatRenderer",
    "detect_format_from_extension",
    "render_all_formats",
]
