"""Core report generation for Oscura.

This module provides the main report generation functionality including
report structure, configuration, and output generation.


Example:
    >>> from oscura.reporting import generate_report
    >>> report = generate_report(results, "report.pdf", verbosity="summary")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Section:
    """A section in a report.

    Attributes:
        title: Section title.
        content: Section content (text, tables, figures).
        level: Heading level (1-4).
        collapsible: Whether section is collapsible in HTML output.
        visible: Whether section is visible in output.
    """

    title: str
    content: str | list[Any] = ""
    level: int = 2
    collapsible: bool = False
    visible: bool = True
    subsections: list[Section] = field(default_factory=list)


@dataclass
class ReportConfig:
    """Report generation configuration.

    Attributes:
        title: Report title.
        author: Report author.
        verbosity: Detail level (executive, summary, standard, detailed, debug).
        format: Output format (pdf, html, markdown, docx).
        template: Template name or path.
        page_size: Page size (letter, A4).
        margins: Page margins in inches.
        logo_path: Path to logo image.
        watermark: Watermark text.
        show_toc: Include table of contents.
        show_page_numbers: Include page numbers.
    """

    title: str = "Oscura Analysis Report"
    author: str = ""
    verbosity: Literal["executive", "summary", "standard", "detailed", "debug"] = "standard"
    format: Literal["pdf", "html", "markdown", "docx"] = "pdf"
    template: str = "default"
    page_size: Literal["letter", "A4"] = "letter"
    margins: float = 1.0
    logo_path: str | None = None
    watermark: str | None = None
    show_toc: bool = True
    show_page_numbers: bool = True
    created: datetime = field(default_factory=datetime.now)


@dataclass
class Report:
    """A generated report.

    Attributes:
        config: Report configuration.
        sections: Report sections.
        metadata: Report metadata.
        figures: Embedded figures.
        tables: Embedded tables.
    """

    config: ReportConfig
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    figures: list[Any] = field(default_factory=list)
    tables: list[Any] = field(default_factory=list)

    def add_section(
        self,
        title: str,
        content: str | list[Any] = "",
        level: int = 2,
        **kwargs: Any,
    ) -> Section:
        """Add a section to the report.

        Args:
            title: Section title.
            content: Section content.
            level: Heading level.
            **kwargs: Additional section options.

        Returns:
            The created Section.
        """
        section = Section(title=title, content=content, level=level, **kwargs)
        self.sections.append(section)
        return section

    def add_table(
        self,
        data: list[list[Any]] | NDArray[Any],
        headers: list[str] | None = None,
        caption: str = "",
    ) -> dict:  # type: ignore[type-arg]
        """Add a table to the report.

        Args:
            data: Table data as 2D list or array.
            headers: Column headers.
            caption: Table caption.

        Returns:
            Table reference dictionary.
        """
        table = {
            "type": "table",
            "data": data if isinstance(data, list) else data.tolist(),
            "headers": headers,
            "caption": caption,
            "id": len(self.tables),
        }
        self.tables.append(table)
        return table

    def add_figure(
        self,
        figure: Any,
        caption: str = "",
        width: str = "100%",
    ) -> dict:  # type: ignore[type-arg]
        """Add a figure to the report.

        Args:
            figure: Matplotlib figure or image path.
            caption: Figure caption.
            width: Figure width.

        Returns:
            Figure reference dictionary.
        """
        fig = {
            "type": "figure",
            "figure": figure,
            "caption": caption,
            "width": width,
            "id": len(self.figures),
        }
        self.figures.append(fig)
        return fig

    def generate_executive_summary(
        self,
        results: dict[str, Any],
        key_findings: list[str] | None = None,
    ) -> str:
        """Generate executive summary from results.

        Args:
            results: Analysis results dictionary.
            key_findings: List of key findings to highlight.

        Returns:
            Executive summary text.
        """
        summary_parts = []

        # Overall status
        if "pass_count" in results and "total_count" in results:
            pass_count = results["pass_count"]
            total = results["total_count"]
            if pass_count == total:
                summary_parts.append(f"All {total} tests passed.")
            else:
                fail_count = total - pass_count
                summary_parts.append(
                    f"{fail_count} of {total} tests failed ({fail_count / total * 100:.0f}%)."
                )

        # Key findings
        if key_findings:
            summary_parts.append("\nKey Findings:")
            for finding in key_findings[:5]:  # Top 5
                summary_parts.append(f"- {finding}")

        # Margin summary
        if "min_margin" in results:
            margin = results["min_margin"]
            if margin < 10:
                summary_parts.append(f"\nWarning: Minimum margin is {margin:.1f}%")
            elif margin < 20:
                summary_parts.append(f"\nNote: Minimum margin is {margin:.1f}%")

        return "\n".join(summary_parts)

    def to_markdown(self) -> str:
        """Convert report to Markdown format.

        Returns:
            Markdown string.
        """
        lines = []

        # Title
        lines.append(f"# {self.config.title}")
        lines.append("")

        if self.config.author:
            lines.append(f"**Author:** {self.config.author}")
        lines.append(f"**Date:** {self.config.created.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Sections
        for section in self.sections:
            if not section.visible:
                continue

            prefix = "#" * (section.level + 1)
            lines.append(f"{prefix} {section.title}")
            lines.append("")

            if isinstance(section.content, str):
                lines.append(section.content)
            elif isinstance(section.content, list):
                for item in section.content:
                    if isinstance(item, dict) and item.get("type") == "table":
                        lines.extend(self._table_to_markdown(item))
                    else:
                        lines.append(str(item))
            lines.append("")

            # Subsections
            for subsec in section.subsections:
                if not subsec.visible:
                    continue
                prefix = "#" * (subsec.level + 1)
                lines.append(f"{prefix} {subsec.title}")
                lines.append("")
                if isinstance(subsec.content, str):
                    lines.append(subsec.content)
                lines.append("")

        return "\n".join(lines)

    def _table_to_markdown(self, table: dict) -> list[str]:  # type: ignore[type-arg]
        """Convert table to Markdown format."""
        lines = []
        headers = table.get("headers", [])
        data = table.get("data", [])

        if headers:
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")

        for row in data:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        if table.get("caption"):
            lines.append("")
            lines.append(f"*{table['caption']}*")

        return lines

    def save(self, path: str | Path) -> None:
        """Save report to file.

        Args:
            path: Output file path.
        """
        path = Path(path)

        if path.suffix == ".md":
            content = self.to_markdown()
            path.write_text(content)
        elif path.suffix == ".html":
            content = self.to_html()
            path.write_text(content)
        else:
            # For PDF and other formats, use Markdown as intermediate
            content = self.to_markdown()
            path.with_suffix(".md").write_text(content)

    def to_html(self) -> str:
        """Convert report to HTML format.

        Returns:
            HTML string.
        """
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.config.title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #333; }",
            "h2 { color: #555; border-bottom: 1px solid #ddd; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".pass { color: green; }",
            ".fail { color: red; }",
            ".warning { color: orange; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.config.title}</h1>",
        ]

        if self.config.author:
            lines.append(f"<p><strong>Author:</strong> {self.config.author}</p>")
        lines.append(
            f"<p><strong>Date:</strong> {self.config.created.strftime('%Y-%m-%d %H:%M')}</p>"
        )

        for section in self.sections:
            if not section.visible:
                continue

            tag = f"h{min(section.level + 1, 6)}"
            lines.append(f"<{tag}>{section.title}</{tag}>")

            if isinstance(section.content, str):
                lines.append(f"<p>{section.content}</p>")
            elif isinstance(section.content, list):
                for item in section.content:
                    if isinstance(item, dict) and item.get("type") == "table":
                        lines.extend(self._table_to_html(item))
                    else:
                        lines.append(f"<p>{item}</p>")

        lines.extend(["</body>", "</html>"])
        return "\n".join(lines)

    def _table_to_html(self, table: dict) -> list[str]:  # type: ignore[type-arg]
        """Convert table to HTML format."""
        lines = ["<table>"]
        headers = table.get("headers", [])
        data = table.get("data", [])

        if headers:
            lines.append("<thead><tr>")
            for h in headers:
                lines.append(f"<th>{h}</th>")
            lines.append("</tr></thead>")

        lines.append("<tbody>")
        for row in data:
            lines.append("<tr>")
            for cell in row:
                lines.append(f"<td>{cell}</td>")
            lines.append("</tr>")
        lines.append("</tbody>")

        lines.append("</table>")

        if table.get("caption"):
            lines.append(f"<p><em>{table['caption']}</em></p>")

        return lines


def generate_report(
    results: dict[str, Any],
    output_path: str | Path | None = None,
    *,
    title: str = "Oscura Analysis Report",
    verbosity: Literal["executive", "summary", "standard", "detailed", "debug"] = ("standard"),
    template: str = "default",
    formats: list[str] | None = None,
    **kwargs: Any,
) -> Report:
    """Generate a report from analysis results.

    Creates a formatted report from analysis results with configurable
    verbosity and output formats.

    Args:
        results: Analysis results dictionary.
        output_path: Output file path (optional).
        title: Report title.
        verbosity: Detail level.
        template: Template name.
        formats: Output formats (pdf, html, markdown).
        **kwargs: Additional configuration options.

    Returns:
        Generated Report object.

    Example:
        >>> report = generate_report(results, "report.pdf", verbosity="summary")
    """
    config = ReportConfig(
        title=title,
        verbosity=verbosity,
        template=template,
        **{k: v for k, v in kwargs.items() if hasattr(ReportConfig, k)},
    )

    report = Report(config=config, metadata={"source": "Oscura"})

    # Add executive summary
    if verbosity in ("executive", "summary", "standard", "detailed", "debug"):
        summary = report.generate_executive_summary(results)
        report.add_section("Executive Summary", summary, level=1)

    # Add results section
    if verbosity in ("summary", "standard", "detailed", "debug"):
        _add_results_section(report, results, verbosity)

    # Add methodology section
    if verbosity in ("standard", "detailed", "debug"):
        _add_methodology_section(report, results)

    # Add raw data section
    if verbosity in ("detailed", "debug"):
        _add_raw_data_section(report, results)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        if formats:
            for fmt in formats:
                path = output_path.with_suffix(f".{fmt}")
                report.save(path)
        else:
            report.save(output_path)

    return report


def _add_results_section(
    report: Report,
    results: dict[str, Any],
    verbosity: str,
) -> None:
    """Add results section to report."""
    report.add_section("Test Results", level=1)

    # Create results table
    if "measurements" in results:
        measurements = results["measurements"]
        headers = ["Parameter", "Value", "Specification", "Status"]
        data = []

        for name, meas in measurements.items():
            value = meas.get("value", "N/A")
            spec = meas.get("specification", "N/A")
            status = "PASS" if meas.get("passed", True) else "FAIL"
            data.append([name, value, spec, status])

        report.add_table(data, headers, "Measurement Results")


def _add_methodology_section(
    report: Report,
    results: dict[str, Any],
) -> None:
    """Add methodology section to report."""
    content = []

    if "sample_rate" in results:
        content.append(f"Sample rate: {results['sample_rate']} Hz")
    if "num_samples" in results:
        content.append(f"Number of samples: {results['num_samples']}")
    if "analysis_time" in results:
        content.append(f"Analysis time: {results['analysis_time']:.3f} seconds")

    report.add_section(
        "Methodology",
        "\n".join(content) if content else "Standard analysis methodology applied.",
        level=1,
    )


def _add_raw_data_section(
    report: Report,
    results: dict[str, Any],
) -> None:
    """Add raw data section to report."""
    content = []

    for key, value in results.items():
        if isinstance(value, int | float | str):
            content.append(f"{key}: {value}")

    report.add_section(
        "Raw Data",
        "\n".join(content) if content else "No raw data available.",
        level=1,
        collapsible=True,
    )
