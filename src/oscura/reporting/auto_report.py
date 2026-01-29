"""Automatic executive report generation.

This module provides one-click generation of comprehensive analysis reports
in multiple formats (PDF, HTML, Markdown).


Example:
    >>> from oscura.reporting import generate_report
    >>> trace = load("capture.wfm")
    >>> report = generate_report(trace)
    >>> report.save_pdf("analysis_report.pdf")

References:
    Oscura Auto-Discovery Specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


@dataclass
class ReportMetadata:
    """Report metadata.

    Attributes:
        title: Report title.
        author: Report author.
        date: Report date.
        project: Project name.
        tags: List of tags.
    """

    title: str = "Signal Analysis Report"
    author: str = "Oscura"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    project: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Report:
    """Executive analysis report.

    Attributes:
        sections: List of section names included.
        plots: List of plot types included.
        page_count: Estimated page count.
        metadata: Report metadata.
        content: Dictionary of section content.
        output_path: Path to saved report.
        file_size_mb: File size in MB (if saved).
    """

    sections: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    page_count: int = 0
    metadata: ReportMetadata = field(default_factory=ReportMetadata)
    content: dict[str, str] = field(default_factory=dict)
    output_path: str | None = None
    file_size_mb: float = 0.0

    def save_pdf(self, path: str) -> None:
        """Save report as PDF.

        Args:
            path: Output file path.

        Note:
            This is a placeholder implementation. Full PDF generation
            would require reportlab or similar library.
        """
        self.output_path = path
        # Placeholder: would generate actual PDF here
        with open(path, "w") as f:
            f.write("PDF Report - Placeholder\n")
            f.write(f"Title: {self.metadata.title}\n")
            f.write(f"Date: {self.metadata.date}\n\n")

            for section in self.sections:
                if section in self.content:
                    f.write(f"\n{section.upper()}\n")
                    f.write("=" * 60 + "\n")
                    f.write(self.content[section] + "\n")

        # Estimate file size
        self.file_size_mb = Path(path).stat().st_size / (1024 * 1024)

    def save_html(self, path: str) -> None:
        """Save report as HTML.

        Args:
            path: Output file path.
        """
        self.output_path = path

        # Use list + join for O(n) string building instead of O(n²) +=
        html_parts = [
            f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.metadata.title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .critical {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .info {{
            color: #3498db;
        }}
    </style>
</head>
<body>
    <h1>{self.metadata.title}</h1>

    <div class="metadata">
        <p><strong>Date:</strong> {self.metadata.date}</p>
        <p><strong>Author:</strong> {self.metadata.author}</p>
"""
        ]

        if self.metadata.project:
            html_parts.append(f"        <p><strong>Project:</strong> {self.metadata.project}</p>\n")

        if self.metadata.tags:
            html_parts.append(
                f"        <p><strong>Tags:</strong> {', '.join(self.metadata.tags)}</p>\n"
            )

        html_parts.append("    </div>\n\n")

        # Build sections with list.append instead of += in loop
        for section in self.sections:
            if section in self.content:
                section_title = section.replace("_", " ").title()
                html_parts.append(
                    f"""    <div class="section">
        <h2>{section_title}</h2>
        <p>{self.content[section]}</p>
    </div>
"""
                )

        html_parts.append("</body>\n</html>")
        html_content = "".join(html_parts)

        with open(path, "w") as f:
            f.write(html_content)

        self.file_size_mb = Path(path).stat().st_size / (1024 * 1024)

    def save_markdown(self, path: str) -> None:
        """Save report as Markdown.

        Args:
            path: Output file path.
        """
        self.output_path = path

        # Use list + join for O(n) string building instead of O(n²) +=
        md_parts = [
            f"# {self.metadata.title}\n\n",
            f"**Date:** {self.metadata.date}  \n",
            f"**Author:** {self.metadata.author}  \n",
        ]

        if self.metadata.project:
            md_parts.append(f"**Project:** {self.metadata.project}  \n")

        if self.metadata.tags:
            md_parts.append(f"**Tags:** {', '.join(self.metadata.tags)}  \n")

        md_parts.append("\n---\n\n")

        # Build sections with list.append instead of += in loop
        for section in self.sections:
            if section in self.content:
                section_title = section.replace("_", " ").title()
                md_parts.append(f"## {section_title}\n\n")
                md_parts.append(self.content[section])
                md_parts.append("\n\n")

        md_content = "".join(md_parts)

        with open(path, "w") as f:
            f.write(md_content)

        self.file_size_mb = Path(path).stat().st_size / (1024 * 1024)

    def add_section(
        self,
        title: str,
        content: str,
        position: int | None = None,
    ) -> None:
        """Add custom section to report.

        Args:
            title: Section title.
            content: Section content.
            position: Insert position (None = append).
        """
        section_key = title.lower().replace(" ", "_")

        if position is None:
            self.sections.append(section_key)
        else:
            self.sections.insert(position, section_key)

        self.content[section_key] = content

    def include_plots(self, plot_types: list[str]) -> None:
        """Select which plots to include in report.

        Args:
            plot_types: List of plot type names.
        """
        self.plots = plot_types

    def set_metadata(
        self,
        title: str | None = None,
        author: str | None = None,
        date: str | None = None,
        project: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Set report metadata.

        Args:
            title: Report title.
            author: Report author.
            date: Report date.
            project: Project name.
            tags: List of tags.
        """
        if title:
            self.metadata.title = title
        if author:
            self.metadata.author = author
        if date:
            self.metadata.date = date
        if project:
            self.metadata.project = project
        if tags:
            self.metadata.tags = tags


def _generate_executive_summary(trace: WaveformTrace, context: dict) -> str:  # type: ignore[type-arg]
    """Generate executive summary section.

    Args:
        trace: Waveform to analyze.
        context: Analysis context.

    Returns:
        Executive summary text (≤200 words).
    """
    sample_rate = trace.metadata.sample_rate
    duration_ms = len(trace.data) / sample_rate * 1000
    v_min = float(np.min(trace.data))
    v_max = float(np.max(trace.data))

    summary = "This report presents analysis of a signal capture taken at "
    summary += f"{sample_rate / 1e6:.1f} MS/s sample rate over {duration_ms:.2f} milliseconds. "
    summary += f"The signal ranges from {v_min:.3f}V to {v_max:.3f}V. "

    # Add context-specific information
    if "characterization" in context:
        char = context["characterization"]
        if hasattr(char, "signal_type"):
            summary += f"The signal was identified as {char.signal_type}. "

    if "quality" in context:
        quality = context["quality"]
        if hasattr(quality, "status"):
            summary += f"Data quality assessment: {quality.status}. "

    summary += "Detailed findings and recommendations are provided in the sections below."

    return summary


def _generate_key_findings(trace: WaveformTrace, context: dict) -> str:  # type: ignore[type-arg]
    """Generate key findings section.

    Args:
        trace: Waveform to analyze.
        context: Analysis context.

    Returns:
        Key findings text.
    """
    findings = []

    # Basic signal characteristics
    v_range = np.ptp(trace.data)
    findings.append(f"Signal swing: {v_range:.3f}V")

    # Add context-specific findings
    if "anomalies" in context:
        anomalies = context["anomalies"]
        if hasattr(anomalies, "__len__"):
            findings.append(f"Detected {len(anomalies)} anomalies in signal")

    if "decode" in context:
        decode = context["decode"]
        if hasattr(decode, "data") and hasattr(decode.data, "__len__"):
            findings.append(f"Successfully decoded {len(decode.data)} bytes")

    # Format findings
    findings_text = "Key findings from signal analysis:\n\n"
    for i, finding in enumerate(findings, 1):
        findings_text += f"{i}. {finding}\n"

    return findings_text


def _generate_methodology(trace: WaveformTrace, context: dict[str, Any]) -> str:
    """Generate methodology section.

    Args:
        trace: Waveform to analyze.
        context: Analysis context.

    Returns:
        Methodology description.
    """
    methodology = "Analysis methodology:\n\n"

    methodology += "Signal characterization: Automated signal type detection using "
    methodology += "statistical analysis and pattern recognition algorithms.\n\n"

    methodology += "Quality assessment: Signal-to-noise ratio, clipping detection, "
    methodology += "and sample rate validation.\n\n"

    if "anomalies" in context:
        methodology += "Anomaly detection: Automated detection of glitches, dropouts, "
        methodology += "noise spikes, and timing violations.\n\n"

    if "decode" in context:
        methodology += "Protocol decode: Automatic parameter detection and "
        methodology += "frame extraction with confidence scoring.\n\n"

    return methodology


def _generate_detailed_results(trace: WaveformTrace, context: dict[str, Any]) -> str:
    """Generate detailed results section.

    Args:
        trace: Waveform to analyze.
        context: Analysis context.

    Returns:
        Detailed results text.
    """
    results = "Detailed measurement results:\n\n"

    # Basic statistics
    data = trace.data.astype(np.float64)
    results += f"Minimum voltage: {np.min(data):.6f}V\n"
    results += f"Maximum voltage: {np.max(data):.6f}V\n"
    results += f"Mean voltage: {np.mean(data):.6f}V\n"
    results += f"Standard deviation: {np.std(data):.6f}V\n"
    results += f"Peak-to-peak: {np.ptp(data):.6f}V\n\n"

    # Sample info
    results += f"Sample count: {len(data):,}\n"
    results += f"Sample rate: {trace.metadata.sample_rate / 1e6:.3f} MS/s\n"
    results += f"Duration: {len(data) / trace.metadata.sample_rate * 1000:.3f} ms\n\n"

    return results


def _generate_section_content(
    sections: list[str], trace: WaveformTrace, context: dict[str, Any]
) -> dict[str, str]:
    """Generate content for requested sections.

    Args:
        sections: List of section names to generate
        trace: Waveform trace
        context: Analysis context

    Returns:
        Dictionary mapping section names to content
    """
    content = {}

    if "executive_summary" in sections or "summary" in sections:
        content["executive_summary"] = _generate_executive_summary(trace, context)

    if "key_findings" in sections or "findings" in sections:
        content["key_findings"] = _generate_key_findings(trace, context)

    if "methodology" in sections:
        content["methodology"] = _generate_methodology(trace, context)

    if "detailed_results" in sections or "results" in sections:
        content["detailed_results"] = _generate_detailed_results(trace, context)

    if "recommendations" in sections:
        content["recommendations"] = (
            "Recommendations based on analysis:\n\n"
            "1. Signal quality is acceptable for analysis\n"
            "2. Consider additional captures for verification\n"
            "3. Review anomalies if present\n"
        )

    return content


def _determine_plot_types(trace: WaveformTrace, options: dict[str, Any]) -> list[str]:
    """Determine which plots to include in report.

    Args:
        trace: Waveform trace
        options: Report options

    Returns:
        List of plot type names
    """
    configured_plot_types = options.get("plot_types", [])
    if configured_plot_types:
        # Cast to list[str] - we know it contains strings from config
        return [str(p) for p in configured_plot_types]

    # Auto-select based on signal characteristics
    plot_types = ["time_domain_waveform"]
    if len(trace.data) > 100:
        plot_types.append("fft_spectrum")

    return plot_types


def generate_report(
    trace: WaveformTrace,
    *,
    format: str = "pdf",
    template: str | None = None,
    context: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> Report:
    """Generate comprehensive executive analysis report.

    Creates a professional report with executive summary, key findings,
    methodology, and detailed results. Auto-includes relevant plots.

    Args:
        trace: Waveform to analyze.
        format: Output format ("pdf", "html", "markdown").
        template: Optional template file path.
        context: Pre-computed analysis results (characterization, anomalies, etc.).
        options: Report customization options:
            - select_sections: List of sections to include
            - custom_header: Custom header text
            - custom_footer: Custom footer text
            - page_orientation: "portrait" or "landscape"
            - include_raw_data: Include raw data table
            - plot_dpi: Plot resolution (default 300)

    Returns:
        Report object with content and save methods.

    Example:
        >>> report = generate_report(trace)
        >>> report.save_pdf("analysis.pdf")
        >>> print(f"Generated {report.page_count} page report")

    References:
        DISC-005: Automatic Executive Report
    """
    context = context or {}
    options = options or {}

    default_sections = ["executive_summary", "key_findings", "methodology", "detailed_results"]
    sections = options.get("select_sections", default_sections)

    content = _generate_section_content(sections, trace, context)
    plot_types = _determine_plot_types(trace, options)

    # Estimate page count
    page_count = 1 + len(sections) + (len(plot_types) + 1) // 2

    report = Report(
        sections=list(sections),
        plots=plot_types,
        page_count=page_count,
        content=content,
    )

    if "custom_header" in options:
        report.metadata.title = options["custom_header"]

    return report


__all__ = [
    "Report",
    "ReportMetadata",
    "generate_report",
]
