"""Vintage logic analysis report generation.

This module provides comprehensive reporting capabilities for vintage logic
analysis results, including HTML, PDF, and Markdown formats.

Example:
    >>> from oscura.reporting.vintage_logic_report import generate_vintage_logic_report
    >>> report = generate_vintage_logic_report(result, traces, output_dir="./output")
    >>> report.save_html("analysis.html")
    >>> report.save_markdown("analysis.md")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.analyzers.digital.vintage_result import VintageLogicAnalysisResult
    from oscura.core.types import DigitalTrace, WaveformTrace


@dataclass
class ReportMetadata:
    """Report metadata.

    Attributes:
        title: Report title.
        author: Report author.
        timestamp: Report generation timestamp.
        version: Report format version.
    """

    title: str
    author: str = "Oscura Vintage Logic Analyzer"
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class VintageLogicReport:
    """Vintage logic analysis report.

    Attributes:
        result: Analysis result object.
        plots: Dictionary mapping plot names to file paths.
        metadata: Report metadata.
    """

    result: VintageLogicAnalysisResult
    plots: dict[str, Path]
    metadata: ReportMetadata

    def save_html(self, path: str | Path, **options: Any) -> Path:
        """Generate comprehensive HTML report.

        Args:
            path: Output HTML file path.
            **options: Additional options (currently unused, reserved for future).

        Returns:
            Path to saved HTML file.

        Example:
            >>> report.save_html("analysis.html")
        """

        path = Path(path)

        # Generate HTML content
        html = _generate_html_report(self)

        # Write to file
        path.write_text(html, encoding="utf-8")

        return path

    def save_markdown(self, path: str | Path) -> Path:
        """Generate markdown summary.

        Args:
            path: Output markdown file path.

        Returns:
            Path to saved markdown file.

        Example:
            >>> report.save_markdown("analysis.md")
        """
        path = Path(path)

        # Generate markdown content
        md = _generate_markdown_report(self)

        # Write to file
        path.write_text(md, encoding="utf-8")

        return path

    def save_pdf(self, path: str | Path, **options: Any) -> Path:
        """Generate PDF report.

        Currently generates HTML first, then suggests using browser print
        or external tools for PDF conversion.

        Args:
            path: Output PDF file path.
            **options: Additional options.

        Returns:
            Path to saved PDF file (or HTML file with instructions).

        Example:
            >>> report.save_pdf("analysis.pdf")
        """
        path = Path(path)

        # For now, save as HTML with instructions
        html_path = path.with_suffix(".html")
        self.save_html(html_path)

        # Add note about PDF conversion
        print(f"HTML report generated: {html_path}")
        print("To convert to PDF:")
        print("  1. Open in browser and use Print > Save as PDF")
        print("  2. Use weasyprint: weasyprint {html_path} {path}")
        print("  3. Use wkhtmltopdf: wkhtmltopdf {html_path} {path}")

        return html_path


def generate_vintage_logic_report(
    result: VintageLogicAnalysisResult,
    traces: dict[str, WaveformTrace | DigitalTrace],
    *,
    title: str | None = None,
    include_plots: list[str] | None = None,
    output_dir: Path | None = None,
) -> VintageLogicReport:
    """Generate complete vintage logic report.

    Args:
        result: Analysis result from analyze_vintage_logic().
        traces: Original signal traces.
        title: Report title. Defaults to auto-generated title.
        include_plots: List of plot types to include (None = all).
        output_dir: Directory for saving plot files. If None, uses temp directory.

    Returns:
        VintageLogicReport object ready for export.

    Example:
        >>> from oscura.analyzers.digital.vintage import analyze_vintage_logic
        >>> result = analyze_vintage_logic(traces)
        >>> report = generate_vintage_logic_report(result, traces, output_dir="./output")
        >>> report.save_html("report.html")
    """
    from pathlib import Path
    from tempfile import mkdtemp

    from oscura.visualization.digital_advanced import generate_all_vintage_logic_plots

    # Use temp directory if no output directory specified
    if output_dir is None:
        output_dir = Path(mkdtemp(prefix="oscura_vintage_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate default title if not provided
    if title is None:
        title = f"Vintage Logic Analysis - {result.detected_family}"

    # Create metadata
    metadata = ReportMetadata(title=title)

    # Generate all plots
    plot_objects = generate_all_vintage_logic_plots(
        result,
        traces,
        output_dir=output_dir,
        save_formats=["png"],
    )

    # Get plot paths from output directory
    plot_paths: dict[str, Path] = {}
    for plot_name in plot_objects:
        plot_path = output_dir / f"{plot_name}.png"
        if plot_path.exists():
            plot_paths[plot_name] = plot_path

    # Filter plots if specified
    if include_plots:
        plot_paths = {k: v for k, v in plot_paths.items() if k in include_plots}

    return VintageLogicReport(
        result=result,
        plots=plot_paths,
        metadata=metadata,
    )


def _generate_html_report(report: VintageLogicReport) -> str:
    """Generate HTML report content."""
    result = report.result

    # Embed images as base64
    from base64 import b64encode

    plot_html = []
    for plot_name, plot_path in report.plots.items():
        if plot_path.exists():
            with plot_path.open("rb") as f:
                img_data = b64encode(f.read()).decode("utf-8")
            plot_html.append(
                f'<div class="plot"><h3>{plot_name.replace("_", " ").title()}</h3>'
                f'<img src="data:image/png;base64,{img_data}" alt="{plot_name}" /></div>'
            )

    plots_section = "\n".join(plot_html)

    # Generate IC table
    ic_rows = []
    for ic in result.identified_ics:
        validation_status = (
            "PASS" if all(v.get("passes", True) for v in ic.validation.values()) else "FAIL"
        )
        status_class = "pass" if validation_status == "PASS" else "fail"

        ic_rows.append(
            f"<tr>"
            f"<td>{ic.ic_name}</td>"
            f"<td>{ic.confidence * 100:.1f}%</td>"
            f"<td>{ic.family}</td>"
            f'<td class="{status_class}">{validation_status}</td>'
            f"</tr>"
        )

    ic_table = (
        "<table><tr><th>IC</th><th>Confidence</th><th>Family</th><th>Validation</th></tr>\n"
        + "\n".join(ic_rows)
        + "</table>"
        if ic_rows
        else "<p>No ICs identified</p>"
    )

    # Generate BOM table
    bom_rows = []
    for entry in result.bom:
        bom_rows.append(
            f"<tr>"
            f"<td>{entry.part_number}</td>"
            f"<td>{entry.description}</td>"
            f"<td>{entry.quantity}</td>"
            f"<td>{entry.category}</td>"
            f"<td>{entry.notes or ''}</td>"
            f"</tr>"
        )

    bom_table = (
        "<table><tr><th>Part Number</th><th>Description</th><th>Qty</th><th>Category</th><th>Notes</th></tr>\n"
        + "\n".join(bom_rows)
        + "</table>"
        if bom_rows
        else "<p>No BOM entries</p>"
    )

    # Generate timing measurements table
    timing_rows = []
    for param, value in result.timing_measurements.items():
        timing_rows.append(f"<tr><td>{param}</td><td>{value * 1e9:.2f} ns</td></tr>")

    timing_table = (
        "<table><tr><th>Parameter</th><th>Value</th></tr>\n" + "\n".join(timing_rows) + "</table>"
        if timing_rows
        else "<p>No timing measurements</p>"
    )

    # Generate warnings
    warnings_html = ""
    if result.warnings:
        warnings_list = "\n".join(f"<li>{w}</li>" for w in result.warnings)
        warnings_html = f'<div class="warnings"><h2>Warnings</h2><ul>{warnings_list}</ul></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.metadata.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        .summary-box {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .summary-box strong {{
            color: #2c3e50;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .warnings {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .warnings h2 {{
            color: #856404;
            margin-top: 0;
        }}
        .warnings ul {{
            margin: 10px 0;
        }}
        .warnings li {{
            color: #856404;
        }}
        .metadata {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report.metadata.title}</h1>

        <div class="metadata">
            <p>Generated: {report.metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Analysis Date: {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Analysis Duration: {result.analysis_duration:.2f} seconds</p>
        </div>

        <div class="summary-box">
            <h2>Summary</h2>
            <p><strong>Detected Logic Family:</strong> {result.detected_family} ({result.family_confidence * 100:.1f}% confidence)</p>
            <p><strong>ICs Identified:</strong> {len(result.identified_ics)}</p>
            <p><strong>Timing Measurements:</strong> {len(result.timing_measurements)}</p>
            <p><strong>Open-Collector Detected:</strong> {"Yes" if result.open_collector_detected else "No"}</p>
            {f"<p><strong>Asymmetry Ratio:</strong> {result.asymmetry_ratio:.2f}</p>" if result.open_collector_detected else ""}
        </div>

        {warnings_html}

        <h2>IC Identification</h2>
        {ic_table}

        <h2>Timing Measurements</h2>
        {timing_table}

        <h2>Bill of Materials</h2>
        {bom_table}

        <h2>Visualizations</h2>
        {plots_section}
    </div>
</body>
</html>"""

    return html


def _generate_markdown_report(report: VintageLogicReport) -> str:
    """Generate markdown report content."""
    result = report.result

    # Generate IC table
    ic_rows = []
    for ic in result.identified_ics:
        validation_status = (
            "PASS" if all(v.get("passes", True) for v in ic.validation.values()) else "FAIL"
        )
        ic_rows.append(
            f"| {ic.ic_name} | {ic.confidence * 100:.1f}% | {ic.family} | {validation_status} |"
        )

    ic_table = (
        "| IC | Confidence | Family | Validation |\n|---|---|---|---|\n" + "\n".join(ic_rows)
        if ic_rows
        else "*No ICs identified*"
    )

    # Generate BOM table
    bom_rows = []
    for entry in result.bom:
        bom_rows.append(
            f"| {entry.part_number} | {entry.description} | {entry.quantity} | {entry.category} | {entry.notes or ''} |"
        )

    bom_table = (
        "| Part Number | Description | Qty | Category | Notes |\n|---|---|---|---|---|\n"
        + "\n".join(bom_rows)
        if bom_rows
        else "*No BOM entries*"
    )

    # Generate warnings
    warnings_md = ""
    if result.warnings:
        warnings_list = "\n".join(f"- {w}" for w in result.warnings)
        warnings_md = f"\n## Warnings\n\n{warnings_list}\n"

    md = f"""# {report.metadata.title}

**Generated:** {report.metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Date:** {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Duration:** {result.analysis_duration:.2f} seconds

## Summary

- **Detected Logic Family:** {result.detected_family} ({result.family_confidence * 100:.1f}% confidence)
- **ICs Identified:** {len(result.identified_ics)}
- **Timing Measurements:** {len(result.timing_measurements)}
- **Open-Collector Detected:** {"Yes" if result.open_collector_detected else "No"}
{"- **Asymmetry Ratio:** " + f"{result.asymmetry_ratio:.2f}" if result.open_collector_detected else ""}

{warnings_md}

## IC Identification

{ic_table}

## Timing Measurements

| Parameter | Value |
|---|---|
"""

    for param, value in result.timing_measurements.items():
        md += f"| {param} | {value * 1e9:.2f} ns |\n"

    if not result.timing_measurements:
        md += "*No timing measurements*\n"

    md += f"""
## Bill of Materials

{bom_table}

## Visualizations

"""

    for plot_name, plot_path in report.plots.items():
        md += f"### {plot_name.replace('_', ' ').title()}\n\n"
        md += f"![{plot_name}]({plot_path})\n\n"

    return md


__all__ = [
    "ReportMetadata",
    "VintageLogicReport",
    "generate_vintage_logic_report",
]
