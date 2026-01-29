"""EMC compliance report generation.

This module provides compliance report generation in multiple formats.


Example:
    >>> from oscura.validation.compliance import test_compliance, generate_compliance_report
    >>> result = test_compliance(trace, mask)
    >>> generate_compliance_report(result, 'report.html')

References:
    ANSI C63.4 (Test Methods)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.validation.compliance.testing import ComplianceResult


class ComplianceReportFormat(Enum):
    """Compliance report output formats."""

    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"


def generate_compliance_report(
    result: ComplianceResult,
    output_path: str | Path,
    *,
    format: ComplianceReportFormat | str = ComplianceReportFormat.HTML,
    include_plot: bool = True,
    title: str | None = None,
    company_name: str | None = None,
    dut_info: dict[str, str] | None = None,
) -> Path:
    """Generate EMC compliance report.

    Args:
        result: ComplianceResult from test_compliance().
        output_path: Output file path.
        format: Report format ('html', 'pdf', 'markdown', 'json').
        include_plot: Include spectrum/limit plot in report.
        title: Report title (default: "EMC Compliance Report").
        company_name: Company name for header.
        dut_info: Device Under Test information dict.

    Returns:
        Path to generated report.

    Raises:
        ValueError: If format is unknown.

    Example:
        >>> result = test_compliance(trace, mask)
        >>> report_path = generate_compliance_report(
        ...     result,
        ...     'compliance_report.html',
        ...     title="Product X EMC Test",
        ...     dut_info={'model': 'XYZ-100', 'serial': '12345'}
        ... )
    """
    output_path = Path(output_path)

    # Handle format
    if isinstance(format, str):
        format = ComplianceReportFormat(format.lower())

    if format == ComplianceReportFormat.HTML:
        _generate_html_report(
            result,
            output_path,
            include_plot=include_plot,
            title=title,
            company_name=company_name,
            dut_info=dut_info,
        )
    elif format == ComplianceReportFormat.MARKDOWN:
        _generate_markdown_report(
            result,
            output_path,
            title=title,
            dut_info=dut_info,
        )
    elif format == ComplianceReportFormat.JSON:
        _generate_json_report(result, output_path)
    elif format == ComplianceReportFormat.PDF:
        # Generate HTML first, then convert to PDF
        html_path = output_path.with_suffix(".html")
        _generate_html_report(
            result,
            html_path,
            include_plot=include_plot,
            title=title,
            company_name=company_name,
            dut_info=dut_info,
        )
        _convert_html_to_pdf(html_path, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return output_path


def _generate_dut_section_html(dut_info: dict[str, str] | None) -> str:
    """Generate DUT information section HTML.

    Args:
        dut_info: Device Under Test information dict.

    Returns:
        HTML string for DUT section.
    """
    if not dut_info:
        return ""

    dut_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in dut_info.items())
    return f"""
        <h3>Device Under Test</h3>
        <table class="info-table">
            {dut_rows}
        </table>
        """


def _generate_violations_section_html(result: ComplianceResult) -> str:
    """Generate violations table section HTML.

    Args:
        result: ComplianceResult with violations list.

    Returns:
        HTML string for violations section.
    """
    if not result.violations:
        return ""

    violation_rows = ""
    for v in result.violations:
        freq_mhz = v.frequency / 1e6
        violation_rows += f"""
            <tr>
                <td>{freq_mhz:.3f}</td>
                <td>{v.measured_level:.1f}</td>
                <td>{v.limit_level:.1f}</td>
                <td style="color: red;">{v.excess_db:.1f}</td>
            </tr>
            """

    return f"""
        <h3>Violations ({len(result.violations)})</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Frequency (MHz)</th>
                    <th>Measured ({result.metadata.get("unit", "dBuV")})</th>
                    <th>Limit ({result.metadata.get("unit", "dBuV")})</th>
                    <th>Excess (dB)</th>
                </tr>
            </thead>
            <tbody>
                {violation_rows}
            </tbody>
        </table>
        """


def _generate_report_css(status_color: str) -> str:
    """Generate CSS stylesheet for compliance report.

    Args:
        status_color: Badge color for pass/fail status.

    Returns:
        CSS stylesheet string.
    """
    return f"""
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .company-name {{
            font-size: 12px;
            color: #666;
            text-align: right;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 30px;
            font-size: 24px;
            font-weight: bold;
            color: white;
            background-color: {status_color};
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .summary-card h4 {{
            margin: 0 0 5px 0;
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .info-table, .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .info-table td, .data-table th, .data-table td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        .data-table th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
        .data-table tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    """


def _generate_summary_grid_html(result: ComplianceResult) -> str:
    """Generate summary grid HTML.

    Args:
        result: ComplianceResult with summary metrics.

    Returns:
        HTML string for summary grid.
    """
    return f"""
    <h2>Test Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <h4>Standard</h4>
            <div class="value">{result.mask_name}</div>
        </div>
        <div class="summary-card">
            <h4>Margin to Limit</h4>
            <div class="value">{result.margin_to_limit:.1f} dB</div>
        </div>
        <div class="summary-card">
            <h4>Worst Frequency</h4>
            <div class="value">{result.worst_frequency / 1e6:.3f} MHz</div>
        </div>
        <div class="summary-card">
            <h4>Violations</h4>
            <div class="value">{len(result.violations)}</div>
        </div>
    </div>
    """


def _generate_footer_html(result: ComplianceResult) -> str:
    """Generate footer HTML.

    Args:
        result: ComplianceResult with metadata.

    Returns:
        HTML string for footer.
    """
    return f"""
    <div class="footer">
        <p>Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Detector: {result.detector} | Distance: {result.metadata.get("distance", "N/A")}m</p>
        <p>Generated by Oscura EMC Compliance Module</p>
    </div>
    """


def _generate_html_report(
    result: ComplianceResult,
    output_path: Path,
    *,
    include_plot: bool = True,
    title: str | None = None,
    company_name: str | None = None,
    dut_info: dict[str, str] | None = None,
) -> None:
    """Generate HTML compliance report.

    Args:
        result: ComplianceResult from test_compliance().
        output_path: Output file path.
        include_plot: Include spectrum/limit plot in report.
        title: Report title (default: "EMC Compliance Report").
        company_name: Company name for header.
        dut_info: Device Under Test information dict.
    """
    title = title or "EMC Compliance Report"
    status_color = "#28a745" if result.passed else "#dc3545"
    status_text = "PASS" if result.passed else "FAIL"

    # Build sections
    dut_section = _generate_dut_section_html(dut_info)
    violations_section = _generate_violations_section_html(result)

    plot_section = ""
    if include_plot and len(result.spectrum_freq) > 0:
        plot_section = _generate_plot_html(result)

    company_header = f"<div class='company-name'>{company_name}</div>" if company_name else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        {_generate_report_css(status_color)}
    </style>
</head>
<body>
    {company_header}
    <h1>{title}</h1>

    <div class="status-badge">{status_text}</div>

    {_generate_summary_grid_html(result)}

    {dut_section}

    {violations_section}

    {plot_section}

    {_generate_footer_html(result)}
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)


def _generate_plot_html(result: ComplianceResult) -> str:
    """Generate inline SVG plot for HTML report."""
    import numpy as np

    # Simple ASCII-style data representation for inline embedding
    # In production, would use matplotlib to generate SVG

    freq_mhz = result.spectrum_freq / 1e6
    f_min, f_max = freq_mhz.min(), freq_mhz.max()
    level_min = min(result.spectrum_level.min(), result.limit_level.min()) - 5
    level_max = max(result.spectrum_level.max(), result.limit_level.max()) + 5

    # Create SVG plot
    width, height = 800, 400
    padding = 60

    # Scale functions
    def x_scale(f: float) -> float:
        return padding + (np.log10(f) - np.log10(f_min)) / (np.log10(f_max) - np.log10(f_min)) * (  # type: ignore[no-any-return]
            width - 2 * padding
        )

    def y_scale(l: float) -> float:
        return height - padding - (l - level_min) / (level_max - level_min) * (height - 2 * padding)  # type: ignore[no-any-return]

    # Build spectrum path (downsample for SVG)
    step = max(1, len(freq_mhz) // 500)
    spectrum_points = " ".join(
        f"{x_scale(freq_mhz[i]):.1f},{y_scale(result.spectrum_level[i]):.1f}"
        for i in range(0, len(freq_mhz), step)
    )

    # Build limit path
    limit_points = " ".join(
        f"{x_scale(freq_mhz[i]):.1f},{y_scale(result.limit_level[i]):.1f}"
        for i in range(0, len(freq_mhz), step)
    )

    svg = f"""
    <div class="plot-container">
        <h3>Spectrum vs Limit</h3>
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="100%" height="100%" fill="#f8f9fa"/>

            <!-- Grid -->
            <g stroke="#ddd" stroke-width="1">
                <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}"/>
                <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"/>
            </g>

            <!-- Limit line (red dashed) -->
            <polyline points="{limit_points}"
                      fill="none" stroke="#dc3545" stroke-width="2" stroke-dasharray="5,3"/>

            <!-- Spectrum line (blue) -->
            <polyline points="{spectrum_points}"
                      fill="none" stroke="#007bff" stroke-width="1.5"/>

            <!-- Legend -->
            <g transform="translate({width - 150}, 20)">
                <rect width="130" height="50" fill="white" stroke="#ddd"/>
                <line x1="10" y1="20" x2="40" y2="20" stroke="#007bff" stroke-width="2"/>
                <text x="50" y="24" font-size="12">Spectrum</text>
                <line x1="10" y1="40" x2="40" y2="40" stroke="#dc3545" stroke-width="2" stroke-dasharray="5,3"/>
                <text x="50" y="44" font-size="12">Limit</text>
            </g>

            <!-- Axis labels -->
            <text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="12">Frequency (MHz)</text>
            <text x="15" y="{height / 2}" text-anchor="middle" font-size="12"
                  transform="rotate(-90, 15, {height / 2})">Level ({result.metadata.get("unit", "dBuV")})</text>
        </svg>
    </div>
    """
    return svg


def _generate_markdown_report(
    result: ComplianceResult,
    output_path: Path,
    *,
    title: str | None = None,
    dut_info: dict[str, str] | None = None,
) -> None:
    """Generate Markdown compliance report."""
    title = title or "EMC Compliance Report"
    status = "PASS" if result.passed else "FAIL"

    md = f"""# {title}

## Test Result: **{status}**

## Summary

| Parameter | Value |
|-----------|-------|
| Standard | {result.mask_name} |
| Margin to Limit | {result.margin_to_limit:.1f} dB |
| Worst Frequency | {result.worst_frequency / 1e6:.3f} MHz |
| Worst Margin | {result.worst_margin:.1f} dB |
| Violations | {len(result.violations)} |
| Detector | {result.detector} |
"""
    if dut_info:
        md += "## Device Under Test\n\n"
        md += "| Field | Value |\n|-------|-------|\n"
        for k, v in dut_info.items():
            md += f"| {k} | {v} |\n"
        md += "\n"

    if result.violations:
        md += f"## Violations ({len(result.violations)})\n\n"
        md += "| Frequency (MHz) | Measured | Limit | Excess (dB) |\n"
        md += "|-----------------|----------|-------|-------------|\n"
        for v in result.violations:  # type: ignore[assignment]
            md += f"| {v.frequency / 1e6:.3f} | {v.measured_level:.1f} | {v.limit_level:.1f} | {v.excess_db:.1f} |\n"  # type: ignore[attr-defined]
        md += "\n"

    md += f"""
---
*Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Generated by Oscura EMC Compliance Module*
"""
    with open(output_path, "w") as f:
        f.write(md)


def _generate_json_report(result: ComplianceResult, output_path: Path) -> None:
    """Generate JSON compliance report."""
    import json

    data = {
        "status": result.status,
        "mask_name": result.mask_name,
        "margin_to_limit": result.margin_to_limit,
        "worst_frequency": result.worst_frequency,
        "worst_margin": result.worst_margin,
        "detector": result.detector,
        "violation_count": len(result.violations),
        "violations": [
            {
                "frequency_hz": v.frequency,
                "frequency_mhz": v.frequency / 1e6,
                "measured_level": v.measured_level,
                "limit_level": v.limit_level,
                "excess_db": v.excess_db,
            }
            for v in result.violations
        ],
        "metadata": result.metadata,
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _convert_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Convert HTML to PDF using available tools."""
    try:
        # Try weasyprint first
        from weasyprint import HTML

        HTML(str(html_path)).write_pdf(str(pdf_path))
    except ImportError:
        # Fall back to copying HTML
        import shutil

        shutil.copy(html_path, pdf_path.with_suffix(".html"))
        # Could also try pdfkit, wkhtmltopdf, etc.


__all__ = [
    "ComplianceReportFormat",
    "generate_compliance_report",
]
