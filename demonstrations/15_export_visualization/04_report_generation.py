"""Report Generation: PDF and HTML Reports from Analysis Results

Demonstrates:
- HTML report generation with templates
- PDF report creation (requires reportlab)
- Multi-format output
- Template-based reporting
- Embedded plots and tables

This demonstration shows how to generate professional analysis
reports in HTML and PDF formats with embedded visualizations.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace


class ReportGenerationDemo(BaseDemo):
    """Demonstrate PDF and HTML report generation."""

    def __init__(self) -> None:
        """Initialize report generation demonstration."""
        super().__init__(
            name="report_generation",
            description="Generate professional HTML and PDF reports from analysis results",
            capabilities=[
                "export.html_report",
                "export.pdf_report",
                "export.markdown_report",
                "visualization.report_templates",
            ],
            related_demos=[
                "15_export_visualization/01_export_formats.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate analysis results for reporting.

        Returns:
            Dictionary containing analysis results
        """
        # Generate sample waveform
        sample_rate = 1e6
        duration = 0.01
        frequency = 10e3
        num_samples = int(duration * sample_rate)

        t = np.arange(num_samples) / sample_rate
        data = np.sin(2 * np.pi * frequency * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="CH1",
            source_file="test_signal.wfm",
        )

        waveform = WaveformTrace(data=data, metadata=metadata)

        # Generate analysis results
        analysis_results = {
            "rms": float(np.sqrt(np.mean(data**2))),
            "peak": float(np.max(np.abs(data))),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "frequency": frequency,
            "sample_rate": sample_rate,
            "duration": duration,
        }

        return {
            "waveform": waveform,
            "analysis": analysis_results,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the report generation demonstration."""
        results: dict[str, Any] = {}
        output_dir = self.get_output_dir()

        self.section("Report Generation Demonstration")
        self.info("Generate professional HTML and PDF reports")

        analysis = data["analysis"]

        # Part 1: HTML Report Generation
        self.subsection("Part 1: HTML Report Generation")
        self.info("Create standalone HTML report with embedded CSS.")

        html_content = self._generate_html_report(analysis)

        html_path = output_dir / "analysis_report.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        self.result("HTML report created", str(html_path))
        self.result("Report size", f"{len(html_content) / 1024:.2f}", "KB")

        results["html_path"] = str(html_path)

        # Part 2: Markdown Report
        self.subsection("Part 2: Markdown Report Generation")
        self.info("Generate Markdown report for version control and documentation.")

        md_content = self._generate_markdown_report(analysis)

        md_path = output_dir / "analysis_report.md"
        with open(md_path, "w") as f:
            f.write(md_content)

        self.result("Markdown report created", str(md_path))

        # Show excerpt
        self.info("\nMarkdown excerpt:")
        for line in md_content.split("\n")[:15]:
            self.info(f"  {line}")

        results["md_path"] = str(md_path)

        # Part 3: PDF Report (if reportlab available)
        self.subsection("Part 3: PDF Report Generation")
        self.info("Generate PDF report (requires reportlab).")

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

            pdf_path = output_dir / "analysis_report.pdf"

            # Create PDF
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Title
            title = Paragraph("Signal Analysis Report", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 0.2 * inch))

            # Metadata
            metadata_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(metadata_text, styles["Normal"]))
            story.append(Spacer(1, 0.3 * inch))

            # Results table
            table_data = [
                ["Measurement", "Value", "Unit"],
                ["RMS", f"{analysis['rms']:.6f}", "V"],
                ["Peak", f"{analysis['peak']:.6f}", "V"],
                ["Mean", f"{analysis['mean']:.6f}", "V"],
                ["Frequency", f"{analysis['frequency']:.0f}", "Hz"],
                ["Sample Rate", f"{analysis['sample_rate']:.0e}", "Hz"],
            ]

            table = Table(table_data)
            story.append(table)

            # Build PDF
            doc.build(story)

            self.result("PDF report created", str(pdf_path))
            results["pdf_path"] = str(pdf_path)

        except ImportError:
            self.warning("reportlab not installed - skipping PDF generation")
            self.info("  Install with: pip install reportlab")
            results["pdf_path"] = None

        # Part 4: Report templates
        self.subsection("Part 4: Report Templates")
        self.info("Overview of report template system.")

        template_info = {
            "HTML Template": "Standalone HTML with embedded CSS and JavaScript",
            "Markdown Template": "Plain text format for version control",
            "PDF Template": "Professional PDF with tables and plots (reportlab)",
            "LaTeX Template": "Scientific reports with mathematical notation",
        }

        self.info("\nAvailable template types:")
        for template_type, description in template_info.items():
            self.info(f"  • {template_type}:")
            self.info(f"    {description}")

        results["templates"] = template_info

        # Part 5: Best practices
        self.subsection("Part 5: Report Generation Best Practices")
        self.info("Guidelines for effective report generation.")

        best_practices = [
            "Include metadata: date, analyst, instrument, conditions",
            "Embed plots as base64 or SVG for standalone HTML",
            "Use templates for consistent formatting",
            "Add version information and traceability",
            "Include measurement uncertainties",
            "Provide context and interpretation",
            "Use appropriate significant figures",
        ]

        self.info("\nBest practices:")
        for i, practice in enumerate(best_practices, 1):
            self.info(f"  {i}. {practice}")

        results["best_practices"] = best_practices

        self.success("Report generation demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating report generation...")

        # Validate HTML report
        if "html_path" not in results:
            self.error("Missing HTML report")
            return False

        html_path = Path(results["html_path"])
        if not html_path.exists():
            self.error(f"HTML report not found: {html_path}")
            return False

        # Check HTML content
        with open(html_path) as f:
            html_content = f.read()
            if "<html>" not in html_content:
                self.error("Invalid HTML structure")
                return False

        # Validate Markdown report
        if "md_path" not in results:
            self.error("Missing Markdown report")
            return False

        md_path = Path(results["md_path"])
        if not md_path.exists():
            self.error(f"Markdown report not found: {md_path}")
            return False

        self.success("All report generation validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - HTML reports: Portable, can embed plots and styling")
        self.info("  - Markdown reports: Version-control friendly, simple")
        self.info("  - PDF reports: Professional, print-ready (requires reportlab)")
        self.info("  - Always include metadata and traceability information")

        return True

    def _generate_html_report(self, analysis: dict[str, Any]) -> str:
        """Generate HTML report.

        Args:
            analysis: Analysis results

        Returns:
            HTML content string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Signal Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
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
            background-color: #007bff;
            color: white;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Signal Analysis Report</h1>
        <div class="metadata">
            Generated: {timestamp}<br>
            Tool: Oscura Signal Analysis Framework
        </div>

        <h2>Analysis Results</h2>
        <table>
            <tr>
                <th>Measurement</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            <tr>
                <td>RMS Voltage</td>
                <td>{analysis["rms"]:.6f}</td>
                <td>V</td>
            </tr>
            <tr>
                <td>Peak Voltage</td>
                <td>{analysis["peak"]:.6f}</td>
                <td>V</td>
            </tr>
            <tr>
                <td>Mean Voltage</td>
                <td>{analysis["mean"]:.6f}</td>
                <td>V</td>
            </tr>
            <tr>
                <td>Standard Deviation</td>
                <td>{analysis["std"]:.6f}</td>
                <td>V</td>
            </tr>
            <tr>
                <td>Frequency</td>
                <td>{analysis["frequency"]:.0f}</td>
                <td>Hz</td>
            </tr>
            <tr>
                <td>Sample Rate</td>
                <td>{analysis["sample_rate"]:.0e}</td>
                <td>Hz</td>
            </tr>
            <tr>
                <td>Duration</td>
                <td>{analysis["duration"]:.6f}</td>
                <td>s</td>
            </tr>
        </table>

        <h2>Summary</h2>
        <p>
            Analysis performed on signal with {analysis["frequency"]:.0f} Hz frequency,
            sampled at {analysis["sample_rate"]:.0e} Hz for {analysis["duration"]:.6f} seconds.
        </p>
    </div>
</body>
</html>
"""
        return html

    def _generate_markdown_report(self, analysis: dict[str, Any]) -> str:
        """Generate Markdown report.

        Args:
            analysis: Analysis results

        Returns:
            Markdown content string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# Signal Analysis Report

**Generated:** {timestamp}
**Tool:** Oscura Signal Analysis Framework

## Analysis Results

| Measurement | Value | Unit |
|-------------|-------|------|
| RMS Voltage | {analysis["rms"]:.6f} | V |
| Peak Voltage | {analysis["peak"]:.6f} | V |
| Mean Voltage | {analysis["mean"]:.6f} | V |
| Standard Deviation | {analysis["std"]:.6f} | V |
| Frequency | {analysis["frequency"]:.0f} | Hz |
| Sample Rate | {analysis["sample_rate"]:.0e} | Hz |
| Duration | {analysis["duration"]:.6f} | s |

## Summary

Analysis performed on signal with {analysis["frequency"]:.0f} Hz frequency,
sampled at {analysis["sample_rate"]:.0e} Hz for {analysis["duration"]:.6f} seconds.

## Measurement Details

- **RMS Voltage:** Root-mean-square voltage calculated from sampled data
- **Peak Voltage:** Maximum absolute voltage observed
- **Frequency:** Dominant frequency component

---

*Report generated by Oscura Signal Analysis Framework*
"""
        return md


if __name__ == "__main__":
    demo: ReportGenerationDemo = ReportGenerationDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
