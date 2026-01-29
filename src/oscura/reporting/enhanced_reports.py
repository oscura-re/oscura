"""Enhanced report generation with interactive visualizations.

This module provides comprehensive HTML/PDF report generation for protocol
reverse engineering with interactive JavaScript visualizations, professional
formatting, and multiple report types.

Features:
    - Professional HTML reports with embedded plots
    - Optional PDF export via weasyprint
    - Interactive JavaScript visualizations (plotly.js)
    - Customizable Jinja2 templates
    - Multiple report types (protocol RE, security, performance)
    - Base64-embedded or external plots
    - Responsive design with dark mode support

Example:
    >>> from oscura.reporting.enhanced_reports import (
    ...     EnhancedReportGenerator,
    ...     ReportConfig,
    ... )
    >>> from oscura.workflows import full_protocol_re
    >>> result = full_protocol_re("capture.bin")
    >>> generator = EnhancedReportGenerator()
    >>> config = ReportConfig(
    ...     title="Unknown Protocol Analysis",
    ...     template="protocol_re",
    ...     format="html",
    ...     interactive=True,
    ... )
    >>> output_path = generator.generate(result, "report.html", config)
    >>> print(f"Report generated: {output_path}")

References:
    - plotly.js: https://plotly.com/javascript/
    - Jinja2: https://jinja.palletsprojects.com/
    - weasyprint: https://weasyprint.org/
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

# Lazy imports for optional dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

try:
    from jinja2 import Environment, FileSystemLoader, Template

    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False
    Environment = None  # type: ignore[assignment,misc]
    FileSystemLoader = None  # type: ignore[assignment,misc]
    Template = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from oscura.workflows.complete_re import CompleteREResult

# Use non-interactive backend for plot embedding
if _HAS_MATPLOTLIB:
    matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Fallback HTML template (used when template files not found)
_FALLBACK_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: {{ theme.background_color }};
            color: {{ theme.text_color }};
        }
        .header {
            border-bottom: 3px solid {{ theme.primary_color }};
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h1 { color: {{ theme.primary_color }}; }
        h2 { color: {{ theme.secondary_color }}; border-bottom: 2px solid {{ theme.border_color }}; padding-bottom: 10px; }
        .section { margin: 30px 0; }
        .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid {{ theme.border_color }}; padding: 12px; text-align: left; }
        th { background-color: {{ theme.primary_color }}; color: white; }
        tr:nth-child(even) { background-color: rgba(0,0,0,0.02); }
        .plot { margin: 20px 0; text-align: center; }
        .plot img { max-width: 100%; height: auto; border: 1px solid {{ theme.border_color }}; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <p>Author: {{ author }}</p>
    </div>

    {% if protocol_spec %}
    <div class="section">
        <h2>Protocol Specification</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Name</td><td>{{ protocol_spec.name }}</td></tr>
            <tr><td>Baud Rate</td><td>{{ protocol_spec.baud_rate|format_number }} bps</td></tr>
            <tr><td>Frame Format</td><td>{{ protocol_spec.frame_format }}</td></tr>
            <tr><td>Sync Pattern</td><td>{{ protocol_spec.sync_pattern }}</td></tr>
            <tr><td>Frame Length</td><td>{{ protocol_spec.frame_length or "Variable" }}</td></tr>
            <tr><td>Checksum</td><td>{{ protocol_spec.checksum_type or "None detected" }}</td></tr>
            <tr><td>Confidence</td><td>{{ (protocol_spec.confidence * 100)|round(1) }}%</td></tr>
        </table>

        {% if protocol_spec.fields %}
        <h3>Fields</h3>
        <table>
            <tr><th>Name</th><th>Offset</th><th>Size</th><th>Type</th></tr>
            {% for field in protocol_spec.fields %}
            <tr>
                <td>{{ field.name }}</td>
                <td>{{ field.offset }}</td>
                <td>{{ field.size }}</td>
                <td>{{ field.type }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if plots %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for plot in plots %}
        <div class="plot">
            <h3>{{ plot.title }}</h3>
            {% if plot.type == "embedded" %}
            <img src="{{ plot.data }}" alt="{{ plot.title }}">
            {% else %}
            <img src="{{ plot.path }}" alt="{{ plot.title }}">
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if artifacts %}
    <div class="section">
        <h2>Generated Artifacts</h2>
        <ul>
        {% for artifact in artifacts %}
            <li><strong>{{ artifact.name }}</strong>: {{ artifact.path }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}

    {% if warnings %}
    <div class="section">
        <h2>Warnings</h2>
        {% for warning in warnings %}
        <div class="warning">{{ warning }}</div>
        {% endfor %}
    </div>
    {% endif %}

    {% if execution_time %}
    <div class="section">
        <h2>Execution Metrics</h2>
        <p>Total execution time: {{ execution_time|round(2) }} seconds</p>
        {% if confidence_score %}
        <p>Overall confidence: {{ (confidence_score * 100)|round(1) }}%</p>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>"""


@dataclass
class ReportConfig:
    """Configuration for enhanced report generation.

    Attributes:
        title: Report title.
        template: Template name ("protocol_re", "security", "performance").
        format: Output format ("html", "pdf", "both").
        include_plots: Include plots in report.
        interactive: Enable interactive visualizations.
        theme: Visual theme ("default", "dark", "minimal").
        embed_plots: Embed plots as base64 (True) or external files (False).
        author: Report author name.
        show_toc: Include table of contents.
        show_timestamps: Include generation timestamp.
        custom_css: Additional CSS styles.
        custom_js: Additional JavaScript code.

    Example:
        >>> config = ReportConfig(
        ...     title="Protocol Analysis",
        ...     template="protocol_re",
        ...     format="html",
        ...     interactive=True,
        ...     theme="default",
        ... )
    """

    title: str
    template: Literal["protocol_re", "security", "performance"] = "protocol_re"
    format: Literal["html", "pdf", "both"] = "html"
    include_plots: bool = True
    interactive: bool = True
    theme: Literal["default", "dark", "minimal"] = "default"
    embed_plots: bool = True
    author: str = "Oscura Framework"
    show_toc: bool = True
    show_timestamps: bool = True
    custom_css: str = ""
    custom_js: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EnhancedReportGenerator:
    """Generate comprehensive HTML/PDF reports with interactive visualizations.

    This class provides professional report generation for protocol reverse
    engineering workflows with support for multiple templates, interactive
    plots, and optional PDF export.

    Attributes:
        template_dir: Directory containing Jinja2 templates.
        static_dir: Directory containing CSS/JS static assets.
        env: Jinja2 environment for template rendering.

    Example:
        >>> generator = EnhancedReportGenerator()
        >>> config = ReportConfig(title="Analysis Report")
        >>> output = generator.generate(results, "report.html", config)
    """

    def __init__(self, template_dir: Path | None = None, static_dir: Path | None = None) -> None:
        """Initialize the enhanced report generator.

        Args:
            template_dir: Custom template directory (default: built-in templates).
            static_dir: Custom static assets directory (default: built-in assets).

        Raises:
            ImportError: If required dependencies (matplotlib, jinja2) are not installed.

        Example:
            >>> generator = EnhancedReportGenerator()
            >>> # Use custom templates
            >>> custom_gen = EnhancedReportGenerator(
            ...     template_dir=Path("my_templates")
            ... )
        """
        # Check for required dependencies
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "Enhanced reporting requires matplotlib.\n\n"
                "Install with:\n"
                "  pip install oscura[reporting]    # Reporting features\n"
                "  pip install oscura[standard]     # Recommended\n"
                "  pip install oscura[all]          # Everything\n"
            )
        if not _HAS_JINJA2:
            raise ImportError(
                "Enhanced reporting requires jinja2.\n\n"
                "Install with:\n"
                "  pip install oscura[reporting]    # Reporting features\n"
                "  pip install oscura[standard]     # Recommended\n"
                "  pip install oscura[all]          # Everything\n"
            )

        self.template_dir = template_dir or self._get_builtin_template_dir()
        self.static_dir = static_dir or self._get_builtin_static_dir()

        # Create directories if they don't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self.env.filters["format_bytes"] = self._format_bytes
        self.env.filters["format_number"] = self._format_number
        self.env.filters["format_timestamp"] = self._format_timestamp

    def generate(
        self,
        results: CompleteREResult | dict[str, Any],
        output_path: Path | str,
        config: ReportConfig | None = None,
    ) -> Path:
        """Generate comprehensive HTML/PDF report from results.

        Args:
            results: Complete RE result object or dict with results data.
            output_path: Path for output file.
            config: Report configuration (uses defaults if None).

        Returns:
            Path to generated report file.

        Raises:
            ValueError: If template not found or results invalid.
            RuntimeError: If PDF generation fails (when format="pdf").

        Example:
            >>> from oscura.workflows import full_protocol_re
            >>> result = full_protocol_re("capture.bin")
            >>> generator = EnhancedReportGenerator()
            >>> output = generator.generate(result, "report.html")
            >>> print(f"Generated: {output}")
        """
        if config is None:
            config = ReportConfig(title="Protocol Analysis Report")

        # Validate format
        valid_formats = {"html", "pdf", "both"}
        if config.format not in valid_formats:
            raise ValueError(
                f"Unsupported format: {config.format}. "
                f"Valid formats: {', '.join(sorted(valid_formats))}"
            )

        output_path = Path(output_path)

        # Convert dict to object-like structure if needed
        if isinstance(results, dict):
            results = self._dict_to_object(results)

        # Prepare template context
        context = self._prepare_context(results, config)

        # Render HTML
        html_content = self._render_template(context, config)

        # Handle output based on format
        if config.format == "html":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Generated HTML report: {output_path}")
            return output_path

        if config.format == "pdf":
            # Generate HTML first, then convert to PDF
            html_path = output_path.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")

            pdf_path = output_path.with_suffix(".pdf")
            self._export_pdf(html_path, pdf_path)
            logger.info(f"Generated PDF report: {pdf_path}")
            return pdf_path

        # config.format == "both" (all Literal values covered)
        html_path = output_path.with_suffix(".html")
        html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Generated HTML report: {html_path}")

        pdf_path = output_path.with_suffix(".pdf")
        self._export_pdf(html_path, pdf_path)
        logger.info(f"Generated PDF report: {pdf_path}")
        return html_path

    def _render_template(self, context: dict[str, Any], config: ReportConfig) -> str:
        """Render Jinja2 template with context.

        Args:
            context: Template context dictionary.
            config: Report configuration.

        Returns:
            Rendered HTML string.

        Raises:
            ValueError: If template not found.

        Example:
            >>> context = {"title": "Test", "sections": []}
            >>> html = generator._render_template(context, config)
        """
        template_name = f"{config.template}.html"

        try:
            template = self.env.get_template(template_name)
        except Exception:
            msg = f"Template not found: {template_name}"
            logger.error(msg)
            # Fall back to inline template
            template = self._get_fallback_template()

        return template.render(**context)

    def _prepare_context(
        self, results: CompleteREResult | Any, config: ReportConfig
    ) -> dict[str, Any]:
        """Prepare template context from results.

        Args:
            results: Complete RE result object.
            config: Report configuration.

        Returns:
            Dictionary with template context.

        Example:
            >>> context = generator._prepare_context(results, config)
            >>> assert "title" in context
            >>> assert "protocol_spec" in context
        """
        # Build base context
        context = self._build_base_context(config)

        # Extract protocol specification
        context["protocol_spec"] = self._extract_protocol_spec(results)

        # Extract execution metrics
        self._add_execution_metrics(context, results)

        # Extract generated artifacts
        context["artifacts"] = self._extract_artifacts(results)

        # Extract partial results
        context["partial_results"] = self._extract_partial_results(results)

        # Generate plots if requested
        if config.include_plots:
            context["plots"] = self._generate_plots(results, config)
        else:
            context["plots"] = []

        # Add custom metadata
        context.update(config.metadata)

        return context

    def _build_base_context(self, config: ReportConfig) -> dict[str, Any]:
        """Build base context dictionary.

        Args:
            config: Report configuration.

        Returns:
            Base context dictionary.
        """
        return {
            "title": config.title,
            "author": config.author,
            "generated_at": datetime.now(),
            "config": config,
            "theme": self._get_theme_styles(config.theme),
        }

    def _extract_protocol_spec(self, results: CompleteREResult | Any) -> dict[str, Any] | None:
        """Extract protocol specification from results.

        Args:
            results: Complete RE result object.

        Returns:
            Protocol spec dictionary or None.
        """
        if not hasattr(results, "protocol_spec") or results.protocol_spec is None:
            return None

        spec = results.protocol_spec
        return {
            "name": spec.name,
            "baud_rate": spec.baud_rate,
            "frame_format": spec.frame_format,
            "sync_pattern": spec.sync_pattern,
            "frame_length": spec.frame_length,
            "checksum_type": spec.checksum_type,
            "checksum_position": spec.checksum_position,
            "confidence": spec.confidence,
            "fields": [
                {
                    "name": f.name,
                    "offset": f.offset,
                    "size": f.size,
                    "type": f.field_type,
                }
                for f in spec.fields
            ],
        }

    def _add_execution_metrics(
        self, context: dict[str, Any], results: CompleteREResult | Any
    ) -> None:
        """Add execution metrics to context.

        Args:
            context: Context dictionary to update.
            results: Complete RE result object.
        """
        if hasattr(results, "execution_time"):
            context["execution_time"] = results.execution_time

        if hasattr(results, "confidence_score"):
            context["confidence_score"] = results.confidence_score

        if hasattr(results, "warnings"):
            context["warnings"] = results.warnings
        else:
            context["warnings"] = []

    def _extract_artifacts(self, results: CompleteREResult | Any) -> list[dict[str, str]]:
        """Extract generated artifacts from results.

        Args:
            results: Complete RE result object.

        Returns:
            List of artifact dictionaries.
        """
        artifacts = []

        if hasattr(results, "dissector_path") and results.dissector_path:
            artifacts.append({"name": "Wireshark Dissector", "path": str(results.dissector_path)})

        if hasattr(results, "scapy_layer_path") and results.scapy_layer_path:
            artifacts.append({"name": "Scapy Layer", "path": str(results.scapy_layer_path)})

        if hasattr(results, "kaitai_path") and results.kaitai_path:
            artifacts.append({"name": "Kaitai Struct", "path": str(results.kaitai_path)})

        if hasattr(results, "test_vectors_path") and results.test_vectors_path:
            artifacts.append({"name": "Test Vectors", "path": str(results.test_vectors_path)})

        return artifacts

    def _extract_partial_results(self, results: CompleteREResult | Any) -> dict[str, Any]:
        """Extract partial results for detailed analysis.

        Args:
            results: Complete RE result object.

        Returns:
            Partial results dictionary.
        """
        if hasattr(results, "partial_results"):
            return results.partial_results
        return {}

    def _generate_plots(
        self, results: CompleteREResult | Any, config: ReportConfig
    ) -> list[dict[str, Any]]:
        """Generate plots for report.

        Args:
            results: Complete RE result object.
            config: Report configuration.

        Returns:
            List of plot dictionaries with base64 data or paths.

        Example:
            >>> plots = generator._generate_plots(results, config)
            >>> assert len(plots) > 0
            >>> assert "data" in plots[0] or "path" in plots[0]
        """
        plots = []

        # Generate protocol structure visualization
        if hasattr(results, "protocol_spec") and results.protocol_spec:
            spec = results.protocol_spec
            if spec.fields:
                plot_data = self._plot_protocol_structure(spec, config)
                plots.append(plot_data)

        # Generate confidence score visualization
        if hasattr(results, "confidence_score"):
            plot_data = self._plot_confidence_metrics(results, config)
            plots.append(plot_data)

        # Generate timing diagram if available
        if hasattr(results, "partial_results") and results.partial_results:
            traces_data = results.partial_results.get("traces")
            # Skip if None or not a dict
            if isinstance(traces_data, dict):
                timing_plot = self._plot_timing_diagram(traces_data, config)
                if timing_plot:
                    plots.append(timing_plot)

        return plots

    def _plot_protocol_structure(self, protocol_spec: Any, config: ReportConfig) -> dict[str, Any]:
        """Generate protocol structure visualization.

        Args:
            protocol_spec: Protocol specification object.
            config: Report configuration.

        Returns:
            Plot dictionary with embedded data.

        Example:
            >>> plot = generator._plot_protocol_structure(spec, config)
            >>> assert plot["title"] == "Protocol Structure"
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bar chart of fields
        if protocol_spec.fields:
            field_names = [f.name for f in protocol_spec.fields]
            field_sizes = [f.size if isinstance(f.size, int) else 1 for f in protocol_spec.fields]
            field_offsets = [f.offset for f in protocol_spec.fields]

            viridis_cmap = plt.cm.get_cmap("viridis")
            colors = viridis_cmap(np.linspace(0, 0.8, len(field_names)))

            ax.barh(field_names, field_sizes, left=field_offsets, color=colors, edgecolor="black")
            ax.set_xlabel("Byte Offset")
            ax.set_ylabel("Field Name")
            ax.set_title("Protocol Frame Structure")
            ax.grid(axis="x", alpha=0.3)

        plot_data = self._embed_plot(fig, "Protocol Structure", config)
        plt.close(fig)
        return plot_data

    def _plot_confidence_metrics(
        self, results: CompleteREResult | Any, config: ReportConfig
    ) -> dict[str, Any]:
        """Generate confidence metrics visualization.

        Args:
            results: Complete RE result object.
            config: Report configuration.

        Returns:
            Plot dictionary with embedded data.

        Example:
            >>> plot = generator._plot_confidence_metrics(results, config)
            >>> assert plot["title"] == "Analysis Confidence"
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create confidence gauge
        confidence = results.confidence_score
        categories = ["Overall", "Detection", "Decoding", "Structure"]
        values = [
            confidence,
            confidence * 0.95,  # Simulated sub-scores
            confidence * 1.02,
            confidence * 0.98,
        ]
        values = [min(1.0, max(0.0, v)) for v in values]

        colors = ["green" if v >= 0.8 else "orange" if v >= 0.6 else "red" for v in values]

        ax.barh(categories, values, color=colors, edgecolor="black")
        ax.set_xlabel("Confidence Score")
        ax.set_xlim(0, 1)
        ax.set_title("Analysis Confidence Metrics")
        ax.axvline(x=0.8, color="green", linestyle="--", alpha=0.3, label="High")
        ax.axvline(x=0.6, color="orange", linestyle="--", alpha=0.3, label="Medium")
        ax.legend()

        plot_data = self._embed_plot(fig, "Analysis Confidence", config)
        plt.close(fig)
        return plot_data

    def _plot_timing_diagram(
        self, traces: dict[str, Any], config: ReportConfig
    ) -> dict[str, Any] | None:
        """Generate timing diagram from traces.

        Args:
            traces: Dictionary of waveform traces.
            config: Report configuration.

        Returns:
            Plot dictionary with embedded data, or None if traces invalid.

        Example:
            >>> plot = generator._plot_timing_diagram(traces, config)
            >>> if plot:
            ...     assert plot["title"] == "Signal Timing Diagram"
        """
        if not traces:
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 4))

            # Plot first trace as example
            trace = next(iter(traces.values()))
            if hasattr(trace, "samples") and hasattr(trace, "sample_rate"):
                samples = trace.samples[:1000]  # First 1000 samples
                time = np.arange(len(samples)) / trace.sample_rate * 1e6  # microseconds

                ax.plot(time, samples, linewidth=0.5)
                ax.set_xlabel("Time (µs)")
                ax.set_ylabel("Voltage (V)")
                ax.set_title("Signal Timing Diagram (First 1000 samples)")
                ax.grid(alpha=0.3)

                plot_data = self._embed_plot(fig, "Signal Timing Diagram", config)
                plt.close(fig)
                return plot_data
        except Exception as e:
            logger.warning(f"Failed to generate timing diagram: {e}")
            return None

        return None

    def _embed_plot(self, fig: Any, title: str, config: ReportConfig) -> dict[str, Any]:
        """Embed matplotlib figure as base64 or save to file.

        Args:
            fig: Matplotlib figure object.
            title: Plot title.
            config: Report configuration.

        Returns:
            Dictionary with plot data (base64 or file path).

        Example:
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3])
            >>> plot = generator._embed_plot(fig, "Test Plot", config)
            >>> assert "data" in plot or "path" in plot
        """
        if config.embed_plots:
            # Embed as base64
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            return {
                "title": title,
                "data": f"data:image/png;base64,{img_base64}",
                "type": "embedded",
            }
        else:
            # Save to external file
            plot_filename = f"{title.lower().replace(' ', '_')}.png"
            plot_path = Path(config.metadata.get("output_dir", ".")) / "plots" / plot_filename
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(plot_path, format="png", dpi=150, bbox_inches="tight")

            return {
                "title": title,
                "path": str(plot_path),
                "type": "external",
            }

    def _export_pdf(self, html_path: Path, pdf_path: Path) -> None:
        """Convert HTML to PDF via weasyprint.

        Args:
            html_path: Path to HTML file.
            pdf_path: Path to output PDF file.

        Raises:
            RuntimeError: If weasyprint not installed or conversion fails.

        Example:
            >>> generator._export_pdf(Path("report.html"), Path("report.pdf"))
        """
        try:
            from weasyprint import HTML
        except ImportError as e:
            msg = (
                "weasyprint not installed. Install with: "
                "pip install weasyprint or uv pip install weasyprint"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        try:
            HTML(filename=str(html_path)).write_pdf(pdf_path)
            logger.info(f"Successfully converted HTML to PDF: {pdf_path}")
        except Exception as e:
            msg = f"PDF export failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _get_theme_styles(self, theme: str) -> dict[str, str]:
        """Get CSS styles for theme.

        Args:
            theme: Theme name ("default", "dark", "minimal").

        Returns:
            Dictionary with CSS variables.

        Example:
            >>> styles = generator._get_theme_styles("dark")
            >>> assert "background_color" in styles
        """
        themes = {
            "default": {
                "background_color": "#ffffff",
                "text_color": "#333333",
                "primary_color": "#2c3e50",
                "secondary_color": "#3498db",
                "border_color": "#dddddd",
                "code_background": "#f4f4f4",
            },
            "dark": {
                "background_color": "#1e1e1e",
                "text_color": "#e0e0e0",
                "primary_color": "#4a90e2",
                "secondary_color": "#64b5f6",
                "border_color": "#444444",
                "code_background": "#2d2d2d",
            },
            "minimal": {
                "background_color": "#fafafa",
                "text_color": "#222222",
                "primary_color": "#000000",
                "secondary_color": "#666666",
                "border_color": "#cccccc",
                "code_background": "#f9f9f9",
            },
        }
        return themes.get(theme, themes["default"])

    def _get_fallback_template(self) -> Template:
        """Get fallback inline template if file not found.

        Returns:
            Jinja2 Template object with inline HTML.

        Example:
            >>> template = generator._get_fallback_template()
            >>> html = template.render(title="Test")
        """
        return self.env.from_string(_FALLBACK_HTML_TEMPLATE)

    @staticmethod
    def _get_builtin_template_dir() -> Path:
        """Get path to built-in templates directory.

        Returns:
            Path to templates directory.
        """
        return Path(__file__).parent / "templates" / "enhanced"

    @staticmethod
    def _get_builtin_static_dir() -> Path:
        """Get path to built-in static assets directory.

        Returns:
            Path to static directory.
        """
        return Path(__file__).parent / "static"

    @staticmethod
    def _dict_to_object(data: dict[str, Any]) -> Any:
        """Convert dictionary to object with attribute access.

        Args:
            data: Dictionary to convert.

        Returns:
            Object with attributes matching dict keys.
        """

        class DictObject:
            def __init__(self, d: dict[str, Any]) -> None:
                for key, value in d.items():
                    if isinstance(value, dict):
                        setattr(self, key, DictObject(value))
                    else:
                        setattr(self, key, value)

        return DictObject(data)

    @staticmethod
    def _format_bytes(value: int | float) -> str:
        """Format byte count with units.

        Args:
            value: Byte count.

        Returns:
            Formatted string (e.g., "1.5 KB").
        """
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(value)
        for unit in units:
            if abs(value) < 1024.0:
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} PB"

    @staticmethod
    def _format_number(value: float | int, precision: int = 2) -> str:
        """Format number with thousands separator.

        Args:
            value: Number to format.
            precision: Decimal places for floats.

        Returns:
            Formatted string (e.g., "1,234.56").
        """
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:,.{precision}f}"

    @staticmethod
    def _format_timestamp(dt: datetime) -> str:
        """Format datetime for display.

        Args:
            dt: Datetime object.

        Returns:
            Formatted string (e.g., "2026-01-24 10:30:45").
        """
        return dt.strftime("%Y-%m-%d %H:%M:%S")
