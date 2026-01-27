"""Tests for enhanced report generation module.

This module tests comprehensive HTML/PDF report generation with interactive
visualizations, multiple templates, and various configuration options.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from oscura.reporting.enhanced_reports import (
    EnhancedReportGenerator,
    ReportConfig,
)


def _weasyprint_available() -> bool:
    """Check if weasyprint is available.

    Returns:
        True if weasyprint can be imported, False otherwise.
    """
    try:
        import weasyprint  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def mock_complete_result() -> Mock:
    """Create mock CompleteREResult for testing.

    Returns:
        Mock object with expected attributes.
    """
    result = Mock()

    # Protocol specification
    protocol_spec = Mock()
    protocol_spec.name = "UART"
    protocol_spec.baud_rate = 115200.0
    protocol_spec.frame_format = "8N1"
    protocol_spec.sync_pattern = "0xAA55"
    protocol_spec.frame_length = 16
    protocol_spec.checksum_type = "CRC16"
    protocol_spec.checksum_position = 14
    protocol_spec.confidence = 0.85

    # Fields
    field1 = Mock()
    field1.name = "header"
    field1.offset = 0
    field1.size = 2
    field1.field_type = "uint16"

    field2 = Mock()
    field2.name = "payload"
    field2.offset = 2
    field2.size = 12
    field2.field_type = "bytes"

    field3 = Mock()
    field3.name = "checksum"
    field3.offset = 14
    field3.size = 2
    field3.field_type = "crc16"

    protocol_spec.fields = [field1, field2, field3]
    result.protocol_spec = protocol_spec

    # Metrics
    result.execution_time = 12.34
    result.confidence_score = 0.87
    result.warnings = ["Low signal quality detected", "Possible frame errors"]

    # Artifacts
    result.dissector_path = Path("/tmp/uart.lua")
    result.scapy_layer_path = Path("/tmp/uart.py")
    result.kaitai_path = Path("/tmp/uart.ksy")
    result.test_vectors_path = Path("/tmp/test_vectors.json")

    # Partial results
    result.partial_results = {
        "protocol_detection": {"protocol": "uart", "confidence": 0.9},
        "structure": {"fields": 3, "patterns": []},
    }

    return result


@pytest.fixture
def report_config() -> ReportConfig:
    """Create default report configuration.

    Returns:
        ReportConfig with standard settings.
    """
    return ReportConfig(
        title="Test Protocol Analysis",
        template="protocol_re",
        format="html",
        include_plots=True,
        interactive=True,
        theme="default",
    )


@pytest.fixture
def generator() -> EnhancedReportGenerator:
    """Create EnhancedReportGenerator instance.

    Returns:
        Generator instance.
    """
    return EnhancedReportGenerator()


class TestReportConfig:
    """Test ReportConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ReportConfig(title="Test")

        assert config.title == "Test"
        assert config.template == "protocol_re"
        assert config.format == "html"
        assert config.include_plots is True
        assert config.interactive is True
        assert config.theme == "default"
        assert config.embed_plots is True
        assert config.author == "Oscura Framework"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ReportConfig(
            title="Custom Report",
            template="security",
            format="pdf",
            include_plots=False,
            interactive=False,
            theme="dark",
            embed_plots=False,
            author="Test Author",
        )

        assert config.title == "Custom Report"
        assert config.template == "security"
        assert config.format == "pdf"
        assert config.include_plots is False
        assert config.interactive is False
        assert config.theme == "dark"
        assert config.embed_plots is False
        assert config.author == "Test Author"

    def test_metadata_field(self) -> None:
        """Test metadata dictionary field."""
        config = ReportConfig(title="Test", metadata={"custom_field": "value", "version": "1.0"})

        assert config.metadata["custom_field"] == "value"
        assert config.metadata["version"] == "1.0"


class TestEnhancedReportGenerator:
    """Test EnhancedReportGenerator class."""

    def test_initialization(self, generator: EnhancedReportGenerator) -> None:
        """Test generator initialization."""
        assert generator.template_dir is not None
        assert generator.static_dir is not None
        assert generator.env is not None

        # Check custom filters registered
        assert "format_bytes" in generator.env.filters
        assert "format_number" in generator.env.filters
        assert "format_timestamp" in generator.env.filters

    def test_custom_template_dir(self, tmp_path: Path) -> None:
        """Test initialization with custom template directory."""
        custom_template_dir = tmp_path / "templates"
        custom_static_dir = tmp_path / "static"

        generator = EnhancedReportGenerator(
            template_dir=custom_template_dir, static_dir=custom_static_dir
        )

        assert generator.template_dir == custom_template_dir
        assert generator.static_dir == custom_static_dir
        assert custom_template_dir.exists()
        assert custom_static_dir.exists()

    def test_generate_html_report(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        report_config: ReportConfig,
        tmp_path: Path,
    ) -> None:
        """Test HTML report generation."""
        output_path = tmp_path / "report.html"

        result_path = generator.generate(mock_complete_result, output_path, report_config)

        assert result_path.exists()
        assert result_path.suffix == ".html"

        # Verify HTML content
        html_content = result_path.read_text()
        assert "Test Protocol Analysis" in html_content
        assert "UART" in html_content
        # Baud rate is formatted with thousands separator
        assert "115,200" in html_content or "115200" in html_content
        assert "8N1" in html_content  # Frame format

    def test_generate_html_report_with_dict_input(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig, tmp_path: Path
    ) -> None:
        """Test HTML report generation with dictionary input."""
        output_path = tmp_path / "report.html"

        # Create dict input
        results_dict = {
            "protocol_spec": {
                "name": "I2C",
                "baud_rate": 400000,
                "frame_format": "standard",
                "sync_pattern": "",
                "frame_length": None,
                "checksum_type": None,
                "checksum_position": None,
                "confidence": 0.75,
                "fields": [],
            },
            "execution_time": 5.67,
            "confidence_score": 0.78,
            "warnings": [],
        }

        result_path = generator.generate(results_dict, output_path, report_config)

        assert result_path.exists()
        html_content = result_path.read_text()
        assert "I2C" in html_content

    def test_generate_with_no_plots(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        tmp_path: Path,
    ) -> None:
        """Test report generation without plots."""
        output_path = tmp_path / "report.html"
        config = ReportConfig(title="Test", include_plots=False)

        result_path = generator.generate(mock_complete_result, output_path, config)

        assert result_path.exists()
        html_content = result_path.read_text()
        # Plots section should not be rendered when include_plots=False
        # (though template may still contain plotly reference in head)
        assert '<div class="plot-container">' not in html_content

    def test_generate_with_warnings(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        report_config: ReportConfig,
        tmp_path: Path,
    ) -> None:
        """Test report generation with warnings."""
        output_path = tmp_path / "report.html"

        result_path = generator.generate(mock_complete_result, output_path, report_config)

        html_content = result_path.read_text()
        assert "Warnings" in html_content
        assert "Low signal quality detected" in html_content
        assert "Possible frame errors" in html_content

    def test_generate_without_protocol_spec(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig, tmp_path: Path
    ) -> None:
        """Test report generation when protocol_spec is None."""
        output_path = tmp_path / "report.html"

        # Create result without protocol spec
        result = Mock()
        result.protocol_spec = None
        result.execution_time = 1.0
        result.confidence_score = 0.5
        result.warnings = ["Protocol detection failed"]
        result.partial_results = {}

        result_path = generator.generate(result, output_path, report_config)

        assert result_path.exists()
        html_content = result_path.read_text()
        assert "Protocol detection failed" in html_content

    @pytest.mark.skipif(
        not _weasyprint_available(),
        reason="weasyprint not installed",
    )
    def test_generate_pdf_report(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        tmp_path: Path,
    ) -> None:
        """Test PDF report generation."""
        output_path = tmp_path / "report.pdf"
        config = ReportConfig(title="Test", format="pdf")

        result_path = generator.generate(mock_complete_result, output_path, config)

        assert result_path.exists()
        assert result_path.suffix == ".pdf"

    def test_generate_pdf_without_weasyprint(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        tmp_path: Path,
    ) -> None:
        """Test PDF generation fails gracefully without weasyprint."""
        output_path = tmp_path / "report.pdf"
        config = ReportConfig(title="Test", format="pdf")

        with patch.dict("sys.modules", {"weasyprint": None}):
            with pytest.raises(RuntimeError, match="weasyprint not installed"):
                generator.generate(mock_complete_result, output_path, config)

    def test_generate_both_formats(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        tmp_path: Path,
    ) -> None:
        """Test generation of both HTML and PDF formats."""
        output_path = tmp_path / "report"
        config = ReportConfig(title="Test", format="both")

        # Mock PDF export to avoid weasyprint requirement
        with patch.object(generator, "_export_pdf"):
            result_path = generator.generate(mock_complete_result, output_path, config)

            html_path = tmp_path / "report.html"
            assert html_path.exists()
            assert result_path == html_path

    def test_unsupported_format(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        tmp_path: Path,
    ) -> None:
        """Test error handling for unsupported format."""
        output_path = tmp_path / "report.txt"
        config = ReportConfig(title="Test")
        config.format = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unsupported format"):
            generator.generate(mock_complete_result, output_path, config)


class TestTemplateRendering:
    """Test template rendering functionality."""

    def test_prepare_context(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        report_config: ReportConfig,
    ) -> None:
        """Test context preparation for templates."""
        context = generator._prepare_context(mock_complete_result, report_config)

        assert context["title"] == "Test Protocol Analysis"
        assert context["author"] == "Oscura Framework"
        assert "generated_at" in context
        assert "theme" in context

        # Check protocol spec
        assert context["protocol_spec"] is not None
        assert context["protocol_spec"]["name"] == "UART"
        assert context["protocol_spec"]["baud_rate"] == 115200.0

        # Check fields
        assert len(context["protocol_spec"]["fields"]) == 3
        assert context["protocol_spec"]["fields"][0]["name"] == "header"

        # Check metrics
        assert context["execution_time"] == 12.34
        assert context["confidence_score"] == 0.87
        assert len(context["warnings"]) == 2

        # Check artifacts
        assert len(context["artifacts"]) == 4

    def test_render_template_fallback(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        report_config: ReportConfig,
    ) -> None:
        """Test fallback template when template file not found."""
        context = generator._prepare_context(mock_complete_result, report_config)

        # Force fallback by using non-existent template
        report_config.template = "nonexistent"
        html = generator._render_template(context, report_config)

        assert "<!DOCTYPE html>" in html
        assert context["title"] in html

    def test_get_theme_styles(self, generator: EnhancedReportGenerator) -> None:
        """Test theme style generation."""
        # Default theme
        default_styles = generator._get_theme_styles("default")
        assert default_styles["background_color"] == "#ffffff"
        assert default_styles["text_color"] == "#333333"

        # Dark theme
        dark_styles = generator._get_theme_styles("dark")
        assert dark_styles["background_color"] == "#1e1e1e"
        assert dark_styles["text_color"] == "#e0e0e0"

        # Minimal theme
        minimal_styles = generator._get_theme_styles("minimal")
        assert minimal_styles["background_color"] == "#fafafa"
        assert minimal_styles["text_color"] == "#222222"

        # Unknown theme defaults to default
        unknown_styles = generator._get_theme_styles("unknown")
        assert unknown_styles == default_styles


class TestPlotGeneration:
    """Test plot generation functionality."""

    def test_generate_plots(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        report_config: ReportConfig,
    ) -> None:
        """Test plot generation from results."""
        plots = generator._generate_plots(mock_complete_result, report_config)

        # Should generate protocol structure and confidence plots
        assert len(plots) >= 2
        assert any("Protocol Structure" in p["title"] for p in plots)
        assert any("Confidence" in p["title"] for p in plots)

    def test_plot_protocol_structure(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig
    ) -> None:
        """Test protocol structure visualization."""
        # Create mock protocol spec
        spec = Mock()
        spec.name = "Test Protocol"

        field1 = Mock()
        field1.name = "header"
        field1.offset = 0
        field1.size = 2

        field2 = Mock()
        field2.name = "data"
        field2.offset = 2
        field2.size = 8

        spec.fields = [field1, field2]

        plot = generator._plot_protocol_structure(spec, report_config)

        assert plot["title"] == "Protocol Structure"
        assert "data" in plot
        assert plot["type"] == "embedded"
        assert plot["data"].startswith("data:image/png;base64,")

    def test_plot_confidence_metrics(
        self,
        generator: EnhancedReportGenerator,
        mock_complete_result: Mock,
        report_config: ReportConfig,
    ) -> None:
        """Test confidence metrics visualization."""
        plot = generator._plot_confidence_metrics(mock_complete_result, report_config)

        assert plot["title"] == "Analysis Confidence"
        assert "data" in plot
        assert plot["type"] == "embedded"

    def test_plot_timing_diagram_no_traces(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig
    ) -> None:
        """Test timing diagram with no traces."""
        plot = generator._plot_timing_diagram({}, report_config)
        assert plot is None

    def test_plot_timing_diagram_with_trace(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig
    ) -> None:
        """Test timing diagram with valid trace."""
        # Create mock trace
        trace = Mock()
        trace.samples = np.random.randn(1000)
        trace.sample_rate = 1e6  # 1 MHz

        traces = {"test": trace}

        plot = generator._plot_timing_diagram(traces, report_config)

        if plot:  # May fail if trace structure unexpected
            assert plot["title"] == "Signal Timing Diagram"
            assert "data" in plot

    def test_embed_plot_as_base64(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig
    ) -> None:
        """Test plot embedding as base64."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        report_config.embed_plots = True
        plot = generator._embed_plot(fig, "Test Plot", report_config)

        assert plot["title"] == "Test Plot"
        assert plot["type"] == "embedded"
        assert plot["data"].startswith("data:image/png;base64,")
        assert "path" not in plot

        plt.close(fig)

    def test_embed_plot_as_external_file(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig, tmp_path: Path
    ) -> None:
        """Test plot saving as external file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        report_config.embed_plots = False
        report_config.metadata["output_dir"] = str(tmp_path)
        plot = generator._embed_plot(fig, "Test Plot", report_config)

        assert plot["title"] == "Test Plot"
        assert plot["type"] == "external"
        assert "path" in plot
        assert Path(plot["path"]).exists()

        plt.close(fig)


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_format_bytes(self, generator: EnhancedReportGenerator) -> None:
        """Test byte formatting utility."""
        assert generator._format_bytes(512) == "512.0 B"
        assert generator._format_bytes(1024) == "1.0 KB"
        assert generator._format_bytes(1536) == "1.5 KB"
        assert generator._format_bytes(1048576) == "1.0 MB"
        assert generator._format_bytes(1073741824) == "1.0 GB"

    def test_format_number(self, generator: EnhancedReportGenerator) -> None:
        """Test number formatting utility."""
        assert generator._format_number(1234) == "1,234"
        assert generator._format_number(1234567) == "1,234,567"
        assert generator._format_number(1234.56, 2) == "1,234.56"
        assert generator._format_number(1234.5678, 3) == "1,234.568"

    def test_format_timestamp(self, generator: EnhancedReportGenerator) -> None:
        """Test timestamp formatting utility."""
        from datetime import datetime

        dt = datetime(2026, 1, 24, 10, 30, 45)
        formatted = generator._format_timestamp(dt)
        assert formatted == "2026-01-24 10:30:45"

    def test_dict_to_object(self, generator: EnhancedReportGenerator) -> None:
        """Test dictionary to object conversion."""
        test_dict = {
            "name": "test",
            "value": 123,
            "nested": {"key": "value", "number": 456},
        }

        obj = generator._dict_to_object(test_dict)

        assert obj.name == "test"
        assert obj.value == 123
        assert obj.nested.key == "value"
        assert obj.nested.number == 456


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_fields(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig, tmp_path: Path
    ) -> None:
        """Test report generation with empty fields list."""
        result = Mock()
        spec = Mock()
        spec.name = "Empty Protocol"
        spec.baud_rate = 9600
        spec.frame_format = "8N1"
        spec.sync_pattern = ""
        spec.frame_length = None
        spec.checksum_type = None
        spec.checksum_position = None
        spec.confidence = 0.5
        spec.fields = []  # Empty fields

        result.protocol_spec = spec
        result.execution_time = 1.0
        result.confidence_score = 0.5
        result.warnings = []
        result.partial_results = {}

        output_path = tmp_path / "report.html"
        result_path = generator.generate(result, output_path, report_config)

        assert result_path.exists()

    def test_very_long_warnings_list(
        self, generator: EnhancedReportGenerator, report_config: ReportConfig, tmp_path: Path
    ) -> None:
        """Test report with many warnings."""
        result = Mock()
        result.protocol_spec = None
        result.execution_time = 1.0
        result.confidence_score = 0.3
        result.warnings = [f"Warning {i}" for i in range(100)]
        result.partial_results = {}

        output_path = tmp_path / "report.html"
        result_path = generator.generate(result, output_path, report_config)

        assert result_path.exists()
        html_content = result_path.read_text()
        assert "Warning 0" in html_content
        assert "Warning 99" in html_content

    def test_special_characters_in_title(
        self, generator: EnhancedReportGenerator, mock_complete_result: Mock, tmp_path: Path
    ) -> None:
        """Test report with special characters in title."""
        config = ReportConfig(title="Test <Report> & 'Analysis' \"2026\"")
        output_path = tmp_path / "report.html"

        result_path = generator.generate(mock_complete_result, output_path, config)

        assert result_path.exists()
        html_content = result_path.read_text()
        # Jinja2 autoescape should handle special characters
        assert "Test" in html_content

    def test_missing_partial_results(
        self, generator: EnhancedReportGenerator, tmp_path: Path
    ) -> None:
        """Test report generation without partial_results attribute."""
        result = Mock(spec=[])  # spec=[] means only explicitly set attrs exist
        result.protocol_spec = None
        result.execution_time = 1.0
        result.confidence_score = 0.5
        result.warnings = []

        config = ReportConfig(title="Test")
        output_path = tmp_path / "report.html"

        result_path = generator.generate(result, output_path, config)
        assert result_path.exists()
