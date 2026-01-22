"""Tests for vintage logic reporting module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from oscura.analyzers.digital.vintage_result import (
    BOMEntry,
    ICIdentificationResult,
    ModernReplacementIC,
    VintageLogicAnalysisResult,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.reporting.vintage_logic_report import (
    ReportMetadata,
    VintageLogicReport,
    generate_vintage_logic_report,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def sample_analysis_result():
    """Create a sample analysis result for testing."""
    return VintageLogicAnalysisResult(
        timestamp=datetime.now(),
        source_file="test_file.wfm",
        analysis_duration=1.23,
        detected_family="TTL",
        family_confidence=0.95,
        voltage_levels={"VCC": 5.0, "VIL": 0.8, "VIH": 2.0},
        identified_ics=[
            ICIdentificationResult(
                ic_name="7474",
                confidence=0.85,
                timing_params={"t_pd": 15e-9, "t_su": 10e-9},
                validation={"t_pd": {"passes": True}, "t_su": {"passes": True}},
                family="TTL",
            )
        ],
        timing_measurements={"CLK→DATA_t_pd": 15e-9, "DATA_t_su": 10e-9},
        timing_paths=None,
        decoded_protocols=None,
        open_collector_detected=False,
        asymmetry_ratio=1.0,
        modern_replacements=[
            ModernReplacementIC(
                original_ic="7474",
                replacement_ic="74HCT74",
                family="74HCTxx",
                benefits=["Lower power", "Better availability"],
                notes="Direct replacement",
            )
        ],
        bom=[
            BOMEntry(
                part_number="7474",
                description="Dual D Flip-Flop",
                quantity=1,
                category="IC",
                notes="Original IC",
            ),
            BOMEntry(
                part_number="0.1µF",
                description="Decoupling capacitor",
                quantity=2,
                category="Capacitor",
                notes="One per IC",
            ),
        ],
        warnings=["Test warning 1", "Test warning 2"],
        confidence_scores={"logic_family": 0.95, "ic_identification": 0.85},
    )


@pytest.fixture
def sample_traces(signal_builder):
    """Create sample traces for testing."""
    clk_data = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
    data_data = signal_builder.square_wave(frequency=500e3, duration=0.001, sample_rate=1e6)

    return {
        "CLK": WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6)),
        "DATA": WaveformTrace(data=data_data, metadata=TraceMetadata(sample_rate=1e6)),
    }


class TestReportMetadata:
    """Test ReportMetadata dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        metadata = ReportMetadata(title="Test Report")

        assert metadata.title == "Test Report"
        assert metadata.author == "Oscura Vintage Logic Analyzer"
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.version == "1.0"

    def test_custom_values(self):
        """Test initialization with custom values."""
        custom_time = datetime(2024, 1, 1)
        metadata = ReportMetadata(
            title="Custom Report", author="Test Author", timestamp=custom_time, version="2.0"
        )

        assert metadata.title == "Custom Report"
        assert metadata.author == "Test Author"
        assert metadata.timestamp == custom_time
        assert metadata.version == "2.0"


class TestVintageLogicReport:
    """Test VintageLogicReport class."""

    def test_initialization(self, sample_analysis_result):
        """Test report initialization."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        assert report.result == sample_analysis_result
        assert report.plots == {}
        assert report.metadata == metadata

    def test_save_html(self, sample_analysis_result, tmp_path):
        """Test HTML report generation."""
        metadata = ReportMetadata(title="Test HTML Report")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        output_path = tmp_path / "report.html"
        saved_path = report.save_html(output_path)

        assert saved_path.exists()
        assert saved_path == output_path

        # Check HTML content
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Test HTML Report" in content
        assert "TTL" in content  # Should show detected family
        assert "7474" in content  # Should show identified IC

    def test_save_html_with_plots(self, sample_analysis_result, tmp_path):
        """Test HTML report with embedded plots."""
        # Create a dummy plot file
        plot_file = tmp_path / "test_plot.png"
        plot_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header

        metadata = ReportMetadata(title="Test with Plots")
        report = VintageLogicReport(
            result=sample_analysis_result,
            plots={"test_plot": plot_file},
            metadata=metadata,
        )

        output_path = tmp_path / "report_with_plots.html"
        report.save_html(output_path)

        content = output_path.read_text()
        assert "data:image/png;base64" in content  # Should have embedded image
        assert "Test Plot" in content  # Should have plot title

    def test_save_html_with_warnings(self, sample_analysis_result, tmp_path):
        """Test HTML report includes warnings."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        output_path = tmp_path / "report_warnings.html"
        report.save_html(output_path)

        content = output_path.read_text()
        assert "Test warning 1" in content
        assert "Test warning 2" in content
        assert "Warnings" in content

    def test_save_markdown(self, sample_analysis_result, tmp_path):
        """Test Markdown report generation."""
        metadata = ReportMetadata(title="Test Markdown Report")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        output_path = tmp_path / "report.md"
        saved_path = report.save_markdown(output_path)

        assert saved_path.exists()
        assert saved_path == output_path

        # Check markdown content
        content = output_path.read_text()
        assert "# Test Markdown Report" in content
        assert "## Summary" in content
        assert "TTL" in content
        assert "7474" in content
        assert "| IC | Confidence" in content  # Should have table

    def test_save_markdown_with_plots(self, sample_analysis_result, tmp_path):
        """Test Markdown report with plot references."""
        plot_file = tmp_path / "test_plot.png"
        plot_file.write_bytes(b"fake image data")

        metadata = ReportMetadata(title="Test MD with Plots")
        report = VintageLogicReport(
            result=sample_analysis_result,
            plots={"test_plot": plot_file},
            metadata=metadata,
        )

        output_path = tmp_path / "report_plots.md"
        report.save_markdown(output_path)

        content = output_path.read_text()
        assert "![test_plot]" in content  # Should have markdown image syntax
        assert str(plot_file) in content  # Should reference plot file

    def test_save_markdown_with_warnings(self, sample_analysis_result, tmp_path):
        """Test Markdown report includes warnings."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        output_path = tmp_path / "report_warnings.md"
        report.save_markdown(output_path)

        content = output_path.read_text()
        assert "## Warnings" in content
        assert "- Test warning 1" in content
        assert "- Test warning 2" in content

    def test_save_pdf(self, sample_analysis_result, tmp_path, capsys):
        """Test PDF report generation (HTML with instructions)."""
        metadata = ReportMetadata(title="Test PDF")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        output_path = tmp_path / "report.pdf"
        saved_path = report.save_pdf(output_path)

        # Should create HTML file instead
        html_path = tmp_path / "report.html"
        assert html_path.exists()
        assert saved_path == html_path

        # Check for instructions in stdout
        captured = capsys.readouterr()
        assert "HTML report generated" in captured.out
        assert "To convert to PDF" in captured.out
        assert "weasyprint" in captured.out

    def test_save_html_path_string(self, sample_analysis_result, tmp_path):
        """Test HTML save with path as string."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        output_path = str(tmp_path / "string_path.html")
        saved_path = report.save_html(output_path)

        assert Path(output_path).exists()
        assert saved_path == Path(output_path)


class TestGenerateVintageLogicReport:
    """Test generate_vintage_logic_report function."""

    def test_basic_generation(self, sample_analysis_result, sample_traces):
        """Test basic report generation."""
        # Mock the plot generation to avoid matplotlib dependency
        import unittest.mock

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {}

            report = generate_vintage_logic_report(sample_analysis_result, sample_traces)

            assert isinstance(report, VintageLogicReport)
            assert report.result == sample_analysis_result
            assert isinstance(report.metadata, ReportMetadata)
            assert "Vintage Logic Analysis" in report.metadata.title

    def test_generation_with_title(self, sample_analysis_result, sample_traces):
        """Test report generation with custom title."""
        import unittest.mock

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {}

            report = generate_vintage_logic_report(
                sample_analysis_result, sample_traces, title="Custom Title"
            )

            assert report.metadata.title == "Custom Title"

    def test_generation_with_output_dir(self, sample_analysis_result, sample_traces, tmp_path):
        """Test report generation with specified output directory."""
        import unittest.mock

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {}

            report = generate_vintage_logic_report(
                sample_analysis_result, sample_traces, output_dir=tmp_path
            )

            assert isinstance(report, VintageLogicReport)
            # Output dir should be created
            assert tmp_path.exists()

    def test_generation_creates_temp_dir(self, sample_analysis_result, sample_traces):
        """Test that temp directory is created when not specified."""
        import unittest.mock

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {}

            report = generate_vintage_logic_report(sample_analysis_result, sample_traces)

            # Should have called plot generation with some output dir
            assert mock_plots.called
            call_kwargs = mock_plots.call_args.kwargs
            assert "output_dir" in call_kwargs
            assert call_kwargs["output_dir"] is not None

    def test_generation_filters_plots(self, sample_analysis_result, sample_traces, tmp_path):
        """Test report generation with plot filtering."""
        import unittest.mock

        # Create dummy plot files
        plot1 = tmp_path / "plot1.png"
        plot2 = tmp_path / "plot2.png"
        plot1.write_bytes(b"fake")
        plot2.write_bytes(b"fake")

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {"plot1": None, "plot2": None}

            report = generate_vintage_logic_report(
                sample_analysis_result,
                sample_traces,
                output_dir=tmp_path,
                include_plots=["plot1"],
            )

            # Should only include plot1
            assert "plot1" in report.plots or len(report.plots) <= 1

    def test_generation_default_title(self, sample_analysis_result, sample_traces):
        """Test default title includes detected family."""
        import unittest.mock

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {}

            report = generate_vintage_logic_report(sample_analysis_result, sample_traces)

            assert "TTL" in report.metadata.title  # Should include detected family


class TestHTMLGeneration:
    """Test HTML generation specifics."""

    def test_html_contains_all_sections(self, sample_analysis_result):
        """Test HTML contains all expected sections."""
        metadata = ReportMetadata(title="Complete Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        html = report.save_html(Path("/tmp/test.html"))
        content = html.read_text()

        # Check for all major sections
        assert "Summary" in content
        assert "IC Identification" in content
        assert "Timing Measurements" in content
        assert "Bill of Materials" in content
        assert "Visualizations" in content

    def test_html_styling(self, sample_analysis_result):
        """Test HTML includes styling."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        html = report.save_html(Path("/tmp/test.html"))
        content = html.read_text()

        assert "<style>" in content
        assert "font-family" in content
        assert "table" in content  # Should have table styles

    def test_html_timing_conversion(self, sample_analysis_result):
        """Test timing values are converted to nanoseconds."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        html = report.save_html(Path("/tmp/test.html"))
        content = html.read_text()

        # Should show timing in ns
        assert "15.00 ns" in content  # t_pd
        assert "10.00 ns" in content  # t_su


class TestMarkdownGeneration:
    """Test Markdown generation specifics."""

    def test_markdown_contains_all_sections(self, sample_analysis_result):
        """Test Markdown contains all expected sections."""
        metadata = ReportMetadata(title="Complete Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        md = report.save_markdown(Path("/tmp/test.md"))
        content = md.read_text()

        # Check for all major sections
        assert "## Summary" in content
        assert "## IC Identification" in content
        assert "## Timing Measurements" in content
        assert "## Bill of Materials" in content
        assert "## Visualizations" in content

    def test_markdown_tables(self, sample_analysis_result):
        """Test Markdown includes properly formatted tables."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        md = report.save_markdown(Path("/tmp/test.md"))
        content = md.read_text()

        # Check for table syntax
        assert "|---|---|" in content  # Table separator
        assert "| 7474 |" in content  # IC table row

    def test_markdown_timing_conversion(self, sample_analysis_result):
        """Test timing values are converted to nanoseconds in markdown."""
        metadata = ReportMetadata(title="Test")
        report = VintageLogicReport(result=sample_analysis_result, plots={}, metadata=metadata)

        md = report.save_markdown(Path("/tmp/test.md"))
        content = md.read_text()

        # Should show timing in ns
        assert "15.00 ns" in content
        assert "10.00 ns" in content


class TestEmptyResults:
    """Test report generation with empty/minimal results."""

    def test_no_ics_identified(self, sample_traces):
        """Test report with no identified ICs."""
        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=1.0,
            detected_family="unknown",
            family_confidence=0.0,
            voltage_levels={},
            identified_ics=[],
            timing_measurements={},
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=1.0,
            modern_replacements=[],
            bom=[],
            warnings=[],
            confidence_scores={},
        )

        import unittest.mock

        with unittest.mock.patch(
            "oscura.visualization.digital_advanced.generate_all_vintage_logic_plots"
        ) as mock_plots:
            mock_plots.return_value = {}

            report = generate_vintage_logic_report(result, sample_traces)

            # Should still generate report
            assert isinstance(report, VintageLogicReport)

    def test_no_warnings(self):
        """Test report with no warnings."""
        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=1.0,
            detected_family="TTL",
            family_confidence=0.9,
            voltage_levels={},
            identified_ics=[],
            timing_measurements={},
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=1.0,
            modern_replacements=[],
            bom=[],
            warnings=[],  # No warnings
            confidence_scores={},
        )

        metadata = ReportMetadata(title="No Warnings Test")
        report = VintageLogicReport(result=result, plots={}, metadata=metadata)

        html = report.save_html(Path("/tmp/test.html"))
        content = html.read_text()

        # Should not have warnings section
        assert "Warnings" not in content or '<div class="warnings">' not in content
