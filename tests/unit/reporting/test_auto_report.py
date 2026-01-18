"""Comprehensive unit tests for oscura.reporting.auto_report module.

Tests automatic executive report generation with various configurations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from oscura.reporting.auto_report import (
    Report,
    ReportMetadata,
    _generate_detailed_results,
    _generate_executive_summary,
    _generate_key_findings,
    _generate_methodology,
    generate_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_waveform_trace():
    """Create a mock WaveformTrace for testing."""
    trace = MagicMock()
    trace.data = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0])
    trace.metadata = MagicMock()
    trace.metadata.sample_rate = 1e6  # 1 MS/s
    return trace


@pytest.fixture
def large_waveform_trace():
    """Create a larger mock waveform for more realistic testing."""
    trace = MagicMock()
    # Generate 10000 samples of a sine wave
    t = np.linspace(0, 2 * np.pi * 10, 10000)
    trace.data = 2.5 + 1.0 * np.sin(t)  # Centered at 2.5V with 1V amplitude
    trace.metadata = MagicMock()
    trace.metadata.sample_rate = 10e6  # 10 MS/s
    return trace


class TestReportMetadata:
    """Tests for ReportMetadata dataclass."""

    def test_default_metadata(self):
        """Test creating metadata with defaults."""
        metadata = ReportMetadata()

        assert metadata.title == "Signal Analysis Report"
        assert metadata.author == "Oscura"
        assert metadata.project is None
        assert metadata.tags == []
        # Date should be set to today
        assert len(metadata.date) == 10  # YYYY-MM-DD format

    def test_custom_metadata(self):
        """Test creating metadata with custom values."""
        metadata = ReportMetadata(
            title="Custom Report",
            author="Test User",
            date="2024-01-15",
            project="Project X",
            tags=["test", "signal", "analysis"],
        )

        assert metadata.title == "Custom Report"
        assert metadata.author == "Test User"
        assert metadata.date == "2024-01-15"
        assert metadata.project == "Project X"
        assert metadata.tags == ["test", "signal", "analysis"]


class TestReport:
    """Tests for Report dataclass."""

    def test_default_report(self):
        """Test creating report with defaults."""
        report = Report()

        assert report.sections == []
        assert report.plots == []
        assert report.page_count == 0
        assert isinstance(report.metadata, ReportMetadata)
        assert report.content == {}
        assert report.output_path is None
        assert report.file_size_mb == 0.0

    def test_report_with_sections(self):
        """Test report with custom sections."""
        report = Report(
            sections=["summary", "results", "conclusions"],
            page_count=5,
        )

        assert len(report.sections) == 3
        assert "summary" in report.sections
        assert report.page_count == 5

    def test_add_section_append(self):
        """Test adding section at the end."""
        report = Report(sections=["intro"])
        report.add_section("conclusions", "This is the conclusion.")

        assert len(report.sections) == 2
        assert report.sections[-1] == "conclusions"
        assert "conclusions" in report.content
        assert report.content["conclusions"] == "This is the conclusion."

    def test_add_section_with_position(self):
        """Test adding section at specific position."""
        report = Report(sections=["intro", "results"])
        report.add_section("methodology", "Methods used.", position=1)

        assert len(report.sections) == 3
        assert report.sections[1] == "methodology"
        assert report.sections[0] == "intro"
        assert report.sections[2] == "results"

    def test_add_section_normalizes_title(self):
        """Test that section title is normalized."""
        report = Report()
        report.add_section("Key Findings", "Some findings.")

        assert "key_findings" in report.sections
        assert "key_findings" in report.content

    def test_include_plots(self):
        """Test including specific plot types."""
        report = Report()
        report.include_plots(["time_domain", "fft_spectrum", "histogram"])

        assert len(report.plots) == 3
        assert "time_domain" in report.plots

    def test_set_metadata_partial(self):
        """Test setting partial metadata."""
        report = Report()
        original_date = report.metadata.date

        report.set_metadata(title="New Title", author="New Author")

        assert report.metadata.title == "New Title"
        assert report.metadata.author == "New Author"
        assert report.metadata.date == original_date  # Unchanged

    def test_set_metadata_full(self):
        """Test setting all metadata fields."""
        report = Report()
        report.set_metadata(
            title="Full Report",
            author="Jane Doe",
            date="2024-06-01",
            project="Test Project",
            tags=["tag1", "tag2"],
        )

        assert report.metadata.title == "Full Report"
        assert report.metadata.author == "Jane Doe"
        assert report.metadata.date == "2024-06-01"
        assert report.metadata.project == "Test Project"
        assert report.metadata.tags == ["tag1", "tag2"]

    def test_set_metadata_none_values_ignored(self):
        """Test that None values don't override existing metadata."""
        report = Report()
        report.metadata.title = "Existing Title"

        report.set_metadata(title=None, author="New Author")

        assert report.metadata.title == "Existing Title"
        assert report.metadata.author == "New Author"


class TestReportSavePDF:
    """Tests for Report.save_pdf method."""

    def test_save_pdf_basic(self, tmp_path):
        """Test saving report as PDF."""
        report = Report(sections=["summary"])
        report.content["summary"] = "This is a summary."

        output_path = tmp_path / "report.pdf"
        report.save_pdf(str(output_path))

        assert output_path.exists()
        assert report.output_path == str(output_path)
        assert report.file_size_mb > 0

    def test_save_pdf_with_metadata(self, tmp_path):
        """Test that PDF includes metadata."""
        report = Report(sections=["summary"])
        report.content["summary"] = "Content here."
        report.metadata.title = "Test PDF Report"
        report.metadata.date = "2024-01-01"

        output_path = tmp_path / "report.pdf"
        report.save_pdf(str(output_path))

        content = output_path.read_text()
        assert "Test PDF Report" in content
        assert "2024-01-01" in content

    def test_save_pdf_multiple_sections(self, tmp_path):
        """Test PDF with multiple sections."""
        report = Report(sections=["intro", "results", "conclusion"])
        report.content["intro"] = "Introduction text."
        report.content["results"] = "Results text."
        report.content["conclusion"] = "Conclusion text."

        output_path = tmp_path / "report.pdf"
        report.save_pdf(str(output_path))

        content = output_path.read_text()
        assert "INTRO" in content
        assert "RESULTS" in content
        assert "CONCLUSION" in content


class TestReportSaveHTML:
    """Tests for Report.save_html method."""

    def test_save_html_basic(self, tmp_path):
        """Test saving report as HTML."""
        report = Report(sections=["summary"])
        report.content["summary"] = "This is a summary."

        output_path = tmp_path / "report.html"
        report.save_html(str(output_path))

        assert output_path.exists()
        assert report.output_path == str(output_path)

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content

    def test_save_html_includes_styles(self, tmp_path):
        """Test that HTML includes CSS styles."""
        report = Report(sections=["summary"])
        report.content["summary"] = "Content."

        output_path = tmp_path / "report.html"
        report.save_html(str(output_path))

        content = output_path.read_text()
        assert "<style>" in content
        assert "font-family" in content

    def test_save_html_metadata(self, tmp_path):
        """Test HTML includes metadata."""
        report = Report(sections=[])
        report.metadata.title = "HTML Test Report"
        report.metadata.author = "Test Author"
        report.metadata.project = "Test Project"
        report.metadata.tags = ["test", "html"]

        output_path = tmp_path / "report.html"
        report.save_html(str(output_path))

        content = output_path.read_text()
        assert "HTML Test Report" in content
        assert "Test Author" in content
        assert "Test Project" in content
        assert "test, html" in content

    def test_save_html_without_project_and_tags(self, tmp_path):
        """Test HTML without optional project and tags."""
        report = Report(sections=["summary"])
        report.content["summary"] = "Content."
        report.metadata.project = None
        report.metadata.tags = []

        output_path = tmp_path / "report.html"
        report.save_html(str(output_path))

        content = output_path.read_text()
        assert "Project:" not in content
        assert "Tags:" not in content


class TestReportSaveMarkdown:
    """Tests for Report.save_markdown method."""

    def test_save_markdown_basic(self, tmp_path):
        """Test saving report as Markdown."""
        report = Report(sections=["summary"])
        report.content["summary"] = "This is a summary."

        output_path = tmp_path / "report.md"
        report.save_markdown(str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "# " in content  # Heading
        assert "## " in content  # Section heading

    def test_save_markdown_metadata(self, tmp_path):
        """Test Markdown includes metadata."""
        report = Report(sections=[])
        report.metadata.title = "MD Test Report"
        report.metadata.author = "MD Author"

        output_path = tmp_path / "report.md"
        report.save_markdown(str(output_path))

        content = output_path.read_text()
        assert "# MD Test Report" in content
        assert "**Author:** MD Author" in content

    def test_save_markdown_with_project_and_tags(self, tmp_path):
        """Test Markdown with project and tags."""
        report = Report(sections=[])
        report.metadata.project = "Test Project"
        report.metadata.tags = ["markdown", "test"]

        output_path = tmp_path / "report.md"
        report.save_markdown(str(output_path))

        content = output_path.read_text()
        assert "**Project:** Test Project" in content
        assert "**Tags:** markdown, test" in content


class TestGenerateExecutiveSummary:
    """Tests for _generate_executive_summary function."""

    def test_executive_summary_basic(self, mock_waveform_trace):
        """Test generating basic executive summary."""
        summary = _generate_executive_summary(mock_waveform_trace, {})

        assert isinstance(summary, str)
        assert "sample rate" in summary.lower()
        assert "milliseconds" in summary

    def test_executive_summary_voltage_range(self, mock_waveform_trace):
        """Test that summary includes voltage range."""
        summary = _generate_executive_summary(mock_waveform_trace, {})

        assert "-1.0" in summary or "-1.00" in summary
        assert "1.0" in summary or "1.00" in summary

    def test_executive_summary_with_characterization(self, mock_waveform_trace):
        """Test summary with characterization context."""
        context = {
            "characterization": MagicMock(signal_type="digital_clock"),
        }

        summary = _generate_executive_summary(mock_waveform_trace, context)

        assert "digital_clock" in summary

    def test_executive_summary_with_quality(self, mock_waveform_trace):
        """Test summary with quality context."""
        context = {
            "quality": MagicMock(status="good"),
        }

        summary = _generate_executive_summary(mock_waveform_trace, context)

        assert "good" in summary.lower()


class TestGenerateKeyFindings:
    """Tests for _generate_key_findings function."""

    def test_key_findings_basic(self, mock_waveform_trace):
        """Test generating basic key findings."""
        findings = _generate_key_findings(mock_waveform_trace, {})

        assert isinstance(findings, str)
        assert "Signal swing" in findings

    def test_key_findings_with_anomalies(self, mock_waveform_trace):
        """Test findings with anomalies context."""
        context = {
            "anomalies": [1, 2, 3, 4, 5],  # 5 anomalies
        }

        findings = _generate_key_findings(mock_waveform_trace, context)

        assert "5 anomalies" in findings

    def test_key_findings_with_decode(self, mock_waveform_trace):
        """Test findings with decode context."""
        decode_mock = MagicMock()
        decode_mock.data = bytes(100)  # 100 bytes decoded

        context = {"decode": decode_mock}

        findings = _generate_key_findings(mock_waveform_trace, context)

        assert "100 bytes" in findings


class TestGenerateMethodology:
    """Tests for _generate_methodology function."""

    def test_methodology_basic(self, mock_waveform_trace):
        """Test generating basic methodology."""
        methodology = _generate_methodology(mock_waveform_trace, {})

        assert isinstance(methodology, str)
        assert "Signal characterization" in methodology
        assert "Quality assessment" in methodology

    def test_methodology_with_anomalies(self, mock_waveform_trace):
        """Test methodology with anomaly detection context."""
        context = {"anomalies": []}

        methodology = _generate_methodology(mock_waveform_trace, context)

        assert "Anomaly detection" in methodology

    def test_methodology_with_decode(self, mock_waveform_trace):
        """Test methodology with protocol decode context."""
        context = {"decode": MagicMock()}

        methodology = _generate_methodology(mock_waveform_trace, context)

        assert "Protocol decode" in methodology


class TestGenerateDetailedResults:
    """Tests for _generate_detailed_results function."""

    def test_detailed_results_basic(self, mock_waveform_trace):
        """Test generating detailed results."""
        results = _generate_detailed_results(mock_waveform_trace, {})

        assert isinstance(results, str)
        assert "Minimum voltage" in results
        assert "Maximum voltage" in results
        assert "Mean voltage" in results
        assert "Standard deviation" in results
        assert "Peak-to-peak" in results

    def test_detailed_results_sample_info(self, mock_waveform_trace):
        """Test that results include sample information."""
        results = _generate_detailed_results(mock_waveform_trace, {})

        assert "Sample count" in results
        assert "Sample rate" in results
        assert "Duration" in results

    def test_detailed_results_accurate_statistics(self, large_waveform_trace):
        """Test that statistics are accurately calculated."""
        results = _generate_detailed_results(large_waveform_trace, {})

        # For sine wave centered at 2.5V with 1V amplitude:
        # Min should be ~1.5V, Max ~3.5V, Mean ~2.5V
        assert "1.5" in results  # min
        assert "3.5" in results  # max
        assert "2.5" in results  # mean


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_report_basic(self, mock_waveform_trace):
        """Test generating basic report."""
        report = generate_report(mock_waveform_trace)

        assert isinstance(report, Report)
        assert len(report.sections) > 0
        assert len(report.content) > 0

    def test_generate_report_default_sections(self, mock_waveform_trace):
        """Test default sections are included."""
        report = generate_report(mock_waveform_trace)

        assert "executive_summary" in report.sections
        assert "key_findings" in report.sections
        assert "methodology" in report.sections
        assert "detailed_results" in report.sections

    def test_generate_report_custom_sections(self, mock_waveform_trace):
        """Test selecting custom sections."""
        options = {"select_sections": ["summary", "results", "recommendations"]}

        report = generate_report(mock_waveform_trace, options=options)

        # Should include executive_summary (maps from "summary")
        assert "executive_summary" in report.content

    def test_generate_report_with_context(self, mock_waveform_trace):
        """Test report generation with pre-computed context."""
        context = {
            "characterization": MagicMock(signal_type="i2c"),
            "anomalies": [1, 2, 3],
        }

        report = generate_report(mock_waveform_trace, context=context)

        # Context should influence content
        summary = report.content.get("executive_summary", "")
        assert "i2c" in summary

    def test_generate_report_auto_plots(self, large_waveform_trace):
        """Test that plots are auto-selected."""
        report = generate_report(large_waveform_trace)

        assert len(report.plots) > 0
        assert "time_domain_waveform" in report.plots

    def test_generate_report_with_custom_header(self, mock_waveform_trace):
        """Test report with custom header option."""
        options = {"custom_header": "Custom Report Title"}

        report = generate_report(mock_waveform_trace, options=options)

        assert report.metadata.title == "Custom Report Title"

    def test_generate_report_page_count(self, mock_waveform_trace):
        """Test that page count is estimated."""
        report = generate_report(mock_waveform_trace)

        assert report.page_count > 0

    def test_generate_report_format_options(self, mock_waveform_trace):
        """Test format option is accepted."""
        report = generate_report(mock_waveform_trace, format="html")

        # Format doesn't affect content, just validates API
        assert isinstance(report, Report)

    def test_generate_report_recommendations_section(self, mock_waveform_trace):
        """Test recommendations section when included."""
        options = {"select_sections": ["recommendations"]}

        report = generate_report(mock_waveform_trace, options=options)

        assert "recommendations" in report.content
        assert "Recommendations" in report.content["recommendations"]


class TestReportIntegration:
    """Integration tests for complete report workflows."""

    def test_full_report_workflow(self, large_waveform_trace, tmp_path):
        """Test complete report generation workflow."""
        # Generate report
        report = generate_report(large_waveform_trace)

        # Verify report structure
        assert len(report.sections) >= 4
        assert len(report.content) >= 4

        # Save to all formats
        pdf_path = tmp_path / "report.pdf"
        html_path = tmp_path / "report.html"
        md_path = tmp_path / "report.md"

        report.save_pdf(str(pdf_path))
        report.save_html(str(html_path))
        report.save_markdown(str(md_path))

        assert pdf_path.exists()
        assert html_path.exists()
        assert md_path.exists()

    def test_customized_report(self, mock_waveform_trace, tmp_path):
        """Test fully customized report."""
        report = generate_report(mock_waveform_trace)

        # Customize metadata
        report.set_metadata(
            title="Customized Analysis Report",
            author="Test Engineer",
            project="Signal Analysis Project",
            tags=["custom", "test"],
        )

        # Add custom section
        report.add_section("Custom Analysis", "Custom analysis content here.")

        # Save and verify
        output_path = tmp_path / "custom_report.html"
        report.save_html(str(output_path))

        content = output_path.read_text()
        assert "Customized Analysis Report" in content
        assert "Custom Analysis" in content

    def test_report_with_all_context(self, large_waveform_trace):
        """Test report with full context."""
        context = {
            "characterization": MagicMock(signal_type="spi_data"),
            "quality": MagicMock(status="excellent"),
            "anomalies": [1, 2],
            "decode": MagicMock(data=bytes(256)),
        }

        report = generate_report(large_waveform_trace, context=context)

        # All context should be reflected
        summary = report.content["executive_summary"]
        assert "spi_data" in summary
        assert "excellent" in summary.lower()

        findings = report.content["key_findings"]
        assert "2 anomalies" in findings
        assert "256 bytes" in findings

        methodology = report.content["methodology"]
        assert "Anomaly detection" in methodology
        assert "Protocol decode" in methodology
