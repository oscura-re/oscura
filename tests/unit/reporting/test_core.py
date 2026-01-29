"""Tests for core report generation.

Tests the main report generation functionality including report structure,
configuration, and output generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.reporting.core import (
    Report,
    ReportConfig,
    Section,
    generate_report,
)

pytestmark = pytest.mark.unit


class TestSection:
    """Test Section dataclass."""

    def test_create_simple_section(self) -> None:
        """Test creating a simple section."""
        section = Section(title="Introduction", content="This is the intro")
        assert section.title == "Introduction"
        assert section.content == "This is the intro"
        assert section.level == 2
        assert section.visible is True

    def test_create_section_with_level(self) -> None:
        """Test creating section with custom level."""
        section = Section(title="Subsection", content="Text", level=3)
        assert section.level == 3

    def test_create_collapsible_section(self) -> None:
        """Test creating collapsible section."""
        section = Section(title="Details", collapsible=True)
        assert section.collapsible is True

    def test_section_with_subsections(self) -> None:
        """Test section with subsections."""
        subsection = Section(title="Sub", content="Sub content", level=3)
        section = Section(title="Main", subsections=[subsection])
        assert len(section.subsections) == 1
        assert section.subsections[0].title == "Sub"

    def test_section_invisible(self) -> None:
        """Test invisible section."""
        section = Section(title="Hidden", visible=False)
        assert section.visible is False


class TestReportConfig:
    """Test ReportConfig dataclass."""

    def test_create_default_config(self) -> None:
        """Test creating default configuration."""
        config = ReportConfig()
        assert config.title == "Oscura Analysis Report"
        assert config.verbosity == "standard"
        assert config.format == "pdf"
        assert config.show_toc is True

    def test_create_custom_config(self) -> None:
        """Test creating custom configuration."""
        config = ReportConfig(
            title="Custom Report",
            author="John Doe",
            verbosity="detailed",
            format="html",
        )
        assert config.title == "Custom Report"
        assert config.author == "John Doe"
        assert config.verbosity == "detailed"
        assert config.format == "html"

    def test_config_page_settings(self) -> None:
        """Test page settings."""
        config = ReportConfig(page_size="A4", margins=2.0)
        assert config.page_size == "A4"
        assert config.margins == 2.0

    def test_config_with_logo(self) -> None:
        """Test configuration with logo."""
        config = ReportConfig(logo_path="/path/to/logo.png")
        assert config.logo_path == "/path/to/logo.png"

    def test_config_with_watermark(self) -> None:
        """Test configuration with watermark."""
        config = ReportConfig(watermark="CONFIDENTIAL")
        assert config.watermark == "CONFIDENTIAL"

    def test_config_verbosity_levels(self) -> None:
        """Test all verbosity levels."""
        for level in ["executive", "summary", "standard", "detailed", "debug"]:
            config = ReportConfig(verbosity=level)  # type: ignore[arg-type]
            assert config.verbosity == level

    def test_config_formats(self) -> None:
        """Test all output formats."""
        for fmt in ["pdf", "html", "markdown", "docx"]:
            config = ReportConfig(format=fmt)  # type: ignore[arg-type]
            assert config.format == fmt


class TestReport:
    """Test Report class."""

    def test_create_empty_report(self) -> None:
        """Test creating empty report."""
        config = ReportConfig()
        report = Report(config=config)
        assert len(report.sections) == 0
        assert len(report.figures) == 0
        assert len(report.tables) == 0

    def test_add_section(self) -> None:
        """Test adding section to report."""
        config = ReportConfig()
        report = Report(config=config)

        section = report.add_section("Introduction", "Welcome to the report")
        assert len(report.sections) == 1
        assert section.title == "Introduction"
        assert section.content == "Welcome to the report"

    def test_add_multiple_sections(self) -> None:
        """Test adding multiple sections."""
        config = ReportConfig()
        report = Report(config=config)

        report.add_section("Section 1", "Content 1")
        report.add_section("Section 2", "Content 2")
        report.add_section("Section 3", "Content 3")

        assert len(report.sections) == 3

    def test_add_section_with_kwargs(self) -> None:
        """Test adding section with additional kwargs."""
        config = ReportConfig()
        report = Report(config=config)

        section = report.add_section("Detailed Section", "Content", level=3, collapsible=True)
        assert section.level == 3
        assert section.collapsible is True

    def test_add_table_simple(self) -> None:
        """Test adding simple table."""
        config = ReportConfig()
        report = Report(config=config)

        data = [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
        headers = ["Column 1", "Column 2"]

        table = report.add_table(data, headers, "Test Table")

        assert len(report.tables) == 1
        assert table["type"] == "table"
        assert table["headers"] == headers
        assert table["caption"] == "Test Table"

    def test_add_table_numpy_array(self) -> None:
        """Test adding table from numpy array."""
        config = ReportConfig()
        report = Report(config=config)

        data = np.array([[1, 2, 3], [4, 5, 6]])
        table = report.add_table(data, ["A", "B", "C"])

        assert len(report.tables) == 1
        assert isinstance(table["data"], list)
        assert len(table["data"]) == 2
        assert len(table["data"][0]) == 3

    def test_add_multiple_tables(self) -> None:
        """Test adding multiple tables."""
        config = ReportConfig()
        report = Report(config=config)

        report.add_table([[1, 2]], ["A", "B"])
        report.add_table([[3, 4]], ["C", "D"])

        assert len(report.tables) == 2
        assert report.tables[0]["id"] == 0
        assert report.tables[1]["id"] == 1

    def test_add_figure(self) -> None:
        """Test adding figure to report."""
        config = ReportConfig()
        report = Report(config=config)

        # Mock figure (could be matplotlib figure or path)
        mock_fig = "path/to/figure.png"
        figure = report.add_figure(mock_fig, "Test Figure", "80%")

        assert len(report.figures) == 1
        assert figure["type"] == "figure"
        assert figure["caption"] == "Test Figure"
        assert figure["width"] == "80%"

    def test_add_multiple_figures(self) -> None:
        """Test adding multiple figures."""
        config = ReportConfig()
        report = Report(config=config)

        report.add_figure("fig1.png", "Figure 1")
        report.add_figure("fig2.png", "Figure 2")

        assert len(report.figures) == 2
        assert report.figures[0]["id"] == 0
        assert report.figures[1]["id"] == 1

    def test_generate_executive_summary_all_pass(self) -> None:
        """Test generating executive summary when all tests pass."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"pass_count": 10, "total_count": 10}
        summary = report.generate_executive_summary(results)

        assert "All 10 tests passed" in summary

    def test_generate_executive_summary_with_failures(self) -> None:
        """Test generating executive summary with failures."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"pass_count": 7, "total_count": 10}
        summary = report.generate_executive_summary(results)

        assert "3 of 10 tests failed" in summary
        assert "30%" in summary

    def test_generate_executive_summary_with_key_findings(self) -> None:
        """Test executive summary with key findings."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"pass_count": 10, "total_count": 10}
        findings = ["Finding 1", "Finding 2", "Finding 3"]
        summary = report.generate_executive_summary(results, findings)

        assert "Key Findings:" in summary
        assert "Finding 1" in summary
        assert "Finding 2" in summary

    def test_generate_executive_summary_limits_findings(self) -> None:
        """Test that executive summary limits findings to 5."""
        config = ReportConfig()
        report = Report(config=config)

        findings = [f"Finding {i}" for i in range(10)]
        summary = report.generate_executive_summary({}, findings)

        # Should only show first 5
        assert "Finding 0" in summary
        assert "Finding 4" in summary
        assert "Finding 9" not in summary

    def test_generate_executive_summary_low_margin_warning(self) -> None:
        """Test executive summary with low margin warning."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"min_margin": 8.5}
        summary = report.generate_executive_summary(results)

        assert "Warning" in summary
        assert "8.5%" in summary

    def test_generate_executive_summary_medium_margin_note(self) -> None:
        """Test executive summary with medium margin note."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"min_margin": 15.0}
        summary = report.generate_executive_summary(results)

        assert "Note" in summary
        assert "15.0%" in summary

    def test_generate_executive_summary_good_margin(self) -> None:
        """Test executive summary with good margin (no warning)."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"min_margin": 25.0}
        summary = report.generate_executive_summary(results)

        # Should not mention margin
        assert "margin" not in summary.lower()

    def test_to_markdown_simple(self) -> None:
        """Test converting report to Markdown."""
        config = ReportConfig(title="Test Report", author="John Doe")
        report = Report(config=config)
        report.add_section("Section 1", "Content 1")

        markdown = report.to_markdown()

        assert "# Test Report" in markdown
        assert "**Author:** John Doe" in markdown
        assert "## Section 1" in markdown
        assert "Content 1" in markdown

    def test_to_markdown_with_table(self) -> None:
        """Test Markdown output with table."""
        config = ReportConfig()
        report = Report(config=config)

        data = [["A", "B"], ["C", "D"]]
        headers = ["Col1", "Col2"]
        table = report.add_table(data, headers)

        section = report.add_section("Data", [table])
        markdown = report.to_markdown()

        assert "| Col1 | Col2 |" in markdown
        assert "| --- | --- |" in markdown
        assert "| A | B |" in markdown

    def test_to_markdown_with_subsections(self) -> None:
        """Test Markdown output with subsections."""
        config = ReportConfig()
        report = Report(config=config)

        subsection = Section(title="Subsection", content="Sub content", level=3)
        section = Section(title="Main", content="Main content", subsections=[subsection])
        report.sections.append(section)

        markdown = report.to_markdown()

        assert "## Main" in markdown
        assert "Main content" in markdown
        assert "### Subsection" in markdown
        assert "Sub content" in markdown

    def test_to_markdown_skips_invisible_sections(self) -> None:
        """Test that invisible sections are skipped in Markdown."""
        config = ReportConfig()
        report = Report(config=config)

        report.add_section("Visible", "You can see this")
        report.add_section("Hidden", "You cannot see this", visible=False)

        markdown = report.to_markdown()

        assert "Visible" in markdown
        assert "Hidden" not in markdown

    def test_to_html_simple(self) -> None:
        """Test converting report to HTML."""
        config = ReportConfig(title="HTML Report", author="Jane Doe")
        report = Report(config=config)
        report.add_section("Introduction", "Welcome")

        html = report.to_html()

        assert "<!DOCTYPE html>" in html
        assert "<title>HTML Report</title>" in html
        assert "<strong>Author:</strong> Jane Doe" in html
        assert "<h3>Introduction</h3>" in html  # level 2 becomes h3
        assert "<p>Welcome</p>" in html

    def test_to_html_with_table(self) -> None:
        """Test HTML output with table."""
        config = ReportConfig()
        report = Report(config=config)

        data = [["A", "B"], ["C", "D"]]
        headers = ["Col1", "Col2"]
        table = report.add_table(data, headers, "Data Table")

        section = report.add_section("Data", [table])
        html = report.to_html()

        assert "<table>" in html
        assert "<th>Col1</th>" in html
        assert "<td>A</td>" in html
        assert "<em>Data Table</em>" in html

    def test_to_html_includes_styles(self) -> None:
        """Test that HTML includes CSS styles."""
        config = ReportConfig()
        report = Report(config=config)

        html = report.to_html()

        assert "<style>" in html
        assert "font-family" in html
        assert ".pass" in html
        assert ".fail" in html

    def test_to_html_skips_invisible_sections(self) -> None:
        """Test that invisible sections are skipped in HTML."""
        config = ReportConfig()
        report = Report(config=config)

        report.add_section("Visible", "You can see this")
        report.add_section("Hidden", "You cannot see this", visible=False)

        html = report.to_html()

        assert "Visible" in html
        assert "Hidden" not in html

    def test_save_markdown(self, tmp_path: Path) -> None:
        """Test saving report as Markdown."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("Test", "Content")

        output_path = tmp_path / "report.md"
        report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Oscura Analysis Report" in content

    def test_save_html(self, tmp_path: Path) -> None:
        """Test saving report as HTML."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("Test", "Content")

        output_path = tmp_path / "report.html"
        report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_other_format_falls_back_to_markdown(self, tmp_path: Path) -> None:
        """Test saving unsupported format falls back to Markdown."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("Test", "Content")

        output_path = tmp_path / "report.pdf"
        report.save(output_path)

        # Should create .md file instead
        md_path = tmp_path / "report.md"
        assert md_path.exists()


class TestGenerateReport:
    """Test generate_report function."""

    def test_generate_basic_report(self) -> None:
        """Test generating basic report."""
        results = {"pass_count": 10, "total_count": 10}
        report = generate_report(results)

        assert report.config.title == "Oscura Analysis Report"
        assert len(report.sections) > 0

    def test_generate_with_custom_title(self) -> None:
        """Test generating report with custom title."""
        results = {}
        report = generate_report(results, title="Custom Title")

        assert report.config.title == "Custom Title"

    def test_generate_with_verbosity_executive(self) -> None:
        """Test generating report with executive verbosity."""
        results = {"pass_count": 10, "total_count": 10}
        report = generate_report(results, verbosity="executive")

        assert report.config.verbosity == "executive"
        # Should have executive summary
        assert any("Executive Summary" in s.title for s in report.sections)

    def test_generate_with_verbosity_summary(self) -> None:
        """Test generating report with summary verbosity."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "measurements": {"test1": {"value": 100, "specification": 90, "passed": True}},
        }
        report = generate_report(results, verbosity="summary")

        # Should have executive summary and results
        titles = [s.title for s in report.sections]
        assert "Executive Summary" in titles
        assert "Test Results" in titles

    def test_generate_with_verbosity_standard(self) -> None:
        """Test generating report with standard verbosity."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "sample_rate": 1e9,
            "measurements": {},
        }
        report = generate_report(results, verbosity="standard")

        # Should have executive summary, results, and methodology
        titles = [s.title for s in report.sections]
        assert "Executive Summary" in titles
        assert "Methodology" in titles

    def test_generate_with_verbosity_detailed(self) -> None:
        """Test generating report with detailed verbosity."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "sample_rate": 1e9,
            "measurements": {},
        }
        report = generate_report(results, verbosity="detailed")

        # Should have all sections including raw data
        titles = [s.title for s in report.sections]
        assert "Executive Summary" in titles
        assert "Test Results" in titles
        assert "Methodology" in titles
        assert "Raw Data" in titles

    def test_generate_with_measurements(self) -> None:
        """Test generating report with measurements."""
        results = {
            "measurements": {
                "Voltage": {"value": 3.3, "specification": 3.0, "passed": True},
                "Current": {"value": 150, "specification": 200, "passed": True},
            }
        }
        report = generate_report(results, verbosity="summary")

        # Should have created table with measurements
        assert len(report.tables) > 0

    def test_generate_with_methodology_data(self) -> None:
        """Test generating report with methodology data."""
        results = {
            "sample_rate": 1e9,
            "num_samples": 100000,
            "analysis_time": 0.523,
        }
        report = generate_report(results, verbosity="standard")

        # Find methodology section
        methodology = next(s for s in report.sections if "Methodology" in s.title)
        content = str(methodology.content)

        assert "1000000000.0 Hz" in content
        assert "100000" in content
        assert "0.523" in content

    def test_generate_and_save(self, tmp_path: Path) -> None:
        """Test generating and saving report."""
        results = {"pass_count": 10, "total_count": 10}
        output_path = tmp_path / "report.md"

        report = generate_report(results, output_path)

        assert output_path.exists()
        assert len(report.sections) > 0

    def test_generate_multiple_formats(self, tmp_path: Path) -> None:
        """Test generating report in multiple formats."""
        results = {"pass_count": 10, "total_count": 10}
        output_path = tmp_path / "report.pdf"

        report = generate_report(results, output_path, formats=["md", "html"])

        md_path = tmp_path / "report.md"
        html_path = tmp_path / "report.html"

        assert md_path.exists()
        assert html_path.exists()

    def test_generate_with_template(self) -> None:
        """Test generating report with specific template."""
        results = {}
        report = generate_report(results, template="detailed")

        assert report.config.template == "detailed"

    def test_generate_with_extra_kwargs(self) -> None:
        """Test generating report with extra kwargs."""
        results = {}
        report = generate_report(results, author="John Doe", page_size="A4", show_toc=False)

        assert report.config.author == "John Doe"
        assert report.config.page_size == "A4"
        assert report.config.show_toc is False

    def test_generate_filters_invalid_kwargs(self) -> None:
        """Test that invalid kwargs are filtered out."""
        results = {}
        # Should not raise error even with invalid kwargs
        report = generate_report(results, invalid_param="value")

        assert report is not None


class TestReportSections:
    """Test report section generation functions."""

    def test_add_results_section_with_measurements(self) -> None:
        """Test adding results section with measurements."""
        config = ReportConfig()
        report = Report(config=config)

        results = {
            "measurements": {
                "Voltage": {"value": 3.3, "specification": 3.0, "passed": True},
                "Current": {"value": 250, "specification": 200, "passed": False},
            }
        }

        from oscura.reporting.core import _add_results_section

        _add_results_section(report, results, "standard")

        # Should have created a table
        assert len(report.tables) == 1
        table = report.tables[0]
        assert table["headers"] == ["Parameter", "Value", "Specification", "Status"]

    def test_add_methodology_section_complete(self) -> None:
        """Test adding complete methodology section."""
        config = ReportConfig()
        report = Report(config=config)

        results = {
            "sample_rate": 1e9,
            "num_samples": 100000,
            "analysis_time": 1.234,
        }

        from oscura.reporting.core import _add_methodology_section

        _add_methodology_section(report, results)

        # Find methodology section
        methodology = next(s for s in report.sections if "Methodology" in s.title)
        content = str(methodology.content)

        assert "1000000000.0 Hz" in content
        assert "100000" in content
        assert "1.234" in content

    def test_add_methodology_section_empty(self) -> None:
        """Test adding methodology section with no data."""
        config = ReportConfig()
        report = Report(config=config)

        from oscura.reporting.core import _add_methodology_section

        _add_methodology_section(report, {})

        methodology = next(s for s in report.sections if "Methodology" in s.title)
        assert "Standard analysis methodology" in str(methodology.content)

    def test_add_raw_data_section_with_data(self) -> None:
        """Test adding raw data section with data."""
        config = ReportConfig()
        report = Report(config=config)

        results = {"count": 100, "average": 42.5, "status": "complete"}

        from oscura.reporting.core import _add_raw_data_section

        _add_raw_data_section(report, results)

        raw_data = next(s for s in report.sections if "Raw Data" in s.title)
        content = str(raw_data.content)

        assert "count: 100" in content
        assert "average: 42.5" in content
        assert "status: complete" in content

    def test_add_raw_data_section_empty(self) -> None:
        """Test adding raw data section with no data."""
        config = ReportConfig()
        report = Report(config=config)

        from oscura.reporting.core import _add_raw_data_section

        _add_raw_data_section(report, {})

        raw_data = next(s for s in report.sections if "Raw Data" in s.title)
        assert "No raw data available" in str(raw_data.content)

    def test_add_raw_data_section_collapsible(self) -> None:
        """Test that raw data section is collapsible."""
        config = ReportConfig()
        report = Report(config=config)

        from oscura.reporting.core import _add_raw_data_section

        _add_raw_data_section(report, {"data": "value"})

        raw_data = next(s for s in report.sections if "Raw Data" in s.title)
        assert raw_data.collapsible is True


class TestTableFormatting:
    """Test table formatting in different outputs."""

    def test_table_to_markdown_with_caption(self) -> None:
        """Test Markdown table formatting with caption."""
        config = ReportConfig()
        report = Report(config=config)

        table = {
            "type": "table",
            "headers": ["A", "B"],
            "data": [[1, 2], [3, 4]],
            "caption": "Test Table",
        }

        lines = report._table_to_markdown(table)
        output = "\n".join(lines)

        assert "| A | B |" in output
        assert "| 1 | 2 |" in output
        assert "*Test Table*" in output

    def test_table_to_markdown_without_headers(self) -> None:
        """Test Markdown table without headers."""
        config = ReportConfig()
        report = Report(config=config)

        table = {"type": "table", "headers": None, "data": [[1, 2], [3, 4]]}

        lines = report._table_to_markdown(table)
        output = "\n".join(lines)

        # Should still have data rows
        assert "| 1 | 2 |" in output

    def test_table_to_html_with_caption(self) -> None:
        """Test HTML table formatting with caption."""
        config = ReportConfig()
        report = Report(config=config)

        table = {
            "type": "table",
            "headers": ["A", "B"],
            "data": [[1, 2], [3, 4]],
            "caption": "Test Table",
        }

        lines = report._table_to_html(table)
        output = "\n".join(lines)

        assert "<table>" in output
        assert "<th>A</th>" in output
        assert "<td>1</td>" in output
        assert "<em>Test Table</em>" in output

    def test_table_to_html_without_headers(self) -> None:
        """Test HTML table without headers."""
        config = ReportConfig()
        report = Report(config=config)

        table = {"type": "table", "headers": None, "data": [[1, 2], [3, 4]]}

        lines = report._table_to_html(table)
        output = "\n".join(lines)

        # Should not have thead
        assert "<thead>" not in output
        assert "<td>1</td>" in output
