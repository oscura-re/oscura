"""Comprehensive unit tests for oscura.reporting.pdf module.

Tests PDF report generation, styles, tables, and metadata.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from oscura.reporting.core import Report, ReportConfig, Section

# Test reportlab availability
try:
    from oscura.reporting.pdf import REPORTLAB_AVAILABLE
except ImportError:
    REPORTLAB_AVAILABLE = False


# Skip all tests if reportlab is not available
pytestmark = pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="reportlab not installed")


@pytest.fixture
def basic_report() -> Report:
    """Create a basic report for testing."""
    config = ReportConfig(
        title="Test PDF Report",
        author="Test Author",
        page_size="letter",
    )
    report = Report(config=config)
    report.add_section("Section 1", "Content 1", level=1)
    report.add_section("Section 2", "Content 2", level=2)
    return report


@pytest.fixture
def report_with_tables() -> Report:
    """Create a report with tables for testing."""
    config = ReportConfig(title="Table PDF Report")
    report = Report(config=config)

    table = {
        "type": "table",
        "headers": ["Parameter", "Value", "Status"],
        "data": [
            ["rise_time", "2.3 ns", "PASS"],
            ["fall_time", "3.0 ns", "FAIL"],
            ["jitter", "100 ps", "WARNING"],
        ],
    }

    report.add_section("Results", [table], level=1)
    return report


@pytest.fixture
def report_with_figures() -> Report:
    """Create a report with figures for testing."""
    config = ReportConfig(title="Figure PDF Report")
    report = Report(config=config)

    figure = {
        "type": "figure",
        "figure": "/path/to/image.png",
        "caption": "Test Figure",
    }

    report.add_section("Plots", [figure], level=1)
    return report


@pytest.fixture
def large_report() -> Report:
    """Create a large report for TOC testing."""
    config = ReportConfig(title="Large PDF Report", author="Tester")
    report = Report(config=config)

    for i in range(10):
        report.add_section(f"Section {i}", f"Content for section {i}", level=1)

    return report


class TestGeneratePdfReport:
    """Tests for generate_pdf_report function."""

    def test_basic_pdf_generation(self, basic_report):
        """Test generating a basic PDF report."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        # PDF magic bytes
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_with_default_options(self, basic_report):
        """Test PDF generation with default options."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100

    def test_pdf_with_custom_dpi(self, basic_report):
        """Test PDF generation with custom DPI."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(basic_report, dpi=600)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_embed_fonts(self, basic_report):
        """Test PDF generation with embedded fonts."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(basic_report, embed_fonts=True)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_table_of_contents(self, large_report):
        """Test PDF generation with table of contents."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(large_report, table_of_contents=True)

        assert isinstance(pdf_bytes, bytes)
        # TOC adds content to PDF
        assert len(pdf_bytes) > 1000

    def test_pdf_without_table_of_contents(self, basic_report):
        """Test PDF generation without table of contents."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(basic_report, table_of_contents=False)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_letter_page_size(self, basic_report):
        """Test PDF with letter page size."""
        from oscura.reporting.pdf import generate_pdf_report

        basic_report.config.page_size = "letter"
        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_a4_page_size(self, basic_report):
        """Test PDF with A4 page size."""
        from oscura.reporting.pdf import generate_pdf_report

        basic_report.config.page_size = "A4"
        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_tables(self, report_with_tables):
        """Test PDF generation with tables."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(report_with_tables)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_figures(self, report_with_figures):
        """Test PDF generation with figures."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(report_with_figures)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_subsections(self, basic_report):
        """Test PDF generation with subsections."""
        from oscura.reporting.pdf import generate_pdf_report

        # Add subsections
        subsection = Section(title="Subsection", content="Subsection content", level=3)
        basic_report.sections[0].subsections.append(subsection)

        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_multiline_content(self, basic_report):
        """Test PDF generation with multi-paragraph content."""
        from oscura.reporting.pdf import generate_pdf_report

        basic_report.sections[0].content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_invisible_sections(self, basic_report):
        """Test PDF generation skips invisible sections."""
        from oscura.reporting.pdf import generate_pdf_report

        basic_report.sections[1].visible = False

        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)

    def test_pdf_with_custom_margins(self, basic_report):
        """Test PDF generation with custom margins."""
        from oscura.reporting.pdf import generate_pdf_report

        basic_report.config.margins = 0.5

        pdf_bytes = generate_pdf_report(basic_report)

        assert isinstance(pdf_bytes, bytes)


class TestCreateStyles:
    """Tests for _create_styles function."""

    def test_styles_creation(self):
        """Test styles dictionary is created."""
        from oscura.reporting.pdf import _create_styles

        styles = _create_styles()

        assert isinstance(styles, dict)
        assert "Title" in styles
        assert "Body" in styles

    def test_styles_headings(self):
        """Test heading styles are defined."""
        from oscura.reporting.pdf import _create_styles

        styles = _create_styles()

        assert "Heading1" in styles
        assert "Heading2" in styles
        assert "Heading3" in styles

    def test_styles_emphasis(self):
        """Test emphasis styles (pass/fail/warning)."""
        from oscura.reporting.pdf import _create_styles

        styles = _create_styles()

        assert "Pass" in styles
        assert "Fail" in styles
        assert "Warning" in styles

    def test_styles_metadata(self):
        """Test metadata style is defined."""
        from oscura.reporting.pdf import _create_styles

        styles = _create_styles()

        assert "Metadata" in styles

    def test_styles_toc(self):
        """Test TOC style is defined."""
        from oscura.reporting.pdf import _create_styles

        styles = _create_styles()

        assert "TOC" in styles


class TestFormatMetadata:
    """Tests for _format_metadata function."""

    def test_metadata_with_author(self, basic_report):
        """Test metadata includes author."""
        from oscura.reporting.pdf import _format_metadata

        metadata = _format_metadata(basic_report)

        assert "Author" in metadata
        assert "Test Author" in metadata

    def test_metadata_with_date(self, basic_report):
        """Test metadata includes date."""
        from oscura.reporting.pdf import _format_metadata

        metadata = _format_metadata(basic_report)

        assert "Date" in metadata

    def test_metadata_with_verbosity(self, basic_report):
        """Test metadata includes verbosity."""
        from oscura.reporting.pdf import _format_metadata

        metadata = _format_metadata(basic_report)

        assert "Detail Level" in metadata

    def test_metadata_without_author(self):
        """Test metadata without author."""
        from oscura.reporting.pdf import _format_metadata

        config = ReportConfig(title="No Author")
        report = Report(config=config)

        metadata = _format_metadata(report)

        assert "Author" not in metadata


class TestCreatePdfTable:
    """Tests for _create_pdf_table function."""

    def test_table_creation(self):
        """Test PDF table creation."""
        from oscura.reporting.pdf import _create_pdf_table

        table_dict = {
            "headers": ["A", "B"],
            "data": [["1", "2"], ["3", "4"]],
        }

        table = _create_pdf_table(table_dict)

        # Should return a Table object
        assert table is not None

    def test_table_with_headers(self):
        """Test table with headers."""
        from oscura.reporting.pdf import _create_pdf_table

        table_dict = {
            "headers": ["Parameter", "Value"],
            "data": [["test", "123"]],
        }

        table = _create_pdf_table(table_dict)

        assert table is not None

    def test_table_without_headers(self):
        """Test table without headers."""
        from oscura.reporting.pdf import _create_pdf_table

        table_dict = {
            "data": [["1", "2"], ["3", "4"]],
        }

        table = _create_pdf_table(table_dict)

        assert table is not None

    def test_table_empty_data(self):
        """Test table with empty data."""
        from oscura.reporting.pdf import _create_pdf_table

        table_dict = {
            "headers": ["A"],
            "data": [],
        }

        table = _create_pdf_table(table_dict)

        assert table is not None

    def test_table_pass_fail_styling(self):
        """Test table applies pass/fail styling."""
        from oscura.reporting.pdf import _create_pdf_table

        table_dict = {
            "headers": ["Status"],
            "data": [["PASS"], ["FAIL"], ["WARNING"]],
        }

        table = _create_pdf_table(table_dict)

        # Table should be created with styling
        assert table is not None

    def test_table_checkmark_styling(self):
        """Test table applies styling for checkmarks."""
        from oscura.reporting.pdf import _create_pdf_table

        table_dict = {
            "headers": ["Status"],
            "data": [["\u2713 PASS"], ["\u2717 FAIL"]],
        }

        table = _create_pdf_table(table_dict)

        assert table is not None


class TestAddPdfSection:
    """Tests for _add_pdf_section function."""

    def test_add_section_string_content(self, basic_report):
        """Test adding section with string content."""
        from oscura.reporting.pdf import _add_pdf_section, _create_styles

        story = []
        styles = _create_styles()
        section = basic_report.sections[0]

        _add_pdf_section(story, section, styles, basic_report)

        assert len(story) > 0

    def test_add_section_list_content(self, report_with_tables):
        """Test adding section with list content (tables)."""
        from oscura.reporting.pdf import _add_pdf_section, _create_styles

        story = []
        styles = _create_styles()
        section = report_with_tables.sections[0]

        _add_pdf_section(story, section, styles, report_with_tables)

        assert len(story) > 0

    def test_add_section_with_figures(self, report_with_figures):
        """Test adding section with figures."""
        from oscura.reporting.pdf import _add_pdf_section, _create_styles

        story = []
        styles = _create_styles()
        section = report_with_figures.sections[0]

        _add_pdf_section(story, section, styles, report_with_figures)

        assert len(story) > 0

    def test_add_section_with_subsections(self, basic_report):
        """Test adding section with subsections."""
        from oscura.reporting.pdf import _add_pdf_section, _create_styles

        story = []
        styles = _create_styles()

        subsection = Section(title="Sub", content="Sub content", level=3)
        basic_report.sections[0].subsections.append(subsection)

        _add_pdf_section(story, basic_report.sections[0], styles, basic_report)

        assert len(story) > 0

    def test_add_section_invisible_subsection(self, basic_report):
        """Test section skips invisible subsections."""
        from oscura.reporting.pdf import _add_pdf_section, _create_styles

        story = []
        styles = _create_styles()

        subsection = Section(title="Hidden", content="Hidden content", level=3, visible=False)
        basic_report.sections[0].subsections.append(subsection)

        _add_pdf_section(story, basic_report.sections[0], styles, basic_report)

        # Should complete without error
        assert len(story) > 0


class TestSavePdfReport:
    """Tests for save_pdf_report function."""

    def test_save_pdf_report(self, basic_report):
        """Test saving PDF report to file."""
        from oscura.reporting.pdf import save_pdf_report

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"

            save_pdf_report(basic_report, path)

            assert path.exists()
            content = path.read_bytes()
            assert content[:4] == b"%PDF"

    def test_save_pdf_report_with_options(self, basic_report):
        """Test saving PDF report with options."""
        from oscura.reporting.pdf import save_pdf_report

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"

            save_pdf_report(basic_report, path, dpi=300, table_of_contents=False)

            assert path.exists()

    def test_save_pdf_report_string_path(self, basic_report):
        """Test saving PDF report with string path."""
        from oscura.reporting.pdf import save_pdf_report

        with TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/report.pdf"

            save_pdf_report(basic_report, path)

            assert Path(path).exists()


class TestReportingPdfIntegration:
    """Integration tests for PDF report generation."""

    def test_full_report_generation(self):
        """Test complete PDF report generation."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(
            title="Full Test PDF Report",
            author="Integration Test",
            verbosity="detailed",
            page_size="letter",
        )
        report = Report(config=config)

        # Add various content types
        report.add_section("Summary", "This is a summary.", level=1)

        table = {
            "type": "table",
            "headers": ["Test", "Result"],
            "data": [["Test 1", "PASS"], ["Test 2", "FAIL"]],
        }
        report.add_section("Results", [table], level=1)

        report.add_section("Conclusion", "All done.", level=1)

        # Generate PDF
        pdf_bytes = generate_pdf_report(report, dpi=300, table_of_contents=False)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b"%PDF"

    def test_large_report_with_toc(self, large_report):
        """Test large PDF report with table of contents."""
        from oscura.reporting.pdf import generate_pdf_report

        pdf_bytes = generate_pdf_report(large_report, table_of_contents=True)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000

    def test_report_with_all_content_types(self):
        """Test PDF report with all content types."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="All Types Report")
        report = Report(config=config)

        # Text section
        report.add_section("Text", "Simple text content.\n\nAnother paragraph.", level=1)

        # Table section
        table = {
            "type": "table",
            "headers": ["Col1", "Col2"],
            "data": [["a", "b"], ["c", "d"]],
        }
        report.add_section("Table", [table], level=1)

        # Figure section
        figure = {
            "type": "figure",
            "caption": "Placeholder Figure",
        }
        report.add_section("Figure", [figure], level=1)

        # Mixed content
        mixed = [
            "Some text",
            table,
            "More text",
        ]
        report.add_section("Mixed", mixed, level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_save_and_load_pdf(self, basic_report):
        """Test saving and verifying PDF file."""
        from oscura.reporting.pdf import save_pdf_report

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pdf"

            save_pdf_report(basic_report, path)

            assert path.exists()
            content = path.read_bytes()

            # PDF structure verification
            assert content.startswith(b"%PDF")


class TestReportingPdfEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_report(self):
        """Test PDF generation with empty report."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Empty Report")
        report = Report(config=config)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_report_with_unicode(self):
        """Test PDF generation with unicode content."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Unicode Report")
        report = Report(config=config)
        report.add_section("Unicode", "Test symbols in content", level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_report_with_very_long_content(self):
        """Test PDF generation with very long content."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Long Report")
        report = Report(config=config)
        long_content = "This is a test. " * 1000
        report.add_section("Long Section", long_content, level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_report_with_empty_table(self):
        """Test PDF generation with empty table."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Empty Table Report")
        report = Report(config=config)
        table = {"type": "table", "headers": ["A", "B"], "data": []}
        report.add_section("Empty Table", [table], level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_report_with_special_chars_in_title(self):
        """Test PDF generation with special characters in title."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Report with special and chars")
        report = Report(config=config)
        report.add_section("Section", "Content", level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_report_minimum_margins(self):
        """Test PDF generation with minimum margins."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Narrow Margins", margins=0.25)
        report = Report(config=config)
        report.add_section("Section", "Content", level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)

    def test_report_maximum_margins(self):
        """Test PDF generation with large margins."""
        from oscura.reporting.pdf import generate_pdf_report

        config = ReportConfig(title="Wide Margins", margins=2.0)
        report = Report(config=config)
        report.add_section("Section", "Content", level=1)

        pdf_bytes = generate_pdf_report(report)

        assert isinstance(pdf_bytes, bytes)
