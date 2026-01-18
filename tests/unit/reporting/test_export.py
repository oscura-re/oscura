"""Comprehensive unit tests for export module.

Tests for:

Coverage targets:
- export_report: Multiple format export functionality
- _export_pdf: PDF export
- _export_html: HTML export
- _export_markdown: Markdown export
- _export_docx: DOCX export with tables
- export_multiple_reports: Batch export to directory
- batch_export_formats: Single report to multiple formats
- create_archive: Archive creation (zip, tar, tar.gz)
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oscura.reporting.core import Report, ReportConfig, Section
from oscura.reporting.export import (
    _add_table_to_docx,
    _export_docx,
    _export_html,
    _export_markdown,
    _export_pdf,
    batch_export_formats,
    create_archive,
    export_multiple_reports,
    export_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_report():
    """Create a sample report for testing."""
    config = ReportConfig(title="Test Report", author="Test Author")
    report = Report(config=config)
    report.add_section("Introduction", "This is a test report.", level=1)
    report.add_section("Results", "Test results section.", level=1)
    return report


@pytest.fixture
def report_with_tables():
    """Create a report with tables for testing."""
    config = ReportConfig(title="Table Report", author="Test Author")
    report = Report(config=config)

    # Add section with table content
    table_dict = {
        "type": "table",
        "headers": ["Parameter", "Value", "Status"],
        "data": [
            ["Voltage", "5.0V", "PASS"],
            ["Current", "1.2A", "PASS"],
        ],
        "caption": "Test Measurements",
    }

    section = Section(
        title="Test Results",
        content=[table_dict],
        level=1,
    )
    report.sections.append(section)

    return report


@pytest.fixture
def report_with_subsections():
    """Create a report with subsections."""
    config = ReportConfig(title="Complex Report")
    report = Report(config=config)

    section = Section(title="Main Section", content="Main content", level=1)
    subsection = Section(title="Subsection", content="Sub content", level=2)
    section.subsections.append(subsection)

    report.sections.append(section)
    return report


@pytest.mark.unit
class TestExportReport:
    """Test main export_report function."""

    def test_export_default_formats(self, sample_report, tmp_path):
        """Test export with default formats (PDF and HTML)."""
        output_path = tmp_path / "report"

        with (
            patch("oscura.reporting.export._export_pdf") as mock_pdf,
            patch("oscura.reporting.export._export_html") as mock_html,
        ):
            mock_pdf.return_value = Path(str(output_path) + ".pdf")
            mock_html.return_value = Path(str(output_path) + ".html")

            result = export_report(sample_report, output_path)

            assert "pdf" in result
            assert "html" in result
            mock_pdf.assert_called_once()
            mock_html.assert_called_once()

    def test_export_single_format(self, sample_report, tmp_path):
        """Test export with single format."""
        output_path = tmp_path / "report"

        with patch("oscura.reporting.export._export_markdown") as mock_md:
            mock_md.return_value = Path(str(output_path) + ".md")

            result = export_report(sample_report, output_path, formats=["markdown"])

            assert "markdown" in result
            assert len(result) == 1
            mock_md.assert_called_once()

    def test_export_multiple_formats(self, sample_report, tmp_path):
        """Test export with multiple formats."""
        output_path = tmp_path / "report"

        with (
            patch("oscura.reporting.export._export_pdf") as mock_pdf,
            patch("oscura.reporting.export._export_html") as mock_html,
            patch("oscura.reporting.export._export_markdown") as mock_md,
            patch("oscura.reporting.export._export_docx") as mock_docx,
        ):
            mock_pdf.return_value = Path(str(output_path) + ".pdf")
            mock_html.return_value = Path(str(output_path) + ".html")
            mock_md.return_value = Path(str(output_path) + ".md")
            mock_docx.return_value = Path(str(output_path) + ".docx")

            result = export_report(
                sample_report,
                output_path,
                formats=["pdf", "html", "markdown", "docx"],
            )

            assert len(result) == 4
            assert "pdf" in result
            assert "html" in result
            assert "markdown" in result
            assert "docx" in result

    def test_export_with_format_options(self, sample_report, tmp_path):
        """Test export with format-specific options."""
        output_path = tmp_path / "report"

        format_options = {
            "pdf": {"dpi": 300, "pdfa_compliance": True},
            "html": {"interactive": True, "dark_mode": True},
        }

        with (
            patch("oscura.reporting.export._export_pdf") as mock_pdf,
            patch("oscura.reporting.export._export_html") as mock_html,
        ):
            mock_pdf.return_value = Path(str(output_path) + ".pdf")
            mock_html.return_value = Path(str(output_path) + ".html")

            export_report(
                sample_report,
                output_path,
                formats=["pdf", "html"],
                format_options=format_options,
            )

            # Check that format options were passed
            mock_pdf.assert_called_once()
            call_kwargs = mock_pdf.call_args[1]
            assert call_kwargs.get("dpi") == 300
            assert call_kwargs.get("pdfa_compliance") is True

            mock_html.assert_called_once()
            call_kwargs = mock_html.call_args[1]
            assert call_kwargs.get("interactive") is True
            assert call_kwargs.get("dark_mode") is True

    def test_export_unsupported_format(self, sample_report, tmp_path):
        """Test export with unsupported format raises error."""
        output_path = tmp_path / "report"

        with pytest.raises(ValueError, match="Unsupported format: invalid"):
            export_report(sample_report, output_path, formats=["invalid"])

    def test_export_with_string_path(self, sample_report, tmp_path):
        """Test export with string path instead of Path object."""
        output_path = str(tmp_path / "report")

        with patch("oscura.reporting.export._export_pdf") as mock_pdf:
            mock_pdf.return_value = Path(output_path + ".pdf")

            result = export_report(sample_report, output_path, formats=["pdf"])

            assert "pdf" in result
            mock_pdf.assert_called_once()


@pytest.mark.unit
class TestExportPDF:
    """Test PDF export functionality."""

    def test_export_pdf_basic(self, sample_report, tmp_path):
        """Test basic PDF export."""
        output_path = tmp_path / "report"

        with patch("oscura.reporting.pdf.save_pdf_report") as mock_save:
            result = _export_pdf(sample_report, output_path)

            assert result == output_path.with_suffix(".pdf")
            mock_save.assert_called_once_with(
                sample_report,
                output_path.with_suffix(".pdf"),
            )

    def test_export_pdf_with_options(self, sample_report, tmp_path):
        """Test PDF export with options."""
        output_path = tmp_path / "report"

        with patch("oscura.reporting.pdf.save_pdf_report") as mock_save:
            result = _export_pdf(
                sample_report,
                output_path,
                dpi=300,
                pdfa_compliance=True,
            )

            assert result == output_path.with_suffix(".pdf")
            mock_save.assert_called_once_with(
                sample_report,
                output_path.with_suffix(".pdf"),
                dpi=300,
                pdfa_compliance=True,
            )

    def test_export_pdf_preserves_extension(self, sample_report, tmp_path):
        """Test PDF export overwrites any existing extension."""
        output_path = tmp_path / "report.txt"

        with patch("oscura.reporting.pdf.save_pdf_report") as mock_save:
            result = _export_pdf(sample_report, output_path)

            assert result.suffix == ".pdf"


@pytest.mark.unit
class TestExportHTML:
    """Test HTML export functionality."""

    def test_export_html_basic(self, sample_report, tmp_path):
        """Test basic HTML export."""
        output_path = tmp_path / "report"

        with patch("oscura.reporting.html.save_html_report") as mock_save:
            result = _export_html(sample_report, output_path)

            assert result == output_path.with_suffix(".html")
            mock_save.assert_called_once_with(
                sample_report,
                output_path.with_suffix(".html"),
            )

    def test_export_html_with_options(self, sample_report, tmp_path):
        """Test HTML export with options."""
        output_path = tmp_path / "report"

        with patch("oscura.reporting.html.save_html_report") as mock_save:
            result = _export_html(
                sample_report,
                output_path,
                interactive=True,
                dark_mode=True,
            )

            assert result == output_path.with_suffix(".html")
            mock_save.assert_called_once_with(
                sample_report,
                output_path.with_suffix(".html"),
                interactive=True,
                dark_mode=True,
            )

    def test_export_html_preserves_extension(self, sample_report, tmp_path):
        """Test HTML export overwrites any existing extension."""
        output_path = tmp_path / "report.pdf"

        with patch("oscura.reporting.html.save_html_report") as mock_save:
            result = _export_html(sample_report, output_path)

            assert result.suffix == ".html"


@pytest.mark.unit
class TestExportMarkdown:
    """Test Markdown export functionality."""

    def test_export_markdown_basic(self, sample_report, tmp_path):
        """Test basic Markdown export."""
        output_path = tmp_path / "report"

        result = _export_markdown(sample_report, output_path)

        assert result == output_path.with_suffix(".md")
        assert result.exists()

        content = result.read_text(encoding="utf-8")
        assert "# Test Report" in content
        assert "Test Author" in content
        assert "Introduction" in content

    def test_export_markdown_with_sections(self, sample_report, tmp_path):
        """Test Markdown export includes all sections."""
        output_path = tmp_path / "report"

        result = _export_markdown(sample_report, output_path)

        content = result.read_text(encoding="utf-8")
        assert "Introduction" in content
        assert "Results" in content
        assert "This is a test report." in content

    def test_export_markdown_preserves_extension(self, sample_report, tmp_path):
        """Test Markdown export overwrites any existing extension."""
        output_path = tmp_path / "report.html"

        result = _export_markdown(sample_report, output_path)

        assert result.suffix == ".md"

    def test_export_markdown_with_options(self, sample_report, tmp_path):
        """Test Markdown export ignores extra options gracefully."""
        output_path = tmp_path / "report"

        # Should not raise error even with unknown options
        result = _export_markdown(sample_report, output_path, unknown_option=True)

        assert result.exists()


@pytest.mark.unit
class TestExportDOCX:
    """Test DOCX export functionality."""

    def test_export_docx_basic(self, sample_report, tmp_path):
        """Test basic DOCX export."""
        import sys

        output_path = tmp_path / "report"

        # Create mock modules
        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            result = _export_docx(sample_report, output_path)

            assert result == output_path.with_suffix(".docx")
            mock_doc.save.assert_called_once()

    def test_export_docx_missing_library(self, sample_report, tmp_path):
        """Test DOCX export raises error when python-docx not installed."""
        output_path = tmp_path / "report"

        # Mock the import to fail
        import sys

        original_modules = sys.modules.copy()
        try:
            # Remove docx from modules if it exists
            for key in list(sys.modules.keys()):
                if key.startswith("docx"):
                    del sys.modules[key]

            # Make import fail
            with patch.dict("sys.modules", {"docx": None}):
                with pytest.raises(ImportError, match="python-docx is required"):
                    _export_docx(sample_report, output_path)
        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def test_export_docx_with_sections(self, sample_report, tmp_path):
        """Test DOCX export includes all sections."""
        import sys

        output_path = tmp_path / "report"

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            _export_docx(sample_report, output_path)

            # Check that sections were added
            assert mock_doc.add_heading.call_count >= 2
            assert mock_doc.add_paragraph.call_count >= 2

    def test_export_docx_with_table(self, report_with_tables, tmp_path):
        """Test DOCX export with tables."""
        import sys

        output_path = tmp_path / "report"

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            _export_docx(report_with_tables, output_path)

            # Should call add_table
            mock_doc.add_table.assert_called()

    def test_export_docx_with_subsections(self, report_with_subsections, tmp_path):
        """Test DOCX export includes subsections."""
        import sys

        output_path = tmp_path / "report"

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            _export_docx(report_with_subsections, output_path)

            # Should add headings for main section and subsection
            assert mock_doc.add_heading.call_count >= 3  # Title + Main + Sub

    def test_export_docx_invisible_sections(self, sample_report, tmp_path):
        """Test DOCX export skips invisible sections."""
        import sys

        sample_report.add_section("Hidden Section", "Should not appear", visible=False)
        output_path = tmp_path / "report"

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            _export_docx(sample_report, output_path)

            # Check that "Hidden Section" was not added
            heading_calls = [str(call) for call in mock_doc.add_heading.call_args_list]
            assert not any("Hidden Section" in str(call) for call in heading_calls)

    def test_export_docx_with_figure_placeholder(self, sample_report, tmp_path):
        """Test DOCX export creates placeholder for figures."""
        import sys

        # Add section with figure
        figure_item = {"type": "figure", "caption": "Test Figure"}
        section = Section(title="Figures", content=[figure_item], level=1)
        sample_report.sections.append(section)

        output_path = tmp_path / "report"

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            _export_docx(sample_report, output_path)

            # Should add paragraph placeholder for figure
            paragraph_calls = [str(call) for call in mock_doc.add_paragraph.call_args_list]
            assert any("[Figure: Test Figure]" in str(call) for call in paragraph_calls)

    def test_export_docx_list_content(self, tmp_path):
        """Test DOCX export with list content (non-dict items)."""
        import sys

        config = ReportConfig(title="List Report")
        report = Report(config=config)
        section = Section(title="Items", content=["Item 1", "Item 2", "Item 3"], level=1)
        report.sections.append(section)

        output_path = tmp_path / "report"

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc

        with patch.dict(
            sys.modules,
            {
                "docx": mock_docx,
                "docx.enum": MagicMock(),
                "docx.enum.text": MagicMock(),
                "docx.shared": MagicMock(),
            },
        ):
            _export_docx(report, output_path)

            # Should add paragraphs for list items
            assert mock_doc.add_paragraph.call_count >= 3


@pytest.mark.unit
class TestAddTableToDOCX:
    """Test DOCX table helper function."""

    def test_add_table_with_headers_and_data(self):
        """Test adding table with headers and data."""
        mock_doc = MagicMock()
        mock_table = MagicMock()
        mock_doc.add_table.return_value = mock_table

        # Setup mock table structure
        header_row = MagicMock()
        header_cells = [MagicMock() for _ in range(3)]
        for cell in header_cells:
            cell.paragraphs = [MagicMock()]
            cell.paragraphs[0].runs = [MagicMock()]
        header_row.cells = header_cells

        data_row = MagicMock()
        data_cells = [MagicMock() for _ in range(3)]
        data_row.cells = data_cells

        mock_table.rows = [header_row, data_row]

        table_dict = {
            "headers": ["Col1", "Col2", "Col3"],
            "data": [["A", "B", "C"]],
            "caption": "Test Table",
        }

        _add_table_to_docx(mock_doc, table_dict)

        mock_doc.add_table.assert_called_once_with(rows=2, cols=3)
        mock_doc.add_paragraph.assert_called_once_with("Test Table", style="Caption")

    def test_add_table_without_headers(self):
        """Test adding table without headers."""
        mock_doc = MagicMock()
        mock_table = MagicMock()
        mock_doc.add_table.return_value = mock_table

        data_row = MagicMock()
        data_cells = [MagicMock() for _ in range(2)]
        data_row.cells = data_cells
        mock_table.rows = [data_row]

        table_dict = {
            "headers": [],
            "data": [["A", "B"]],
        }

        _add_table_to_docx(mock_doc, table_dict)

        # Should create table with 1 row (no header row)
        mock_doc.add_table.assert_called_once_with(rows=1, cols=2)

    def test_add_empty_table(self):
        """Test adding empty table does nothing."""
        mock_doc = MagicMock()

        table_dict = {
            "headers": [],
            "data": [],
        }

        _add_table_to_docx(mock_doc, table_dict)

        mock_doc.add_table.assert_not_called()

    def test_add_table_without_caption(self):
        """Test adding table without caption."""
        mock_doc = MagicMock()
        mock_table = MagicMock()
        mock_doc.add_table.return_value = mock_table

        header_row = MagicMock()
        header_cells = [MagicMock() for _ in range(2)]
        for cell in header_cells:
            cell.paragraphs = [MagicMock()]
            cell.paragraphs[0].runs = [MagicMock()]
        header_row.cells = header_cells
        mock_table.rows = [header_row]

        table_dict = {
            "headers": ["Col1", "Col2"],
            "data": [],
        }

        _add_table_to_docx(mock_doc, table_dict)

        # Should not add caption paragraph
        mock_doc.add_paragraph.assert_not_called()

    def test_add_table_with_multiple_rows(self):
        """Test adding table with multiple data rows."""
        mock_doc = MagicMock()
        mock_table = MagicMock()
        mock_doc.add_table.return_value = mock_table

        # Create mock rows
        rows = []
        for _ in range(4):  # 1 header + 3 data rows
            row = MagicMock()
            cells = [MagicMock() for _ in range(2)]
            if _ == 0:  # header row
                for cell in cells:
                    cell.paragraphs = [MagicMock()]
                    cell.paragraphs[0].runs = [MagicMock()]
            row.cells = cells
            rows.append(row)
        mock_table.rows = rows

        table_dict = {
            "headers": ["Col1", "Col2"],
            "data": [["A", "B"], ["C", "D"], ["E", "F"]],
        }

        _add_table_to_docx(mock_doc, table_dict)

        mock_doc.add_table.assert_called_once_with(rows=4, cols=2)


@pytest.mark.unit
class TestExportMultipleReports:
    """Test batch export of multiple reports."""

    def test_export_multiple_reports_pdf(self, sample_report, tmp_path):
        """Test exporting multiple reports to PDF."""
        report2 = Report(config=ReportConfig(title="Report 2"))
        reports = {"report1": sample_report, "report2": report2}

        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.side_effect = [
                {"pdf": tmp_path / "report1.pdf"},
                {"pdf": tmp_path / "report2.pdf"},
            ]

            result = export_multiple_reports(reports, tmp_path, format="pdf")

            assert len(result) == 2
            assert "report1" in result
            assert "report2" in result
            assert mock_export.call_count == 2

    def test_export_multiple_reports_creates_directory(self, sample_report, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dir" / "reports"
        reports = {"report1": sample_report}

        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {"pdf": output_dir / "report1.pdf"}

            export_multiple_reports(reports, output_dir, format="pdf")

            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_export_multiple_reports_html(self, sample_report, tmp_path):
        """Test exporting multiple reports to HTML."""
        reports = {"report1": sample_report}

        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {"html": tmp_path / "report1.html"}

            result = export_multiple_reports(reports, tmp_path, format="html")

            assert "report1" in result
            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args[1]
            assert call_kwargs["formats"] == ["html"]

    def test_export_multiple_reports_with_options(self, sample_report, tmp_path):
        """Test exporting multiple reports with format options."""
        reports = {"report1": sample_report}

        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {"pdf": tmp_path / "report1.pdf"}

            export_multiple_reports(
                reports,
                tmp_path,
                format="pdf",
                dpi=300,
                pdfa_compliance=True,
            )

            call_kwargs = mock_export.call_args[1]
            assert call_kwargs["format_options"] == {"pdf": {"dpi": 300, "pdfa_compliance": True}}


@pytest.mark.unit
class TestBatchExportFormats:
    """Test batch export of single report to multiple formats."""

    def test_batch_export_all_formats(self, sample_report, tmp_path):
        """Test exporting to all supported formats."""
        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {
                "pdf": tmp_path / "test_report.pdf",
                "html": tmp_path / "test_report.html",
                "docx": tmp_path / "test_report.docx",
                "markdown": tmp_path / "test_report.md",
            }

            result = batch_export_formats(sample_report, tmp_path)

            assert len(result) == 4
            assert "pdf" in result
            assert "html" in result
            assert "docx" in result
            assert "markdown" in result

    def test_batch_export_creates_directory(self, sample_report, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "output"

        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {"pdf": output_dir / "test_report.pdf"}

            batch_export_formats(sample_report, output_dir, formats=["pdf"])

            assert output_dir.exists()

    def test_batch_export_custom_formats(self, sample_report, tmp_path):
        """Test exporting to custom format list."""
        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {
                "pdf": tmp_path / "test_report.pdf",
                "html": tmp_path / "test_report.html",
            }

            result = batch_export_formats(
                sample_report,
                tmp_path,
                formats=["pdf", "html"],
            )

            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args[1]
            assert call_kwargs["formats"] == ["pdf", "html"]

    def test_batch_export_filename_from_title(self, sample_report, tmp_path):
        """Test that filename is derived from report title."""
        sample_report.config.title = "My Test Report"

        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {}

            batch_export_formats(sample_report, tmp_path, formats=["pdf"])

            call_args = mock_export.call_args[0]
            output_path = call_args[1]
            assert "my_test_report" in str(output_path).lower()

    def test_batch_export_with_options(self, sample_report, tmp_path):
        """Test batch export with format options."""
        with patch("oscura.reporting.export.export_report") as mock_export:
            mock_export.return_value = {"pdf": tmp_path / "test_report.pdf"}

            batch_export_formats(
                sample_report,
                tmp_path,
                formats=["pdf"],
                dpi=300,
                pdfa_compliance=True,
            )

            call_kwargs = mock_export.call_args[1]
            assert "dpi" in call_kwargs["format_options"]
            assert "pdfa_compliance" in call_kwargs["format_options"]


@pytest.mark.unit
class TestCreateArchive:
    """Test archive creation functionality."""

    def test_create_zip_archive(self, tmp_path):
        """Test creating ZIP archive."""
        # Create test files
        file1 = tmp_path / "report1.pdf"
        file2 = tmp_path / "report2.html"
        file1.write_text("PDF content")
        file2.write_text("HTML content")

        files = {"pdf": file1, "html": file2}
        archive_path = tmp_path / "archive"

        result = create_archive(files, archive_path, format="zip")

        assert result.suffix == ".zip"
        assert result.exists()

        # Verify archive contents
        with zipfile.ZipFile(result, "r") as zipf:
            names = zipf.namelist()
            assert "report1.pdf" in names
            assert "report2.html" in names

    def test_create_tar_archive(self, tmp_path):
        """Test creating TAR archive."""
        import tarfile

        # Create test files
        file1 = tmp_path / "report.pdf"
        file1.write_text("Content")

        files = {"pdf": file1}
        archive_path = tmp_path / "archive"

        result = create_archive(files, archive_path, format="tar")

        assert result.suffix == ".tar"
        assert result.exists()

        # Verify archive contents
        with tarfile.open(result, "r") as tar:
            names = tar.getnames()
            assert "report.pdf" in names

    def test_create_tar_gz_archive(self, tmp_path):
        """Test creating TAR.GZ archive."""
        import tarfile

        # Create test files
        file1 = tmp_path / "report.pdf"
        file1.write_text("Content")

        files = {"pdf": file1}
        archive_path = tmp_path / "archive"

        result = create_archive(files, archive_path, format="tar.gz")

        assert result.suffix == ".gz"
        assert str(result).endswith(".tar.gz")
        assert result.exists()

        # Verify archive contents
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "report.pdf" in names

    def test_create_archive_unsupported_format(self, tmp_path):
        """Test unsupported archive format raises error."""
        files = {"pdf": tmp_path / "report.pdf"}
        archive_path = tmp_path / "archive"

        with pytest.raises(ValueError, match="Unsupported archive format: invalid"):
            create_archive(files, archive_path, format="invalid")

    def test_create_archive_with_string_path(self, tmp_path):
        """Test archive creation with string path."""
        file1 = tmp_path / "report.pdf"
        file1.write_text("Content")

        files = {"pdf": file1}
        archive_path = str(tmp_path / "archive")

        result = create_archive(files, archive_path, format="zip")

        assert result.exists()
        assert result.suffix == ".zip"

    def test_create_archive_multiple_files(self, tmp_path):
        """Test archive with multiple files."""
        # Create multiple test files
        files = {}
        for i in range(5):
            file = tmp_path / f"report{i}.txt"
            file.write_text(f"Content {i}")
            files[f"file{i}"] = file

        archive_path = tmp_path / "archive"

        result = create_archive(files, archive_path, format="zip")

        with zipfile.ZipFile(result, "r") as zipf:
            names = zipf.namelist()
            assert len(names) == 5
            for i in range(5):
                assert f"report{i}.txt" in names

    def test_create_archive_preserves_basename(self, tmp_path):
        """Test that archive only includes file basename, not full path."""
        # Create file in subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file1 = subdir / "report.pdf"
        file1.write_text("Content")

        files = {"pdf": file1}
        archive_path = tmp_path / "archive"

        result = create_archive(files, archive_path, format="zip")

        with zipfile.ZipFile(result, "r") as zipf:
            names = zipf.namelist()
            # Should only have basename, not full path
            assert "report.pdf" in names
            assert "subdir/report.pdf" not in names
