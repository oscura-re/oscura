"""Comprehensive tests for PDF renderer.


Test Coverage:
- PDFRenderer dataclass initialization and defaults
- render_to_pdf function with various options
- Output to file vs bytes
- Integration with underlying PDF generation
- Error handling for missing dependencies
- Custom rendering options
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.reporting.core import Report, ReportConfig
from oscura.reporting.renderers.pdf import PDFRenderer, render_to_pdf

pytestmark = pytest.mark.unit


class TestPDFRenderer:
    """Tests for PDFRenderer dataclass (REPORT-008)."""

    def test_default_initialization(self):
        """Test PDFRenderer initializes with expected defaults."""
        renderer = PDFRenderer()

        assert renderer.dpi == 300
        assert renderer.embed_fonts is True
        assert renderer.vector_graphics is True
        assert renderer.table_of_contents is True
        assert renderer.pdfa_compliance is False
        assert renderer.page_numbering is True

    def test_custom_initialization(self):
        """Test PDFRenderer accepts custom values."""
        renderer = PDFRenderer(
            dpi=600,
            embed_fonts=False,
            vector_graphics=False,
            table_of_contents=False,
            pdfa_compliance=True,
            page_numbering=False,
        )

        assert renderer.dpi == 600
        assert renderer.embed_fonts is False
        assert renderer.vector_graphics is False
        assert renderer.table_of_contents is False
        assert renderer.pdfa_compliance is True
        assert renderer.page_numbering is False

    def test_partial_custom_initialization(self):
        """Test PDFRenderer with some custom values keeps other defaults."""
        renderer = PDFRenderer(dpi=150, pdfa_compliance=True)

        assert renderer.dpi == 150
        assert renderer.pdfa_compliance is True
        assert renderer.embed_fonts is True  # Still default
        assert renderer.vector_graphics is True  # Still default

    def test_dataclass_equality(self):
        """Test PDFRenderer instances compare correctly."""
        renderer1 = PDFRenderer(dpi=300)
        renderer2 = PDFRenderer(dpi=300)
        renderer3 = PDFRenderer(dpi=600)

        assert renderer1 == renderer2
        assert renderer1 != renderer3

    def test_dataclass_repr(self):
        """Test PDFRenderer has meaningful repr."""
        renderer = PDFRenderer(dpi=150)
        repr_str = repr(renderer)

        assert "PDFRenderer" in repr_str
        assert "dpi=150" in repr_str


class TestRenderToPDFBasics:
    """Basic tests for render_to_pdf function (REPORT-008)."""

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_returns_bytes(self, mock_generate):
        """Test render_to_pdf returns bytes when no output path given."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        result = render_to_pdf(report)

        assert result == b"PDF content"
        assert isinstance(result, bytes)
        mock_generate.assert_called_once()

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_passes_report_to_generator(self, mock_generate):
        """Test render_to_pdf passes report to underlying generator."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        render_to_pdf(report)

        # First argument should be the report
        call_args = mock_generate.call_args
        assert call_args[0][0] == report

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_default_options(self, mock_generate):
        """Test render_to_pdf uses default options when none specified."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        render_to_pdf(report)

        call_args = mock_generate.call_args
        assert call_args[1]["dpi"] == 300
        assert call_args[1]["embed_fonts"] is True
        assert call_args[1]["vector_graphics"] is True
        assert call_args[1]["table_of_contents"] is True
        assert call_args[1]["pdfa_compliance"] is False

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_custom_options(self, mock_generate):
        """Test render_to_pdf accepts custom rendering options."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        render_to_pdf(
            report,
            dpi=600,
            embed_fonts=False,
            pdfa_compliance=True,
        )

        call_args = mock_generate.call_args
        assert call_args[1]["dpi"] == 600
        assert call_args[1]["embed_fonts"] is False
        assert call_args[1]["pdfa_compliance"] is True
        # Other defaults should still be used
        assert call_args[1]["vector_graphics"] is True
        assert call_args[1]["table_of_contents"] is True


class TestRenderToPDFFileOutput:
    """Tests for file output functionality (REPORT-008)."""

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_saves_to_file(self, mock_generate):
        """Test render_to_pdf saves to file when output_path provided."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_report.pdf")

            result = render_to_pdf(report, output_path=output_path)

            # Should still return bytes
            assert result == b"PDF content"

            # Should write to file
            assert Path(output_path).exists()
            assert Path(output_path).read_bytes() == b"PDF content"

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_creates_parent_directories(self, mock_generate):
        """Test render_to_pdf works when parent directory exists."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested path
            output_path = str(Path(tmpdir) / "subdir" / "test_report.pdf")

            # Create parent directory first
            Path(tmpdir, "subdir").mkdir()

            render_to_pdf(report, output_path=output_path)

            assert Path(output_path).exists()
            assert Path(output_path).read_bytes() == b"PDF content"

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_overwrites_existing_file(self, mock_generate):
        """Test render_to_pdf overwrites existing file."""
        mock_generate.return_value = b"New PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_report.pdf")

            # Create existing file
            Path(output_path).write_bytes(b"Old content")

            render_to_pdf(report, output_path=output_path)

            # Should be overwritten
            assert Path(output_path).read_bytes() == b"New PDF content"

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_relative_path(self, mock_generate):
        """Test render_to_pdf works with relative paths."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use relative path within tmpdir
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                output_path = "test_report.pdf"

                render_to_pdf(report, output_path=output_path)

                assert Path(output_path).exists()
                assert Path(output_path).read_bytes() == b"PDF content"
            finally:
                os.chdir(old_cwd)


class TestRenderToPDFIntegration:
    """Integration tests with report structures (REPORT-008)."""

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_complex_report(self, mock_generate):
        """Test render_to_pdf with report containing sections."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(
            title="Complex Report",
            author="Test Author",
            verbosity="detailed",
        )
        report = Report(config=config)

        # Add sections
        report.add_section("Introduction", "This is the introduction", level=1)
        report.add_section("Results", "These are the results", level=1)

        result = render_to_pdf(report)

        assert result == b"PDF content"
        mock_generate.assert_called_once()

        # Verify report was passed correctly
        call_args = mock_generate.call_args
        passed_report = call_args[0][0]
        assert len(passed_report.sections) == 2
        assert passed_report.config.title == "Complex Report"

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_tables_and_figures(self, mock_generate):
        """Test render_to_pdf with report containing tables and figures."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Data Report")
        report = Report(config=config)

        # Add table
        table_data = [["A", "B"], ["C", "D"]]
        report.add_table(table_data, headers=["Col1", "Col2"])

        # Add figure (mock)
        report.add_figure(Mock(), caption="Test Figure")

        result = render_to_pdf(report)

        assert result == b"PDF content"

        # Verify report structure was passed
        call_args = mock_generate.call_args
        passed_report = call_args[0][0]
        assert len(passed_report.tables) == 1
        assert len(passed_report.figures) == 1

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_preserves_all_config_settings(self, mock_generate):
        """Test render_to_pdf preserves all report config settings."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(
            title="Full Config Report",
            author="Test Author",
            verbosity="debug",
            page_size="A4",
            margins=1.5,
            watermark="DRAFT",
            show_toc=False,
            show_page_numbers=False,
        )
        report = Report(config=config)

        render_to_pdf(report)

        call_args = mock_generate.call_args
        passed_report = call_args[0][0]

        assert passed_report.config.title == "Full Config Report"
        assert passed_report.config.author == "Test Author"
        assert passed_report.config.verbosity == "debug"
        assert passed_report.config.page_size == "A4"
        assert passed_report.config.margins == 1.5
        assert passed_report.config.watermark == "DRAFT"
        assert passed_report.config.show_toc is False
        assert passed_report.config.show_page_numbers is False


class TestRenderToPDFErrorHandling:
    """Error handling tests (REPORT-008)."""

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_propagates_import_error(self, mock_generate):
        """Test render_to_pdf propagates ImportError from underlying module."""
        mock_generate.side_effect = ImportError("reportlab not installed")

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        with pytest.raises(ImportError, match="reportlab not installed"):
            render_to_pdf(report)

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_handles_file_write_error(self, mock_generate):
        """Test render_to_pdf handles file write errors appropriately."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        # Try to write to invalid path
        with pytest.raises((OSError, PermissionError)):
            render_to_pdf(report, output_path="/invalid/path/test.pdf")

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_empty_report(self, mock_generate):
        """Test render_to_pdf works with minimal/empty report."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Empty Report")
        report = Report(config=config)
        # No sections added

        result = render_to_pdf(report)

        assert result == b"PDF content"
        mock_generate.assert_called_once()


class TestRenderOptionsKwargs:
    """Tests for kwargs handling in render_to_pdf (REPORT-008)."""

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_merges_kwargs_with_defaults(self, mock_generate):
        """Test render_to_pdf properly merges kwargs with defaults."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        # Only override some options
        render_to_pdf(report, dpi=450, table_of_contents=False)

        call_args = mock_generate.call_args
        assert call_args[1]["dpi"] == 450
        assert call_args[1]["table_of_contents"] is False
        # Others should be defaults
        assert call_args[1]["embed_fonts"] is True
        assert call_args[1]["vector_graphics"] is True
        assert call_args[1]["pdfa_compliance"] is False

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_accepts_all_renderer_options(self, mock_generate):
        """Test render_to_pdf accepts all PDFRenderer options."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        render_to_pdf(
            report,
            dpi=100,
            embed_fonts=False,
            vector_graphics=False,
            table_of_contents=False,
            pdfa_compliance=True,
            page_numbering=False,
        )

        call_args = mock_generate.call_args
        assert call_args[1]["dpi"] == 100
        assert call_args[1]["embed_fonts"] is False
        assert call_args[1]["vector_graphics"] is False
        assert call_args[1]["table_of_contents"] is False
        assert call_args[1]["pdfa_compliance"] is True
        # Note: page_numbering is not passed to generate_pdf_report

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_creates_renderer_from_kwargs(self, mock_generate):
        """Test render_to_pdf creates PDFRenderer instance from kwargs."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        # These kwargs should be used to create PDFRenderer
        with patch("oscura.reporting.renderers.pdf.PDFRenderer") as mock_renderer_class:
            mock_renderer = PDFRenderer(dpi=200, pdfa_compliance=True)
            mock_renderer_class.return_value = mock_renderer

            render_to_pdf(report, dpi=200, pdfa_compliance=True)

            # PDFRenderer should be called with the kwargs
            mock_renderer_class.assert_called_once_with(dpi=200, pdfa_compliance=True)


class TestRenderToPDFEdgeCases:
    """Edge case tests for render_to_pdf (REPORT-008)."""

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_none_output_path(self, mock_generate):
        """Test render_to_pdf with explicitly None output path."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        result = render_to_pdf(report, output_path=None)

        assert result == b"PDF content"
        # Should not attempt to write file

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_empty_pdf_bytes(self, mock_generate):
        """Test render_to_pdf handles empty PDF bytes."""
        mock_generate.return_value = b""

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        result = render_to_pdf(report)

        assert result == b""
        assert isinstance(result, bytes)

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_large_pdf_bytes(self, mock_generate):
        """Test render_to_pdf handles large PDF content."""
        # Simulate large PDF
        large_content = b"X" * 10_000_000  # 10 MB
        mock_generate.return_value = large_content

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        result = render_to_pdf(report)

        assert result == large_content
        assert len(result) == 10_000_000

    @patch("oscura.reporting.pdf.generate_pdf_report")
    def test_render_with_special_characters_in_path(self, mock_generate):
        """Test render_to_pdf with special characters in file path."""
        mock_generate.return_value = b"PDF content"

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with spaces and special chars
            output_path = str(Path(tmpdir) / "test report (final).pdf")

            render_to_pdf(report, output_path=output_path)

            assert Path(output_path).exists()
            assert Path(output_path).read_bytes() == b"PDF content"


class TestModuleExports:
    """Tests for module-level exports."""

    def test_module_exports_all(self):
        """Test __all__ includes expected exports."""
        from oscura.reporting.renderers.pdf import __all__

        assert "PDFRenderer" in __all__
        assert "render_to_pdf" in __all__
        assert len(__all__) == 2

    def test_imports_work_correctly(self):
        """Test that public API can be imported."""
        from oscura.reporting.renderers import pdf

        assert hasattr(pdf, "PDFRenderer")
        assert hasattr(pdf, "render_to_pdf")

        # Should be callable
        assert callable(pdf.render_to_pdf)


class TestPDFRendererDocumentation:
    """Tests for documentation and type hints."""

    def test_pdfrrenderer_has_docstring(self):
        """Test PDFRenderer has docstring."""
        assert PDFRenderer.__doc__ is not None
        assert "PDF report renderer" in PDFRenderer.__doc__

    def test_render_to_pdf_has_docstring(self):
        """Test render_to_pdf has docstring."""
        assert render_to_pdf.__doc__ is not None
        assert "REPORT-008" in render_to_pdf.__doc__

    def test_render_to_pdf_has_type_hints(self):
        """Test render_to_pdf has proper type hints."""
        import inspect

        sig = inspect.signature(render_to_pdf)

        # Check return type (string annotation in __future__ imports)
        assert sig.return_annotation in (bytes, "bytes")

        # Check parameter types
        assert "report" in sig.parameters
        assert "output_path" in sig.parameters
        assert "kwargs" in sig.parameters


class TestRequirementTraceability:
    """Tests for requirement traceability (REPORT-008)."""

    def test_pdfrrenderer_references_requirements(self):
        """Test PDFRenderer docstring references REPORT-008."""
        assert "REPORT-008" in PDFRenderer.__doc__

    def test_render_to_pdf_references_requirements(self):
        """Test render_to_pdf docstring references REPORT-008."""
        assert "REPORT-008" in render_to_pdf.__doc__

    def test_module_docstring_references_requirements(self):
        """Test module docstring references REPORT-008."""
        from oscura.reporting.renderers import pdf

        assert "REPORT-008" in pdf.__doc__
