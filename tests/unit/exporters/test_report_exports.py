"""Tests for report export modules (EXP-006, EXP-007).

Requirements tested:
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.exporter]


@pytest.mark.unit
@pytest.mark.exporter
class TestMarkdownExport:
    """Tests for Markdown export functionality (EXP-006)."""

    def test_export_markdown_basic(self) -> None:
        """Test basic Markdown export."""
        from oscura.exporters.markdown_export import export_markdown

        data = {
            "measurements": {
                "Rise Time": {"value": 1.23e-9, "unit": "s", "status": "PASS"},
                "Amplitude": {"value": 3.3, "unit": "V", "status": "PASS"},
            },
            "metadata": {
                "filename": "test.csv",
                "sample_rate": 1e9,
                "samples": 10000,
            },
            "summary": "All measurements within specification.",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            export_markdown(data, path, title="Test Report")

            assert path.exists()
            content = path.read_text()

            # Check key elements
            assert "# Test Report" in content
            assert "Rise Time" in content
            assert "Amplitude" in content
            assert "PASS" in content
            assert "Executive Summary" in content
            assert "All measurements within specification" in content

    def test_export_markdown_sections(self) -> None:
        """Test Markdown export with specific sections."""
        from oscura.exporters.markdown_export import export_markdown

        data = {
            "measurements": {"Value": 42.0},
            "conclusions": "Test completed successfully.",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            export_markdown(data, path, sections=["measurements", "conclusions"])

            content = path.read_text()
            assert "Measurement Results" in content
            assert "Conclusions" in content
            # Metadata section should NOT be present
            assert "Report Information" not in content

    def test_generate_markdown_report_string(self) -> None:
        """Test Markdown generation to string."""
        from oscura.exporters.markdown_export import generate_markdown_report

        data = {
            "measurements": {"Frequency": {"value": 1e6, "unit": "Hz", "status": "PASS"}},
        }

        content = generate_markdown_report(data, title="String Report")

        assert "# String Report" in content
        assert "Frequency" in content
        assert "PASS" in content


@pytest.mark.unit
@pytest.mark.exporter
class TestHTMLExport:
    """Tests for HTML export functionality (EXP-007)."""

    def test_export_html_basic(self) -> None:
        """Test basic HTML export."""
        from oscura.exporters.html_export import export_html

        data = {
            "measurements": {
                "Rise Time": {"value": 1.23e-9, "unit": "s", "status": "PASS"},
                "Amplitude": {"value": 3.3, "unit": "V", "status": "FAIL"},
            },
            "metadata": {
                "filename": "test.csv",
                "sample_rate": 1e9,
            },
            "summary": "Review required.",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            export_html(data, path, title="HTML Test Report")

            assert path.exists()
            content = path.read_text()

            # Check key elements
            assert "<html" in content
            assert "HTML Test Report" in content
            assert "Rise Time" in content
            assert 'class="pass"' in content
            assert 'class="fail"' in content
            assert "Executive Summary" in content

    def test_export_html_dark_mode(self) -> None:
        """Test HTML export with dark mode enabled."""
        from oscura.exporters.html_export import export_html

        data = {"measurements": {"Value": 42.0}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            export_html(data, path, dark_mode=True)

            content = path.read_text()
            assert 'class="dark-mode"' in content

    def test_generate_html_report_string(self) -> None:
        """Test HTML generation to string."""
        from oscura.exporters.html_export import generate_html_report

        data = {
            "measurements": {"Frequency": {"value": 1e6, "unit": "Hz", "status": "PASS"}},
            "conclusions": "All tests passed.",
        }

        content = generate_html_report(data, title="String HTML Report")

        assert "<html" in content
        assert "String HTML Report" in content
        assert "Frequency" in content
        assert "All tests passed" in content


@pytest.mark.unit
@pytest.mark.exporter
class TestExportersInit:
    """Test that report exports are accessible from the package."""

    def test_markdown_export_accessible(self) -> None:
        """Test Markdown export accessible from package."""
        from oscura.exporters import export_markdown, generate_markdown_report

        assert callable(export_markdown)
        assert callable(generate_markdown_report)

    def test_html_export_accessible(self) -> None:
        """Test HTML export accessible from package."""
        from oscura.exporters import export_html, generate_html_report

        assert callable(export_html)
        assert callable(generate_html_report)
