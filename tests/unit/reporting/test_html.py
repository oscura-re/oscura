"""Comprehensive unit tests for oscura.reporting.html module.

Tests HTML report generation, styles, navigation, and interactive features.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from oscura.reporting.core import Report, ReportConfig, Section
from oscura.reporting.html import (
    _figure_to_html,
    _generate_html_content,
    _generate_html_header,
    _generate_html_nav,
    _generate_html_scripts,
    _generate_html_styles,
    _generate_metadata_section,
    _table_to_html,
    generate_html_report,
    save_html_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def basic_report() -> Report:
    """Create a basic report for testing."""
    config = ReportConfig(title="Test Report", author="Test Author")
    report = Report(config=config)
    report.add_section("Section 1", "Content 1", level=1)
    report.add_section("Section 2", "Content 2", level=2)
    return report


@pytest.fixture
def report_with_tables() -> Report:
    """Create a report with tables for testing."""
    config = ReportConfig(title="Table Report")
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
    config = ReportConfig(title="Figure Report")
    report = Report(config=config)

    figure = {
        "type": "figure",
        "figure": "/path/to/image.png",
        "caption": "Test Figure",
        "width": "80%",
    }

    report.add_section("Plots", [figure], level=1)
    return report


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    def test_basic_html_report(self, basic_report):
        """Test generating a basic HTML report."""
        html = generate_html_report(basic_report)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "Test Report" in html

    def test_html_report_includes_title(self, basic_report):
        """Test that HTML report includes the title."""
        html = generate_html_report(basic_report)

        assert "<title>Test Report</title>" in html
        assert "<h1>Test Report</h1>" in html

    def test_html_report_includes_sections(self, basic_report):
        """Test that HTML report includes all sections."""
        html = generate_html_report(basic_report)

        assert "Section 1" in html
        assert "Content 1" in html
        assert "Section 2" in html
        assert "Content 2" in html

    def test_html_report_interactive_mode(self, basic_report):
        """Test HTML report with interactive features enabled."""
        html = generate_html_report(basic_report, interactive=True)

        assert "<script>" in html
        assert "sortTable" in html

    def test_html_report_non_interactive_mode(self, basic_report):
        """Test HTML report with interactive features disabled."""
        # Small report with <= 3 sections to avoid nav
        config = ReportConfig(title="Simple Report")
        simple_report = Report(config=config)
        simple_report.add_section("Section", "Content", level=1)

        html = generate_html_report(simple_report, interactive=False, collapsible_sections=False)

        # Should not include the scripts block
        assert "sortTable" not in html

    def test_html_report_dark_mode(self, basic_report):
        """Test HTML report with dark mode support."""
        html = generate_html_report(basic_report, dark_mode=True)

        assert "dark-mode" in html or "prefers-color-scheme: dark" in html

    def test_html_report_collapsible_sections(self, basic_report):
        """Test HTML report with collapsible sections."""
        # Add a collapsible section
        basic_report.sections[0].collapsible = True

        html = generate_html_report(basic_report, collapsible_sections=True)

        assert "collapsible" in html

    def test_html_report_responsive(self, basic_report):
        """Test HTML report with responsive design."""
        html = generate_html_report(basic_report, responsive=True)

        assert "viewport" in html
        assert "@media" in html

    def test_html_report_self_contained(self, basic_report):
        """Test HTML report is self-contained (inline assets)."""
        html = generate_html_report(basic_report, self_contained=True)

        assert "<style>" in html
        # All CSS is inline, no external stylesheet links
        assert 'rel="stylesheet" href=' not in html

    def test_html_report_with_tables(self, report_with_tables):
        """Test HTML report includes tables."""
        html = generate_html_report(report_with_tables)

        assert "<table" in html
        assert "<th>" in html
        assert "<td>" in html
        assert "Parameter" in html
        assert "rise_time" in html

    def test_html_report_with_figures(self, report_with_figures):
        """Test HTML report includes figures."""
        html = generate_html_report(report_with_figures)

        assert "<figure" in html or "figure-placeholder" in html
        assert "Test Figure" in html


class TestGenerateHtmlHeader:
    """Tests for _generate_html_header function."""

    def test_header_structure(self, basic_report):
        """Test HTML header has correct structure."""
        header = _generate_html_header(basic_report, dark_mode=False, responsive=True)

        assert "<!DOCTYPE html>" in header
        assert '<html lang="en">' in header
        assert "<head>" in header
        assert '<meta charset="UTF-8">' in header

    def test_header_includes_viewport(self, basic_report):
        """Test header includes viewport meta tag."""
        header = _generate_html_header(basic_report, dark_mode=False, responsive=True)

        assert "viewport" in header

    def test_header_includes_author(self, basic_report):
        """Test header includes author meta tag."""
        header = _generate_html_header(basic_report, dark_mode=False, responsive=True)

        assert "Test Author" in header or "Oscura" in header

    def test_header_includes_generator(self, basic_report):
        """Test header includes generator meta tag."""
        header = _generate_html_header(basic_report, dark_mode=False, responsive=True)

        assert "Oscura" in header


class TestGenerateHtmlStyles:
    """Tests for _generate_html_styles function."""

    def test_styles_include_css(self):
        """Test styles include CSS."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert "<style>" in styles
        assert "</style>" in styles

    def test_styles_include_variables(self):
        """Test styles include CSS variables."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert ":root" in styles
        assert "--primary-color" in styles

    def test_styles_dark_mode_support(self):
        """Test dark mode styles are included."""
        styles = _generate_html_styles(dark_mode=True, responsive=True)

        assert "prefers-color-scheme: dark" in styles or "dark-mode" in styles

    def test_styles_responsive_media_queries(self):
        """Test responsive media queries are included."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert "@media" in styles
        assert "max-width:" in styles

    def test_styles_typography(self):
        """Test typography styles are included."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert "font-family" in styles
        assert "font-size" in styles

    def test_styles_visual_emphasis(self):
        """Test visual emphasis classes (pass/fail/warning)."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert ".pass" in styles
        assert ".fail" in styles
        assert ".warning" in styles

    def test_styles_table_formatting(self):
        """Test table formatting styles."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert "table" in styles
        assert "border-collapse" in styles

    def test_styles_collapsible(self):
        """Test collapsible section styles."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert ".collapsible" in styles

    def test_styles_print_media(self):
        """Test print media styles are included."""
        styles = _generate_html_styles(dark_mode=False, responsive=True)

        assert "@media print" in styles


class TestGenerateHtmlScripts:
    """Tests for _generate_html_scripts function."""

    def test_scripts_include_javascript(self):
        """Test scripts include JavaScript."""
        scripts = _generate_html_scripts()

        assert "<script>" in scripts
        assert "</script>" in scripts

    def test_scripts_collapsible_logic(self):
        """Test collapsible section JavaScript."""
        scripts = _generate_html_scripts()

        assert "collapsible" in scripts
        assert "classList" in scripts

    def test_scripts_table_sorting(self):
        """Test table sorting JavaScript."""
        scripts = _generate_html_scripts()

        assert "sortTable" in scripts

    def test_scripts_dark_mode_toggle(self):
        """Test dark mode toggle JavaScript."""
        scripts = _generate_html_scripts()

        assert "dark-mode" in scripts


class TestGenerateHtmlNav:
    """Tests for _generate_html_nav function."""

    def test_nav_structure(self, basic_report):
        """Test navigation structure."""
        nav = _generate_html_nav(basic_report)

        assert "<nav>" in nav
        assert "</nav>" in nav
        assert "<ul>" in nav

    def test_nav_includes_section_links(self, basic_report):
        """Test navigation includes section links."""
        nav = _generate_html_nav(basic_report)

        assert "Section 1" in nav
        assert "Section 2" in nav
        assert "href=" in nav

    def test_nav_section_ids(self, basic_report):
        """Test navigation generates correct section IDs."""
        nav = _generate_html_nav(basic_report)

        assert "section-1" in nav.lower()
        assert "section-2" in nav.lower()

    def test_nav_skips_invisible_sections(self, basic_report):
        """Test navigation skips invisible sections."""
        basic_report.sections[1].visible = False

        nav = _generate_html_nav(basic_report)

        assert "Section 1" in nav
        assert "Section 2" not in nav


class TestGenerateMetadataSection:
    """Tests for _generate_metadata_section function."""

    def test_metadata_structure(self, basic_report):
        """Test metadata section structure."""
        metadata = _generate_metadata_section(basic_report)

        assert 'class="metadata"' in metadata

    def test_metadata_includes_author(self, basic_report):
        """Test metadata includes author."""
        metadata = _generate_metadata_section(basic_report)

        assert "Author" in metadata
        assert "Test Author" in metadata

    def test_metadata_includes_date(self, basic_report):
        """Test metadata includes date."""
        metadata = _generate_metadata_section(basic_report)

        assert "Date" in metadata

    def test_metadata_includes_verbosity(self, basic_report):
        """Test metadata includes verbosity level."""
        metadata = _generate_metadata_section(basic_report)

        assert "Detail Level" in metadata
        assert basic_report.config.verbosity in metadata


class TestGenerateHtmlContent:
    """Tests for _generate_html_content function."""

    def test_content_structure(self, basic_report):
        """Test content generation structure."""
        content = _generate_html_content(basic_report, collapsible=False)

        assert "<section" in content
        assert "</section>" in content

    def test_content_includes_sections(self, basic_report):
        """Test content includes all visible sections."""
        content = _generate_html_content(basic_report, collapsible=False)

        assert "Section 1" in content
        assert "Content 1" in content

    def test_content_skips_invisible_sections(self, basic_report):
        """Test content skips invisible sections."""
        basic_report.sections[1].visible = False

        content = _generate_html_content(basic_report, collapsible=False)

        assert "Section 1" in content
        assert "Section 2" not in content

    def test_content_collapsible_sections(self, basic_report):
        """Test collapsible section HTML."""
        basic_report.sections[0].collapsible = True

        content = _generate_html_content(basic_report, collapsible=True)

        assert "collapsible" in content
        assert "collapsible-content" in content

    def test_content_section_ids(self, basic_report):
        """Test sections have IDs for navigation."""
        content = _generate_html_content(basic_report, collapsible=False)

        assert 'id="section-1"' in content.lower()

    def test_content_with_tables(self, report_with_tables):
        """Test content includes tables."""
        content = _generate_html_content(report_with_tables, collapsible=False)

        assert "<table" in content
        assert "Parameter" in content

    def test_content_with_figures(self, report_with_figures):
        """Test content includes figures."""
        content = _generate_html_content(report_with_figures, collapsible=False)

        assert "figure" in content.lower()

    def test_content_subsections(self, basic_report):
        """Test content includes subsections."""
        subsection = Section(title="Subsection", content="Sub content", level=3)
        basic_report.sections[0].subsections.append(subsection)

        content = _generate_html_content(basic_report, collapsible=False)

        assert "Subsection" in content
        assert "Sub content" in content


class TestTableToHtml:
    """Tests for _table_to_html function."""

    def test_table_structure(self):
        """Test table HTML structure."""
        table = {
            "headers": ["A", "B"],
            "data": [["1", "2"], ["3", "4"]],
        }

        html = _table_to_html(table)

        assert "<table" in html
        assert "</table>" in html
        assert "<thead>" in html
        assert "<tbody>" in html

    def test_table_headers(self):
        """Test table headers are included."""
        table = {
            "headers": ["Parameter", "Value"],
            "data": [],
        }

        html = _table_to_html(table)

        assert "<th>Parameter</th>" in html
        assert "<th>Value</th>" in html

    def test_table_data_rows(self):
        """Test table data rows."""
        table = {
            "headers": ["A"],
            "data": [["value1"], ["value2"]],
        }

        html = _table_to_html(table)

        assert "<td>value1</td>" in html
        assert "<td>value2</td>" in html

    def test_table_pass_class(self):
        """Test PASS cells get pass class."""
        table = {
            "headers": ["Status"],
            "data": [["PASS"]],
        }

        html = _table_to_html(table)

        assert 'class="pass"' in html

    def test_table_fail_class(self):
        """Test FAIL cells get fail class."""
        table = {
            "headers": ["Status"],
            "data": [["FAIL"]],
        }

        html = _table_to_html(table)

        assert 'class="fail"' in html

    def test_table_warning_class(self):
        """Test WARNING cells get warning class."""
        table = {
            "headers": ["Status"],
            "data": [["WARNING"]],
        }

        html = _table_to_html(table)

        assert 'class="warning"' in html

    def test_table_caption(self):
        """Test table caption is included."""
        table = {
            "headers": ["A"],
            "data": [["1"]],
            "caption": "Test Caption",
        }

        html = _table_to_html(table)

        assert "<caption>Test Caption</caption>" in html

    def test_table_sortable_class(self):
        """Test table has sortable class."""
        table = {
            "headers": ["A"],
            "data": [],
        }

        html = _table_to_html(table)

        assert 'class="sortable"' in html


class TestFigureToHtml:
    """Tests for _figure_to_html function."""

    def test_figure_structure(self):
        """Test figure HTML structure."""
        figure = {
            "figure": "/path/to/image.png",
            "caption": "Test Figure",
        }

        html = _figure_to_html(figure)

        assert "<figure" in html
        assert "</figure>" in html

    def test_figure_with_image_path(self):
        """Test figure with image path."""
        figure = {
            "figure": "/path/to/image.png",
            "caption": "Image",
        }

        html = _figure_to_html(figure)

        assert "<img" in html
        assert 'src="/path/to/image.png"' in html

    def test_figure_caption(self):
        """Test figure caption."""
        figure = {
            "figure": "/path/to/image.png",
            "caption": "My Caption",
        }

        html = _figure_to_html(figure)

        assert "<figcaption>" in html
        assert "My Caption" in html

    def test_figure_width(self):
        """Test figure width."""
        figure = {
            "figure": "/path/to/image.png",
            "width": "50%",
        }

        html = _figure_to_html(figure)

        assert "50%" in html

    def test_figure_default_width(self):
        """Test figure default width."""
        figure = {
            "figure": "/path/to/image.png",
        }

        html = _figure_to_html(figure)

        assert "100%" in html

    def test_figure_placeholder(self):
        """Test figure with non-string object."""
        figure = {
            "figure": object(),  # Not a string path
            "caption": "Placeholder Figure",
        }

        html = _figure_to_html(figure)

        assert "figure-placeholder" in html


class TestSaveHtmlReport:
    """Tests for save_html_report function."""

    def test_save_html_report(self, basic_report):
        """Test saving HTML report to file."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"

            save_html_report(basic_report, path)

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "Test Report" in content

    def test_save_html_report_with_options(self, basic_report):
        """Test saving HTML report with options."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"

            save_html_report(basic_report, path, interactive=True, dark_mode=True)

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "<script>" in content


class TestReportingHtmlIntegration:
    """Integration tests for HTML report generation."""

    def test_full_report_generation(self):
        """Test complete HTML report generation."""
        config = ReportConfig(
            title="Full Test Report",
            author="Integration Test",
            verbosity="detailed",
        )
        report = Report(config=config)

        # Add various content types
        report.add_section("Summary", "This is a summary.", level=1)

        table = {
            "type": "table",
            "headers": ["Test", "Result"],
            "data": [["Test 1", "PASS"], ["Test 2", "FAIL"]],
            "caption": "Test Results",
        }
        report.add_section("Results", [table], level=1)

        report.add_section("Conclusion", "All done.", level=1, collapsible=True)

        # Generate HTML
        html = generate_html_report(
            report, interactive=True, dark_mode=True, collapsible_sections=True
        )

        # Verify all components
        assert "<!DOCTYPE html>" in html
        assert "Full Test Report" in html
        assert "Integration Test" in html
        assert "Summary" in html
        assert "<table" in html
        assert "Test 1" in html
        assert 'class="pass"' in html
        assert 'class="fail"' in html
        assert "collapsible" in html

    def test_report_with_many_sections(self):
        """Test HTML report with many sections (triggers navigation)."""
        config = ReportConfig(title="Many Sections Report")
        report = Report(config=config)

        for i in range(10):
            report.add_section(f"Section {i}", f"Content {i}", level=1)

        html = generate_html_report(report)

        assert "<nav>" in html
        assert "Section 0" in html
        assert "Section 9" in html

    def test_report_with_nested_subsections(self):
        """Test HTML report with nested subsections."""
        config = ReportConfig(title="Nested Report")
        report = Report(config=config)

        section = report.add_section("Main", "Main content", level=1)
        section.subsections.append(Section(title="Sub 1", content="Sub content 1", level=2))
        section.subsections.append(Section(title="Sub 2", content="Sub content 2", level=2))

        html = generate_html_report(report)

        assert "Main" in html
        assert "Sub 1" in html
        assert "Sub 2" in html
