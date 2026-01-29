"""Tests for reporting module.

Tests requirements:
"""

import pytest

from oscura.reporting.core import (
    Report,
    ReportConfig,
    Section,
    generate_report,
)
from oscura.reporting.formatting import (
    NumberFormatter,
    format_margin,
    format_pass_fail,
    format_value,
    format_with_context,
    format_with_units,
)
from oscura.reporting.template_system import (
    get_template_info,
    list_templates,
    load_template,
)

pytestmark = pytest.mark.unit


class TestNumberFormatter:
    """Tests for number formatting (REPORT-006)."""

    def test_format_nanoseconds(self):
        """Test formatting nanosecond values."""
        fmt = NumberFormatter()
        result = fmt.format(2.3e-9, "s")
        assert "ns" in result or "n" in result

    def test_format_megahertz(self):
        """Test formatting MHz values."""
        fmt = NumberFormatter()
        result = fmt.format(2.3e6, "Hz")
        assert "MHz" in result or "M" in result

    def test_format_zero(self):
        """Test formatting zero."""
        fmt = NumberFormatter()
        result = fmt.format(0, "V")
        assert "0" in result

    def test_format_percentage(self):
        """Test percentage formatting."""
        fmt = NumberFormatter()
        result = fmt.format_percentage(0.543)
        assert "54" in result
        assert "%" in result

    def test_format_range(self):
        """Test min/typ/max range formatting."""
        fmt = NumberFormatter()
        result = fmt.format_range(1.0, 2.0, 3.0, "V")
        assert "min" in result
        assert "typ" in result
        assert "max" in result


class TestFormatValue:
    """Tests for format_value function."""

    def test_format_value_basic(self):
        """Test basic value formatting."""
        result = format_value(0.0000023, "s")
        assert "s" in result

    def test_format_with_units(self):
        """Test format_with_units."""
        result = format_with_units(2300000, "Hz")
        assert "Hz" in result


class TestFormatWithContext:
    """Tests for contextual formatting."""

    def test_format_with_context_pass(self):
        """Test formatting with passing context."""
        result = format_with_context(2.3e-9, spec=5e-9, unit="s", spec_type="max")
        assert "PASS" in result or "\u2713" in result

    def test_format_with_context_fail(self):
        """Test formatting with failing context."""
        result = format_with_context(6e-9, spec=5e-9, unit="s", spec_type="max")
        assert "FAIL" in result or "\u2717" in result


class TestFormatPassFail:
    """Tests for pass/fail formatting."""

    def test_format_pass(self):
        """Test pass formatting."""
        result = format_pass_fail(True)
        assert "PASS" in result

    def test_format_fail(self):
        """Test fail formatting."""
        result = format_pass_fail(False)
        assert "FAIL" in result


class TestFormatMargin:
    """Tests for margin formatting."""

    def test_format_margin_good(self):
        """Test formatting good margin."""
        result = format_margin(0.5, 1.0, limit_type="upper")
        assert "50" in result
        assert "good" in result

    def test_format_margin_violation(self):
        """Test formatting margin violation."""
        result = format_margin(1.2, 1.0, limit_type="upper")
        assert "violation" in result


class TestSection:
    """Tests for report sections."""

    def test_create_section(self):
        """Test creating a section."""
        section = Section(title="Test", content="Content", level=2)
        assert section.title == "Test"
        assert section.content == "Content"
        assert section.level == 2
        assert section.visible


class TestReportConfig:
    """Tests for report configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReportConfig()
        assert config.verbosity == "standard"
        assert config.format == "pdf"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            title="Custom Report",
            verbosity="executive",
            format="html",
        )
        assert config.title == "Custom Report"
        assert config.verbosity == "executive"


class TestReport:
    """Tests for Report class."""

    def test_create_report(self):
        """Test creating a report."""
        config = ReportConfig(title="Test Report")
        report = Report(config=config)
        assert report.config.title == "Test Report"

    def test_add_section(self):
        """Test adding sections."""
        report = Report(config=ReportConfig())
        section = report.add_section("Results", "Test results here")
        assert len(report.sections) == 1
        assert section.title == "Results"

    def test_add_table(self):
        """Test adding tables."""
        report = Report(config=ReportConfig())
        table = report.add_table(
            data=[[1, 2], [3, 4]],
            headers=["A", "B"],
            caption="Test table",
        )
        assert len(report.tables) == 1
        assert table["caption"] == "Test table"

    def test_to_markdown(self):
        """Test Markdown conversion."""
        report = Report(config=ReportConfig(title="Test"))
        report.add_section("Summary", "This is a test.")
        md = report.to_markdown()
        assert "# Test" in md
        assert "Summary" in md

    def test_to_html(self):
        """Test HTML conversion."""
        report = Report(config=ReportConfig(title="Test"))
        report.add_section("Summary", "This is a test.")
        html = report.to_html()
        assert "<html>" in html
        assert "Test" in html

    def test_executive_summary(self):
        """Test executive summary generation."""
        report = Report(config=ReportConfig())
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 25,
        }
        summary = report.generate_executive_summary(results)
        assert "passed" in summary.lower()


class TestGenerateReport:
    """Tests for report generation function."""

    def test_generate_basic_report(self):
        """Test basic report generation."""
        results = {
            "measurements": {
                "rise_time": {
                    "value": "2.3 ns",
                    "specification": "<5 ns",
                    "passed": True,
                }
            },
            "sample_rate": 1e9,
        }
        report = generate_report(results, title="Test Report")
        assert len(report.sections) > 0

    def test_generate_with_verbosity(self):
        """Test report generation with different verbosities."""
        results = {"pass_count": 5, "total_count": 5}

        report_exec = generate_report(results, verbosity="executive")
        report_debug = generate_report(results, verbosity="debug")

        # Debug should have more sections
        assert len(report_debug.sections) >= len(report_exec.sections)


class TestTemplates:
    """Tests for template system (REPORT-007)."""

    def test_list_templates(self):
        """Test listing available templates."""
        templates = list_templates()
        assert len(templates) > 0
        assert "default" in templates

    def test_load_builtin_template(self):
        """Test loading built-in template."""
        template = load_template("default")
        assert template.name == "Default Report"
        assert len(template.sections) > 0

    def test_load_compliance_template(self):
        """Test loading compliance template."""
        template = load_template("compliance")
        assert "Compliance" in template.name

    def test_get_template_info(self):
        """Test getting template info."""
        info = get_template_info("default")
        assert "name" in info
        assert "version" in info

    def test_template_sections(self):
        """Test template section structure."""
        template = load_template("characterization")
        assert any("Summary" in s.title for s in template.sections)

    def test_unknown_template(self):
        """Test loading unknown template."""
        with pytest.raises(ValueError):
            load_template("nonexistent_template")
