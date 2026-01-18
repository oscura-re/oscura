"""Tests for formatting standards module.

Tests for reporting formatting standards and color schemes.
"""

from __future__ import annotations

import pytest

from oscura.reporting.formatting.standards import (
    ColorScheme,
    FormatStandards,
    Severity,
    apply_formatting_standards,
)

pytestmark = pytest.mark.unit


class TestSeverity:
    """Test Severity enum."""

    def test_severity_values(self) -> None:
        """Test severity enum values."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"
        assert Severity.SUCCESS.value == "success"

    def test_severity_members(self) -> None:
        """Test severity enum has all expected members."""
        severities = [s.value for s in Severity]
        assert "info" in severities
        assert "warning" in severities
        assert "error" in severities
        assert "critical" in severities
        assert "success" in severities

    def test_severity_count(self) -> None:
        """Test number of severity levels."""
        assert len(Severity) == 5

    def test_severity_comparison(self) -> None:
        """Test severity enum comparison."""
        assert Severity.INFO == Severity.INFO
        assert Severity.WARNING != Severity.ERROR

    def test_severity_from_value(self) -> None:
        """Test creating severity from value."""
        assert Severity("info") == Severity.INFO
        assert Severity("warning") == Severity.WARNING


class TestColorScheme:
    """Test ColorScheme dataclass."""

    def test_default_colors(self) -> None:
        """Test default color values."""
        scheme = ColorScheme()
        assert scheme.primary == "#007ACC"
        assert scheme.secondary == "#6C757D"
        assert scheme.success == "#28A745"
        assert scheme.warning == "#FFC107"
        assert scheme.error == "#DC3545"
        assert scheme.info == "#17A2B8"

    def test_custom_colors(self) -> None:
        """Test custom color values."""
        scheme = ColorScheme(
            primary="#FF0000",
            secondary="#00FF00",
            success="#0000FF",
            warning="#FFFF00",
            error="#FF00FF",
            info="#00FFFF",
        )
        assert scheme.primary == "#FF0000"
        assert scheme.secondary == "#00FF00"
        assert scheme.success == "#0000FF"
        assert scheme.warning == "#FFFF00"
        assert scheme.error == "#FF00FF"
        assert scheme.info == "#00FFFF"

    def test_partial_custom_colors(self) -> None:
        """Test partial custom colors with defaults."""
        scheme = ColorScheme(primary="#000000", error="#FFFFFF")
        assert scheme.primary == "#000000"
        assert scheme.error == "#FFFFFF"
        assert scheme.secondary == "#6C757D"  # Default
        assert scheme.success == "#28A745"  # Default

    def test_color_scheme_equality(self) -> None:
        """Test color scheme equality."""
        scheme1 = ColorScheme()
        scheme2 = ColorScheme()
        assert scheme1 == scheme2

        scheme3 = ColorScheme(primary="#000000")
        assert scheme1 != scheme3

    def test_color_scheme_repr(self) -> None:
        """Test color scheme string representation."""
        scheme = ColorScheme()
        repr_str = repr(scheme)
        assert "ColorScheme" in repr_str
        assert "#007ACC" in repr_str


class TestFormatStandards:
    """Test FormatStandards dataclass."""

    def test_default_standards(self) -> None:
        """Test default formatting standards."""
        standards = FormatStandards()
        assert standards.title_size == 18
        assert standards.heading_size == 14
        assert standards.body_size == 10
        assert standards.code_font == "Courier New"
        assert standards.body_font == "Arial"
        assert isinstance(standards.colors, ColorScheme)

    def test_custom_standards(self) -> None:
        """Test custom formatting standards."""
        custom_colors = ColorScheme(primary="#FF0000")
        standards = FormatStandards(
            title_size=24,
            heading_size=18,
            body_size=12,
            code_font="Monaco",
            body_font="Helvetica",
            colors=custom_colors,
        )
        assert standards.title_size == 24
        assert standards.heading_size == 18
        assert standards.body_size == 12
        assert standards.code_font == "Monaco"
        assert standards.body_font == "Helvetica"
        assert standards.colors.primary == "#FF0000"

    def test_partial_custom_standards(self) -> None:
        """Test partial custom standards with defaults."""
        standards = FormatStandards(title_size=20, code_font="Consolas")
        assert standards.title_size == 20
        assert standards.code_font == "Consolas"
        assert standards.heading_size == 14  # Default
        assert standards.body_size == 10  # Default
        assert standards.body_font == "Arial"  # Default

    def test_default_color_scheme_created(self) -> None:
        """Test default color scheme is created automatically."""
        standards = FormatStandards()
        assert standards.colors.primary == "#007ACC"
        assert standards.colors.error == "#DC3545"

    def test_format_standards_equality(self) -> None:
        """Test format standards equality."""
        standards1 = FormatStandards()
        standards2 = FormatStandards()
        assert standards1 == standards2

        standards3 = FormatStandards(title_size=20)
        assert standards1 != standards3

    def test_format_standards_repr(self) -> None:
        """Test format standards string representation."""
        standards = FormatStandards()
        repr_str = repr(standards)
        assert "FormatStandards" in repr_str
        assert "18" in repr_str  # title_size
        assert "Arial" in repr_str


class TestApplyFormattingStandards:
    """Test apply_formatting_standards function."""

    def test_apply_with_defaults(self) -> None:
        """Test applying formatting with default standards."""
        content = "test content"
        result = apply_formatting_standards(content)
        # Placeholder implementation just returns content
        assert result == content

    def test_apply_with_custom_standards(self) -> None:
        """Test applying formatting with custom standards."""
        content = {"text": "test"}
        standards = FormatStandards(title_size=20)
        result = apply_formatting_standards(content, standards)
        # Placeholder implementation just returns content
        assert result == content

    def test_apply_with_none_standards(self) -> None:
        """Test applying formatting with None creates defaults."""
        content = [1, 2, 3]
        result = apply_formatting_standards(content, standards=None)
        assert result == content

    def test_apply_preserves_content_type(self) -> None:
        """Test that apply_formatting_standards preserves content type."""
        # String
        assert isinstance(apply_formatting_standards("test"), str)

        # Dict
        assert isinstance(apply_formatting_standards({"a": 1}), dict)

        # List
        assert isinstance(apply_formatting_standards([1, 2, 3]), list)

        # None
        assert apply_formatting_standards(None) is None

    def test_apply_with_empty_content(self) -> None:
        """Test applying formatting to empty content."""
        assert apply_formatting_standards("") == ""
        assert apply_formatting_standards([]) == []
        assert apply_formatting_standards({}) == {}


class TestReportingFormattingStandardsIntegration:
    """Integration tests for formatting standards."""

    def test_full_workflow(self) -> None:
        """Test complete workflow with custom colors and standards."""
        # Create custom color scheme
        colors = ColorScheme(primary="#1E88E5", secondary="#424242", error="#F44336")

        # Create formatting standards with custom colors
        standards = FormatStandards(title_size=22, heading_size=16, body_size=11, colors=colors)

        # Verify all properties
        assert standards.title_size == 22
        assert standards.colors.primary == "#1E88E5"
        assert standards.colors.error == "#F44336"

        # Apply standards
        content = {"title": "Test Report", "body": "Content"}
        result = apply_formatting_standards(content, standards)
        assert result == content

    def test_severity_with_color_mapping(self) -> None:
        """Test typical usage pattern: mapping severity to colors."""
        scheme = ColorScheme()

        # Map severity levels to colors
        severity_colors = {
            Severity.INFO: scheme.info,
            Severity.WARNING: scheme.warning,
            Severity.ERROR: scheme.error,
            Severity.CRITICAL: scheme.error,  # Use error color
            Severity.SUCCESS: scheme.success,
        }

        assert severity_colors[Severity.INFO] == "#17A2B8"
        assert severity_colors[Severity.WARNING] == "#FFC107"
        assert severity_colors[Severity.ERROR] == "#DC3545"
        assert severity_colors[Severity.SUCCESS] == "#28A745"

    def test_multiple_standards_independence(self) -> None:
        """Test that multiple standards instances are independent."""
        standards1 = FormatStandards(title_size=20)
        standards2 = FormatStandards(title_size=24)

        # Modifying one doesn't affect the other
        assert standards1.title_size == 20
        assert standards2.title_size == 24

        # Color schemes are also independent
        standards1.colors.primary = "#000000"
        assert standards2.colors.primary == "#007ACC"  # Still default
