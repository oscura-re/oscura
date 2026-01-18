"""Tests for text emphasis formatting utilities.

Tests for terminal text formatting and visual emphasis.
"""

from __future__ import annotations

import pytest

from oscura.reporting.formatting.emphasis import (
    VisualEmphasis,
    bold,
    color,
    format_callout_box,
    format_severity,
    italic,
    underline,
)

pytestmark = pytest.mark.unit


class TestVisualEmphasis:
    """Test VisualEmphasis enum."""

    def test_emphasis_values(self) -> None:
        """Test visual emphasis enum values."""
        assert VisualEmphasis.NONE.value == "none"
        assert VisualEmphasis.SUBTLE.value == "subtle"
        assert VisualEmphasis.MODERATE.value == "moderate"
        assert VisualEmphasis.STRONG.value == "strong"
        assert VisualEmphasis.CRITICAL.value == "critical"

    def test_emphasis_members(self) -> None:
        """Test visual emphasis enum members."""
        levels = [e.value for e in VisualEmphasis]
        assert "none" in levels
        assert "subtle" in levels
        assert "moderate" in levels
        assert "strong" in levels
        assert "critical" in levels

    def test_emphasis_count(self) -> None:
        """Test number of emphasis levels."""
        assert len(VisualEmphasis) == 5

    def test_emphasis_comparison(self) -> None:
        """Test emphasis enum comparison."""
        assert VisualEmphasis.NONE == VisualEmphasis.NONE
        assert VisualEmphasis.SUBTLE != VisualEmphasis.STRONG


class TestBold:
    """Test bold() function."""

    def test_bold_simple_text(self) -> None:
        """Test bold formatting on simple text."""
        result = bold("Hello")
        assert result == "\033[1mHello\033[0m"

    def test_bold_empty_string(self) -> None:
        """Test bold with empty string."""
        result = bold("")
        assert result == "\033[1m\033[0m"

    def test_bold_multiline(self) -> None:
        """Test bold with multiline text."""
        result = bold("Line 1\nLine 2")
        assert result == "\033[1mLine 1\nLine 2\033[0m"

    def test_bold_special_characters(self) -> None:
        """Test bold with special characters."""
        result = bold("Test @#$%!")
        assert result == "\033[1mTest @#$%!\033[0m"

    def test_bold_unicode(self) -> None:
        """Test bold with unicode characters."""
        result = bold("Hello 世界")
        assert result == "\033[1mHello 世界\033[0m"


class TestItalic:
    """Test italic() function."""

    def test_italic_simple_text(self) -> None:
        """Test italic formatting on simple text."""
        result = italic("Hello")
        assert result == "\033[3mHello\033[0m"

    def test_italic_empty_string(self) -> None:
        """Test italic with empty string."""
        result = italic("")
        assert result == "\033[3m\033[0m"

    def test_italic_multiline(self) -> None:
        """Test italic with multiline text."""
        result = italic("Line 1\nLine 2")
        assert result == "\033[3mLine 1\nLine 2\033[0m"


class TestUnderline:
    """Test underline() function."""

    def test_underline_simple_text(self) -> None:
        """Test underline formatting on simple text."""
        result = underline("Hello")
        assert result == "\033[4mHello\033[0m"

    def test_underline_empty_string(self) -> None:
        """Test underline with empty string."""
        result = underline("")
        assert result == "\033[4m\033[0m"

    def test_underline_multiline(self) -> None:
        """Test underline with multiline text."""
        result = underline("Line 1\nLine 2")
        assert result == "\033[4mLine 1\nLine 2\033[0m"


class TestColor:
    """Test color() function."""

    def test_color_red(self) -> None:
        """Test red color."""
        result = color("Error", "red")
        assert result == "\033[31mError\033[0m"

    def test_color_green(self) -> None:
        """Test green color."""
        result = color("Success", "green")
        assert result == "\033[32mSuccess\033[0m"

    def test_color_yellow(self) -> None:
        """Test yellow color."""
        result = color("Warning", "yellow")
        assert result == "\033[33mWarning\033[0m"

    def test_color_blue(self) -> None:
        """Test blue color."""
        result = color("Info", "blue")
        assert result == "\033[34mInfo\033[0m"

    def test_color_magenta(self) -> None:
        """Test magenta color."""
        result = color("Debug", "magenta")
        assert result == "\033[35mDebug\033[0m"

    def test_color_cyan(self) -> None:
        """Test cyan color."""
        result = color("Data", "cyan")
        assert result == "\033[36mData\033[0m"

    def test_color_white(self) -> None:
        """Test white color."""
        result = color("Text", "white")
        assert result == "\033[37mText\033[0m"

    def test_color_case_insensitive(self) -> None:
        """Test that color names are case-insensitive."""
        result1 = color("Test", "RED")
        result2 = color("Test", "red")
        assert result1 == result2

    def test_color_unknown_defaults_to_white(self) -> None:
        """Test unknown color defaults to white."""
        result = color("Test", "purple")
        assert result == "\033[37mTest\033[0m"  # White (37)

    def test_color_empty_string(self) -> None:
        """Test color with empty string."""
        result = color("", "red")
        assert result == "\033[31m\033[0m"

    def test_color_all_supported_colors(self) -> None:
        """Test all supported colors."""
        colors = {
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
            "white": "37",
        }
        for color_name, code in colors.items():
            result = color("Test", color_name)
            expected = f"\033[{code}mTest\033[0m"
            assert result == expected


class TestFormatSeverity:
    """Test format_severity() function."""

    def test_severity_critical(self) -> None:
        """Test critical severity formatting."""
        result = format_severity("Critical Issue", "critical")
        assert result == "\033[31mCritical Issue\033[0m"  # Red

    def test_severity_error(self) -> None:
        """Test error severity formatting."""
        result = format_severity("Error occurred", "error")
        assert result == "\033[31mError occurred\033[0m"  # Red

    def test_severity_warning(self) -> None:
        """Test warning severity formatting."""
        result = format_severity("Warning", "warning")
        assert result == "\033[33mWarning\033[0m"  # Yellow

    def test_severity_info(self) -> None:
        """Test info severity formatting."""
        result = format_severity("Information", "info")
        assert result == "\033[34mInformation\033[0m"  # Blue

    def test_severity_success(self) -> None:
        """Test success severity formatting."""
        result = format_severity("Success!", "success")
        assert result == "\033[32mSuccess!\033[0m"  # Green

    def test_severity_case_insensitive(self) -> None:
        """Test severity is case-insensitive."""
        result1 = format_severity("Test", "ERROR")
        result2 = format_severity("Test", "error")
        assert result1 == result2

    def test_severity_unknown_defaults_to_white(self) -> None:
        """Test unknown severity defaults to white."""
        result = format_severity("Unknown", "debug")
        assert result == "\033[37mUnknown\033[0m"  # White

    def test_severity_empty_string(self) -> None:
        """Test severity formatting with empty text."""
        result = format_severity("", "error")
        assert result == "\033[31m\033[0m"


class TestFormatCalloutBox:
    """Test format_callout_box() function."""

    def test_callout_box_basic(self) -> None:
        """Test basic callout box."""
        result = format_callout_box("Notice", "This is a notice")
        lines = result.split("\n")
        assert len(lines) == 5
        assert "=" * 60 in lines[0]
        assert "NOTICE" in lines[1]
        assert "-" * 60 in lines[2]
        assert "This is a notice" in lines[3]
        assert "=" * 60 in lines[4]

    def test_callout_box_with_severity(self) -> None:
        """Test callout box with severity."""
        result = format_callout_box("Error", "Something went wrong", severity="error")
        lines = result.split("\n")
        # Title should be formatted with error color (red)
        assert "\033[31m" in lines[1]  # Red color code
        assert "ERROR" in lines[1]

    def test_callout_box_default_severity(self) -> None:
        """Test callout box with default severity (info)."""
        result = format_callout_box("Information", "Details here")
        lines = result.split("\n")
        # Default is info, which should be blue
        assert "\033[34m" in lines[1]  # Blue color code

    def test_callout_box_warning(self) -> None:
        """Test callout box with warning severity."""
        result = format_callout_box("Warning", "Be careful", severity="warning")
        lines = result.split("\n")
        assert "\033[33m" in lines[1]  # Yellow color code

    def test_callout_box_success(self) -> None:
        """Test callout box with success severity."""
        result = format_callout_box("Success", "All done", severity="success")
        lines = result.split("\n")
        assert "\033[32m" in lines[1]  # Green color code

    def test_callout_box_multiline_content(self) -> None:
        """Test callout box with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        result = format_callout_box("Title", content)
        # Content should be preserved as-is
        assert "Line 1\nLine 2\nLine 3" in result

    def test_callout_box_empty_content(self) -> None:
        """Test callout box with empty content."""
        result = format_callout_box("Title", "")
        lines = result.split("\n")
        assert len(lines) == 5
        assert lines[3] == ""  # Empty content line

    def test_callout_box_title_uppercased(self) -> None:
        """Test that title is uppercased."""
        result = format_callout_box("my title", "Content")
        assert "MY TITLE" in result
        assert "my title" not in result  # Original case should not appear


class TestReportingFormattingEmphasisIntegration:
    """Integration tests for emphasis formatting."""

    def test_nested_formatting(self) -> None:
        """Test combining multiple formatting functions."""
        # Bold red text
        text = "Important"
        colored = color(text, "red")
        result = bold(colored)
        # Should contain both bold and color codes
        assert "\033[1m" in result
        assert "\033[31m" in result

    def test_formatted_callout_chain(self) -> None:
        """Test creating formatted callouts for different severities."""
        severities = ["critical", "error", "warning", "info", "success"]
        for sev in severities:
            result = format_callout_box("Test", "Content", severity=sev)
            assert "TEST" in result
            assert "Content" in result
            assert "=" * 60 in result

    def test_color_all_emphasis_levels(self) -> None:
        """Test coloring text for each emphasis level."""
        # Map emphasis to colors (example usage pattern)
        emphasis_colors = {
            VisualEmphasis.NONE: "white",
            VisualEmphasis.SUBTLE: "cyan",
            VisualEmphasis.MODERATE: "blue",
            VisualEmphasis.STRONG: "yellow",
            VisualEmphasis.CRITICAL: "red",
        }

        for emphasis, color_name in emphasis_colors.items():
            result = color(f"Level: {emphasis.value}", color_name)
            assert "\033[" in result  # Has ANSI code
            assert result.endswith("\033[0m")  # Ends with reset

    def test_realistic_error_message(self) -> None:
        """Test realistic error message formatting."""
        title = "Configuration Error"
        message = "Invalid value for 'sample_rate': must be positive"
        result = format_callout_box(title, message, severity="error")

        # Verify structure
        assert "CONFIGURATION ERROR" in result
        assert "Invalid value" in result
        assert "\033[31m" in result  # Red color for error

    def test_realistic_success_message(self) -> None:
        """Test realistic success message formatting."""
        title = "Test Complete"
        message = "All 156 tests passed successfully"
        result = format_callout_box(title, message, severity="success")

        assert "TEST COMPLETE" in result
        assert "156 tests" in result
        assert "\033[32m" in result  # Green color for success
