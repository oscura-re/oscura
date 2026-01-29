"""Comprehensive tests for UI formatting utilities.

Tests requirements:
  - Color and text formatting
  - Text alignment and truncation
  - Table, list, and key-value formatting
  - Status, percentage, duration, and size formatting
"""

from __future__ import annotations

import pytest

from oscura.jupyter.ui.formatters import (
    Color,
    FormattedText,
    TextAlignment,
    align_text,
    colorize,
    format_code_block,
    format_duration,
    format_key_value_pairs,
    format_list,
    format_percentage,
    format_size,
    format_status,
    format_table,
    format_text,
    truncate,
)

pytestmark = pytest.mark.unit


class TestColorEnum:
    """Test Color enum."""

    def test_color_values(self) -> None:
        """Test that all colors have ANSI codes."""
        assert Color.RED.value == "\033[31m"
        assert Color.GREEN.value == "\033[32m"
        assert Color.BLUE.value == "\033[34m"
        assert Color.RESET.value == "\033[0m"

    def test_all_colors_defined(self) -> None:
        """Test that all expected colors are defined."""
        expected = ["BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE", "RESET"]
        for color_name in expected:
            assert hasattr(Color, color_name)


class TestTextAlignment:
    """Test TextAlignment enum."""

    def test_alignment_values(self) -> None:
        """Test alignment enum values."""
        assert TextAlignment.LEFT.value == "left"
        assert TextAlignment.CENTER.value == "center"
        assert TextAlignment.RIGHT.value == "right"


class TestFormattedText:
    """Test FormattedText dataclass."""

    def test_init(self) -> None:
        """Test FormattedText initialization."""
        text = FormattedText(content="Hello", color=Color.GREEN, bold=True, width=10)

        assert text.content == "Hello"
        assert text.color == Color.GREEN
        assert text.bold is True
        assert text.width == 10

    def test_str_plain(self) -> None:
        """Test string representation without formatting."""
        text = FormattedText(content="Hello")
        result = str(text)
        assert "Hello" in result

    def test_str_with_color(self) -> None:
        """Test string representation with color."""
        text = FormattedText(content="Hello", color=Color.RED)
        result = str(text)
        assert Color.RED.value in result
        assert Color.RESET.value in result

    def test_str_with_bold(self) -> None:
        """Test string representation with bold."""
        text = FormattedText(content="Hello", bold=True)
        result = str(text)
        assert "\033[1m" in result
        assert "\033[0m" in result


class TestColorize:
    """Test colorize function."""

    def test_colorize_red(self) -> None:
        """Test colorizing text red."""
        result = colorize("Error", color="red")
        assert "\033[31m" in result
        assert "Error" in result
        assert "\033[0m" in result

    def test_colorize_green(self) -> None:
        """Test colorizing text green."""
        result = colorize("Success", color="green")
        assert "\033[32m" in result

    def test_colorize_bold(self) -> None:
        """Test colorizing with bold."""
        result = colorize("Important", color="yellow", bold=True)
        assert "\033[1m" in result
        assert "\033[33m" in result

    def test_colorize_invalid_color(self) -> None:
        """Test colorize with invalid color defaults to white."""
        result = colorize("Text", color="invalid")  # type: ignore[arg-type]
        # Should default to white
        assert "\033[0m" in result


class TestTruncate:
    """Test truncate function."""

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        result = truncate("Hello", max_length=10)
        assert result == "Hello"

    def test_truncate_long_text(self) -> None:
        """Test truncating long text."""
        result = truncate("This is a very long text", max_length=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_custom_suffix(self) -> None:
        """Test truncate with custom suffix."""
        result = truncate("Long text here", max_length=10, suffix=">>")
        assert len(result) == 10
        assert result.endswith(">>")

    def test_truncate_exact_length(self) -> None:
        """Test text exactly at max length."""
        result = truncate("Exactly10!", max_length=10)
        assert result == "Exactly10!"


class TestAlignText:
    """Test align_text function."""

    def test_align_left(self) -> None:
        """Test left alignment."""
        result = align_text("Hello", width=10, alignment="left")
        assert result == "Hello     "
        assert len(result) == 10

    def test_align_center(self) -> None:
        """Test center alignment."""
        result = align_text("Hello", width=11, alignment="center")
        assert result == "   Hello   "
        assert len(result) == 11

    def test_align_right(self) -> None:
        """Test right alignment."""
        result = align_text("Hello", width=10, alignment="right")
        assert result == "     Hello"
        assert len(result) == 10

    def test_align_custom_fill_char(self) -> None:
        """Test alignment with custom fill character."""
        result = align_text("Hi", width=5, alignment="center", fill_char="-")
        assert "-" in result
        assert len(result) == 5

    def test_align_text_longer_than_width(self) -> None:
        """Test that text longer than width is not modified."""
        result = align_text("Very long text", width=5)
        assert result == "Very long text"


class TestFormatText:
    """Test format_text function."""

    def test_format_text_basic(self) -> None:
        """Test basic label-value formatting."""
        result = format_text("Name", "Alice")
        assert "Name" in result
        assert "Alice" in result
        assert ":" in result

    def test_format_text_custom_separator(self) -> None:
        """Test formatting with custom separator."""
        result = format_text("Name", "Alice", separator=" = ")
        assert "Name = Alice" in result

    def test_format_text_with_width(self) -> None:
        """Test formatting with fixed width."""
        result = format_text("Name", "Alice", width=20)
        assert len(result) == 20

    def test_format_text_with_color(self) -> None:
        """Test formatting with color."""
        result = format_text("Status", "active", color="green")
        assert "Status" in result
        assert "\033[" in result


class TestFormatTable:
    """Test format_table function."""

    def test_format_table_basic(self) -> None:
        """Test basic table formatting."""
        data = [["Alice", 85], ["Bob", 92]]
        result = format_table(data, headers=["Name", "Score"])

        assert "Alice" in result
        assert "Bob" in result
        assert "85" in result
        assert "92" in result

    def test_format_table_empty(self) -> None:
        """Test formatting empty table."""
        result = format_table([])
        assert result == ""

    def test_format_table_no_headers(self) -> None:
        """Test table without headers."""
        data = [["A", "B"], ["C", "D"]]
        result = format_table(data)
        assert "A" in result
        assert "B" in result

    def test_format_table_custom_widths(self) -> None:
        """Test table with custom column widths."""
        data = [["A", "B"]]
        result = format_table(data, column_widths=[10, 5])
        # Should respect custom widths
        assert "A" in result

    def test_format_table_custom_alignment(self) -> None:
        """Test table with custom alignment."""
        data = [["Left", "Right"]]
        result = format_table(data, align_columns=["left", "right"])
        assert "Left" in result


class TestFormatStatus:
    """Test format_status function."""

    def test_format_status_pass(self) -> None:
        """Test pass status formatting."""
        result = format_status("pass", "All tests passed")
        assert "✓" in result
        assert "All tests passed" in result

    def test_format_status_fail(self) -> None:
        """Test fail status formatting."""
        result = format_status("fail", "Error occurred")
        assert "✗" in result
        assert "Error occurred" in result

    def test_format_status_warning(self) -> None:
        """Test warning status formatting."""
        result = format_status("warning", "Check this")
        assert "⚠" in result

    def test_format_status_info(self) -> None:
        """Test info status formatting."""
        result = format_status("info", "Information")
        assert "ℹ" in result

    def test_format_status_pending(self) -> None:
        """Test pending status formatting."""
        result = format_status("pending", "In progress")
        assert "⏳" in result

    def test_format_status_no_symbols(self) -> None:
        """Test status formatting without symbols."""
        result = format_status("pass", "Success", use_symbols=False)
        assert "PASS" in result
        assert "Success" in result


class TestFormatPercentage:
    """Test format_percentage function."""

    def test_format_percentage_basic(self) -> None:
        """Test basic percentage formatting."""
        result = format_percentage(75.5)
        assert "75.5%" in result

    def test_format_percentage_normalized(self) -> None:
        """Test percentage with 0-1 value."""
        result = format_percentage(0.755)
        assert "75.5%" in result

    def test_format_percentage_with_bar(self) -> None:
        """Test percentage with progress bar."""
        result = format_percentage(50, show_bar=True)
        assert "50.0%" in result
        assert "[" in result
        assert "]" in result
        assert "█" in result or "░" in result

    def test_format_percentage_decimals(self) -> None:
        """Test percentage decimal precision."""
        result = format_percentage(75.123, decimals=2)
        assert "75.12%" in result

    def test_format_percentage_zero(self) -> None:
        """Test zero percentage."""
        result = format_percentage(0, show_bar=True)
        assert "0.0%" in result

    def test_format_percentage_hundred(self) -> None:
        """Test 100% percentage."""
        result = format_percentage(100, show_bar=True)
        assert "100.0%" in result


class TestFormatDuration:
    """Test format_duration function."""

    def test_format_duration_hours(self) -> None:
        """Test duration in hours."""
        result = format_duration(5025)  # 1h 23m 45s
        assert "1h" in result
        assert "23m" in result
        assert "45s" in result

    def test_format_duration_minutes(self) -> None:
        """Test duration in minutes."""
        result = format_duration(125)  # 2m 5s
        assert "2m" in result
        assert "5s" in result
        assert "h" not in result

    def test_format_duration_seconds(self) -> None:
        """Test duration in seconds."""
        result = format_duration(45)
        assert "45s" in result
        assert "m" not in result

    def test_format_duration_milliseconds(self) -> None:
        """Test duration in milliseconds."""
        result = format_duration(0.250)
        assert "250ms" in result

    def test_format_duration_negative(self) -> None:
        """Test negative duration."""
        result = format_duration(-10)
        assert result == "invalid"


class TestFormatSize:
    """Test format_size function."""

    def test_format_size_bytes(self) -> None:
        """Test formatting bytes."""
        result = format_size(500)
        assert "500.00 B" in result

    def test_format_size_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        result = format_size(2048)
        assert "2.00 KB" in result

    def test_format_size_megabytes(self) -> None:
        """Test formatting megabytes."""
        result = format_size(1234567)
        assert "1.18 MB" in result

    def test_format_size_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        result = format_size(2 * 1024**3)
        assert "2.00 GB" in result

    def test_format_size_precision(self) -> None:
        """Test size formatting with custom precision."""
        result = format_size(1234567, precision=1)
        assert "1.2 MB" in result


class TestFormatList:
    """Test format_list function."""

    def test_format_list_bullet(self) -> None:
        """Test bullet list formatting."""
        items = ["apple", "banana", "cherry"]
        result = format_list(items, style="bullet")
        assert "• apple" in result
        assert "• banana" in result
        assert "• cherry" in result

    def test_format_list_numbered(self) -> None:
        """Test numbered list formatting."""
        items = ["first", "second", "third"]
        result = format_list(items, style="numbered")
        assert "1. first" in result
        assert "2. second" in result
        assert "3. third" in result

    def test_format_list_comma(self) -> None:
        """Test comma-separated list."""
        items = ["a", "b", "c"]
        result = format_list(items, style="comma")
        assert result == "a, b, c"

    def test_format_list_newline(self) -> None:
        """Test newline-separated list."""
        items = ["line1", "line2"]
        result = format_list(items, style="newline")
        assert "line1" in result
        assert "line2" in result
        assert "\n" in result

    def test_format_list_with_prefix(self) -> None:
        """Test list with prefix."""
        items = ["a", "b"]
        result = format_list(items, style="bullet", prefix="  ")
        assert "  •" in result

    def test_format_list_empty(self) -> None:
        """Test empty list."""
        result = format_list([])
        assert result == ""


class TestFormatKeyValuePairs:
    """Test format_key_value_pairs function."""

    def test_format_key_value_basic(self) -> None:
        """Test basic key-value formatting."""
        pairs = {"name": "Alice", "age": 30}
        result = format_key_value_pairs(pairs)
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_format_key_value_custom_indent(self) -> None:
        """Test key-value with custom indent."""
        pairs = {"key": "value"}
        result = format_key_value_pairs(pairs, indent=4)
        assert "    key" in result

    def test_format_key_value_custom_separator(self) -> None:
        """Test key-value with custom separator."""
        pairs = {"key": "value"}
        result = format_key_value_pairs(pairs, separator=" = ")
        assert "key = value" in result

    def test_format_key_value_empty(self) -> None:
        """Test empty dictionary."""
        result = format_key_value_pairs({})
        assert result == ""


class TestFormatCodeBlock:
    """Test format_code_block function."""

    def test_format_code_basic(self) -> None:
        """Test basic code block formatting."""
        code = "x = 1\nprint(x)"
        result = format_code_block(code)
        assert "x = 1" in result
        assert "print(x)" in result

    def test_format_code_line_numbers(self) -> None:
        """Test code block with line numbers."""
        code = "x = 1\nprint(x)"
        result = format_code_block(code, line_numbers=True)
        assert "1 |" in result
        assert "2 |" in result

    def test_format_code_indent(self) -> None:
        """Test code block with indentation."""
        code = "x = 1"
        result = format_code_block(code, indent=4)
        assert "    x" in result

    def test_format_code_language(self) -> None:
        """Test code block with language hint (currently unused)."""
        code = "x = 1"
        result = format_code_block(code, language="python")
        # Language parameter exists but doesn't affect output currently
        assert "x = 1" in result
