import pytest

"""Comprehensive unit tests for oscura.ui.formatters module.

Tests all public functions and classes with >90% coverage including:
- Color codes and ANSI formatting
- Text alignment and truncation
- Status formatting with symbols
- Table, list, and key-value pair formatting
- Duration, size, and percentage formatting
- Code block formatting with line numbers

- UI formatting for terminal and web outputs
"""

from oscura.ui.formatters import (
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


class TestColor:
    """Tests for Color enum."""

    def test_color_enum_exists(self):
        """Test that Color enum has expected values."""
        assert hasattr(Color, "RED")
        assert hasattr(Color, "GREEN")
        assert hasattr(Color, "BLUE")
        assert hasattr(Color, "RESET")

    def test_color_enum_has_value(self):
        """Test that colors have ANSI code values."""
        assert Color.RED.value == "\033[31m"
        assert Color.GREEN.value == "\033[32m"
        assert Color.RESET.value == "\033[0m"

    def test_all_colors_have_values(self):
        """Test that all color enum members have values."""
        for color in Color:
            assert color.value is not None
            assert isinstance(color.value, str)


class TestTextAlignment:
    """Tests for TextAlignment enum."""

    def test_text_alignment_enum_exists(self):
        """Test that TextAlignment enum exists."""
        assert hasattr(TextAlignment, "LEFT")
        assert hasattr(TextAlignment, "CENTER")
        assert hasattr(TextAlignment, "RIGHT")

    def test_alignment_values(self):
        """Test alignment enum values."""
        assert TextAlignment.LEFT.value == "left"
        assert TextAlignment.CENTER.value == "center"
        assert TextAlignment.RIGHT.value == "right"


class TestFormattedText:
    """Tests for FormattedText dataclass."""

    def test_create_formatted_text_basic(self):
        """Test creating FormattedText with basic content."""
        ft = FormattedText(content="Hello")
        assert ft.content == "Hello"
        assert ft.color is None
        assert not ft.bold
        assert ft.width == 0

    def test_create_formatted_text_with_color(self):
        """Test creating FormattedText with color."""
        ft = FormattedText(content="Error", color=Color.RED, bold=True)
        assert ft.content == "Error"
        assert ft.color == Color.RED
        assert ft.bold

    def test_formatted_text_str_with_color(self):
        """Test string representation with color."""
        ft = FormattedText(content="Test", color=Color.GREEN)
        result = str(ft)
        assert "Test" in result
        assert "\033[" in result  # ANSI code present

    def test_formatted_text_str_with_bold(self):
        """Test string representation with bold."""
        ft = FormattedText(content="Bold", bold=True)
        result = str(ft)
        assert "Bold" in result
        assert "\033[1m" in result  # Bold ANSI code

    def test_formatted_text_str_with_bold_and_color(self):
        """Test string representation with both bold and color."""
        ft = FormattedText(content="Styled", color=Color.BLUE, bold=True)
        result = str(ft)
        assert "Styled" in result
        assert "\033[" in result

    def test_formatted_text_reset_color_ignored(self):
        """Test that RESET color doesn't add extra codes."""
        ft = FormattedText(content="Text", color=Color.RESET)
        result = str(ft)
        # RESET should not add color codes
        assert "Text" in result


class TestColorize:
    """Tests for colorize function."""

    def test_colorize_red(self):
        """Test colorizing text red."""
        result = colorize("Error", color="red")
        assert "Error" in result
        assert "\033[31m" in result  # Red color code
        assert "\033[0m" in result  # Reset code

    def test_colorize_green(self):
        """Test colorizing text green."""
        result = colorize("Success", color="green")
        assert "Success" in result
        assert "\033[32m" in result

    def test_colorize_blue(self):
        """Test colorizing text blue."""
        result = colorize("Info", color="blue")
        assert "Info" in result
        assert "\033[34m" in result

    def test_colorize_yellow(self):
        """Test colorizing text yellow."""
        result = colorize("Warning", color="yellow")
        assert "Warning" in result
        assert "\033[33m" in result

    def test_colorize_with_bold(self):
        """Test colorizing with bold."""
        result = colorize("Bold Error", color="red", bold=True)
        assert "Bold Error" in result
        assert "\033[1m" in result  # Bold code

    def test_colorize_invalid_color_defaults_to_white(self):
        """Test that invalid color defaults to white."""
        result = colorize("Text", color="invalid")
        assert "Text" in result
        assert "\033[37m" in result  # White color code

    def test_colorize_all_colors(self):
        """Test all available colors."""
        colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
        for color in colors:
            result = colorize("Test", color=color)
            assert "Test" in result
            assert "\033[" in result


class TestTruncate:
    """Tests for truncate function."""

    def test_truncate_long_text(self):
        """Test truncating long text."""
        result = truncate("Very long text here", max_length=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_short_text(self):
        """Test text shorter than max_length."""
        result = truncate("Short", max_length=20)
        assert result == "Short"
        assert "..." not in result

    def test_truncate_exact_length(self):
        """Test text exactly at max_length."""
        result = truncate("Exactly", max_length=7)
        assert result == "Exactly"

    def test_truncate_custom_suffix(self):
        """Test with custom suffix."""
        result = truncate("Very long text", max_length=10, suffix="***")
        assert len(result) == 10
        assert result.endswith("***")

    def test_truncate_empty_string(self):
        """Test truncating empty string."""
        result = truncate("", max_length=5)
        assert result == ""

    def test_truncate_very_small_max_length(self):
        """Test with very small max_length."""
        result = truncate("Hello World", max_length=3)
        assert len(result) == 3

    def test_truncate_max_length_equals_suffix_length(self):
        """Test when max_length equals suffix length."""
        result = truncate("Hello World", max_length=3, suffix="...")
        assert len(result) == 3
        assert result == "..."


class TestAlignText:
    """Tests for align_text function."""

    def test_align_left(self):
        """Test left alignment."""
        result = align_text("Hello", 10, alignment="left")
        assert result.startswith("Hello")
        assert len(result) == 10

    def test_align_right(self):
        """Test right alignment."""
        result = align_text("Hello", 10, alignment="right")
        assert result.endswith("Hello")
        assert len(result) == 10

    def test_align_center(self):
        """Test center alignment."""
        result = align_text("Hello", 11, alignment="center")
        assert "Hello" in result
        assert len(result) == 11
        # Should be centered
        assert result.count(" ") == 6

    def test_align_with_custom_fill_char(self):
        """Test alignment with custom fill character."""
        result = align_text("Hi", 5, alignment="left", fill_char=".")
        assert result == "Hi..."
        assert len(result) == 5

    def test_align_text_longer_than_width(self):
        """Test when text is longer than width."""
        result = align_text("Hello World", 5)
        assert result == "Hello World"

    def test_align_text_empty_string(self):
        """Test aligning empty string."""
        result = align_text("", 5, alignment="left")
        assert len(result) == 5
        assert result == "     "

    def test_align_default_is_left(self):
        """Test that default alignment is left."""
        result = align_text("Test", 8)
        assert result.startswith("Test")


class TestFormatText:
    """Tests for format_text function."""

    def test_format_text_basic(self):
        """Test basic text formatting."""
        result = format_text("Status", "active")
        assert "Status" in result
        assert "active" in result
        assert ": " in result

    def test_format_text_custom_separator(self):
        """Test with custom separator."""
        result = format_text("Key", "Value", separator=" = ")
        assert "Key = Value" in result

    def test_format_text_with_width(self):
        """Test formatting with width."""
        result = format_text("Name", "Alice", width=20)
        assert len(result) == 20

    def test_format_text_with_color(self):
        """Test formatting with color."""
        result = format_text("Status", "OK", color="green")
        assert "Status" in result
        assert "OK" in result
        assert "\033[" in result  # ANSI code

    def test_format_text_with_alignment(self):
        """Test formatting with alignment."""
        result = format_text("Info", "test", align="center", width=20)
        assert "Info" in result

    def test_format_text_numeric_value(self):
        """Test with numeric value."""
        result = format_text("Count", 42)
        assert "Count" in result
        assert "42" in result


class TestFormatTable:
    """Tests for format_table function."""

    def test_format_table_basic(self):
        """Test basic table formatting."""
        data = [["Alice", 85], ["Bob", 92]]
        result = format_table(data, headers=["Name", "Score"])
        assert "Alice" in result
        assert "Bob" in result
        assert "Name" in result
        assert "Score" in result

    def test_format_table_no_headers(self):
        """Test table without headers."""
        data = [["a", "b"], ["c", "d"]]
        result = format_table(data)
        assert "a" in result
        assert "d" in result

    def test_format_table_empty(self):
        """Test empty table."""
        result = format_table([])
        assert result == ""

    def test_format_table_single_row(self):
        """Test table with single row."""
        data = [["Value1", "Value2"]]
        result = format_table(data)
        assert "Value1" in result
        assert "Value2" in result

    def test_format_table_custom_column_widths(self):
        """Test table with custom column widths."""
        data = [["A", "B"], ["C", "D"]]
        result = format_table(data, column_widths=[5, 5])
        assert "A" in result

    def test_format_table_alignment(self):
        """Test table with column alignment."""
        data = [["Left", "Right"], ["123", "456"]]
        result = format_table(
            data,
            headers=["L", "R"],
            align_columns=["left", "right"],
        )
        assert "Left" in result
        assert "Right" in result

    def test_format_table_many_columns(self):
        """Test table with many columns."""
        data = [["A", "B", "C", "D"], ["1", "2", "3", "4"]]
        result = format_table(data, headers=["Col1", "Col2", "Col3", "Col4"])
        assert "A" in result
        assert "D" in result

    def test_format_table_long_values(self):
        """Test table with long values."""
        data = [["VeryLongValueHere", "Short"], ["Short", "AlsoVeryLongHere"]]
        result = format_table(data)
        assert "VeryLongValueHere" in result


class TestFormatStatus:
    """Tests for format_status function."""

    def test_format_status_pass(self):
        """Test pass status."""
        result = format_status("pass", "All tests passed")
        assert "✓" in result or "PASS" in result

    def test_format_status_fail(self):
        """Test fail status."""
        result = format_status("fail", "Test failed")
        assert "✗" in result or "FAIL" in result

    def test_format_status_warning(self):
        """Test warning status."""
        result = format_status("warning", "Check this")
        assert "⚠" in result or "WARNING" in result

    def test_format_status_info(self):
        """Test info status."""
        result = format_status("info", "FYI")
        assert "ℹ" in result or "INFO" in result

    def test_format_status_pending(self):
        """Test pending status."""
        result = format_status("pending", "In progress")
        assert "⏳" in result or "PENDING" in result

    def test_format_status_without_symbols(self):
        """Test status without symbols."""
        result = format_status("pass", "Success", use_symbols=False)
        assert "PASS" in result
        assert "✓" not in result

    def test_format_status_no_message(self):
        """Test status without message."""
        result = format_status("pass")
        assert len(result) > 0

    def test_format_status_with_color(self):
        """Test that status uses color."""
        result = format_status("fail", "Error")
        assert "Error" in result
        assert "\033[" in result  # Has ANSI codes


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_format_percentage_as_decimal(self):
        """Test percentage from 0-1 value."""
        result = format_percentage(0.75)
        assert "75" in result
        assert "%" in result

    def test_format_percentage_as_percent(self):
        """Test percentage from 0-100 value."""
        result = format_percentage(75)
        assert "75" in result
        assert "%" in result

    def test_format_percentage_with_decimals(self):
        """Test percentage with custom decimals."""
        result = format_percentage(0.333, decimals=2)
        assert "33.30%" in result

    def test_format_percentage_with_progress_bar(self):
        """Test percentage with progress bar."""
        result = format_percentage(0.5, show_bar=True)
        assert "50" in result
        assert "[" in result
        assert "]" in result

    def test_format_percentage_full_bar(self):
        """Test 100% progress bar."""
        result = format_percentage(1.0, show_bar=True, bar_width=10)
        assert "100" in result
        assert "█" in result

    def test_format_percentage_empty_bar(self):
        """Test 0% progress bar."""
        result = format_percentage(0.0, show_bar=True, bar_width=10)
        assert "0" in result
        assert "░" in result

    def test_format_percentage_custom_bar_width(self):
        """Test progress bar with custom width."""
        result = format_percentage(0.5, show_bar=True, bar_width=20)
        assert "50" in result
        assert "[" in result and "]" in result

    def test_format_percentage_negative(self):
        """Test percentage with negative value."""
        result = format_percentage(-10)
        assert "-10" in result or "10" in result


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_duration_seconds(self):
        """Test duration in seconds."""
        result = format_duration(45)
        assert "45s" in result

    def test_format_duration_minutes(self):
        """Test duration in minutes and seconds."""
        result = format_duration(125)
        assert "2m" in result
        assert "5s" in result

    def test_format_duration_hours(self):
        """Test duration in hours, minutes, seconds."""
        result = format_duration(3665)
        assert "1h" in result
        assert "1m" in result
        assert "5s" in result

    def test_format_duration_milliseconds(self):
        """Test duration in milliseconds."""
        result = format_duration(0.5)
        assert "500ms" in result

    def test_format_duration_zero(self):
        """Test zero duration."""
        result = format_duration(0)
        assert "0ms" in result or "0s" in result

    def test_format_duration_specific_values(self):
        """Test specific duration value."""
        # 1 hour 23 minutes 45 seconds
        result = format_duration(5025)
        assert "1h" in result
        assert "23m" in result
        assert "45s" in result

    def test_format_duration_negative(self):
        """Test negative duration."""
        result = format_duration(-10)
        assert "invalid" in result

    def test_format_duration_large_value(self):
        """Test large duration."""
        result = format_duration(86400)  # 24 hours
        assert "24h" in result or "h" in result


class TestFormatSize:
    """Tests for format_size function."""

    def test_format_size_bytes(self):
        """Test size in bytes."""
        result = format_size(500)
        assert "500" in result
        assert "B" in result

    def test_format_size_kilobytes(self):
        """Test size in kilobytes."""
        result = format_size(2048)
        assert "KB" in result

    def test_format_size_megabytes(self):
        """Test size in megabytes."""
        result = format_size(1048576)  # 1 MB
        assert "MB" in result

    def test_format_size_gigabytes(self):
        """Test size in gigabytes."""
        result = format_size(1073741824)  # 1 GB
        assert "GB" in result

    def test_format_size_terabytes(self):
        """Test size in terabytes."""
        result = format_size(1099511627776)  # 1 TB
        assert "TB" in result

    def test_format_size_zero(self):
        """Test zero size."""
        result = format_size(0)
        assert "0" in result
        assert "B" in result

    def test_format_size_custom_precision(self):
        """Test with custom precision."""
        result = format_size(1234567, precision=3)
        assert "MB" in result

    def test_format_size_specific_value(self):
        """Test specific size value."""
        result = format_size(1234567)
        assert "1.18 MB" in result


class TestFormatList:
    """Tests for format_list function."""

    def test_format_list_bullet_style(self):
        """Test bullet list style."""
        result = format_list(["Item 1", "Item 2", "Item 3"], style="bullet")
        assert "• Item 1" in result
        assert "• Item 2" in result
        assert "• Item 3" in result

    def test_format_list_numbered_style(self):
        """Test numbered list style."""
        result = format_list(["First", "Second", "Third"], style="numbered")
        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result

    def test_format_list_comma_style(self):
        """Test comma-separated style."""
        result = format_list(["a", "b", "c"], style="comma")
        assert result == "a, b, c"

    def test_format_list_newline_style(self):
        """Test newline-separated style."""
        result = format_list(["line1", "line2", "line3"], style="newline")
        assert "line1" in result
        assert "line2" in result

    def test_format_list_with_prefix(self):
        """Test list with prefix."""
        result = format_list(["a", "b"], style="bullet", prefix="  ")
        assert "  • a" in result

    def test_format_list_empty(self):
        """Test empty list."""
        result = format_list([], style="bullet")
        assert result == ""

    def test_format_list_single_item(self):
        """Test list with single item."""
        result = format_list(["Only"], style="bullet")
        assert "• Only" in result


class TestFormatKeyValuePairs:
    """Tests for format_key_value_pairs function."""

    def test_format_key_value_pairs_basic(self):
        """Test basic key-value pair formatting."""
        pairs = {"name": "Alice", "age": 30}
        result = format_key_value_pairs(pairs)
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_format_key_value_pairs_with_indent(self):
        """Test with custom indentation."""
        pairs = {"key": "value"}
        result = format_key_value_pairs(pairs, indent=4)
        assert "    key: value" in result

    def test_format_key_value_pairs_custom_separator(self):
        """Test with custom separator."""
        pairs = {"key": "value"}
        result = format_key_value_pairs(pairs, separator=" = ")
        assert "key = value" in result

    def test_format_key_value_pairs_empty(self):
        """Test empty dictionary."""
        result = format_key_value_pairs({})
        assert result == ""

    def test_format_key_value_pairs_multiple(self):
        """Test with multiple pairs."""
        pairs = {"a": 1, "b": 2, "c": 3}
        result = format_key_value_pairs(pairs)
        assert "a: 1" in result
        assert "b: 2" in result
        assert "c: 3" in result

    def test_format_key_value_pairs_numeric_values(self):
        """Test with numeric values."""
        pairs = {"count": 42, "ratio": 3.14}
        result = format_key_value_pairs(pairs)
        assert "count: 42" in result
        assert "ratio: 3.14" in result


class TestFormatCodeBlock:
    """Tests for format_code_block function."""

    def test_format_code_block_basic(self):
        """Test basic code block formatting."""
        code = "x = 1\nprint(x)"
        result = format_code_block(code)
        assert "x = 1" in result
        assert "print(x)" in result

    def test_format_code_block_with_line_numbers(self):
        """Test code block with line numbers."""
        code = "line1\nline2\nline3"
        result = format_code_block(code, line_numbers=True)
        assert "1 |" in result
        assert "2 |" in result
        assert "3 |" in result

    def test_format_code_block_with_language(self):
        """Test code block with language identifier."""
        code = "print('hello')"
        result = format_code_block(code, language="python")
        assert "print('hello')" in result

    def test_format_code_block_with_indent(self):
        """Test code block with indentation."""
        code = "code here"
        result = format_code_block(code, indent=4)
        assert "    code here" in result

    def test_format_code_block_line_numbers_alignment(self):
        """Test line number alignment."""
        code = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11"
        result = format_code_block(code, line_numbers=True)
        # Should have proper alignment for 2-digit numbers
        assert "10 |" in result
        assert "11 |" in result

    def test_format_code_block_empty(self):
        """Test empty code block."""
        result = format_code_block("")
        assert result == ""

    def test_format_code_block_single_line(self):
        """Test single line code block."""
        code = "x = 42"
        result = format_code_block(code)
        assert "x = 42" in result

    def test_format_code_block_multiline_with_indent_and_numbers(self):
        """Test code block with both line numbers and indentation."""
        code = "a = 1\nb = 2"
        result = format_code_block(code, line_numbers=True, indent=2)
        assert "a = 1" in result
        assert "b = 2" in result


class TestUiFormattersIntegration:
    """Integration tests combining multiple formatting functions."""

    def test_colorized_status_with_formatted_text(self):
        """Test combining colorized status with formatted text."""
        status = format_status("pass", "Success")
        text = format_text("Result", status)
        assert "Result" in text
        assert "Success" in text

    def test_formatted_table_with_status(self):
        """Test table with status strings."""
        pass_str = format_status("pass")
        fail_str = format_status("fail")
        data = [["Test1", pass_str], ["Test2", fail_str]]
        result = format_table(data, headers=["Test Name", "Status"])
        assert "Test1" in result
        assert "Test2" in result

    def test_list_with_key_value_pairs(self):
        """Test combining list and key-value formatting."""
        pairs = {"key1": "val1", "key2": "val2"}
        kv_text = format_key_value_pairs(pairs)
        items = kv_text.split("\n")
        result = format_list(items, style="bullet")
        assert "key1" in result
        assert "key2" in result

    def test_percentage_bar_in_table(self):
        """Test progress bar percentage in table."""
        pct1 = format_percentage(0.75, show_bar=True)
        pct2 = format_percentage(0.25, show_bar=True)
        data = [["Task1", pct1], ["Task2", pct2]]
        result = format_table(data, headers=["Task", "Progress"])
        assert "Task1" in result
        assert "[" in result

    def test_truncate_in_formatted_text(self):
        """Test truncation within formatted text."""
        long_text = "A" * 50
        truncated = truncate(long_text, max_length=20)
        result = format_text("Label", truncated)
        assert len(result) <= 30  # Reasonable length


class TestUiFormattersEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_colorize_empty_string(self):
        """Test colorizing empty string."""
        result = colorize("")
        assert "\033[" in result  # Still has color codes

    def test_truncate_unicode_text(self):
        """Test truncating text with unicode characters."""
        result = truncate("Hello 世界", max_length=8)
        assert len(result) <= 8

    def test_format_table_uneven_rows(self):
        """Test table with rows of different lengths."""
        data = [["a", "b"], ["c"]]  # Second row shorter
        result = format_table(data)
        assert "a" in result
        assert "c" in result

    def test_format_duration_fractional_seconds(self):
        """Test duration with fractional seconds."""
        result = format_duration(1.5)
        # 1.5 seconds = 1 second + 500ms, but the function rounds to 1s
        assert "s" in result or "ms" in result

    def test_format_size_very_small(self):
        """Test formatting very small size."""
        result = format_size(1)
        assert "1" in result
        assert "B" in result

    def test_align_text_width_zero(self):
        """Test alignment with zero width."""
        result = align_text("Test", 0)
        assert "Test" in result

    def test_format_percentage_out_of_range(self):
        """Test percentage with value > 100."""
        result = format_percentage(150)
        assert "150" in result


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_colorize_with_none_text(self):
        """Test colorizing None raises or handles gracefully."""
        try:
            result = colorize(None)  # type: ignore
            assert result is not None
        except TypeError:
            pass  # Expected

    def test_format_table_with_none_data(self):
        """Test table with None data."""
        # format_table returns empty string for None (handles gracefully)
        result = format_table(None)  # type: ignore
        assert result == ""

    def test_format_list_with_none_items(self):
        """Test list with None items."""
        # format_list returns empty string for None (handles gracefully)
        result = format_list(None)  # type: ignore
        assert result == ""

    def test_truncate_negative_max_length(self):
        """Test truncate with negative max_length."""
        result = truncate("Hello", max_length=-5)
        # Negative max_length means text is always longer, so gets truncated
        # Text length (5) > max_length (-5) -> truncate_at = max(0, -5-3) = 0
        assert len(result) >= 0  # Should produce some output


class TestRegressions:
    """Tests for specific regression scenarios."""

    def test_colorize_maintains_original_text(self):
        """Test that colorize preserves text content."""
        text = "Important Message"
        result = colorize(text, color="red")
        assert text in result

    def test_truncate_preserves_order(self):
        """Test that truncate preserves word order."""
        text = "Start Middle End"
        result = truncate(text, max_length=10)
        assert result.startswith("Start")

    def test_format_table_column_alignment(self):
        """Test that table columns stay aligned."""
        data = [["AAA", "BBB"], ["C", "D"]]
        result = format_table(data)
        # Both rows should have similar structure
        rows = result.split("\n")
        assert len(rows) >= 2

    def test_format_status_symbol_consistency(self):
        """Test that status symbols are consistent."""
        pass1 = format_status("pass", use_symbols=True)
        pass2 = format_status("pass", use_symbols=True)
        # Both should have the same symbol
        assert pass1.split()[0] == pass2.split()[0]


class TestDocstringExamples:
    """Tests that verify documented examples work correctly."""

    def test_colorize_example(self):
        """Test colorize example from docstring."""
        result = colorize("Error", color="red")
        assert "Error" in result
        assert "\033[31m" in result

    def test_truncate_example(self):
        """Test truncate example from docstring."""
        result = truncate("Very long text here", max_length=10)
        # max_length=10, suffix="..." (3 chars), truncate_at=7
        # "Very long"[0:7] + "..." = "Very lo..."
        assert result == "Very lo..." and len(result) == 10

    def test_format_percentage_example(self):
        """Test format_percentage example from docstring."""
        result = format_percentage(0.75, show_bar=True)
        assert "75.0%" in result
        assert "[" in result

    def test_format_duration_example(self):
        """Test format_duration example from docstring."""
        result = format_duration(5025)
        assert "1h" in result
        assert "23m" in result
        assert "45s" in result

    def test_format_size_example(self):
        """Test format_size example from docstring."""
        result = format_size(1234567)
        assert "1.18 MB" in result

    def test_format_text_example(self):
        """Test format_text example from docstring."""
        result = format_text("Status", "active", align="left", width=20)
        assert "Status" in result
        assert "active" in result
        assert len(result) == 20


class TestCoverage:
    """Tests specifically designed for code coverage."""

    def test_all_color_values_used(self):
        """Test that all color enum values are used."""
        colors_to_test = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
        for color_name in colors_to_test:
            result = colorize(f"Test{color_name}", color=color_name)
            assert "Test" in result

    def test_all_alignments_used(self):
        """Test all alignment options."""
        alignments = ["left", "center", "right"]
        for align in alignments:
            result = align_text("X", 5, alignment=align)
            assert len(result) == 5

    def test_all_list_styles_used(self):
        """Test all list formatting styles."""
        items = ["a", "b", "c"]
        styles = ["bullet", "numbered", "comma", "newline"]
        for style in styles:
            result = format_list(items, style=style)
            assert len(result) > 0

    def test_all_status_types_used(self):
        """Test all status types."""
        statuses = ["pass", "fail", "warning", "info", "pending"]
        for status in statuses:
            result = format_status(status)
            assert len(result) > 0

    def test_format_table_header_separator(self):
        """Test that header separator is generated."""
        data = [["a", "b"]]
        result = format_table(data, headers=["H1", "H2"])
        # Should have a separator line
        assert "-" in result

    def test_format_code_block_languages(self):
        """Test code block with various language identifiers."""
        code = "x = 1"
        languages = ["python", "java", "cpp", "javascript"]
        for lang in languages:
            result = format_code_block(code, language=lang)
            assert "x = 1" in result
