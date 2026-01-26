"""UI-specific formatting utilities for Oscura displays.

This module provides formatting functions tailored for user interface display,
including text alignment, truncation, color codes, and structured output.

- UI formatting for terminal and web outputs

Example:
    >>> from oscura.jupyter.ui.formatters import format_text, truncate, colorize
    >>> format_text("Status", "active", align="left", width=20)
    '  Status: active'
    >>> truncate("Very long text here", max_length=10)
    'Very lon...'
    >>> colorize("Success", color="green")
    '\x1b[32mSuccess\x1b[0m'
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias

# Type alias for color names accepted by colorize()
ColorName: TypeAlias = Literal[
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
]


class Color(Enum):
    """ANSI color codes for terminal output."""

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"


class TextAlignment(Enum):
    """Text alignment options."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class FormattedText:
    """Container for formatted text output.

    Attributes:
        content: The formatted text content.
        color: Optional color code.
        bold: Whether text should be bold.
        width: Display width of the text.
    """

    content: str
    color: Color | None = None
    bold: bool = False
    width: int = 0

    def __str__(self) -> str:
        """Get string representation with ANSI codes."""
        result = self.content
        if self.bold:
            result = f"\033[1m{result}\033[0m"
        if self.color and self.color != Color.RESET:
            result = f"{self.color.value}{result}{Color.RESET.value}"
        return result


def colorize(
    text: str,
    color: ColorName = "white",
    bold: bool = False,
) -> str:
    """Apply ANSI color codes to text.

    Args:
        text: Text to colorize.
        color: Color name (black, red, green, yellow, blue, magenta, cyan, white).
        bold: Apply bold formatting.

    Returns:
        Text with ANSI color codes.

    Example:
        >>> colorize("Error", color="red")
        '\x1b[31mError\x1b[0m'
    """
    try:
        color_enum = Color[color.upper()]
    except KeyError:
        color_enum = Color.WHITE

    result = text
    if bold:
        result = f"\033[1m{result}\033[0m"
    result = f"{color_enum.value}{result}{Color.RESET.value}"
    return result


def truncate(
    text: str,
    max_length: int,
    suffix: str = "...",
) -> str:
    """Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to append if truncated (default "...").

    Returns:
        Truncated text.

    Example:
        >>> truncate("Very long text here", max_length=10)
        'Very lon...'
    """
    if len(text) <= max_length:
        return text

    # Account for suffix length
    truncate_at = max(0, max_length - len(suffix))
    return text[:truncate_at] + suffix


def align_text(
    text: str,
    width: int,
    alignment: Literal["left", "center", "right"] = "left",
    fill_char: str = " ",
) -> str:
    """Align text within a given width.

    Args:
        text: Text to align.
        width: Target width.
        alignment: How to align (left, center, right).
        fill_char: Character to use for padding.

    Returns:
        Aligned text.

    Example:
        >>> align_text("Hello", 10, "center")
        '   Hello  '
    """
    if len(text) >= width:
        return text

    if alignment == "center":
        return text.center(width, fill_char)
    elif alignment == "right":
        return text.rjust(width, fill_char)
    else:  # left
        return text.ljust(width, fill_char)


def format_text(
    label: str,
    value: Any,
    align: Literal["left", "center", "right"] = "left",
    width: int | None = None,
    separator: str = ": ",
    color: ColorName | None = None,
) -> str:
    """Format a label-value pair for display.

    Args:
        label: Label/key.
        value: Value to display.
        align: Text alignment.
        width: Total width for alignment (including label, separator, value).
        separator: Separator between label and value.
        color: Optional color for the value.

    Returns:
        Formatted text.

    Example:
        >>> format_text("Status", "active", align="left", width=20)
        '  Status: active'
    """
    result = f"{label}{separator}{value}"

    if color:
        result = f"{label}{separator}{colorize(str(value), color=color)}"

    if width:
        result = align_text(result, width, alignment=align)

    return result


def format_table(
    data: list[list[Any]],
    headers: list[str] | None = None,
    column_widths: list[int] | None = None,
    align_columns: list[Literal["left", "center", "right"]] | None = None,
) -> str:
    """Format data as a simple table.

    Args:
        data: List of rows (each row is a list of values).
        headers: Optional header row.
        column_widths: Optional column widths (auto-calculated if not provided).
        align_columns: Alignment for each column.

    Returns:
        Formatted table as string.

    Example:
        >>> data = [["Alice", 85], ["Bob", 92]]
        >>> format_table(data, headers=["Name", "Score"])
    """
    if not data:
        return ""

    # Calculate column widths
    num_cols = len(data[0]) if data else 0
    if headers:
        num_cols = len(headers)

    if column_widths is None:
        column_widths = []
        for col_idx in range(num_cols):
            max_width = 0
            if headers and col_idx < len(headers):
                max_width = len(str(headers[col_idx]))
            for row in data:
                if col_idx < len(row):
                    max_width = max(max_width, len(str(row[col_idx])))
            column_widths.append(max_width + 2)

    # Default alignment
    if align_columns is None:
        align_columns = ["left"] * num_cols

    output_lines = []

    # Add headers if provided
    if headers:
        header_cells = []
        for col_idx, header in enumerate(headers):
            width = column_widths[col_idx] if col_idx < len(column_widths) else 10
            aligned = align_text(str(header), width, align_columns[col_idx])
            header_cells.append(aligned)
        output_lines.append(" ".join(header_cells))
        # Add separator
        separator = "-" * sum(column_widths) + ("-" * (num_cols - 1))
        output_lines.append(separator)

    # Add data rows
    for row in data:
        row_cells = []
        for col_idx, cell in enumerate(row):
            width = column_widths[col_idx] if col_idx < len(column_widths) else 10
            aligned = align_text(str(cell), width, align_columns[col_idx])
            row_cells.append(aligned)
        output_lines.append(" ".join(row_cells))

    return "\n".join(output_lines)


def format_status(
    status: Literal["pass", "fail", "warning", "info", "pending"],
    message: str = "",
    use_symbols: bool = True,
) -> str:
    """Format a status message with optional symbol.

    Args:
        status: Status type (pass, fail, warning, info, pending).
        message: Optional message text.
        use_symbols: Use Unicode symbols or text.

    Returns:
        Formatted status string.

    Example:
        >>> format_status("pass", "All tests passed")
        '✓ All tests passed'
    """
    symbols = {
        "pass": "✓",
        "fail": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "pending": "⏳",
    }

    colors: dict[str, ColorName] = {
        "pass": "green",
        "fail": "red",
        "warning": "yellow",
        "info": "blue",
        "pending": "cyan",
    }

    # Get color with default, cast needed because dict.get default is str
    status_color: ColorName = colors.get(status, "white")

    if use_symbols:
        symbol = symbols.get(status, "•")
        colored_symbol = colorize(symbol, color=status_color)
        return f"{colored_symbol} {message}" if message else colored_symbol
    else:
        text = status.upper()
        return (
            colorize(f"{text}: {message}", color=status_color)
            if message
            else colorize(text, color=status_color)
        )


def format_percentage(
    value: float,
    decimals: int = 1,
    show_bar: bool = False,
    bar_width: int = 10,
) -> str:
    """Format a percentage with optional progress bar.

    Args:
        value: Percentage value (0-100 or 0-1).
        decimals: Decimal places.
        show_bar: Include ASCII progress bar.
        bar_width: Width of progress bar.

    Returns:
        Formatted percentage string.

    Example:
        >>> format_percentage(0.75, show_bar=True)
        '75.0% [████████  ]'
    """
    # Normalize to 0-100 range
    if value <= 1:
        pct = value * 100
    else:
        pct = value

    result = f"{pct:.{decimals}f}%"

    if show_bar and 0 <= pct <= 100:
        filled = int((pct / 100) * bar_width)
        bar = "[" + "█" * filled + "░" * (bar_width - filled) + "]"
        result = f"{result} {bar}"

    return result


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration (e.g., "1h 23m 45s").

    Example:
        >>> format_duration(5025)
        '1h 23m 45s'
    """
    if seconds < 0:
        return "invalid"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    elif secs > 0:
        return f"{secs}s"
    else:
        return f"{millis}ms"


def format_size(bytes_value: int, precision: int = 2) -> str:
    """Format byte size as human-readable string.

    Args:
        bytes_value: Size in bytes.
        precision: Decimal precision.

    Returns:
        Human-readable size (e.g., "1.23 MB").

    Example:
        >>> format_size(1234567)
        '1.18 MB'
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_value)

    for unit in units:
        if size < 1024:
            return f"{size:.{precision}f} {unit}"
        size /= 1024

    return f"{size:.{precision}f} PB"


def format_list(
    items: list[str],
    style: Literal["bullet", "numbered", "comma", "newline"] = "bullet",
    prefix: str = "",
) -> str:
    """Format a list of items with various styles.

    Args:
        items: List of items to format.
        style: Formatting style (bullet, numbered, comma, newline).
        prefix: Prefix for each item.

    Returns:
        Formatted list as string.

    Example:
        >>> format_list(["a", "b", "c"], style="bullet")
        '  • a\\n  • b\\n  • c'
    """
    if not items:
        return ""

    if style == "bullet":
        return "\n".join(f"{prefix}• {item}" for item in items)
    elif style == "numbered":
        return "\n".join(f"{prefix}{i}. {item}" for i, item in enumerate(items, 1))
    elif style == "comma":
        return ", ".join(items)
    else:  # newline
        return "\n".join(items)


def format_key_value_pairs(
    pairs: dict[str, Any],
    indent: int = 2,
    separator: str = ": ",
) -> str:
    """Format key-value pairs with indentation.

    Args:
        pairs: Dictionary of key-value pairs.
        indent: Indentation level (spaces).
        separator: Separator between key and value.

    Returns:
        Formatted key-value pairs as string.

    Example:
        >>> format_key_value_pairs({"name": "Alice", "age": 30})
        '  name: Alice\\n  age: 30'
    """
    if not pairs:
        return ""

    indent_str = " " * indent
    lines = []
    for key, value in pairs.items():
        lines.append(f"{indent_str}{key}{separator}{value}")
    return "\n".join(lines)


def format_code_block(
    code: str,
    line_numbers: bool = False,
    indent: int = 0,
    language: str | None = None,
) -> str:
    """Format code with optional line numbers and indentation.

    Args:
        code: Code content.
        line_numbers: Include line numbers.
        indent: Indentation level.
        language: Programming language for syntax highlighting (currently unused).

    Returns:
        Formatted code block.

    Example:
        >>> format_code_block("x = 1\\nprint(x)", line_numbers=True)
    """
    indent_str = " " * indent
    lines = code.split("\n")

    if line_numbers:
        max_num_width = len(str(len(lines)))
        formatted_lines = []
        for num, line in enumerate(lines, 1):
            formatted_lines.append(f"{indent_str}{num:>{max_num_width}} | {line}")
        return "\n".join(formatted_lines)
    else:
        return "\n".join(f"{indent_str}{line}" for line in lines)


__all__ = [
    "Color",
    "FormattedText",
    "TextAlignment",
    "align_text",
    "colorize",
    "format_code_block",
    "format_duration",
    "format_key_value_pairs",
    "format_list",
    "format_percentage",
    "format_size",
    "format_status",
    "format_table",
    "format_text",
    "truncate",
]
