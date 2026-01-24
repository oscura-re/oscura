"""Output formatting utilities for demonstrations."""

from __future__ import annotations

from typing import Any


def format_value(value: Any, unit: str = "", precision: int = 4) -> str:
    """Format a value with optional unit.

    Args:
        value: Value to format
        unit: Optional unit string
        precision: Number of significant figures

    Returns:
        Formatted string

    Example:
        format_value(0.001234, "V", precision=3)  # "1.23e-03 V"
        format_value(1234.5, "Hz")  # "1.235e+03 Hz"
    """
    if isinstance(value, (int, float)):
        if abs(value) >= 1000 or abs(value) < 0.001:
            formatted = f"{value:.{precision - 1}e}"
        else:
            formatted = f"{value:.{precision}g}"
    else:
        formatted = str(value)

    if unit:
        return f"{formatted} {unit}"
    return formatted


def format_percentage(value: float, precision: int = 2) -> str:
    """Format a value as a percentage.

    Args:
        value: Value (0.0 to 1.0)
        precision: Decimal places

    Returns:
        Formatted percentage string

    Example:
        format_percentage(0.85)  # "85.00%"
    """
    return f"{value * 100:.{precision}f}%"


def format_table(rows: list[list[str]], headers: list[str] | None = None) -> str:
    """Format data as an ASCII table.

    Args:
        rows: List of rows, each row is a list of strings
        headers: Optional header row

    Returns:
        Formatted table string

    Example:
        rows = [["Alice", "30"], ["Bob", "25"]]
        headers = ["Name", "Age"]
        print(format_table(rows, headers))
    """
    # Combine headers and rows
    all_rows = [headers] if headers else []
    all_rows.extend(rows)

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(all_rows[0]))]

    # Format rows
    formatted_rows = []
    for i, row in enumerate(all_rows):
        formatted_row = " | ".join(
            str(item).ljust(width) for item, width in zip(row, col_widths, strict=True)
        )
        formatted_rows.append(formatted_row)

        # Add separator after header
        if i == 0 and headers:
            separator = "-+-".join("-" * width for width in col_widths)
            formatted_rows.append(separator)

    return "\n".join(formatted_rows)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Example:
        format_duration(0.001)   # "1.00 ms"
        format_duration(1.5)     # "1.50 s"
        format_duration(90)      # "1m 30s"
    """
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"


def format_size(bytes: int) -> str:
    """Format byte size in human-readable form.

    Args:
        bytes: Size in bytes

    Returns:
        Formatted size string

    Example:
        format_size(1024)       # "1.00 KB"
        format_size(1536000)    # "1.46 MB"
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes)
    unit_idx = 0

    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1

    return f"{size:.2f} {units[unit_idx]}"


def format_list(items: list[str], bullet: str = "-") -> str:
    """Format a list with bullets.

    Args:
        items: List of items
        bullet: Bullet character

    Returns:
        Formatted list string

    Example:
        format_list(["Item 1", "Item 2"])
        # "  - Item 1\n  - Item 2"
    """
    return "\n".join(f"  {bullet} {item}" for item in items)
