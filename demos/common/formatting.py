"""Console output formatting utilities for demos.

This module provides consistent terminal output formatting for all demos,
including colored output, section headers, and result formatting.

Usage:
    from demos.common.formatting import print_header, print_success, print_result

    print_header("OSCURA DEMO")
    print_success("Test passed!")
    print_result("Frequency", 1000.5, "Hz")
"""

from __future__ import annotations

from typing import Any

# ANSI color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def print_header(title: str, width: int = 80) -> None:
    """Print major section header.

    Args:
        title: Header title text
        width: Total width of header
    """
    print(f"\n{BOLD}{BLUE}{'=' * width}{RESET}")
    print(f"{BOLD}{BLUE}{title:^{width}}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * width}{RESET}")


def print_subheader(title: str, width: int = 60) -> None:
    """Print subsection header.

    Args:
        title: Subheader title text
        width: Total width of header
    """
    print(f"\n{BOLD}--- {title} ---{RESET}")


def print_success(message: str) -> None:
    """Print success message with green checkmark.

    Args:
        message: Success message text
    """
    print(f"{GREEN}[PASS]{RESET} {message}")


def print_info(message: str) -> None:
    """Print informational message.

    Args:
        message: Info message text
    """
    print(f"{BLUE}[INFO]{RESET} {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow indicator.

    Args:
        message: Warning message text
    """
    print(f"{YELLOW}[WARN]{RESET} {message}")


def print_error(message: str) -> None:
    """Print error message with red indicator.

    Args:
        message: Error message text
    """
    print(f"{RED}[FAIL]{RESET} {message}")


def print_result(name: str, value: Any, unit: str = "") -> None:
    """Print formatted measurement result.

    Args:
        name: Measurement name
        value: Measurement value
        unit: Unit string (optional)
    """
    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) > 1e6:
            formatted = f"{value:.4e}"
        else:
            formatted = f"{value:.6g}"
    elif isinstance(value, int):
        formatted = f"{value:,}"
    else:
        formatted = str(value)

    if unit:
        print(f"  {name}: {formatted} {unit}")
    else:
        print(f"  {name}: {formatted}")


def print_metric(
    name: str,
    value: float,
    unit: str,
    spec: str = "",
    name_width: int = 25,
    value_width: int = 12,
) -> None:
    """Print metric with optional specification reference.

    Args:
        name: Metric name
        value: Metric value
        unit: Unit string
        spec: Specification reference (optional)
        name_width: Width for name column
        value_width: Width for value column
    """
    if spec:
        print(f"  {name:<{name_width}}: {value:>{value_width}.4f} {unit:<6} {DIM}({spec}){RESET}")
    else:
        print(f"  {name:<{name_width}}: {value:>{value_width}.4f} {unit}")


def print_table(headers: list[str], rows: list[list[Any]], indent: int = 2) -> None:
    """Print formatted table.

    Args:
        headers: List of column headers
        rows: List of row data (each row is a list of values)
        indent: Spaces to indent table
    """
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    indent_str = " " * indent

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"{indent_str}{BOLD}{header_line}{RESET}")

    # Print separator
    separator = "-+-".join("-" * w for w in widths)
    print(f"{indent_str}{separator}")

    # Print rows
    for row in rows:
        row_cells = []
        for i, cell in enumerate(row):
            cell_str = str(cell)
            # Color code pass/fail
            if cell_str.upper() == "PASS":
                cell_str = f"{GREEN}PASS{RESET}"
            elif cell_str.upper() == "FAIL":
                cell_str = f"{RED}FAIL{RESET}"
            elif cell_str.upper() in ("OK", "YES", "TRUE"):
                cell_str = f"{GREEN}{cell_str}{RESET}"
            elif cell_str.upper() in ("NO", "FALSE", "ERROR"):
                cell_str = f"{RED}{cell_str}{RESET}"

            if i < len(widths):
                # Account for ANSI codes in width calculation
                if "\033[" in cell_str:
                    # Add extra padding for ANSI codes
                    raw_len = len(cell.replace("PASS", "").replace("FAIL", ""))
                    extra = widths[i] - len(str(cell)) + len(str(cell))
                    cell_str = cell_str + " " * max(0, widths[i] - len(str(cell)))
                else:
                    cell_str = cell_str.ljust(widths[i])
            row_cells.append(cell_str)

        row_line = " | ".join(row_cells)
        print(f"{indent_str}{row_line}")


def print_progress(
    current: int, total: int, prefix: str = "", suffix: str = "", width: int = 40
) -> None:
    """Print progress bar.

    Args:
        current: Current progress value
        total: Total value
        prefix: Text before bar
        suffix: Text after bar
        width: Bar width in characters
    """
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "=" * filled + "-" * (width - filled)
    print(f"\r{prefix}[{bar}] {percent * 100:.1f}% {suffix}", end="", flush=True)
    if current >= total:
        print()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1.23 s" or "45.6 ms"
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_frequency(hz: float) -> str:
    """Format frequency in human-readable form.

    Args:
        hz: Frequency in Hz

    Returns:
        Formatted string like "1.23 kHz" or "45.6 MHz"
    """
    if hz < 1e3:
        return f"{hz:.2f} Hz"
    elif hz < 1e6:
        return f"{hz / 1e3:.2f} kHz"
    elif hz < 1e9:
        return f"{hz / 1e6:.2f} MHz"
    else:
        return f"{hz / 1e9:.2f} GHz"


def format_bytes(num_bytes: int) -> str:
    """Format byte count in human-readable form.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string like "1.23 KB" or "45.6 MB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
