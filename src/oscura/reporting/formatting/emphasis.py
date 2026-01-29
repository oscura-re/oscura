"""Text emphasis formatting utilities.

Simple text formatting for terminal output.
"""

from __future__ import annotations

from enum import Enum


class VisualEmphasis(Enum):
    """Visual emphasis levels."""

    NONE = "none"
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


def bold(text: str) -> str:
    """Make text bold (terminal)."""
    return f"\033[1m{text}\033[0m"


def italic(text: str) -> str:
    """Make text italic (terminal)."""
    return f"\033[3m{text}\033[0m"


def underline(text: str) -> str:
    """Underline text (terminal)."""
    return f"\033[4m{text}\033[0m"


def color(text: str, color_code: str) -> str:
    """Color text (terminal)."""
    colors = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    code = colors.get(color_code.lower(), "37")
    return f"\033[{code}m{text}\033[0m"


def format_severity(text: str, severity: str) -> str:
    """Format text based on severity level."""
    severity_colors = {
        "critical": "red",
        "error": "red",
        "warning": "yellow",
        "info": "blue",
        "success": "green",
    }
    return color(text, severity_colors.get(severity.lower(), "white"))


def format_callout_box(title: str, content: str, severity: str = "info") -> str:
    """Format a callout box."""
    lines = [
        "=" * 60,
        format_severity(f" {title.upper()} ", severity),
        "-" * 60,
        content,
        "=" * 60,
    ]
    return "\n".join(lines)


__all__ = [
    "VisualEmphasis",
    "bold",
    "color",
    "format_callout_box",
    "format_severity",
    "italic",
    "underline",
]
