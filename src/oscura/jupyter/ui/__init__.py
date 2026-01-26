"""Oscura UI module.

Provides user interface patterns for progressive disclosure and formatting utilities.
"""

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
from oscura.jupyter.ui.progressive_display import (
    ProgressiveDisplay,
    ProgressiveOutput,
    Section,
)

__all__ = [
    "Color",
    "FormattedText",
    "ProgressiveDisplay",
    "ProgressiveOutput",
    "Section",
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
