"""Minimal boilerplate content generation.

This module eliminates unnecessary static text and focuses on data-driven
narrative with compact formatting and automated captions.


Example:
    >>> from oscura.reporting.content import generate_compact_text
    >>> compact = generate_compact_text(value=2.3e-9, spec=5e-9, unit="s")
    >>> print(compact)  # "Rise time: 2.3ns (spec <5ns, ✓ 54% margin)"

References:
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class MinimalContent:
    """Minimal boilerplate content generator.

    Focuses on data-driven content with minimal filler text.

    Attributes:
        auto_units: Auto-scale units (2300ns → 2.3μs).
        data_first: Show results before methodology.
        show_passing: Include passing tests or violations only.
        auto_captions: Generate captions from data, not static templates.

    References:
        REPORT-003: Minimal Boilerplate Content
    """

    auto_units: bool = True
    data_first: bool = True
    show_passing: bool = True
    auto_captions: bool = True


def generate_compact_text(
    value: float,
    spec: float | None = None,
    unit: str = "",
    *,
    spec_type: Literal["max", "min"] = "max",
    name: str = "",
) -> str:
    """Generate compact, data-driven text.

    Avoids filler text like "The measurement was performed and the result was...".
    Instead produces compact format: "Rise time: 2.3ns (spec <5ns, ✓ 54% margin)".

    Args:
        value: Measured value.
        spec: Specification limit (optional).
        unit: Unit string.
        spec_type: Type of specification (max or min).
        name: Measurement name (optional).

    Returns:
        Compact formatted string.

    Example:
        >>> generate_compact_text(2.3e-9, 5e-9, "s", name="Rise time")
        'Rise time: 2.3ns (spec <5ns, ✓ 54% margin)'

    References:
        REPORT-003: Minimal Boilerplate Content
    """
    from oscura.reporting.formatting.numbers import format_with_units

    # Format value with auto-scaled units
    value_str = format_with_units(value, unit)

    # Build compact text
    parts = []
    if name:
        parts.append(f"{name}:")

    parts.append(value_str)

    # Add spec context if provided
    if spec is not None:
        spec_str = format_with_units(spec, unit)
        spec_symbol = "<" if spec_type == "max" else ">"

        # Calculate pass/fail and margin
        if spec_type == "max":
            passed = value <= spec
            margin = (spec - value) / spec * 100 if spec != 0 else 0
        else:
            passed = value >= spec
            margin = (value - spec) / spec * 100 if spec != 0 else 0

        status_symbol = "\u2713" if passed else "\u2717"  # ✓ or ✗
        spec_part = f"(spec {spec_symbol}{spec_str}, {status_symbol} {margin:.0f}% margin)"
        parts.append(spec_part)

    return " ".join(parts)


def auto_caption(
    data_type: str,
    data: dict[str, Any],
    *,
    include_stats: bool = True,
) -> str:
    """Generate automated captions from data.

    Instead of static templates, generates captions based on actual data content.

    Args:
        data_type: Type of data (measurement, plot, table).
        data: Data dictionary.
        include_stats: Include statistics in caption.

    Returns:
        Generated caption string.

    Example:
        >>> data = {"name": "Rise time", "count": 100, "mean": 2.3e-9}
        >>> auto_caption("measurement", data)
        'Rise time measurement (n=100, mean=2.3ns)'

    References:
        REPORT-003: Minimal Boilerplate Content
    """
    parts = []

    # Extract key information
    name = data.get("name", data_type.title())
    parts.append(name)

    # Add data-specific information
    if data_type == "measurement" and include_stats:
        count = data.get("count")
        if count:
            parts.append(f"(n={count}")

            mean = data.get("mean")
            if mean is not None:
                from oscura.reporting.formatting.numbers import format_with_units

                unit = data.get("unit", "")
                mean_str = format_with_units(mean, unit)
                parts[-1] += f", mean={mean_str}"

            parts[-1] += ")"

    elif data_type == "plot":
        plot_type = data.get("type", "plot")
        if plot_type != "plot":
            parts.append(f"- {plot_type}")

    elif data_type == "table":
        rows = data.get("rows")
        cols = data.get("cols")
        if rows and cols:
            parts.append(f"({rows}x{cols})")

    return " ".join(parts)


def remove_filler_text(text: str) -> str:
    """Remove common filler phrases from text.

    Args:
        text: Input text.

    Returns:
        Text with filler removed.

    Example:
        >>> text = "The measurement was performed and the result was 2.3ns."
        >>> remove_filler_text(text)
        'Result: 2.3ns.'

    References:
        REPORT-003: Minimal Boilerplate Content
    """
    # Common filler phrases to remove
    filler_phrases = [
        "The measurement was performed and",
        "The result was",
        "It was found that",
        "The analysis shows that",
        "It can be seen that",
        "As can be observed",
        "The data indicates",
        "It should be noted that",
    ]

    result = text
    for phrase in filler_phrases:
        result = result.replace(phrase, "").strip()

    # Clean up extra spaces
    while "  " in result:
        result = result.replace("  ", " ")

    # Capitalize first letter after removal
    if result and not result[0].isupper():
        result = result[0].upper() + result[1:]

    return result


def conditional_section(
    data: list[Any] | dict[str, Any],
    section_title: str,
) -> tuple[bool, str]:
    """Determine if section should be shown.

    Only show sections if data exists (no empty sections).

    Args:
        data: Section data.
        section_title: Section title.

    Returns:
        Tuple of (should_show, reason).

    Example:
        >>> should_show, reason = conditional_section([], "Violations")
        >>> print(should_show)  # False
        >>> print(reason)  # "No violations found"

    References:
        REPORT-003: Minimal Boilerplate Content
    """
    if isinstance(data, list):
        if not data:
            return False, f"No {section_title.lower()} found"
        return True, f"{len(data)} {section_title.lower()}"

    elif isinstance(data, dict):
        if not data or all(not v for v in data.values()):
            return False, f"No {section_title.lower()} found"
        return True, ""

    else:
        # For other types, check if truthy
        if not data:  # type: ignore[unreachable]
            return False, f"No {section_title.lower()} found"
        return True, ""


__all__ = [
    "MinimalContent",
    "auto_caption",
    "conditional_section",
    "generate_compact_text",
    "remove_filler_text",
]
