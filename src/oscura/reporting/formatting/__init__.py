"""Formatting utilities for reports."""

from oscura.reporting.formatting.emphasis import (
    VisualEmphasis,
    format_callout_box,
    format_severity,
)
from oscura.reporting.formatting.numbers import (
    NumberFormatter,
    format_percentage,
    format_range,
    format_value,
    format_with_context,
    format_with_locale,
    format_with_units,
)
from oscura.reporting.formatting.standards import (
    ColorScheme,
    FormatStandards,
    Severity,
    apply_formatting_standards,
)


def format_margin(
    value: float,
    limit: float,
    unit: str = "",
    limit_type: str = "upper",
) -> str:
    """Format a margin value with pass/fail indication.

    Calculates the margin between a measured value and its specification limit,
    then categorizes it as good (>20%), ok (10-20%), marginal (0-10%), or violation.

    Args:
        value: Measured value.
        limit: Limit value (specification).
        unit: Unit of measurement.
        limit_type: Type of limit ("upper" or "lower").

    Returns:
        Formatted margin string with status indication.

    Examples:
        >>> format_margin(70, 100, limit_type="upper")
        '30.0% margin (good)'
        >>> format_margin(95, 100, limit_type="upper")
        '5.0% margin (marginal)'
    """
    # Calculate margin percentage
    if limit != 0:
        if limit_type == "upper":
            margin = limit - value
            margin_pct = (margin / limit) * 100
            is_passing = value < limit
        else:  # lower
            margin = value - limit
            margin_pct = (margin / limit) * 100
            is_passing = value > limit
    else:
        margin_pct = 0
        is_passing = False

    # Determine status based on margin percentage
    if not is_passing:
        status = "violation"
    elif margin_pct >= 20:
        status = "good"
    elif margin_pct >= 10:
        status = "ok"
    else:
        status = "marginal"

    return f"{abs(margin_pct):.1f}% margin ({status})"


def format_pass_fail(passed: bool, message: str = "", with_symbol: bool = True) -> str:
    """Format pass/fail status.

    Args:
        passed: Whether the test passed.
        message: Optional message to append.
        with_symbol: Whether to include Unicode symbol (checkmark/cross).

    Returns:
        Formatted pass/fail string.

    Examples:
        >>> format_pass_fail(True)
        'PASS ✓'
        >>> format_pass_fail(False, with_symbol=False)
        'FAIL'
    """
    status = "PASS" if passed else "FAIL"

    if with_symbol:
        symbol = "\u2713" if passed else "\u2717"
        result = f"{status} {symbol}"
    else:
        result = status

    if message:
        return f"{result}: {message}"
    return result


__all__ = [
    "ColorScheme",
    # Standards
    "FormatStandards",
    # Numbers
    "NumberFormatter",
    "Severity",
    # Emphasis
    "VisualEmphasis",
    "apply_formatting_standards",
    "format_callout_box",
    # Convenience
    "format_margin",
    "format_pass_fail",
    "format_percentage",
    "format_range",
    "format_severity",
    "format_value",
    "format_with_context",
    "format_with_locale",
    "format_with_units",
]
