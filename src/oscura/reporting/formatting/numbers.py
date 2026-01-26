"""Number formatting utilities for signal analysis reports.

Provides comprehensive number formatting with SI prefixes, engineering notation,
locale support, and specification comparison capabilities.
"""

import math
from datetime import datetime
from typing import ClassVar


class NumberFormatter:
    """Comprehensive number formatting utility with SI prefix support.

    Attributes:
        sig_figs: Number of significant figures to display.
        auto_scale: Whether to automatically scale to appropriate SI prefix.
        engineering_notation: Use engineering notation (powers of 3).
        unicode_prefixes: Use Unicode characters for prefixes (µ vs u).

    Examples:
        >>> fmt = NumberFormatter(sig_figs=3)
        >>> fmt.format(2.3e-9, "s")
        '2.30 ns'
        >>> fmt.format(1.5e6, "Hz")
        '1.50 MHz'
    """

    # SI prefixes (powers of 10)
    # Unicode uses Greek Small Letter Mu (U+03BC), ASCII uses 'u'
    SI_PREFIXES: ClassVar[dict[int, tuple[str, str]]] = {
        -15: ("f", "f"),  # femto
        -12: ("p", "p"),  # pico
        -9: ("n", "n"),  # nano
        -6: ("\u03bc", "u"),  # micro (Greek mu U+03BC, ascii 'u')
        -3: ("m", "m"),  # milli
        0: ("", ""),  # base
        3: ("k", "k"),  # kilo
        6: ("M", "M"),  # mega
        9: ("G", "G"),  # giga
        12: ("T", "T"),  # tera
    }

    def __init__(
        self,
        sig_figs: int = 3,
        auto_scale: bool = True,
        engineering_notation: bool = True,
        unicode_prefixes: bool = True,
        precision: int | None = None,
        use_si: bool | None = None,
    ) -> None:
        """Initialize formatter.

        Args:
            sig_figs: Number of significant figures (default 3).
            auto_scale: Enable automatic SI prefix scaling (default True).
            engineering_notation: Use engineering notation (default True).
            unicode_prefixes: Use Unicode prefixes like µ (default True).
            precision: Alias for sig_figs (backwards compatibility).
            use_si: Alias for auto_scale (backwards compatibility).
        """
        # Handle backwards compatibility aliases
        self.sig_figs = precision if precision is not None else sig_figs
        self.auto_scale = use_si if use_si is not None else auto_scale
        self.engineering_notation = engineering_notation
        self.unicode_prefixes = unicode_prefixes

        # Legacy attribute for backwards compatibility
        self.precision = self.sig_figs
        self.use_si = self.auto_scale

    def _format_without_scaling(self, value: float, places: int, unit: str) -> str:
        """Format value without SI scaling.

        Args:
            value: Value to format
            places: Decimal places
            unit: Unit string

        Returns:
            Formatted string
        """
        abs_val = abs(value)

        if abs_val != 0 and abs_val < 1:
            from math import floor, log10

            order = floor(log10(abs_val))
            decimal_places_needed = max(places, abs(order) + 1)
            return f"{value:.{decimal_places_needed}f} {unit}".strip()

        if abs_val >= 1e6:
            return f"{value:.{places}e} {unit}".strip()

        return f"{value:.{places}f} {unit}".strip()

    def _get_si_scale(self, abs_val: float) -> tuple[float, int] | None:
        """Get SI scale and exponent for value.

        Args:
            abs_val: Absolute value

        Returns:
            Tuple of (scale_factor, exponent) or None for extreme values
        """
        # Find the appropriate SI prefix by checking value ranges
        # Each range is [lower_bound, upper_bound) for the prefix
        if abs_val >= 1e15:
            return (1e-15, 15)
        if abs_val >= 1e12:
            return (1e-12, 12)
        if abs_val >= 1e9:
            return (1e-9, 9)
        if abs_val >= 1e6:
            return (1e-6, 6)
        if abs_val >= 1e3:
            return (1e-3, 3)
        if abs_val >= 1:
            return (1, 0)
        if abs_val >= 1e-3:
            return (1e3, -3)
        if abs_val >= 1e-6:
            return (1e6, -6)
        if abs_val >= 1e-9:
            return (1e9, -9)
        if abs_val >= 1e-12:
            return (1e12, -12)
        if abs_val >= 1e-15:
            return (1e15, -15)

        # Value too small, use scientific notation
        return None

    def format(
        self,
        value: float,
        unit: str = "",
        decimal_places: int | None = None,
    ) -> str:
        """Format a number with optional SI prefix.

        Args:
            value: Numeric value to format.
            unit: Unit of measurement (e.g., "V", "Hz", "s").
            decimal_places: Override number of decimal places.

        Returns:
            Formatted string with value, prefix, and unit.

        Examples:
            >>> fmt = NumberFormatter()
            >>> fmt.format(2.3e-6, "s")
            '2.300 µs'
            >>> fmt.format(1500000, "Hz")
            '1.500 MHz'
        """
        if math.isnan(value):
            return f"NaN {unit}".strip()
        if math.isinf(value):
            sign = "-" if value < 0 else ""
            return f"{sign}Inf {unit}".strip()

        places = decimal_places if decimal_places is not None else self.sig_figs

        if not self.auto_scale:
            return self._format_without_scaling(value, places, unit)

        if value == 0:
            return f"0.{'0' * places} {unit}".strip()

        abs_val = abs(value)
        scale_result = self._get_si_scale(abs_val)

        if scale_result is None:
            return f"{value:.{places}e} {unit}".strip()

        scale, exp = scale_result
        scaled = value * scale

        prefix_idx = 0 if self.unicode_prefixes else 1
        prefix = self.SI_PREFIXES.get(exp, ("", ""))[prefix_idx]

        return f"{scaled:.{places}f} {prefix}{unit}".strip()

    def format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format a value as a percentage.

        Args:
            value: Value as decimal (0.543 = 54.3%) or already percentage.
            decimals: Number of decimal places.

        Returns:
            Formatted percentage string.

        Examples:
            >>> fmt = NumberFormatter()
            >>> fmt.format_percentage(0.543)
            '54.3%'
        """
        # If value > 1, assume it's already a percentage
        if abs(value) > 1:
            return f"{value:.{decimals}f}%"
        return f"{value * 100:.{decimals}f}%"

    def format_range(
        self,
        min_val: float,
        typ_val: float,
        max_val: float,
        unit: str = "",
    ) -> str:
        """Format min/typ/max range.

        Args:
            min_val: Minimum value.
            typ_val: Typical value.
            max_val: Maximum value.
            unit: Unit of measurement.

        Returns:
            Formatted range string.

        Examples:
            >>> fmt = NumberFormatter()
            >>> fmt.format_range(1e-6, 2e-6, 3e-6, "s")
            'min=1.000 µs typ=2.000 µs max=3.000 µs'
        """
        return (
            f"min={self.format(min_val, unit)} "
            f"typ={self.format(typ_val, unit)} "
            f"max={self.format(max_val, unit)}"
        )


def format_value(
    value: float,
    unit_or_precision: str | int = 3,
    unit: str = "",
    sig_figs: int | None = None,
) -> str:
    """Format a numeric value with SI prefix.

    Args:
        value: Value to format.
        unit_or_precision: Either unit (str) or precision (int).
        unit: Unit string (if precision specified first).
        sig_figs: Number of significant figures (alternative to precision arg).

    Returns:
        Formatted value with appropriate SI prefix.

    Examples:
        >>> format_value(2.3e-9, "s")
        '2.300 ns'
        >>> format_value(1.5e6, 4, "Hz")
        '1.5000 MHz'
    """
    # Handle flexible arguments
    if isinstance(unit_or_precision, str):
        precision = sig_figs if sig_figs is not None else 3
        actual_unit = unit_or_precision
    else:
        precision = sig_figs if sig_figs is not None else unit_or_precision
        actual_unit = unit

    fmt = NumberFormatter(sig_figs=precision)
    return fmt.format(value, actual_unit)


def format_with_units(value: float, unit: str, sig_figs: int = 3) -> str:
    """Format value with units and SI prefix.

    Args:
        value: Numeric value.
        unit: Unit of measurement.
        sig_figs: Number of significant figures.

    Returns:
        Formatted string with value and unit.
    """
    fmt = NumberFormatter(sig_figs=sig_figs)
    return fmt.format(value, unit)


def format_with_context(
    value: float,
    context: str = "",
    spec: float | None = None,
    unit: str = "",
    spec_type: str = "max",
    show_margin: bool = True,
) -> str:
    """Format value with context and specification comparison.

    Args:
        value: Value to format.
        context: Context string for display.
        spec: Specification limit for comparison.
        unit: Unit of measurement.
        spec_type: Type of specification ("max", "min", or "exact").
        show_margin: Whether to show margin percentage.

    Returns:
        Formatted string with pass/fail indication if spec provided.

    Examples:
        >>> format_with_context(2.3e-9, spec=5e-9, unit="s", spec_type="max")
        '2.300 ns ✓'
    """
    formatted = format_value(value, unit)

    if spec is not None:
        # Determine pass/fail
        if spec_type == "max":
            passed = value <= spec
            margin = ((spec - value) / spec * 100) if spec != 0 else 0
        elif spec_type == "min":
            passed = value >= spec
            margin = ((value - spec) / spec * 100) if spec != 0 else 0
        else:  # exact
            tolerance = abs(spec * 0.01)  # 1% tolerance
            passed = abs(value - spec) <= tolerance
            margin = 100 - abs((value - spec) / spec * 100) if spec != 0 else 100

        # Unicode checkmark/cross
        status = "\u2713" if passed else "\u2717"

        if show_margin and passed:
            return f"{formatted} {status} ({margin:.1f}% margin)"
        return f"{formatted} {status}"

    if context:
        return f"{formatted} ({context})"
    return formatted


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage.

    Args:
        value: Value as decimal (0.5 = 50%) or already percentage (>1).
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string.
    """
    if abs(value) > 1:
        return f"{value:.{decimals}f}%"
    return f"{value * 100:.{decimals}f}%"


def format_range(min_val: float, max_val: float, unit: str = "") -> str:
    """Format a range of values.

    Args:
        min_val: Minimum value.
        max_val: Maximum value.
        unit: Unit of measurement.

    Returns:
        Formatted range string.
    """
    return f"{format_value(min_val, unit)} to {format_value(max_val, unit)}"


def format_with_locale(
    value: float | None = None,
    locale: str = "en_US",
    date_value: float | None = None,
) -> str:
    """Format value with locale-specific formatting.

    Args:
        value: Numeric value to format.
        locale: Locale string (e.g., 'en_US', 'de_DE', 'fr_FR').
        date_value: Unix timestamp for date formatting.

    Returns:
        Formatted string with locale-specific separators.
        Returns empty string if value is None and no date_value.

    Examples:
        >>> format_with_locale(1234.56, locale="en_US")
        '1,234.56'
        >>> format_with_locale(1234.56, locale="de_DE")
        '1.234,56'
    """
    # Handle date formatting
    if date_value is not None:
        dt = datetime.fromtimestamp(date_value)
        if locale.startswith("en"):
            return dt.strftime("%m/%d/%Y")
        elif locale.startswith("de"):
            return dt.strftime("%d.%m.%Y")
        elif locale.startswith("fr"):
            return dt.strftime("%d/%m/%Y")
        else:
            return dt.strftime("%Y-%m-%d")

    # Handle None value gracefully
    if value is None:
        return ""

    # Handle numeric formatting by locale
    if locale.startswith("en"):
        return f"{value:,.2f}"
    elif locale.startswith("de"):
        # German: 1.234,56
        formatted = f"{value:,.2f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    elif locale.startswith("fr"):
        # French: 1 234,56 (narrow no-break space for thousands)
        formatted = f"{value:,.2f}"
        return formatted.replace(",", " ").replace(".", ",")
    else:
        return f"{value:,.2f}"


__all__ = [
    "NumberFormatter",
    "format_percentage",
    "format_range",
    "format_value",
    "format_with_context",
    "format_with_locale",
    "format_with_units",
]
