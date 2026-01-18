"""Number and value formatting for Oscura reports.

This module provides smart number formatting with SI prefixes,
significant figures, and contextual annotations.


Example:
    >>> from oscura.reporting import format_with_units, format_with_context
    >>> format_with_units(0.0000023, "s")  # "2.3 us"
    >>> format_with_context(2.3e-9, spec=5e-9)  # "2.3 ns (spec <5 ns, PASS)"
"""

from __future__ import annotations

import locale as locale_module
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

# SI prefixes
SI_PREFIXES = {
    24: "Y",
    21: "Z",
    18: "E",
    15: "P",
    12: "T",
    9: "G",
    6: "M",
    3: "k",
    0: "",
    -3: "m",
    -6: "u",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
    -21: "z",
    -24: "y",
}

# Unicode SI prefixes
SI_PREFIXES_UNICODE = {
    **SI_PREFIXES,
    -6: "\u03bc",  # Greek mu
}


@dataclass
class NumberFormatter:
    """Configurable number formatter.

    Attributes:
        sig_figs: Significant figures (default 3).
        auto_scale: Use SI prefixes for scaling.
        engineering_notation: Use engineering notation (10^3, 10^6, etc.).
        unicode_prefixes: Use Unicode characters (e.g., micro symbol).
        min_exp: Minimum exponent before using scientific notation.
        max_exp: Maximum exponent before using scientific notation.
    """

    sig_figs: int = 3
    auto_scale: bool = True
    engineering_notation: bool = True
    unicode_prefixes: bool = True
    min_exp: int = -3
    max_exp: int = 3

    def format(
        self,
        value: float,
        unit: str = "",
        *,
        decimal_places: int | None = None,
    ) -> str:
        """Format a numeric value.

        Args:
            value: Value to format.
            unit: Unit suffix (e.g., "s", "Hz", "V").
            decimal_places: Override significant figures with fixed decimals.

        Returns:
            Formatted string.

        Example:
            >>> fmt = NumberFormatter()
            >>> fmt.format(0.0000023, "s")
            '2.30 us'
        """
        if not np.isfinite(value):
            if np.isnan(value):
                return "NaN"
            elif value > 0:
                return "+Inf"
            else:
                return "-Inf"

        if value == 0:
            return f"0 {unit}".strip()

        if self.auto_scale and self.engineering_notation:
            return self._format_engineering(value, unit, decimal_places)
        elif self.auto_scale:
            return self._format_scaled(value, unit, decimal_places)
        else:
            return self._format_plain(value, unit, decimal_places)

    def _format_engineering(
        self,
        value: float,
        unit: str,
        decimal_places: int | None,
    ) -> str:
        """Format with engineering notation (SI prefixes)."""
        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        # Find appropriate SI prefix
        if abs_value == 0:
            exp = 0
        else:
            exp = int(np.floor(np.log10(abs_value)))
            # Round to nearest multiple of 3
            exp = (exp // 3) * 3

        # Clamp to available prefixes
        exp = max(-24, min(24, exp))

        # Get prefix
        prefixes = SI_PREFIXES_UNICODE if self.unicode_prefixes else SI_PREFIXES
        prefix = prefixes.get(exp, "")

        # Scale value
        scaled = abs_value / (10**exp)

        # Format with significant figures
        if decimal_places is not None:
            formatted = f"{sign}{scaled:.{decimal_places}f}"
        else:
            # Calculate decimal places from significant figures
            if scaled >= 100:
                decimals = max(0, self.sig_figs - 3)
            elif scaled >= 10:
                decimals = max(0, self.sig_figs - 2)
            elif scaled >= 1:
                decimals = max(0, self.sig_figs - 1)
            else:
                decimals = self.sig_figs
            formatted = f"{sign}{scaled:.{decimals}f}"

        return f"{formatted} {prefix}{unit}".strip()

    def _format_scaled(
        self,
        value: float,
        unit: str,
        decimal_places: int | None,
    ) -> str:
        """Format with auto-scaling but not necessarily engineering notation."""
        return self._format_engineering(value, unit, decimal_places)

    def _format_plain(
        self,
        value: float,
        unit: str,
        decimal_places: int | None,
    ) -> str:
        """Format without scaling."""
        if decimal_places is not None:
            formatted = f"{value:.{decimal_places}f}"
        else:
            formatted = f"{value:.{self.sig_figs}g}"

        return f"{formatted} {unit}".strip()

    def format_percentage(self, value: float, *, decimals: int = 1) -> str:
        """Format as percentage.

        Args:
            value: Value (0-1 or 0-100).
            decimals: Decimal places.

        Returns:
            Percentage string.
        """
        # Assume value is already in percent if > 1
        if abs(value) <= 1:
            value = value * 100
        return f"{value:.{decimals}f}%"

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
            unit: Unit suffix.

        Returns:
            Formatted range string.
        """
        # Use same scaling for all values
        abs_max = max(abs(min_val), abs(typ_val), abs(max_val))
        exp = 0 if abs_max == 0 else int(np.floor(np.log10(abs_max))) // 3 * 3
        exp = max(-24, min(24, exp))

        prefixes = SI_PREFIXES_UNICODE if self.unicode_prefixes else SI_PREFIXES
        prefix = prefixes.get(exp, "")

        scale = 10**exp
        decimals = max(0, self.sig_figs - 1)

        min_s = f"{min_val / scale:.{decimals}f}"
        typ_s = f"{typ_val / scale:.{decimals}f}"
        max_s = f"{max_val / scale:.{decimals}f}"

        return f"min/typ/max: {min_s} / {typ_s} / {max_s} {prefix}{unit}".strip()


# Default formatter
_default_formatter = NumberFormatter()


def format_value(
    value: float,
    unit: str = "",
    *,
    sig_figs: int = 3,
) -> str:
    """Format a numeric value with SI prefix.

    Args:
        value: Value to format.
        unit: Unit suffix.
        sig_figs: Significant figures.

    Returns:
        Formatted string.

    Example:
        >>> format_value(0.0000023, "s")
        '2.30 us'
    """
    formatter = NumberFormatter(sig_figs=sig_figs)
    return formatter.format(value, unit)


def format_with_units(
    value: float,
    unit: str,
    *,
    sig_figs: int = 3,
) -> str:
    """Format value with automatic SI prefix scaling.

    Args:
        value: Value to format.
        unit: Base unit (e.g., "s", "Hz", "V").
        sig_figs: Significant figures.

    Returns:
        Formatted string with SI prefix.

    Example:
        >>> format_with_units(2300000, "Hz")
        '2.30 MHz'
    """
    return format_value(value, unit, sig_figs=sig_figs)


def format_with_context(
    value: float,
    *,
    spec: float | None = None,
    spec_type: Literal["max", "min", "exact"] = "max",
    unit: str = "",
    sig_figs: int = 3,
    show_margin: bool = True,
) -> str:
    """Format value with specification context.

    Args:
        value: Measured value.
        spec: Specification limit.
        spec_type: Type of specification (max, min, exact).
        unit: Unit suffix.
        sig_figs: Significant figures.
        show_margin: Show margin percentage.

    Returns:
        Formatted string with context.

    Example:
        >>> format_with_context(2.3e-9, spec=5e-9, unit="s")
        '2.30 ns (spec <5.00 ns, PASS 54%)'
    """
    formatter = NumberFormatter(sig_figs=sig_figs)
    value_str = formatter.format(value, unit)

    if spec is None:
        return value_str

    spec_str = formatter.format(spec, unit)

    # Determine pass/fail
    if spec_type == "max":
        passed = value <= spec
        spec_prefix = "<"
    elif spec_type == "min":
        passed = value >= spec
        spec_prefix = ">"
    else:  # exact
        passed = abs(value - spec) < spec * 0.01  # 1% tolerance
        spec_prefix = "="

    status_char = "\u2713" if passed else "\u2717"  # Check/X marks

    # Calculate margin
    margin_str = ""
    if show_margin and spec != 0:
        if spec_type == "max":
            margin = (spec - value) / spec * 100
        elif spec_type == "min":
            margin = (value - spec) / spec * 100
        else:
            margin = (1 - abs(value - spec) / spec) * 100

        margin_str = f" {margin:.0f}%"

    return f"{value_str} (spec {spec_prefix}{spec_str}, {status_char}{margin_str})"


def format_pass_fail(passed: bool, *, with_symbol: bool = True) -> str:
    """Format pass/fail status.

    Args:
        passed: True for pass, False for fail.
        with_symbol: Include Unicode symbol.

    Returns:
        Formatted status string.
    """
    if with_symbol:
        if passed:
            return "\u2713 PASS"  # Check mark
        else:
            return "\u2717 FAIL"  # X mark
    else:
        return "PASS" if passed else "FAIL"


def format_margin(
    value: float,
    limit: float,
    *,
    limit_type: Literal["upper", "lower"] = "upper",
) -> str:
    """Format margin to limit.

    Args:
        value: Measured value.
        limit: Limit value.
        limit_type: Whether limit is upper or lower bound.

    Returns:
        Margin string with status indicator.
    """
    if limit_type == "upper":
        margin = limit - value
        margin_pct = (margin / limit * 100) if limit != 0 else 0
    else:
        margin = value - limit
        margin_pct = (margin / limit * 100) if limit != 0 else 0

    # Status based on margin
    if margin_pct > 20:
        status = "good"
    elif margin_pct > 10:
        status = "ok"
    elif margin_pct > 0:
        status = "marginal"
    else:
        status = "violation"

    return f"margin: {margin_pct:.1f}% ({status})"


def format_with_locale(
    value: float | None = None,
    locale: str | None = None,
    *,
    date_value: float | None = None,
) -> str:
    """Format numbers/dates with locale-aware formatting.

    Args:
        value: Numeric value to format (mutually exclusive with date_value).
        locale: Locale string (e.g., 'en_US', 'de_DE', 'fr_FR').
                If None, uses system locale.
        date_value: Timestamp to format as date (mutually exclusive with value).

    Returns:
        Formatted string with locale-specific separators and formats.

    Example:
        >>> format_with_locale(1234.56, locale="en_US")
        '1,234.56'
        >>> format_with_locale(1234.56, locale="de_DE")
        '1.234,56'
        >>> format_with_locale(1234.56, locale="fr_FR")
        '1 234,56'

    References:
        REPORT-026: Locale-aware Formatting
    """
    # Determine locale
    current_locale = locale_module.getlocale()[0] or "en_US" if locale is None else locale

    # Format date if date_value provided
    if date_value is not None:
        dt = datetime.fromtimestamp(date_value)
        if current_locale.startswith("en_US"):
            return dt.strftime("%m/%d/%Y")
        elif current_locale.startswith("de_DE"):
            return dt.strftime("%d.%m.%Y")
        elif current_locale.startswith("fr_FR"):
            return dt.strftime("%d/%m/%Y")
        else:  # ISO format as fallback
            return dt.strftime("%Y-%m-%d")

    # Format number
    if value is None:
        return ""

    # Locale-specific decimal and thousands separators
    if current_locale.startswith("en_US"):
        decimal_sep = "."
        thousands_sep = ","
    elif current_locale.startswith("de_DE"):
        decimal_sep = ","
        thousands_sep = "."
    elif current_locale.startswith("fr_FR"):
        decimal_sep = ","
        thousands_sep = " "
    else:  # SI standard (space for thousands)
        decimal_sep = "."
        thousands_sep = " "

    # Format with 2 decimal places
    formatted = f"{value:,.2f}"

    # Replace separators
    formatted = formatted.replace(",", "TEMP")
    formatted = formatted.replace(".", decimal_sep)
    formatted = formatted.replace("TEMP", thousands_sep)

    return formatted
