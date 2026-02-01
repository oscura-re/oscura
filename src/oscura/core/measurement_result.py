"""Helper functions for creating MeasurementResult instances.

This module provides utilities to construct MeasurementResult TypedDict
instances with proper formatting and applicability tracking.

Example:
    >>> from oscura.core.measurement_result import make_measurement, make_inapplicable
    >>> # Applicable measurement
    >>> freq = make_measurement(1000.0, "Hz")
    >>> print(freq["display"])
    1.000 kHz

    >>> # Inapplicable measurement
    >>> period = make_inapplicable("s", "Aperiodic signal")
    >>> print(f"{period['display']} - {period['reason']}")
    N/A - Aperiodic signal
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import MeasurementResult


def format_si_prefix(value: float, unit: str, precision: int = 3) -> str:
    """Format value with appropriate SI prefix.

    Args:
        value: Numeric value to format.
        unit: Base unit (e.g., "Hz", "V", "s").
        precision: Number of significant figures.

    Returns:
        Formatted string with SI prefix (e.g., "1.234 kHz", "5.67 mV").

    Example:
        >>> format_si_prefix(1000, "Hz")
        '1.000 kHz'
        >>> format_si_prefix(0.001, "V")
        '1.000 mV'
        >>> format_si_prefix(1e6, "Hz", precision=2)
        '1.00 MHz'
    """
    if value == 0 or not math.isfinite(value):
        return f"{value:.{precision}g} {unit}"

    # SI prefixes and their powers of 10
    prefixes = [
        (1e24, "Y"),  # yotta
        (1e21, "Z"),  # zetta
        (1e18, "E"),  # exa
        (1e15, "P"),  # peta
        (1e12, "T"),  # tera
        (1e9, "G"),  # giga
        (1e6, "M"),  # mega
        (1e3, "k"),  # kilo
        (1, ""),  # no prefix
        (1e-3, "m"),  # milli
        (1e-6, "µ"),  # micro
        (1e-9, "n"),  # nano
        (1e-12, "p"),  # pico
        (1e-15, "f"),  # femto
        (1e-18, "a"),  # atto
        (1e-21, "z"),  # zepto
        (1e-24, "y"),  # yocto
    ]

    abs_value = abs(value)

    # Find appropriate prefix
    for scale, prefix in prefixes:
        if abs_value >= scale:
            scaled = value / scale
            return f"{scaled:.{precision}f} {prefix}{unit}"

    # Fallback for very small values
    return f"{value:.{precision}e} {unit}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage value.

    Args:
        value: Percentage value (0-100).
        precision: Decimal places.

    Returns:
        Formatted percentage string.

    Example:
        >>> format_percentage(5.2)
        '5.20%'
        >>> format_percentage(0.123, precision=3)
        '0.123%'
    """
    return f"{value:.{precision}f}%"


def format_ratio(value: float, precision: int = 3) -> str:
    """Format ratio value (0-1) as percentage.

    Args:
        value: Ratio value (0.0 to 1.0).
        precision: Decimal places.

    Returns:
        Formatted percentage string.

    Example:
        >>> format_ratio(0.052)
        '5.200%'
        >>> format_ratio(0.5)
        '50.000%'
    """
    return f"{value * 100:.{precision}f}%"


def format_decibel(value: float, precision: int = 1) -> str:
    """Format decibel value.

    Args:
        value: Decibel value.
        precision: Decimal places.

    Returns:
        Formatted dB string.

    Example:
        >>> format_decibel(42.5)
        '42.5 dB'
        >>> format_decibel(-3.01, precision=2)
        '-3.01 dB'
    """
    return f"{value:.{precision}f} dB"


def make_measurement(
    value: float, unit: str, *, precision: int = 3, raw_value: bool = False
) -> MeasurementResult:
    """Create an applicable MeasurementResult.

    Args:
        value: Measurement value.
        unit: Unit string ("V", "Hz", "s", "dB", "%", "ratio", or "").
        precision: Formatting precision (default: 3).
        raw_value: If True, format as raw number (default: False).

    Returns:
        MeasurementResult with applicable=True.

    Example:
        >>> result = make_measurement(1000, "Hz")
        >>> result["value"]
        1000.0
        >>> result["display"]
        '1.000 kHz'
        >>> result["applicable"]
        True

        >>> # Percentage measurement
        >>> thd = make_measurement(5.2, "%")
        >>> thd["display"]
        '5.200%'

        >>> # Ratio measurement (auto-converts to percentage)
        >>> duty = make_measurement(0.3, "ratio")
        >>> duty["display"]
        '30.000%'
    """
    # Handle NaN/Inf values
    if not math.isfinite(value):
        return make_inapplicable(unit, f"Invalid value: {value}")

    # Format display string based on unit type
    if raw_value:
        display = f"{value:.{precision}g}"
    elif unit == "dB":
        display = format_decibel(value, precision=precision)
    elif unit == "%":
        display = format_percentage(value, precision=precision)
    elif unit == "ratio":
        # Convert ratio (0-1) to percentage for display
        display = format_ratio(value, precision=precision)
    elif unit == "":
        # Dimensionless - format as integer or float
        if isinstance(value, (int, np.integer)) or value == int(value):
            display = f"{int(value)}"
        else:
            display = f"{value:.{precision}g}"
    else:
        # SI units (V, Hz, s, A, W, etc.)
        display = format_si_prefix(value, unit, precision=precision)

    return {
        "value": float(value),
        "unit": unit,
        "applicable": True,
        "reason": None,
        "display": display,
    }


def make_inapplicable(unit: str, reason: str) -> MeasurementResult:
    """Create an inapplicable MeasurementResult (replaces NaN).

    Args:
        unit: Unit string for metadata.
        reason: Human-readable explanation (e.g., "Aperiodic signal").

    Returns:
        MeasurementResult with applicable=False, value=None.

    Example:
        >>> result = make_inapplicable("s", "Aperiodic signal (single impulse)")
        >>> result["value"] is None
        True
        >>> result["applicable"]
        False
        >>> result["display"]
        'N/A'
        >>> result["reason"]
        'Aperiodic signal (single impulse)'
    """
    return {
        "value": None,
        "unit": unit,
        "applicable": False,
        "reason": reason,
        "display": "N/A",
    }


def make_measurement_safe(
    value: float | None, unit: str, inapplicable_reason: str | None = None, **kwargs: Any
) -> MeasurementResult:
    """Create MeasurementResult with automatic NaN/None handling.

    Args:
        value: Measurement value (can be NaN or None).
        unit: Unit string.
        inapplicable_reason: Reason if inapplicable (required if value is None/NaN).
        **kwargs: Additional arguments passed to make_measurement().

    Returns:
        MeasurementResult (applicable if value is valid, inapplicable otherwise).

    Example:
        >>> # Valid value
        >>> result = make_measurement_safe(1000, "Hz")
        >>> result["applicable"]
        True

        >>> # NaN value
        >>> result = make_measurement_safe(float('nan'), "s", "Aperiodic signal")
        >>> result["applicable"]
        False
        >>> result["reason"]
        'Aperiodic signal'

        >>> # None value
        >>> result = make_measurement_safe(None, "V", "DC signal")
        >>> result["display"]
        'N/A'
    """
    # Check if value is invalid (None, NaN, Inf)
    if value is None or not math.isfinite(value):
        reason = inapplicable_reason or "Invalid or undefined value"
        return make_inapplicable(unit, reason)

    return make_measurement(value, unit, **kwargs)


__all__ = [
    "format_decibel",
    "format_percentage",
    "format_ratio",
    "format_si_prefix",
    "make_inapplicable",
    "make_measurement",
    "make_measurement_safe",
]
