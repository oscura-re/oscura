"""Time-aware X-axis formatting and optimization.

This module provides intelligent time axis formatting with automatic unit
selection, relative time offsets, and cursor readout with full precision.


Example:
    >>> from oscura.visualization.time_axis import format_time_axis
    >>> labels = format_time_axis(time_values, unit="auto")

References:
    - SI prefixes for time units
    - IEEE publication time axis standards
    - Matplotlib formatter customization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

TimeUnit = Literal["s", "ms", "us", "ns", "ps", "auto"]


def select_time_unit(
    time_range: float,
    *,
    prefer_larger: bool = False,
) -> TimeUnit:
    """Automatically select appropriate time unit based on range.

    Args:
        time_range: Time range in seconds.
        prefer_larger: Prefer larger units when ambiguous.

    Returns:
        Selected time unit ("s", "ms", "us", "ns", "ps").

    Example:
        >>> select_time_unit(0.001)  # 1 ms
        'ms'
        >>> select_time_unit(1e-6)  # 1 us
        'us'

    References:
        VIS-014: Adaptive X-Axis Time Window
    """
    if time_range >= 1.0:
        return "s"
    elif time_range >= 1e-3:
        return "ms" if not prefer_larger else "s"
    elif time_range >= 1e-6:
        return "us" if not prefer_larger else "ms"
    elif time_range >= 1e-9:
        return "ns" if not prefer_larger else "us"
    else:
        return "ps" if not prefer_larger else "ns"


def convert_time_values(
    time: NDArray[np.float64],
    unit: TimeUnit,
) -> NDArray[np.float64]:
    """Convert time values to specified unit.

    Args:
        time: Time array in seconds.
        unit: Target time unit.

    Returns:
        Time array in target unit.

    Raises:
        ValueError: If unit is invalid.

    Example:
        >>> time_s = np.array([0.001, 0.002, 0.003])
        >>> time_ms = convert_time_values(time_s, "ms")
        >>> # Returns [1.0, 2.0, 3.0]

    References:
        VIS-014: Adaptive X-Axis Time Window
    """
    multipliers = {
        "s": 1.0,
        "ms": 1e3,
        "us": 1e6,
        "ns": 1e9,
        "ps": 1e12,
    }

    if unit == "auto":
        time_range = float(np.ptp(time))
        unit = select_time_unit(time_range)

    if unit not in multipliers:
        raise ValueError(f"Invalid time unit: {unit}")

    return time * multipliers[unit]


def format_time_labels(
    time: NDArray[np.float64],
    unit: TimeUnit = "auto",
    *,
    precision: int | None = None,
    scientific_threshold: float = 1e6,
) -> list[str]:
    """Format time values as labels with appropriate precision.

    Args:
        time: Time array in seconds.
        unit: Time unit ("s", "ms", "us", "ns", "ps", "auto").
        precision: Number of decimal places (auto if None).
        scientific_threshold: Use scientific notation above this value.

    Returns:
        List of formatted time labels.

    Example:
        >>> time = np.array([0.0, 0.001, 0.002])
        >>> labels = format_time_labels(time, unit="ms")
        >>> # Returns ['0', '1', '2']

    References:
        VIS-014: Adaptive X-Axis Time Window
    """
    # Convert to target unit
    time_converted = convert_time_values(time, unit)

    # Auto-select precision based on value range
    if precision is None:
        value_range = np.ptp(time_converted)
        if value_range == 0:
            precision = 1
        else:
            # Use enough precision to show differences
            magnitude = np.log10(value_range)
            precision = max(0, int(np.ceil(2 - magnitude)))

    # Format labels
    labels = []
    for val in time_converted:
        if abs(val) >= scientific_threshold:
            # Scientific notation
            labels.append(f"{val:.{precision}e}")
        else:
            # Fixed point
            labels.append(f"{val:.{precision}f}".rstrip("0").rstrip("."))

    return labels


def create_relative_time(
    time: NDArray[np.float64],
    *,
    start_at_zero: bool = True,
    reference_time: float | None = None,
) -> NDArray[np.float64]:
    """Create relative time axis starting at zero or reference.

    Args:
        time: Absolute time array in seconds.
        start_at_zero: Start time axis at t=0.
        reference_time: Reference time (uses first sample if None).

    Returns:
        Relative time array.

    Example:
        >>> time_abs = np.array([1000.5, 1000.6, 1000.7])
        >>> time_rel = create_relative_time(time_abs)
        >>> # Returns [0.0, 0.1, 0.2]

    References:
        VIS-014: Adaptive X-Axis Time Window
    """
    if len(time) == 0:
        return time

    if reference_time is None:
        reference_time = time[0] if start_at_zero else 0.0

    return time - reference_time


def calculate_major_ticks(
    time_min: float,
    time_max: float,
    *,
    target_count: int = 7,
    unit: TimeUnit = "auto",
) -> NDArray[np.float64]:
    """Calculate major tick positions for time axis.

    Args:
        time_min: Minimum time value in seconds.
        time_max: Maximum time value in seconds.
        target_count: Target number of major ticks.
        unit: Time unit for tick alignment.

    Returns:
        Array of major tick positions in seconds.

    Example:
        >>> ticks = calculate_major_ticks(0, 0.01, target_count=5, unit="ms")

    References:
        VIS-014: Adaptive X-Axis Time Window
        VIS-019: Grid Auto-Spacing
    """
    time_range = time_max - time_min

    if time_range <= 0:
        return np.array([time_min])

    # Select unit if auto
    if unit == "auto":
        unit = select_time_unit(time_range)

    # Convert to selected unit
    multipliers = {
        "s": 1.0,
        "ms": 1e3,
        "us": 1e6,
        "ns": 1e9,
        "ps": 1e12,
    }
    multiplier = multipliers[unit]

    time_min_unit = time_min * multiplier
    time_max_unit = time_max * multiplier
    range_unit = time_max_unit - time_min_unit

    # Calculate rough spacing
    rough_spacing = range_unit / target_count

    # Round to nice number
    nice_spacing = _round_to_nice_time(rough_spacing)

    # Generate ticks
    first_tick = np.ceil(time_min_unit / nice_spacing) * nice_spacing
    n_ticks = int((time_max_unit - first_tick) / nice_spacing) + 1

    ticks_unit = first_tick + np.arange(n_ticks) * nice_spacing

    # Convert back to seconds
    ticks = ticks_unit / multiplier

    # Filter to range
    filtered_ticks: NDArray[np.float64] = ticks[(ticks >= time_min) & (ticks <= time_max)]

    return filtered_ticks


def _round_to_nice_time(value: float) -> float:
    """Round to nice time value (1, 2, 5, 10, 20, 50 × 10^n).  # noqa: RUF002

    Args:
        value: Value to round.

    Returns:
        Nice rounded value.
    """
    if value <= 0:
        return 1.0

    exponent = np.floor(np.log10(value))
    mantissa = value / (10**exponent)

    # Nice fractions for time
    nice_fractions = [1.0, 2.0, 5.0, 10.0]

    # Find closest
    distances = [abs(f - mantissa) for f in nice_fractions]
    min_idx = np.argmin(distances)
    nice_mantissa = nice_fractions[min_idx]

    # Handle overflow
    if nice_mantissa >= 10.0:
        nice_mantissa = 1.0
        exponent += 1

    return nice_mantissa * (10**exponent)  # type: ignore[no-any-return]


def format_cursor_readout(
    time_value: float,
    *,
    unit: TimeUnit = "auto",
    full_precision: bool = True,
) -> str:
    """Format time value for cursor readout with full precision.

    Args:
        time_value: Time value in seconds.
        unit: Display unit.
        full_precision: Show full floating-point precision.

    Returns:
        Formatted time string.

    Example:
        >>> readout = format_cursor_readout(1.23456789e-6, unit="us")
        >>> # Returns "1.23456789 μs"

    References:
        VIS-014: Adaptive X-Axis Time Window (cursor readout)
    """
    # Select unit if auto
    if unit == "auto":
        unit = select_time_unit(abs(time_value))

    # Convert to unit
    time_converted = convert_time_values(np.array([time_value]), unit)[0]

    # Unit symbols
    unit_symbols = {
        "s": "s",
        "ms": "ms",
        "us": "μs",
        "ns": "ns",
        "ps": "ps",
    }

    symbol = unit_symbols.get(unit, unit)

    # Format with appropriate precision
    if full_precision:
        # Maximum useful precision (avoid floating point noise)
        formatted = f"{time_converted:.12g}"
    else:
        # Standard precision
        formatted = f"{time_converted:.6g}"

    return f"{formatted} {symbol}"


__all__ = [
    "TimeUnit",
    "calculate_major_ticks",
    "convert_time_values",
    "create_relative_time",
    "format_cursor_readout",
    "format_time_labels",
    "select_time_unit",
]
