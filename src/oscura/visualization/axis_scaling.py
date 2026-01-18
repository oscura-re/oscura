"""Intelligent axis scaling and range optimization.

This module provides enhanced Y-axis scaling with nice number rounding,
outlier exclusion, and per-channel scaling for multi-channel plots.


Example:
    >>> from oscura.visualization.axis_scaling import calculate_axis_limits
    >>> y_min, y_max = calculate_axis_limits(signal, nice_numbers=True)

References:
    - Wilkinson's tick placement algorithm
    - Percentile-based outlier detection
    - IEEE publication best practices
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def calculate_axis_limits(
    data: NDArray[np.float64],
    *,
    nice_numbers: bool = True,
    outlier_percentile: float = 1.0,
    margin_percent: float = 5.0,
    symmetric: bool = False,
    zero_centered: bool = False,
) -> tuple[float, float]:
    """Calculate optimal axis limits with nice number rounding.

    Enhanced version with nice number rounding for publication-quality plots.

    Args:
        data: Signal data array.
        nice_numbers: Round limits to nice numbers (1, 2, 5 × 10^n).  # noqa: RUF002
        outlier_percentile: Percentile for outlier exclusion (default 1% each side).
        margin_percent: Margin as percentage of data range (default 5%).
        symmetric: Use symmetric range ±max for bipolar signals.
        zero_centered: Force zero to be centered in range.

    Returns:
        Tuple of (y_min, y_max) with nice rounded values.

    Raises:
        ValueError: If data is empty or all NaN.

    Example:
        >>> signal = np.array([1.234, 5.678, 9.012])
        >>> y_min, y_max = calculate_axis_limits(signal, nice_numbers=True)
        >>> # Returns nice values like (0.0, 10.0) instead of (1.234, 9.012)

    References:
        VIS-013: Auto Y-Axis Range Optimization
        Wilkinson (1999): The Grammar of Graphics
    """
    if len(data) == 0:
        raise ValueError("Data array is empty")

    # Remove NaN values
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        raise ValueError("Data contains only NaN values")

    # Exclude outliers using percentiles
    lower_pct = outlier_percentile
    upper_pct = 100.0 - outlier_percentile

    data_min = np.percentile(clean_data, lower_pct)
    data_max = np.percentile(clean_data, upper_pct)
    data_range = data_max - data_min

    # Apply margin
    margin = margin_percent / 100.0
    margin_value = data_range * margin

    if symmetric:
        # Symmetric range: ±max
        max_abs = max(abs(data_min), abs(data_max))
        y_min = -(max_abs + margin_value)
        y_max = max_abs + margin_value
    elif zero_centered:
        # Force zero to be centered
        max_extent = max(abs(data_min), abs(data_max)) + margin_value
        y_min = -max_extent
        y_max = max_extent
    else:
        # Asymmetric range
        y_min = data_min - margin_value
        y_max = data_max + margin_value

    # Round to nice numbers if requested
    if nice_numbers:
        y_min = _round_to_nice_number(y_min, direction="down")
        y_max = _round_to_nice_number(y_max, direction="up")

    return (float(y_min), float(y_max))


def calculate_multi_channel_limits(  # type: ignore[no-untyped-def]
    channels: list[NDArray[np.float64]],
    *,
    mode: Literal["per_channel", "common", "grouped"] = "per_channel",
    nice_numbers: bool = True,
    **kwargs,
) -> list[tuple[float, float]]:
    """Calculate axis limits for multiple channels.

    Args:
        channels: List of channel data arrays.
        mode: Scaling mode:
            - "per_channel": Independent ranges per channel
            - "common": Single range for all channels
            - "grouped": Group similar ranges
        nice_numbers: Round to nice numbers.
        **kwargs: Additional arguments passed to calculate_axis_limits.

    Returns:
        List of (y_min, y_max) tuples, one per channel.

    Raises:
        ValueError: If unknown mode specified.

    Example:
        >>> ch1 = np.array([0, 1, 2])
        >>> ch2 = np.array([0, 10, 20])
        >>> limits = calculate_multi_channel_limits([ch1, ch2], mode="per_channel")

    References:
        VIS-013: Auto Y-Axis Range Optimization (per-channel scaling)
        VIS-015: Multi-Channel Stack Optimization
    """
    if len(channels) == 0:
        return []

    if mode == "per_channel":
        # Independent ranges
        return [calculate_axis_limits(ch, nice_numbers=nice_numbers, **kwargs) for ch in channels]

    elif mode == "common":
        # Single range for all channels
        all_data = np.concatenate([ch for ch in channels if len(ch) > 0])
        if len(all_data) == 0:
            return [(0.0, 1.0)] * len(channels)

        common_limits = calculate_axis_limits(all_data, nice_numbers=nice_numbers, **kwargs)
        return [common_limits] * len(channels)

    elif mode == "grouped":
        # Group channels with similar ranges
        # First calculate individual ranges
        individual_limits = [
            calculate_axis_limits(ch, nice_numbers=False, **kwargs) for ch in channels
        ]

        # Simple grouping: group by order of magnitude
        grouped_limits = []
        for y_min, y_max in individual_limits:
            range_mag = np.log10(max(abs(y_max - y_min), 1e-10))
            # Round to nearest integer magnitude
            group_mag = int(np.round(range_mag))

            # Use 10^group_mag as the range scale
            scale = 10.0**group_mag

            # Round to this scale
            grouped_min = np.floor(y_min / scale) * scale
            grouped_max = np.ceil(y_max / scale) * scale

            if nice_numbers:
                grouped_min = _round_to_nice_number(grouped_min, direction="down")
                grouped_max = _round_to_nice_number(grouped_max, direction="up")

            grouped_limits.append((float(grouped_min), float(grouped_max)))

        return grouped_limits

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _round_to_nice_number(
    value: float,
    *,
    direction: Literal["up", "down", "nearest"] = "nearest",
) -> float:
    """Round value to nice number (1, 2, 5 × 10^n).  # noqa: RUF002

    Args:
        value: Value to round.
        direction: Rounding direction ("up", "down", "nearest").

    Returns:
        Rounded nice number.

    Example:
        >>> _round_to_nice_number(3.7, direction="up")
        5.0
        >>> _round_to_nice_number(3.7, direction="down")
        2.0
        >>> _round_to_nice_number(0.037, direction="up")
        0.05
    """
    if value == 0:
        return 0.0

    # Determine sign
    sign = 1 if value >= 0 else -1
    abs_value = abs(value)

    # Find exponent
    exponent = np.floor(np.log10(abs_value))
    mantissa = abs_value / (10**exponent)

    # Round mantissa to nice fraction (1, 2, 5)
    nice_fractions = [1.0, 2.0, 5.0, 10.0]

    if direction == "up":
        # Find smallest nice fraction >= mantissa
        nice_mantissa = next((f for f in nice_fractions if f >= mantissa), 10.0)
    elif direction == "down":
        # Find largest nice fraction <= mantissa
        nice_mantissa = 1.0
        for f in nice_fractions:
            if f <= mantissa:
                nice_mantissa = f
            else:
                break
    else:  # nearest
        # Find closest nice fraction
        distances = [abs(f - mantissa) for f in nice_fractions]
        min_idx = np.argmin(distances)
        nice_mantissa = nice_fractions[min_idx]

    # Handle mantissa = 10 case (move to next exponent)
    if nice_mantissa >= 10.0:
        nice_mantissa = 1.0
        exponent += 1

    return sign * nice_mantissa * (10**exponent)  # type: ignore[no-any-return]


def suggest_tick_spacing(
    y_min: float,
    y_max: float,
    *,
    target_ticks: int = 5,
    minor_ticks: bool = True,
) -> tuple[float, float]:
    """Suggest tick spacing for axis.

    Args:
        y_min: Minimum axis value.
        y_max: Maximum axis value.
        target_ticks: Target number of major ticks.
        minor_ticks: Generate minor tick spacing.

    Returns:
        Tuple of (major_spacing, minor_spacing).

    Example:
        >>> major, minor = suggest_tick_spacing(0, 10, target_ticks=5)
        >>> # Returns (2.0, 0.5) for nice tick marks at 0, 2, 4, 6, 8, 10

    References:
        VIS-019: Grid Auto-Spacing
    """
    axis_range = y_max - y_min

    if axis_range <= 0:
        return (1.0, 0.2)

    # Calculate rough spacing
    rough_spacing = axis_range / target_ticks

    # Round to nice number
    major_spacing = _round_to_nice_number(rough_spacing, direction="nearest")

    # Minor spacing: 1/5 of major for most cases
    if minor_ticks:
        # Use 1/5 for multiples of 5, 1/4 for multiples of 2, 1/2 otherwise
        if major_spacing % 5 == 0:
            minor_spacing = major_spacing / 5
        elif major_spacing % 2 == 0:
            minor_spacing = major_spacing / 4
        else:
            minor_spacing = major_spacing / 2
    else:
        minor_spacing = major_spacing

    return (float(major_spacing), float(minor_spacing))


__all__ = [
    "calculate_axis_limits",
    "calculate_multi_channel_limits",
    "suggest_tick_spacing",
]
