"""Jitter timing measurements.

This module provides cycle-to-cycle jitter, period jitter, and
duty cycle distortion measurements.


Example:
    >>> from oscura.analyzers.jitter.measurements import cycle_to_cycle_jitter
    >>> c2c = cycle_to_cycle_jitter(periods)
    >>> print(f"C2C RMS: {c2c.c2c_rms * 1e12:.2f} ps")

References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CycleJitterResult:
    """Result of cycle-to-cycle or period jitter measurement.

    Attributes:
        c2c_rms: Cycle-to-cycle jitter RMS in seconds.
        c2c_pp: Cycle-to-cycle jitter peak-to-peak in seconds.
        c2c_values: Array of individual C2C jitter values.
        period_mean: Mean period in seconds.
        period_std: Standard deviation of periods in seconds.
        n_cycles: Number of cycles analyzed.
        histogram: Histogram of C2C values.
        bin_centers: Bin centers for histogram.
    """

    c2c_rms: float
    c2c_pp: float
    c2c_values: NDArray[np.float64]
    period_mean: float
    period_std: float
    n_cycles: int
    histogram: NDArray[np.float64] | None = None
    bin_centers: NDArray[np.float64] | None = None


@dataclass
class DutyCycleDistortionResult:
    """Result of duty cycle distortion measurement.

    Attributes:
        dcd_seconds: DCD in seconds.
        dcd_percent: DCD as percentage of period.
        mean_high_time: Mean high time in seconds.
        mean_low_time: Mean low time in seconds.
        duty_cycle: Actual duty cycle as fraction (0.0 to 1.0).
        period: Mean period in seconds.
        n_cycles: Number of cycles analyzed.
    """

    dcd_seconds: float
    dcd_percent: float
    mean_high_time: float
    mean_low_time: float
    duty_cycle: float
    period: float
    n_cycles: int


def tie_from_edges(
    edge_timestamps: NDArray[np.float64],
    nominal_period: float | None = None,
) -> NDArray[np.float64]:
    """Calculate Time Interval Error from edge timestamps.

    TIE is the deviation of each edge from its ideal position
    based on the recovered clock period.

    Args:
        edge_timestamps: Array of edge timestamps in seconds.
        nominal_period: Expected period (computed from data if None).

    Returns:
        Array of TIE values in seconds.

    Example:
        >>> tie = tie_from_edges(rising_edges, nominal_period=1e-9)
        >>> print(f"TIE range: {np.ptp(tie) * 1e12:.2f} ps")
    """
    if len(edge_timestamps) < 3:
        return np.array([], dtype=np.float64)

    # Calculate actual periods
    periods = np.diff(edge_timestamps)

    # Use mean period if nominal not provided
    if nominal_period is None:
        nominal_period = np.mean(periods)

    # Calculate ideal edge positions
    n_edges = len(edge_timestamps)
    start_time = edge_timestamps[0]
    ideal_positions = start_time + np.arange(n_edges) * nominal_period

    # TIE is actual - ideal
    tie: NDArray[np.float64] = edge_timestamps - ideal_positions

    return tie


def cycle_to_cycle_jitter(
    periods: NDArray[np.float64],
    *,
    include_histogram: bool = True,
    n_bins: int = 50,
) -> CycleJitterResult:
    """Measure cycle-to-cycle jitter for clock quality analysis.

    Cycle-to-cycle jitter measures the variation in period from
    one clock cycle to the next: C2C[n] = |Period[n] - Period[n-1]|

    Args:
        periods: Array of measured clock periods in seconds.
        include_histogram: Include histogram in result.
        n_bins: Number of histogram bins.

    Returns:
        CycleJitterResult with C2C jitter statistics.

    Raises:
        InsufficientDataError: If fewer than 3 periods provided.

    Example:
        >>> c2c = cycle_to_cycle_jitter(periods)
        >>> print(f"C2C: {c2c.c2c_rms * 1e12:.2f} ps RMS")

    References:
        IEEE 2414-2020 Section 5.3
    """
    if len(periods) < 3:
        raise InsufficientDataError(
            "Cycle-to-cycle jitter requires at least 3 periods",
            required=3,
            available=len(periods),
            analysis_type="cycle_to_cycle_jitter",
        )

    # Remove NaN values
    valid_periods = periods[~np.isnan(periods)]

    if len(valid_periods) < 3:
        raise InsufficientDataError(
            "Cycle-to-cycle jitter requires at least 3 valid periods",
            required=3,
            available=len(valid_periods),
            analysis_type="cycle_to_cycle_jitter",
        )

    # Calculate cycle-to-cycle differences
    c2c_values = np.abs(np.diff(valid_periods))

    # Statistics
    c2c_rms = float(np.sqrt(np.mean(c2c_values**2)))
    c2c_pp = float(np.max(c2c_values) - np.min(c2c_values))
    period_mean = float(np.mean(valid_periods))
    period_std = float(np.std(valid_periods))

    # Optional histogram
    if include_histogram and len(c2c_values) > 10:
        hist, bin_edges = np.histogram(c2c_values, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        hist = None
        bin_centers = None

    return CycleJitterResult(
        c2c_rms=c2c_rms,
        c2c_pp=c2c_pp,
        c2c_values=c2c_values,
        period_mean=period_mean,
        period_std=period_std,
        n_cycles=len(valid_periods),
        histogram=hist,
        bin_centers=bin_centers,
    )


def period_jitter(
    periods: NDArray[np.float64],
    nominal_period: float | None = None,
) -> CycleJitterResult:
    """Measure period jitter (deviation from nominal period).

    Period jitter is the deviation of each period from the ideal
    or nominal period. Unlike C2C jitter, it measures absolute deviation.

    Args:
        periods: Array of measured clock periods in seconds.
        nominal_period: Expected period (uses mean if None).

    Returns:
        CycleJitterResult with period jitter statistics.

    Raises:
        InsufficientDataError: If fewer than 2 periods provided.

    Example:
        >>> pj = period_jitter(periods, nominal_period=1e-9)
        >>> print(f"Period jitter: {pj.c2c_rms * 1e12:.2f} ps RMS")
    """
    if len(periods) < 2:
        raise InsufficientDataError(
            "Period jitter requires at least 2 periods",
            required=2,
            available=len(periods),
            analysis_type="period_jitter",
        )

    valid_periods = periods[~np.isnan(periods)]

    if nominal_period is None:
        nominal_period = np.mean(valid_periods)

    # Calculate deviations from nominal
    deviations = valid_periods - nominal_period

    return CycleJitterResult(
        c2c_rms=float(np.std(valid_periods)),  # RMS of period variation
        c2c_pp=float(np.max(valid_periods) - np.min(valid_periods)),
        c2c_values=np.abs(deviations),
        period_mean=float(np.mean(valid_periods)),
        period_std=float(np.std(valid_periods)),
        n_cycles=len(valid_periods),
    )


def measure_dcd(
    trace: WaveformTrace | DigitalTrace,
    clock_period: float | None = None,
    *,
    threshold: float = 0.5,
) -> DutyCycleDistortionResult:
    """Measure duty cycle distortion.

    DCD measures the asymmetry between high and low times in a clock signal.
    DCD = |mean_high_time - mean_low_time|

    Args:
        trace: Input waveform or digital trace.
        clock_period: Expected clock period (computed if None).
        threshold: Threshold level as fraction of amplitude (0.0-1.0).

    Returns:
        DutyCycleDistortionResult with DCD metrics.

    Raises:
        InsufficientDataError: If not enough edges found.

    Example:
        >>> dcd = measure_dcd(clock_trace, clock_period=1e-9)
        >>> print(f"DCD: {dcd.dcd_percent:.1f}%")

    References:
        IEEE 2414-2020 Section 5.4
    """
    # Get edge timestamps
    rising_edges, falling_edges = _find_edges(trace, threshold)

    if len(rising_edges) < 2 or len(falling_edges) < 2:
        raise InsufficientDataError(
            "DCD measurement requires at least 2 rising and 2 falling edges",
            required=4,
            available=len(rising_edges) + len(falling_edges),
            analysis_type="dcd_measurement",
        )

    # Measure high times (rising to falling)
    high_times = []
    for r_edge in rising_edges:
        # Find next falling edge
        next_falling = falling_edges[falling_edges > r_edge]
        if len(next_falling) > 0:
            high_times.append(next_falling[0] - r_edge)

    # Measure low times (falling to rising)
    low_times = []
    for f_edge in falling_edges:
        # Find next rising edge
        next_rising = rising_edges[rising_edges > f_edge]
        if len(next_rising) > 0:
            low_times.append(next_rising[0] - f_edge)

    if len(high_times) < 1 or len(low_times) < 1:
        raise InsufficientDataError(
            "Could not measure high/low times",
            required=2,
            available=0,
            analysis_type="dcd_measurement",
        )

    mean_high = float(np.mean(high_times))
    mean_low = float(np.mean(low_times))

    # Calculate DCD
    dcd_seconds = abs(mean_high - mean_low)
    period = mean_high + mean_low

    if clock_period is None:
        clock_period = period

    dcd_percent = (dcd_seconds / clock_period) * 100
    duty_cycle = mean_high / period

    return DutyCycleDistortionResult(
        dcd_seconds=dcd_seconds,
        dcd_percent=dcd_percent,
        mean_high_time=mean_high,
        mean_low_time=mean_low,
        duty_cycle=duty_cycle,
        period=period,
        n_cycles=min(len(high_times), len(low_times)),
    )


def _find_edges(
    trace: WaveformTrace | DigitalTrace,
    threshold_frac: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find rising and falling edge timestamps with sub-sample interpolation.

    Args:
        trace: Input trace.
        threshold_frac: Threshold as fraction of amplitude.

    Returns:
        Tuple of (rising_edges, falling_edges) arrays in seconds.
    """
    data = trace.data.astype(np.float64) if isinstance(trace, DigitalTrace) else trace.data

    sample_rate = trace.metadata.sample_rate
    sample_period = 1.0 / sample_rate

    if len(data) < 3:
        return np.array([]), np.array([])

    # Find amplitude levels - use more extreme percentiles for better accuracy
    low = np.percentile(data, 5)
    high = np.percentile(data, 95)
    threshold = low + threshold_frac * (high - low)

    # Find crossings
    above = data >= threshold
    below = data < threshold

    rising_indices = np.where(below[:-1] & above[1:])[0]
    falling_indices = np.where(above[:-1] & below[1:])[0]

    # Convert to timestamps with linear interpolation
    # For a crossing between samples i and i+1:
    # time = i * dt + (threshold - v[i]) / (v[i+1] - v[i]) * dt

    rising_edges = []
    for idx in rising_indices:
        v1, v2 = data[idx], data[idx + 1]
        dv = v2 - v1
        if abs(dv) > 1e-12:
            # Linear interpolation to find exact crossing time
            frac = (threshold - v1) / dv
            # Clamp to [0, 1] to handle numerical errors
            frac = max(0.0, min(1.0, frac))
            t_offset = frac * sample_period
        else:
            # Values are equal, use midpoint
            t_offset = sample_period / 2
        rising_edges.append(idx * sample_period + t_offset)

    falling_edges = []
    for idx in falling_indices:
        v1, v2 = data[idx], data[idx + 1]
        dv = v2 - v1
        if abs(dv) > 1e-12:
            # Linear interpolation to find exact crossing time
            frac = (threshold - v1) / dv
            # Clamp to [0, 1] to handle numerical errors
            frac = max(0.0, min(1.0, frac))
            t_offset = frac * sample_period
        else:
            # Values are equal, use midpoint
            t_offset = sample_period / 2
        falling_edges.append(idx * sample_period + t_offset)

    return (
        np.array(rising_edges, dtype=np.float64),
        np.array(falling_edges, dtype=np.float64),
    )


__all__ = [
    "CycleJitterResult",
    "DutyCycleDistortionResult",
    "cycle_to_cycle_jitter",
    "measure_dcd",
    "period_jitter",
    "tie_from_edges",
]
