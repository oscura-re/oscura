"""Interpolation and resampling operations for Oscura.

This module provides interpolation, resampling, and trace alignment
functions for waveform data.


Example:
    >>> from oscura.utils.math import resample, align_traces
    >>> resampled = resample(trace, new_sample_rate=1e6)
    >>> aligned = align_traces(trace1, trace2)

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import interpolate as sp_interp
from scipy import signal as sp_signal

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def interpolate(
    trace: WaveformTrace,
    new_time: NDArray[np.float64],
    *,
    method: Literal["linear", "cubic", "nearest", "zero"] = "linear",
    fill_value: float | tuple[float, float] = np.nan,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Interpolate trace to new time points.

    Interpolates the waveform data to a new set of time points using
    the specified interpolation method.

    Args:
        trace: Input trace.
        new_time: New time points in seconds.
        method: Interpolation method ("linear", "cubic", "nearest", "zero").
        fill_value: Value for points outside original range.
        channel_name: Name for the result trace (optional).

    Returns:
        Interpolated WaveformTrace at new time points.

    Raises:
        InsufficientDataError: If trace has insufficient samples.
        ValueError: If interpolation method is unknown.

    Example:
        >>> new_time = np.linspace(0, 1e-3, 2000)
        >>> interpolated = interpolate(trace, new_time, method="cubic")
    """
    if len(trace.data) < 2:
        raise InsufficientDataError(
            "Need at least 2 samples for interpolation",
            required=2,
            available=len(trace.data),
            analysis_type="interpolate",
        )

    # Create interpolator and interpolate
    interp_func = _create_interpolator(
        trace.time_vector, trace.data.astype(np.float64), method, fill_value
    )
    result_data = interp_func(new_time)

    # Build result trace
    new_sample_rate = _calculate_new_sample_rate(new_time, trace.metadata.sample_rate)
    new_metadata = _create_interpolated_metadata(trace, new_sample_rate, channel_name)

    return WaveformTrace(data=result_data.astype(np.float64), metadata=new_metadata)


def _create_interpolator(
    original_time: NDArray[np.float64],
    data: NDArray[np.float64],
    method: str,
    fill_value: float | tuple[float, float],
) -> Any:
    """Create scipy interpolation function."""
    valid_methods = {"linear", "cubic", "nearest", "zero"}
    if method not in valid_methods:
        raise ValueError(f"Unknown interpolation method: {method}")

    return sp_interp.interp1d(
        original_time, data, kind=method, bounds_error=False, fill_value=fill_value
    )


def _calculate_new_sample_rate(new_time: NDArray[np.float64], original_sample_rate: float) -> float:
    """Calculate new sample rate from time points."""
    if len(new_time) > 1:
        mean_diff: np.floating[Any] = np.mean(np.diff(new_time))
        new_rate: float = float(1.0 / mean_diff)
        return new_rate
    return original_sample_rate


def _create_interpolated_metadata(
    trace: WaveformTrace, new_sample_rate: float, channel_name: str | None
) -> TraceMetadata:
    """Create metadata for interpolated trace."""
    return TraceMetadata(
        sample_rate=new_sample_rate,
        vertical_scale=trace.metadata.vertical_scale,
        vertical_offset=trace.metadata.vertical_offset,
        acquisition_time=trace.metadata.acquisition_time,
        trigger_info=trace.metadata.trigger_info,
        source_file=trace.metadata.source_file,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_interp",
    )


def _calculate_target_params(
    new_sample_rate: float | None,
    num_samples: int | None,
    original_rate: float,
    original_samples: int,
) -> tuple[float, int]:
    """Calculate target sample rate and sample count for resampling."""
    if new_sample_rate is not None:
        target_rate = new_sample_rate
        target_samples = round(original_samples * target_rate / original_rate)
    else:
        target_samples = num_samples  # type: ignore[assignment]
        target_rate = original_rate * target_samples / original_samples
    return target_rate, target_samples


def _check_nyquist_violation(
    data: NDArray[np.float64], original_rate: float, target_rate: float
) -> None:
    """Validate Nyquist criterion when downsampling and warn if violated."""
    fft_data = np.fft.rfft(data)
    fft_freqs = np.fft.rfftfreq(len(data), 1 / original_rate)
    power = np.abs(fft_data) ** 2
    power_threshold = 0.01 * np.max(power)
    significant_freqs = fft_freqs[power > power_threshold]

    if len(significant_freqs) > 0:
        max_frequency = np.max(significant_freqs)
        nyquist_required = 2 * max_frequency
        if target_rate < nyquist_required:
            warnings.warn(
                f"Downsampling to {target_rate:.2e} Hz violates Nyquist criterion. "
                f"Maximum signal frequency is ~{max_frequency:.2e} Hz, "
                f"requiring ≥{nyquist_required:.2e} Hz sample rate. "
                f"Aliasing may occur.",
                UserWarning,
                stacklevel=3,
            )


def _apply_anti_alias_filter(
    data: NDArray[np.float64], target_rate: float, original_rate: float
) -> NDArray[np.float64]:
    """Apply lowpass anti-aliasing filter before downsampling."""
    nyquist = target_rate / 2
    cutoff = nyquist / original_rate * 2  # Normalized frequency
    if cutoff < 1.0:
        b, a = sp_signal.butter(8, min(cutoff * 0.9, 0.99), btype="low")
        filtered: NDArray[np.float64] = np.asarray(sp_signal.filtfilt(b, a, data), dtype=np.float64)
        return filtered
    return data


def _perform_resampling(
    data: NDArray[np.float64],
    method: Literal["fft", "polyphase", "interp"],
    target_samples: int,
    original_samples: int,
    original_rate: float,
    target_rate: float,
) -> NDArray[np.float64]:
    """Perform the actual resampling based on selected method."""
    if method == "fft":
        resampled: NDArray[np.float64] = np.asarray(
            sp_signal.resample(data, target_samples), dtype=np.float64
        )
        return resampled
    elif method == "polyphase":
        from fractions import Fraction

        ratio = Fraction(target_samples, original_samples).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        result = sp_signal.resample_poly(data, up, down)
        truncated: NDArray[np.float64] = np.asarray(result[:target_samples], dtype=np.float64)
        return truncated
    elif method == "interp":
        old_time = np.arange(original_samples) / original_rate
        new_time = np.arange(target_samples) / target_rate
        interpolated: NDArray[np.float64] = np.asarray(
            np.interp(new_time, old_time, data), dtype=np.float64
        )
        return interpolated
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def resample(
    trace: WaveformTrace,
    new_sample_rate: float | None = None,
    *,
    num_samples: int | None = None,
    method: Literal["fft", "polyphase", "interp"] = "fft",
    anti_alias: bool = True,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Resample trace to new sample rate or number of samples.

    Resamples the waveform to a different sample rate using high-quality
    resampling algorithms. Applies anti-aliasing filter when downsampling.

    Args:
        trace: Input trace.
        new_sample_rate: Target sample rate in Hz. Mutually exclusive
            with num_samples.
        num_samples: Target number of samples. Mutually exclusive with
            new_sample_rate.
        method: Resampling method:
            - "fft": FFT-based resampling (default, best quality)
            - "polyphase": Polyphase filter resampling (efficient)
            - "interp": Linear interpolation (fastest)
        anti_alias: Apply anti-aliasing filter before downsampling.
        channel_name: Name for the result trace (optional).

    Returns:
        Resampled WaveformTrace.

    Raises:
        ValueError: If neither or both rate/samples specified.
        InsufficientDataError: If trace has insufficient samples.

    Example:
        >>> upsampled = resample(trace, new_sample_rate=2e9)
        >>> downsampled = resample(trace, num_samples=1000)

    References:
        MEM-012 (downsampling for memory management)
    """
    # Validate inputs
    if (new_sample_rate is None) == (num_samples is None):
        raise ValueError("Specify exactly one of new_sample_rate or num_samples")
    if len(trace.data) < 2:
        raise InsufficientDataError(
            "Need at least 2 samples for resampling",
            required=2,
            available=len(trace.data),
            analysis_type="resample",
        )

    # Setup
    data = trace.data.astype(np.float64)
    original_rate = trace.metadata.sample_rate
    original_samples = len(data)

    # Calculate target parameters
    target_rate, target_samples = _calculate_target_params(
        new_sample_rate, num_samples, original_rate, original_samples
    )

    if target_samples < 1:
        raise ValueError("Target number of samples must be at least 1")

    # REQ: API-019 - Validate Nyquist criterion when downsampling
    if target_rate < original_rate:
        _check_nyquist_violation(data, original_rate, target_rate)

    # Apply anti-aliasing filter if downsampling
    if anti_alias and target_samples < original_samples:
        data = _apply_anti_alias_filter(data, target_rate, original_rate)

    # Perform resampling
    result_data = _perform_resampling(
        data, method, target_samples, original_samples, original_rate, target_rate
    )

    # Build output trace
    new_metadata = TraceMetadata(
        sample_rate=target_rate,
        vertical_scale=trace.metadata.vertical_scale,
        vertical_offset=trace.metadata.vertical_offset,
        acquisition_time=trace.metadata.acquisition_time,
        trigger_info=trace.metadata.trigger_info,
        source_file=trace.metadata.source_file,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_resampled",
    )

    return WaveformTrace(data=result_data.astype(np.float64), metadata=new_metadata)


def align_traces(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    method: Literal["interpolate", "resample"] = "interpolate",
    reference: Literal["first", "second", "higher"] = "higher",
    channel_names: tuple[str | None, str | None] | None = None,
) -> tuple[WaveformTrace, WaveformTrace]:
    """Align two traces to have the same sample rate and length.

    Adjusts two traces to be compatible for arithmetic operations by
    resampling to a common sample rate and time base.

    Args:
        trace1: First trace.
        trace2: Second trace.
        method: Alignment method:
            - "interpolate": Interpolate to common time points
            - "resample": Resample to common rate
        reference: Which trace to use as reference:
            - "first": Use trace1's sample rate
            - "second": Use trace2's sample rate
            - "higher": Use the higher sample rate (default)
        channel_names: Optional names for the aligned traces.

    Returns:
        Tuple of (aligned_trace1, aligned_trace2) with matching parameters.

    Example:
        >>> aligned1, aligned2 = align_traces(trace1, trace2)
        >>> diff = subtract(aligned1, aligned2)
    """
    rate1 = trace1.metadata.sample_rate
    rate2 = trace2.metadata.sample_rate

    # Determine reference sample rate
    if reference == "first":
        target_rate = rate1
    elif reference == "second":
        target_rate = rate2
    else:  # "higher"
        target_rate = max(rate1, rate2)

    # Determine time span (use overlapping portion)
    t1_end = len(trace1.data) / rate1
    t2_end = len(trace2.data) / rate2
    common_end = min(t1_end, t2_end)

    # Calculate number of samples
    num_samples = round(common_end * target_rate)

    # Create common time vector
    common_time = np.arange(num_samples) / target_rate

    name1 = channel_names[0] if channel_names else None
    name2 = channel_names[1] if channel_names else None

    if method == "interpolate":
        # Interpolate both traces to common time points
        aligned1 = interpolate(trace1, common_time, channel_name=name1)
        aligned2 = interpolate(trace2, common_time, channel_name=name2)
    else:  # "resample"
        # Resample both to common rate
        aligned1 = resample(trace1, num_samples=num_samples, channel_name=name1)
        aligned2 = resample(trace2, num_samples=num_samples, channel_name=name2)

    return aligned1, aligned2


def downsample(
    trace: WaveformTrace,
    factor: int,
    *,
    anti_alias: bool = True,
    method: Literal["decimate", "average", "max", "min"] = "decimate",
    channel_name: str | None = None,
) -> WaveformTrace:
    """Downsample trace by an integer factor.

    Reduces the sample rate by keeping every Nth sample (decimate)
    or by aggregating N samples (average/max/min).

    Args:
        trace: Input trace.
        factor: Downsampling factor (must be >= 1).
        anti_alias: Apply anti-aliasing filter before decimation.
        method: Downsampling method:
            - "decimate": Keep every Nth sample (default)
            - "average": Average every N samples
            - "max": Maximum of every N samples
            - "min": Minimum of every N samples
        channel_name: Name for the result trace (optional).

    Returns:
        Downsampled WaveformTrace.

    Raises:
        ValueError: If factor is less than 1 or method is unknown.

    Example:
        >>> small = downsample(large_trace, factor=10)

    References:
        MEM-012 (memory management)
    """
    if factor < 1:
        raise ValueError(f"Factor must be >= 1, got {factor}")

    if factor == 1:
        return trace  # No change needed

    data = trace.data.astype(np.float64)

    if anti_alias and method == "decimate":
        # Apply anti-aliasing filter
        nyquist = 0.5 / factor
        b, a = sp_signal.butter(8, min(nyquist * 0.9, 0.99), btype="low")
        data = sp_signal.filtfilt(b, a, data)

    # Truncate to multiple of factor
    n = len(data) // factor * factor
    data = data[:n]

    if method == "decimate":
        result_data = data[::factor]
    elif method == "average":
        result_data = data.reshape(-1, factor).mean(axis=1)
    elif method == "max":
        result_data = data.reshape(-1, factor).max(axis=1)
    elif method == "min":
        result_data = data.reshape(-1, factor).min(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    new_metadata = TraceMetadata(
        sample_rate=trace.metadata.sample_rate / factor,
        vertical_scale=trace.metadata.vertical_scale,
        vertical_offset=trace.metadata.vertical_offset,
        acquisition_time=trace.metadata.acquisition_time,
        trigger_info=trace.metadata.trigger_info,
        source_file=trace.metadata.source_file,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_ds{factor}",
    )

    return WaveformTrace(data=result_data, metadata=new_metadata)
