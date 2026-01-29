"""Convenience filtering functions for Oscura.

Provides simple one-call filter functions for common operations like
moving average, median filter, Savitzky-Golay smoothing, and matched
filtering.

Example:
    >>> from oscura.utils.filtering.convenience import low_pass, moving_average
    >>> filtered = low_pass(trace, cutoff=1e6)
    >>> smoothed = moving_average(trace, window_size=11)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import ndimage, signal

from oscura.core.exceptions import AnalysisError
from oscura.core.types import WaveformTrace
from oscura.utils.filtering.design import (
    BandPassFilter,
    BandStopFilter,
    HighPassFilter,
    LowPassFilter,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def low_pass(
    trace: WaveformTrace,
    cutoff: float,
    *,
    order: int = 4,
    filter_type: Literal[
        "butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"
    ] = "butterworth",
) -> WaveformTrace:
    """Apply low-pass filter to trace.

    Args:
        trace: Input waveform trace.
        cutoff: Cutoff frequency in Hz.
        order: Filter order (default 4).
        filter_type: Type of filter (default Butterworth).

    Returns:
        Filtered waveform trace.

    Example:
        >>> filtered = low_pass(trace, cutoff=1e6)
    """
    filt = LowPassFilter(
        cutoff=cutoff,
        sample_rate=trace.metadata.sample_rate,
        order=order,
        filter_type=filter_type,
    )
    result = filt.apply(trace)
    if isinstance(result, WaveformTrace):
        return result
    return result.trace


def high_pass(
    trace: WaveformTrace,
    cutoff: float,
    *,
    order: int = 4,
    filter_type: Literal[
        "butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"
    ] = "butterworth",
) -> WaveformTrace:
    """Apply high-pass filter to trace.

    Args:
        trace: Input waveform trace.
        cutoff: Cutoff frequency in Hz.
        order: Filter order (default 4).
        filter_type: Type of filter (default Butterworth).

    Returns:
        Filtered waveform trace.

    Example:
        >>> filtered = high_pass(trace, cutoff=100)  # Remove DC and low frequencies
    """
    filt = HighPassFilter(
        cutoff=cutoff,
        sample_rate=trace.metadata.sample_rate,
        order=order,
        filter_type=filter_type,
    )
    result = filt.apply(trace)
    if isinstance(result, WaveformTrace):
        return result
    return result.trace


def band_pass(
    trace: WaveformTrace,
    low: float,
    high: float,
    *,
    order: int = 4,
    filter_type: Literal[
        "butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"
    ] = "butterworth",
) -> WaveformTrace:
    """Apply band-pass filter to trace.

    Args:
        trace: Input waveform trace.
        low: Lower cutoff frequency in Hz.
        high: Upper cutoff frequency in Hz.
        order: Filter order (default 4).
        filter_type: Type of filter (default Butterworth).

    Returns:
        Filtered waveform trace.

    Example:
        >>> filtered = band_pass(trace, low=1e3, high=10e3)
    """
    filt = BandPassFilter(
        low=low,
        high=high,
        sample_rate=trace.metadata.sample_rate,
        order=order,
        filter_type=filter_type,
    )
    result = filt.apply(trace)
    if isinstance(result, WaveformTrace):
        return result
    return result.trace


def band_stop(
    trace: WaveformTrace,
    low: float,
    high: float,
    *,
    order: int = 4,
    filter_type: Literal[
        "butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"
    ] = "butterworth",
) -> WaveformTrace:
    """Apply band-stop (notch) filter to trace.

    Args:
        trace: Input waveform trace.
        low: Lower cutoff frequency in Hz.
        high: Upper cutoff frequency in Hz.
        order: Filter order (default 4).
        filter_type: Type of filter (default Butterworth).

    Returns:
        Filtered waveform trace.

    Example:
        >>> filtered = band_stop(trace, low=59, high=61)  # Remove 60 Hz
    """
    filt = BandStopFilter(
        low=low,
        high=high,
        sample_rate=trace.metadata.sample_rate,
        order=order,
        filter_type=filter_type,
    )
    result = filt.apply(trace)
    if isinstance(result, WaveformTrace):
        return result
    return result.trace


def notch_filter(
    trace: WaveformTrace,
    freq: float,
    *,
    q_factor: float = 30.0,
) -> WaveformTrace:
    """Apply narrow notch filter at specified frequency.

    Uses a band-stop Butterworth filter with bandwidth determined by Q factor.
    Bandwidth (Hz) = freq / Q

    Args:
        trace: Input waveform trace.
        freq: Center frequency to notch out in Hz.
        q_factor: Quality factor (higher = narrower notch). Default 30.

    Returns:
        Filtered waveform trace.

    Raises:
        AnalysisError: If notch frequency exceeds Nyquist frequency.

    Example:
        >>> filtered = notch_filter(trace, freq=60, q_factor=30)  # Remove 60 Hz line noise
    """
    sample_rate = trace.metadata.sample_rate

    if freq >= sample_rate / 2:
        raise AnalysisError(
            f"Notch frequency {freq} Hz must be less than Nyquist {sample_rate / 2} Hz"
        )

    # Calculate bandwidth from Q factor: BW = f0 / Q
    bandwidth = freq / q_factor

    # Design band-stop filter centered at freq with calculated bandwidth
    # Use 4th order Butterworth for good attenuation
    low = max(freq - bandwidth / 2, 0.1)  # Avoid zero frequency
    high = min(freq + bandwidth / 2, sample_rate / 2 - 1)  # Stay below Nyquist

    # Normalize frequencies
    wn = [low / (sample_rate / 2), high / (sample_rate / 2)]

    # Design bandstop filter
    sos = signal.butter(4, wn, btype="bandstop", output="sos")

    # Apply zero-phase filter
    filtered_data = signal.sosfiltfilt(sos, trace.data)

    return WaveformTrace(
        data=filtered_data.astype(np.float64),
        metadata=trace.metadata,
    )


def moving_average(
    trace: WaveformTrace,
    window_size: int,
    *,
    mode: Literal["same", "valid", "full"] = "same",
) -> WaveformTrace:
    """Apply moving average filter.

    Simple FIR filter with uniform weights.

    Args:
        trace: Input waveform trace.
        window_size: Number of samples in averaging window (must be odd for 'same' mode).
        mode: Convolution mode - "same" preserves length.

    Returns:
        Filtered waveform trace.

    Raises:
        AnalysisError: If window_size is not positive or exceeds data length.

    Example:
        >>> smoothed = moving_average(trace, window_size=11)
    """
    if window_size < 1:
        raise AnalysisError(f"Window size must be positive, got {window_size}")

    if window_size > len(trace.data):
        raise AnalysisError(f"Window size {window_size} exceeds data length {len(trace.data)}")

    kernel = np.ones(window_size) / window_size
    filtered_data = np.convolve(trace.data, kernel, mode=mode)

    return WaveformTrace(
        data=filtered_data.astype(np.float64),
        metadata=trace.metadata,
    )


def median_filter(
    trace: WaveformTrace,
    kernel_size: int,
) -> WaveformTrace:
    """Apply median filter for spike/impulse noise removal.

    Non-linear filter that preserves edges while removing outliers.

    Args:
        trace: Input waveform trace.
        kernel_size: Size of the median filter kernel (must be odd).

    Returns:
        Filtered waveform trace.

    Raises:
        AnalysisError: If kernel_size is not positive or not odd.

    Example:
        >>> cleaned = median_filter(trace, kernel_size=5)  # Remove impulse noise
    """
    if kernel_size < 1:
        raise AnalysisError(f"Kernel size must be positive, got {kernel_size}")

    if kernel_size % 2 == 0:
        raise AnalysisError(f"Kernel size must be odd, got {kernel_size}")

    filtered_data = ndimage.median_filter(trace.data, size=kernel_size)

    return WaveformTrace(
        data=filtered_data.astype(np.float64),
        metadata=trace.metadata,
    )


def savgol_filter(
    trace: WaveformTrace,
    window_length: int,
    polyorder: int,
    *,
    deriv: int = 0,
) -> WaveformTrace:
    """Apply Savitzky-Golay smoothing filter.

    Smooths data while preserving higher moments (peaks, etc.) better
    than simple moving average.

    Args:
        trace: Input waveform trace.
        window_length: Length of filter window (must be odd and > polyorder).
        polyorder: Order of polynomial used in fitting.
        deriv: Derivative order (0 = smoothing, 1 = first derivative, etc.).

    Returns:
        Filtered waveform trace.

    Raises:
        AnalysisError: If window_length is not odd or polyorder is invalid.

    Example:
        >>> smoothed = savgol_filter(trace, window_length=11, polyorder=3)
    """
    if window_length % 2 == 0:
        raise AnalysisError(f"Window length must be odd, got {window_length}")

    if polyorder >= window_length:
        raise AnalysisError(
            f"Polynomial order {polyorder} must be less than window length {window_length}"
        )

    filtered_data = signal.savgol_filter(trace.data, window_length, polyorder, deriv=deriv)

    return WaveformTrace(
        data=filtered_data.astype(np.float64),
        metadata=trace.metadata,
    )


def matched_filter(
    trace: WaveformTrace,
    template: NDArray[np.floating[Any]],
    *,
    normalize: bool = True,
) -> WaveformTrace:
    """Apply matched filter for pulse detection.

    Correlates the input with a known pulse template to detect
    occurrences of that pulse shape.

    Args:
        trace: Input waveform trace.
        template: Template pulse to match.
        normalize: If True, normalize template for unit energy.

    Returns:
        Matched filter output trace. Peaks indicate template matches.

    Raises:
        AnalysisError: If template is empty or exceeds data length.

    Example:
        >>> # Detect a specific pulse shape
        >>> pulse_template = np.array([0, 0.5, 1.0, 0.5, 0])
        >>> match_output = matched_filter(trace, pulse_template)
        >>> # Find peaks in match_output for detection
    """
    if len(template) == 0:
        raise AnalysisError("Template cannot be empty")

    if len(template) > len(trace.data):
        raise AnalysisError(
            f"Template length {len(template)} exceeds data length {len(trace.data)}"
        )

    # Matched filter is correlation with time-reversed template
    h = template[::-1].copy()

    if normalize:
        energy = np.sum(h**2)
        if energy > 0:
            h = h / np.sqrt(energy)

    # Correlate (convolve with time-reversed template)
    output = np.convolve(trace.data, h, mode="same")

    return WaveformTrace(
        data=output.astype(np.float64),
        metadata=trace.metadata,
    )


def exponential_moving_average(
    trace: WaveformTrace,
    alpha: float,
) -> WaveformTrace:
    """Apply exponential moving average (EMA) filter.

    IIR filter with exponential decay weighting.

    Args:
        trace: Input waveform trace.
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.

    Returns:
        Filtered waveform trace.

    Raises:
        AnalysisError: If alpha is not in range (0, 1].

    Example:
        >>> smoothed = exponential_moving_average(trace, alpha=0.1)
    """
    if not 0 < alpha <= 1:
        raise AnalysisError(f"Alpha must be in (0, 1], got {alpha}")

    # EMA as IIR filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    # Transfer function: H(z) = alpha / (1 - (1-alpha) * z^-1)
    b = np.array([alpha])
    a = np.array([1.0, -(1 - alpha)])

    filtered_data = signal.lfilter(b, a, trace.data)

    return WaveformTrace(
        data=filtered_data.astype(np.float64),
        metadata=trace.metadata,
    )


def gaussian_filter(
    trace: WaveformTrace,
    sigma: float,
) -> WaveformTrace:
    """Apply Gaussian smoothing filter.

    Smooth with Gaussian kernel of specified standard deviation.

    Args:
        trace: Input waveform trace.
        sigma: Standard deviation of Gaussian kernel in samples.

    Returns:
        Filtered waveform trace.

    Raises:
        AnalysisError: If sigma is not positive.

    Example:
        >>> smoothed = gaussian_filter(trace, sigma=3.0)
    """
    if sigma <= 0:
        raise AnalysisError(f"Sigma must be positive, got {sigma}")

    filtered_data = ndimage.gaussian_filter1d(trace.data, sigma)

    return WaveformTrace(
        data=filtered_data.astype(np.float64),
        metadata=trace.metadata,
    )


def differentiate(
    trace: WaveformTrace,
    *,
    order: int = 1,
) -> WaveformTrace:
    """Compute numerical derivative of trace.

    Uses numpy gradient for smooth differentiation.

    Args:
        trace: Input waveform trace.
        order: Derivative order (1 = first derivative, 2 = second, etc.).

    Returns:
        Differentiated waveform trace. Units change (V -> V/s, etc.).

    Raises:
        AnalysisError: If order is not positive.

    Example:
        >>> velocity = differentiate(position_trace)
        >>> acceleration = differentiate(position_trace, order=2)
    """
    if order < 1:
        raise AnalysisError(f"Derivative order must be positive, got {order}")

    sample_period = trace.metadata.time_base
    result = trace.data.copy()

    for _ in range(order):
        result = np.gradient(result, sample_period)

    return WaveformTrace(
        data=result.astype(np.float64),
        metadata=trace.metadata,
    )


def integrate(
    trace: WaveformTrace,
    *,
    method: Literal["cumtrapz", "cumsum"] = "cumtrapz",
    initial: float = 0.0,
) -> WaveformTrace:
    """Compute numerical integral of trace.

    Args:
        trace: Input waveform trace.
        method: Integration method - "cumtrapz" (trapezoidal) or "cumsum".
        initial: Initial value at t=0.

    Returns:
        Integrated waveform trace. Units change (V -> V*s, etc.).

    Raises:
        AnalysisError: If method is not recognized.

    Example:
        >>> position = integrate(velocity_trace)
    """
    sample_period = trace.metadata.time_base

    if method == "cumtrapz":
        from scipy.integrate import cumulative_trapezoid

        result = cumulative_trapezoid(trace.data, dx=sample_period, initial=initial)
    elif method == "cumsum":
        result = np.cumsum(trace.data) * sample_period + initial
    else:
        raise AnalysisError(f"Unknown integration method: {method}")

    return WaveformTrace(
        data=result.astype(np.float64),
        metadata=trace.metadata,
    )


__all__ = [
    "band_pass",
    "band_stop",
    "differentiate",
    "exponential_moving_average",
    "gaussian_filter",
    "high_pass",
    "integrate",
    "low_pass",
    "matched_filter",
    "median_filter",
    "moving_average",
    "notch_filter",
    "savgol_filter",
]
