"""Auto-detection utilities for signal analysis.

This module provides utilities for automatic detection of signal
parameters such as baud rate, logic levels, and protocol types.


Example:
    >>> from oscura.utils.autodetect import detect_baud_rate
    >>> baudrate = detect_baud_rate(trace)
    >>> print(f"Detected baud rate: {baudrate}")

References:
    Standard baud rates. and UART specifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Standard baud rates (RS-232, UART, CAN, etc.)
STANDARD_BAUD_RATES: tuple[int, ...] = (
    300,
    600,
    1200,
    2400,
    4800,
    9600,
    14400,
    19200,
    28800,
    38400,
    57600,
    76800,
    115200,
    230400,
    250000,  # CAN common
    460800,
    500000,  # CAN common
    576000,
    921600,
    1000000,  # 1 Mbps
    1500000,
    2000000,
    3000000,
    4000000,
)


def detect_baud_rate(
    trace: WaveformTrace | DigitalTrace,
    *,
    threshold: float | Literal["auto"] = "auto",
    method: Literal["pulse_width", "edge_timing", "autocorr"] = "pulse_width",
    tolerance: float = 0.05,
    return_confidence: bool = False,
) -> int | tuple[int, float]:
    """Detect baud rate from signal timing.

    Analyzes pulse widths or edge timing to determine the symbol rate,
    then maps to the nearest standard baud rate.

    Args:
        trace: Input trace (analog or digital).
        threshold: Threshold for analog to digital conversion.
        method: Detection method:
            - "pulse_width": Minimum pulse width (default)
            - "edge_timing": Edge-to-edge timing analysis
            - "autocorr": Autocorrelation peak detection
        tolerance: Tolerance for matching to standard rate (default 5%).
        return_confidence: If True, also return confidence score.

    Returns:
        Detected baud rate (nearest standard), or tuple of (rate, confidence)
        if return_confidence=True.

    Raises:
        ValueError: If unknown detection method specified.

    Example:
        >>> baudrate = detect_baud_rate(trace)
        >>> print(f"Detected: {baudrate} bps")

        >>> baudrate, confidence = detect_baud_rate(trace, return_confidence=True)
        >>> print(f"Detected: {baudrate} bps ({confidence:.0%} confidence)")

    References:
        RS-232 Standard Baud Rates
    """
    # Get digital representation
    if isinstance(trace, WaveformTrace):
        from oscura.analyzers.digital.extraction import to_digital

        digital_trace = to_digital(trace, threshold=threshold)
        data = digital_trace.data
    else:
        data = trace.data

    sample_rate = trace.metadata.sample_rate

    if method == "pulse_width":
        bit_period = _detect_via_pulse_width(data, sample_rate)
    elif method == "edge_timing":
        bit_period = _detect_via_edge_timing(data, sample_rate)
    elif method == "autocorr":
        bit_period = _detect_via_autocorrelation(data, sample_rate)
    else:
        raise ValueError(f"Unknown method: {method}")

    if bit_period <= 0 or np.isnan(bit_period):
        if return_confidence:
            return 0, 0.0
        return 0

    # Convert to baud rate
    measured_rate = 1.0 / bit_period

    # Find nearest standard rate
    best_rate = 0
    best_error = float("inf")

    for std_rate in STANDARD_BAUD_RATES:
        error = abs(measured_rate - std_rate) / std_rate
        if error < best_error:
            best_error = error
            best_rate = std_rate

    # Compute confidence
    confidence = max(0.0, 1.0 - best_error / tolerance) if best_error <= tolerance else 0.0

    if return_confidence:
        return best_rate, confidence

    return best_rate


def _detect_via_pulse_width(data: NDArray[np.bool_], sample_rate: float) -> float:
    """Detect bit period from minimum pulse width.

    Args:
        data: Digital signal data.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated bit period in seconds.
    """
    # Find pulse widths (runs of consecutive values)
    pulse_widths = []

    current_value = data[0]
    run_length = 1

    for i in range(1, len(data)):
        if data[i] == current_value:
            run_length += 1
        else:
            pulse_widths.append(run_length)
            current_value = data[i]
            run_length = 1

    # Add final run
    pulse_widths.append(run_length)

    if len(pulse_widths) == 0:
        return 0.0

    pulse_widths_arr = np.array(pulse_widths, dtype=np.float64)

    # Filter out very short pulses (noise)
    min_pulse = max(2, np.min(pulse_widths_arr[pulse_widths_arr > 1]))

    # The minimum pulse width corresponds to a single bit
    # Use the mode of small pulses for robustness
    small_pulses = pulse_widths_arr[pulse_widths_arr <= min_pulse * 1.5]

    bit_samples = min_pulse if len(small_pulses) == 0 else np.median(small_pulses)

    return float(bit_samples / sample_rate)


def _detect_via_edge_timing(data: NDArray[np.bool_], sample_rate: float) -> float:
    """Detect bit period from edge-to-edge timing.

    Args:
        data: Digital signal data.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated bit period in seconds.
    """
    # Find all edges
    transitions = np.diff(data.astype(np.int8))
    edge_indices = np.where(transitions != 0)[0]

    if len(edge_indices) < 2:
        return 0.0

    # Compute edge intervals
    intervals = np.diff(edge_indices).astype(np.float64)

    if len(intervals) == 0:
        return 0.0

    # Intervals should be multiples of bit period
    # Find GCD-like value using histogram
    min_interval = np.min(intervals)
    max_check = min(min_interval * 2, np.median(intervals))

    # The bit period is the smallest common interval
    # Use histogram to find the cluster
    bins = np.arange(1, max_check + 1)
    hist, _ = np.histogram(intervals, bins=bins)

    if len(hist) == 0 or np.max(hist) == 0:
        bit_samples = min_interval
    else:
        # Find first significant peak
        threshold = np.max(hist) * 0.3
        peaks = np.where(hist >= threshold)[0]

        if len(peaks) > 0:
            bit_samples = peaks[0] + 1  # +1 for bin offset
        else:
            bit_samples = min_interval

    return float(bit_samples / sample_rate)


def _detect_via_autocorrelation(data: NDArray[np.bool_], sample_rate: float) -> float:
    """Detect bit period via autocorrelation.

    Args:
        data: Digital signal data.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated bit period in seconds.
    """
    # Convert to float for correlation
    signal = data.astype(np.float64) * 2 - 1  # Map to [-1, 1]

    # Remove DC
    signal = signal - np.mean(signal)

    # Compute autocorrelation
    n = len(signal)
    max_lag = min(n // 2, int(sample_rate / 300))  # Limit to reasonable range

    autocorr = np.correlate(signal[: max_lag * 2], signal[: max_lag * 2], mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # Keep positive lags

    # Normalize
    autocorr = autocorr / autocorr[0]

    # Find first significant peak after lag 0
    # Skip initial samples to avoid lag-0 region
    min_lag = max(2, max_lag // 100)

    # Find local maxima
    peaks = []
    for i in range(min_lag, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.3:  # Significance threshold
                peaks.append((i, autocorr[i]))

    if len(peaks) == 0:
        return 0.0

    # First significant peak is likely the bit period
    bit_samples = peaks[0][0]

    return float(bit_samples / sample_rate)


def detect_logic_family(
    trace: WaveformTrace,
    *,
    return_confidence: bool = False,
) -> str | tuple[str, float]:
    """Detect logic family from signal levels.

    Analyzes voltage levels to identify TTL, CMOS, LVTTL, LVCMOS variants.

    Args:
        trace: Input analog trace.
        return_confidence: If True, also return confidence score.

    Returns:
        Logic family name (e.g., "TTL", "LVCMOS_3V3"), or tuple of
        (family, confidence) if return_confidence=True.
    """
    from oscura.analyzers.digital.extraction import LOGIC_FAMILIES

    data = trace.data

    # Get voltage levels
    v_low = float(np.percentile(data, 10))
    v_high = float(np.percentile(data, 90))

    # Estimate VCC from high level
    v_cc_est = v_high * 1.1  # Add margin

    best_family = "TTL"
    best_score = 0.0

    for family, levels in LOGIC_FAMILIES.items():
        vcc = levels["VCC"]
        vol = levels["VOL_max"]
        voh = levels["VOH_min"]

        # Score based on how well levels match
        low_match = 1.0 - min(1.0, abs(v_low - vol) / 0.5)
        high_match = 1.0 - min(1.0, abs(v_high - voh) / 0.5)
        vcc_match = 1.0 - min(1.0, abs(v_cc_est - vcc) / vcc)

        score = (low_match + high_match + vcc_match) / 3

        if score > best_score:
            best_score = score
            best_family = family

    if return_confidence:
        return best_family, best_score

    return best_family


__all__ = [
    "STANDARD_BAUD_RATES",
    "detect_baud_rate",
    "detect_logic_family",
]
