"""Advanced timing measurements for digital signals.

This module provides IEEE 181-2011 and JEDEC compliant timing measurements
including propagation delay, setup/hold time, slew rate, phase, and skew.


Example:
    >>> from oscura.analyzers.digital.timing import propagation_delay, setup_time
    >>> delay = propagation_delay(trace1, trace2)
    >>> t_setup = setup_time(data_trace, clock_trace, clock_edge="rising")

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    IEEE 2414-2020: Standard for Jitter and Phase Noise
    JEDEC Standard No. 65B: High-Speed Interface Timing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ClockRecoveryResult:
    """Result of clock recovery analysis.

    Attributes:
        frequency: Recovered clock frequency in Hz.
        period: Recovered clock period in seconds.
        method: Method used for recovery ("fft" or "edge").
        confidence: Confidence score (0.0 to 1.0).
        jitter_rms: RMS jitter in seconds (edge method only).
        jitter_pp: Peak-to-peak jitter in seconds (edge method only).
    """

    frequency: float
    period: float
    method: str
    confidence: float
    jitter_rms: float | None = None
    jitter_pp: float | None = None


@dataclass
class TimingViolation:
    """Represents a timing violation.

    Attributes:
        timestamp: Time of violation in seconds.
        violation_type: Type of violation ("setup" or "hold").
        measured: Measured time in seconds.
        required: Required time (specification) in seconds.
        margin: Margin to specification (negative = violation).
    """

    timestamp: float
    violation_type: str
    measured: float
    required: float
    margin: float


@dataclass
class RMSJitterResult:
    """Result of RMS jitter measurement.

    Attributes:
        rms: RMS jitter in seconds.
        mean: Mean period in seconds.
        samples: Number of edges used.
        uncertainty: Measurement uncertainty (1-sigma) in seconds.
        edge_type: Type of edges used.

    References:
        IEEE 2414-2020 Section 5.1
        TIM-007
    """

    rms: float
    mean: float
    samples: int
    uncertainty: float
    edge_type: str


def propagation_delay(
    input_trace: WaveformTrace | DigitalTrace,
    output_trace: WaveformTrace | DigitalTrace,
    *,
    ref_level: float = 0.5,
    edge_type: Literal["rising", "falling", "both"] = "rising",
    return_all: bool = False,
) -> float | NDArray[np.float64]:
    """Measure propagation delay between two signals.

    Computes the time delay from input edge to corresponding output edge
    at the specified reference level per IEEE 181-2011.

    Args:
        input_trace: Input signal trace.
        output_trace: Output signal trace.
        ref_level: Reference level as fraction (0.0 to 1.0). Default 0.5 (50%).
        edge_type: Type of edges to measure:
            - "rising": Low-to-high transitions
            - "falling": High-to-low transitions
            - "both": All transitions
        return_all: If True, return array of all delays. If False, return mean.

    Returns:
        Propagation delay in seconds (mean if return_all=False), or array of delays.

    Raises:
        InsufficientDataError: If traces have insufficient edges.

    Example:
        >>> delay = propagation_delay(input_trace, output_trace)
        >>> print(f"Propagation delay: {delay * 1e9:.2f} ns")

    References:
        IEEE 181-2011 Section 5.6
    """
    # Get edge timestamps for both signals
    input_edges = _get_edge_timestamps(input_trace, edge_type, ref_level)
    output_edges = _get_edge_timestamps(output_trace, edge_type, ref_level)

    if len(input_edges) == 0:
        raise InsufficientDataError(
            "No edges found in input trace",
            required=1,
            available=0,
            analysis_type="propagation_delay",
        )

    if len(output_edges) == 0:
        raise InsufficientDataError(
            "No edges found in output trace",
            required=1,
            available=0,
            analysis_type="propagation_delay",
        )

    # Match input edges to nearest subsequent output edges
    delays: list[float] = []

    for in_edge in input_edges:
        # Find output edges after this input edge
        subsequent_outputs = output_edges[output_edges > in_edge]
        if len(subsequent_outputs) > 0:
            # Use nearest subsequent output edge
            delay = subsequent_outputs[0] - in_edge
            if delay > 0:
                delays.append(delay)

    if len(delays) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    delays_arr = np.array(delays, dtype=np.float64)

    if return_all:
        return delays_arr
    return float(np.mean(delays_arr))


def setup_time(
    data_trace: WaveformTrace | DigitalTrace,
    clock_trace: WaveformTrace | DigitalTrace,
    *,
    clock_edge: Literal["rising", "falling"] = "rising",
    data_stable_level: float = 0.5,
    return_all: bool = False,
) -> float | NDArray[np.float64]:
    """Measure setup time between data and clock signals.

    Computes the time from when data becomes stable to the clock edge
    per JEDEC timing standards.

    Args:
        data_trace: Data signal trace.
        clock_trace: Clock signal trace.
        clock_edge: Type of clock edge to reference ("rising" or "falling").
        data_stable_level: Reference level for data stability (0.0 to 1.0).
        return_all: If True, return array of all setup times. If False, return mean.

    Returns:
        Setup time in seconds (positive = data stable before clock).

    Example:
        >>> t_setup = setup_time(data_trace, clock_trace, clock_edge="rising")
        >>> print(f"Setup time: {t_setup * 1e9:.2f} ns")

    References:
        JEDEC Standard No. 65B
    """
    # Get clock edges
    clock_edges = _get_edge_timestamps(clock_trace, clock_edge, 0.5)

    if len(clock_edges) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    # Get all data edges (both rising and falling)
    data_edges = _get_edge_timestamps(data_trace, "both", data_stable_level)

    if len(data_edges) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    # For each clock edge, find the most recent data edge
    setup_times: list[float] = []

    for clk_edge in clock_edges:
        # Find data edges before this clock edge
        prior_data_edges = data_edges[data_edges < clk_edge]
        if len(prior_data_edges) > 0:
            # Setup time = clock edge - last data edge
            setup = clk_edge - prior_data_edges[-1]
            setup_times.append(setup)

    if len(setup_times) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    result = np.array(setup_times, dtype=np.float64)

    if return_all:
        return result
    return float(np.mean(result))


def hold_time(
    data_trace: WaveformTrace | DigitalTrace,
    clock_trace: WaveformTrace | DigitalTrace,
    *,
    clock_edge: Literal["rising", "falling"] = "rising",
    data_stable_level: float = 0.5,
    return_all: bool = False,
) -> float | NDArray[np.float64]:
    """Measure hold time between clock and data signals.

    Computes the time from clock edge to when data changes
    per JEDEC timing standards.

    Args:
        data_trace: Data signal trace.
        clock_trace: Clock signal trace.
        clock_edge: Type of clock edge to reference ("rising" or "falling").
        data_stable_level: Reference level for data transition (0.0 to 1.0).
        return_all: If True, return array of all hold times. If False, return mean.

    Returns:
        Hold time in seconds (positive = data stable after clock).

    Example:
        >>> t_hold = hold_time(data_trace, clock_trace, clock_edge="rising")
        >>> print(f"Hold time: {t_hold * 1e9:.2f} ns")

    References:
        JEDEC Standard No. 65B
    """
    # Get clock edges
    clock_edges = _get_edge_timestamps(clock_trace, clock_edge, 0.5)

    if len(clock_edges) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    # Get all data edges
    data_edges = _get_edge_timestamps(data_trace, "both", data_stable_level)

    if len(data_edges) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    # For each clock edge, find the next data edge
    hold_times: list[float] = []

    for clk_edge in clock_edges:
        # Find data edges after this clock edge
        subsequent_data_edges = data_edges[data_edges > clk_edge]
        if len(subsequent_data_edges) > 0:
            # Hold time = next data edge - clock edge
            hold = subsequent_data_edges[0] - clk_edge
            hold_times.append(hold)

    if len(hold_times) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    result = np.array(hold_times, dtype=np.float64)

    if return_all:
        return result
    return float(np.mean(result))


def slew_rate(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.2, 0.8),
    edge_type: Literal["rising", "falling", "both"] = "rising",
    return_all: bool = False,
) -> float | NDArray[np.float64]:
    """Measure slew rate (dV/dt) during signal transitions.

    Computes the rate of voltage change during edge transitions
    per IEEE 181-2011.

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (default 20%-80%).
        edge_type: Type of edges to measure ("rising", "falling", or "both").
        return_all: If True, return array of all slew rates. If False, return mean.

    Returns:
        Slew rate in V/s (positive for rising, negative for falling).
        Returns NaN if no transitions found or amplitude is zero.

    Example:
        >>> sr = slew_rate(trace)
        >>> print(f"Slew rate: {sr / 1e6:.2f} V/us")

    References:
        IEEE 181-2011 Section 5.2
    """
    if len(trace.data) < 3:
        return np.array([], dtype=np.float64) if return_all else np.nan

    data = trace.data
    sample_period = trace.metadata.time_base

    # Find signal levels and validate
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return np.array([], dtype=np.float64) if return_all else np.nan

    # Calculate reference voltages
    v_low = low + ref_levels[0] * amplitude
    v_high = low + ref_levels[1] * amplitude
    dv = v_high - v_low

    # Measure slew rates for requested edge types
    slew_rates: list[float] = []

    if edge_type in ("rising", "both"):
        slew_rates.extend(_measure_rising_slew_rates(data, v_low, v_high, dv, sample_period))

    if edge_type in ("falling", "both"):
        slew_rates.extend(_measure_falling_slew_rates(data, v_low, v_high, dv, sample_period))

    if len(slew_rates) == 0:
        return np.array([], dtype=np.float64) if return_all else np.nan

    result = np.array(slew_rates, dtype=np.float64)
    return result if return_all else float(np.mean(result))


def _measure_rising_slew_rates(
    data: NDArray[np.float64],
    v_low: float,
    v_high: float,
    dv: float,
    sample_period: float,
) -> list[float]:
    """Measure slew rates for rising edges.

    Args:
        data: Signal data.
        v_low: Low reference voltage.
        v_high: High reference voltage.
        dv: Voltage difference between reference levels.
        sample_period: Time between samples.

    Returns:
        List of rising slew rates (V/s).
    """
    slew_rates: list[float] = []
    rising_start = np.where((data[:-1] < v_low) & (data[1:] >= v_low))[0]

    for start_idx in rising_start:
        remaining = data[start_idx:]
        above_high = remaining >= v_high

        if np.any(above_high):
            end_offset = np.argmax(above_high)
            dt = end_offset * sample_period
            if dt > 0:
                slew_rates.append(float(dv / dt))

    return slew_rates


def _measure_falling_slew_rates(
    data: NDArray[np.float64],
    v_low: float,
    v_high: float,
    dv: float,
    sample_period: float,
) -> list[float]:
    """Measure slew rates for falling edges.

    Args:
        data: Signal data.
        v_low: Low reference voltage.
        v_high: High reference voltage.
        dv: Voltage difference between reference levels.
        sample_period: Time between samples.

    Returns:
        List of falling slew rates (negative V/s).
    """
    slew_rates: list[float] = []
    falling_start = np.where((data[:-1] > v_high) & (data[1:] <= v_high))[0]

    for start_idx in falling_start:
        remaining = data[start_idx:]
        below_low = remaining <= v_low

        if np.any(below_low):
            end_offset = np.argmax(below_low)
            dt = end_offset * sample_period
            if dt > 0:
                slew_rates.append(float(-dv / dt))  # Negative for falling

    return slew_rates


def phase(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    method: Literal["edge", "fft"] = "edge",
    unit: Literal["degrees", "radians"] = "degrees",
) -> float:
    """Measure phase difference between two signals.

    Computes the phase relationship between two waveforms using
    either edge-based or FFT-based methods.

    Args:
        trace1: Reference signal trace.
        trace2: Signal to measure phase relative to reference.
        method: Measurement method:
            - "edge": Edge-to-edge timing (default, more accurate for digital)
            - "fft": Cross-spectral phase (better for analog/noisy signals)
        unit: Output unit ("degrees" or "radians").

    Returns:
        Phase difference in specified units. Positive = trace2 leads trace1.

    Raises:
        ValueError: If method is not recognized.

    Example:
        >>> phase_deg = phase(ref_trace, sig_trace)
        >>> print(f"Phase: {phase_deg:.1f} degrees")

    References:
        IEEE 181-2011 Section 5.8
    """
    if method == "edge":
        return _phase_edge(trace1, trace2, unit)
    elif method == "fft":
        return _phase_fft(trace1, trace2, unit)
    else:
        raise ValueError(f"Unknown method: {method}")


def _phase_edge(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    unit: Literal["degrees", "radians"],
) -> float:
    """Compute phase using edge timing."""
    # Get rising edges for both signals
    edges1 = _get_edge_timestamps(trace1, "rising", 0.5)
    edges2 = _get_edge_timestamps(trace2, "rising", 0.5)

    if len(edges1) < 2 or len(edges2) < 2:
        return np.nan

    # Calculate period from first signal
    period1 = np.mean(np.diff(edges1))

    if period1 <= 0:
        return np.nan

    # Calculate phase from edge differences
    phase_times: list[float] = []

    for e1 in edges1:
        # Find nearest edge in trace2
        diffs = edges2 - e1
        # Find closest edge (could be before or after)
        idx = np.argmin(np.abs(diffs))
        phase_times.append(diffs[idx])

    if len(phase_times) == 0:
        return np.nan

    mean_phase_time = np.mean(phase_times)

    # Convert to phase angle
    phase_rad = 2 * np.pi * mean_phase_time / period1

    # Normalize to [-pi, pi]
    phase_rad = (phase_rad + np.pi) % (2 * np.pi) - np.pi

    if unit == "degrees":
        return float(np.degrees(phase_rad))
    return float(phase_rad)


def _phase_fft(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    unit: Literal["degrees", "radians"],
) -> float:
    """Compute phase using FFT cross-spectral analysis."""
    data1 = trace1.data - np.mean(trace1.data)
    data2 = trace2.data - np.mean(trace2.data)

    # Ensure same length
    n = min(len(data1), len(data2))
    data1 = data1[:n]
    data2 = data2[:n]

    if n < 16:
        return np.nan

    # Compute FFTs
    fft1 = np.fft.rfft(data1)
    fft2 = np.fft.rfft(data2)

    # Cross-spectrum
    cross = fft2 * np.conj(fft1)

    # Find fundamental frequency (strongest component after DC)
    magnitudes = np.abs(cross)
    fund_idx = np.argmax(magnitudes[1:]) + 1

    # Phase at fundamental
    phase_rad = np.angle(cross[fund_idx])

    if unit == "degrees":
        return float(np.degrees(phase_rad))
    return float(phase_rad)


def skew(
    traces: list[WaveformTrace | DigitalTrace],
    *,
    reference_idx: int = 0,
    edge_type: Literal["rising", "falling"] = "rising",
) -> dict[str, float | NDArray[np.float64]]:
    """Measure timing skew between multiple signals.

    Computes the timing offset of each signal relative to a reference
    per IEEE 181-2011.

    Args:
        traces: List of signal traces to compare.
        reference_idx: Index of reference signal (default 0).
        edge_type: Type of edges to use for comparison.

    Returns:
        Dictionary with skew statistics (skew_values, min, max, mean, range).

    Raises:
        ValueError: If fewer than 2 traces or reference_idx out of range.

    Example:
        >>> result = skew([clk1, clk2, clk3])
        >>> print(f"Max skew: {result['max'] * 1e12:.0f} ps")

    References:
        IEEE 181-2011 Section 5.7
    """
    if len(traces) < 2:
        raise ValueError("Need at least 2 traces for skew measurement")
    if reference_idx >= len(traces):
        raise ValueError(f"reference_idx {reference_idx} out of range")

    ref_edges = _get_edge_timestamps(traces[reference_idx], edge_type, 0.5)

    if len(ref_edges) == 0:
        return _empty_skew_result()

    all_skews, skew_values = _compute_all_skews(traces, reference_idx, ref_edges, edge_type)

    return _build_skew_result(skew_values, all_skews)


def _empty_skew_result() -> dict[str, float | NDArray[np.float64]]:
    """Return empty skew result dictionary.

    Returns:
        Dictionary with empty/NaN skew values.
    """
    return {
        "skew_values": np.array([], dtype=np.float64),
        "min": float(np.nan),
        "max": float(np.nan),
        "mean": float(np.nan),
        "range": float(np.nan),
    }


def _compute_all_skews(
    traces: list[WaveformTrace | DigitalTrace],
    reference_idx: int,
    ref_edges: NDArray[np.float64],
    edge_type: Literal["rising", "falling"],
) -> tuple[list[float], list[float]]:
    """Compute skew values for all traces.

    Args:
        traces: List of traces to analyze.
        reference_idx: Index of reference trace.
        ref_edges: Reference edge timestamps.
        edge_type: Edge type to analyze.

    Returns:
        Tuple of (all_skews including reference, skew_values excluding reference).
    """
    all_skews: list[float] = []
    skew_values: list[float] = []

    for i, trace in enumerate(traces):
        if i == reference_idx:
            all_skews.append(0.0)
            continue

        trace_edges = _get_edge_timestamps(trace, edge_type, 0.5)
        skew_val = _compute_trace_skew(trace_edges, ref_edges)

        skew_values.append(skew_val)
        all_skews.append(skew_val)

    return all_skews, skew_values


def _compute_trace_skew(trace_edges: NDArray[np.float64], ref_edges: NDArray[np.float64]) -> float:
    """Compute skew for a single trace relative to reference.

    Args:
        trace_edges: Edge timestamps for trace.
        ref_edges: Reference edge timestamps.

    Returns:
        Mean skew value or NaN if no edges.
    """
    if len(trace_edges) == 0:
        return float(np.nan)

    edge_skews = []
    for ref_edge in ref_edges:
        diffs = np.abs(trace_edges - ref_edge)
        nearest_idx = np.argmin(diffs)
        skew_val_edge = trace_edges[nearest_idx] - ref_edge
        edge_skews.append(skew_val_edge)

    return float(np.mean(edge_skews)) if len(edge_skews) > 0 else float(np.nan)


def _build_skew_result(
    skew_values: list[float], all_skews: list[float]
) -> dict[str, float | NDArray[np.float64]]:
    """Build final skew result dictionary.

    Args:
        skew_values: Skew values excluding reference.
        all_skews: Skew values including reference.

    Returns:
        Dictionary with skew statistics.
    """
    skew_arr = np.array(skew_values, dtype=np.float64)
    all_skews_arr = np.array(all_skews, dtype=np.float64)
    valid_all_skews = all_skews_arr[~np.isnan(all_skews_arr)]

    if len(valid_all_skews) == 0:
        return {
            "skew_values": skew_arr,
            "min": float(np.nan),
            "max": float(np.nan),
            "mean": float(np.nan),
            "range": float(np.nan),
        }

    return {
        "skew_values": skew_arr,
        "min": float(np.min(valid_all_skews)),
        "max": float(np.max(valid_all_skews)),
        "mean": float(np.mean(valid_all_skews)),
        "range": float(np.max(valid_all_skews) - np.min(valid_all_skews)),
    }


def recover_clock_fft(
    trace: WaveformTrace | DigitalTrace,
    *,
    min_freq: float | None = None,
    max_freq: float | None = None,
) -> ClockRecoveryResult:
    """Recover clock frequency using FFT peak detection.

    Detects the dominant frequency component in the signal using
    FFT analysis, suitable for periodic digital signals.

    **Best for**: Long signals (>64 samples) with clear periodicity.
    **Not recommended for**: Short random data, aperiodic signals.
    For short signals, use recover_clock_edge() instead.

    Args:
        trace: Input trace (analog or digital). Should have at least
            4-5 cycles of the clock signal for reliable detection.
        min_freq: Minimum frequency to consider (Hz). Default: sample_rate/1000.
        max_freq: Maximum frequency to consider (Hz). Default: sample_rate/2.

    Returns:
        ClockRecoveryResult with recovered frequency and confidence.
        Confidence < 0.5 indicates unreliable detection (warning issued).

    Raises:
        InsufficientDataError: If trace has fewer than 64 samples.
        ValueError: If no frequency components found in specified range.

    Warnings:
        UserWarning: Issued when confidence < 0.5 (unreliable result).

    Example:
        >>> result = recover_clock_fft(trace)
        >>> if result.confidence > 0.7:
        ...     print(f"Clock: {result.frequency / 1e6:.3f} MHz")
        >>> else:
        ...     print("Low confidence - try edge-based recovery")

    References:
        IEEE 1241-2010 Section 4.1
    """
    # Prepare data and validate
    data = trace.data.astype(np.float64) if isinstance(trace, DigitalTrace) else trace.data
    sample_rate = trace.metadata.sample_rate
    _validate_fft_requirements(len(data))

    # Set frequency range
    min_freq_val, max_freq_val = _determine_frequency_range(min_freq, max_freq, sample_rate)

    # Compute FFT spectrum
    freq, magnitude = _compute_fft_spectrum(data, sample_rate)

    # Find peak frequency
    peak_freq, peak_mag, valid_indices = _find_peak_frequency(
        freq, magnitude, min_freq_val, max_freq_val
    )

    # Calculate confidence
    confidence = _calculate_fft_confidence(magnitude, peak_mag, valid_indices)

    # Refine frequency with interpolation
    peak_freq_refined = _refine_peak_frequency(peak_freq, magnitude, freq, sample_rate, len(data))

    # Warn if low confidence
    _check_confidence_and_warn(confidence, peak_freq_refined)

    period = 1.0 / peak_freq_refined if peak_freq_refined > 0 else np.nan

    return ClockRecoveryResult(
        frequency=float(peak_freq_refined),
        period=float(period),
        method="fft",
        confidence=float(confidence),
    )


def _validate_fft_requirements(n_samples: int) -> None:
    """Validate trace has enough samples for FFT."""
    min_samples = 64
    if n_samples < min_samples:
        raise InsufficientDataError(
            f"FFT clock recovery requires at least {min_samples} samples for reliable frequency detection",
            required=min_samples,
            available=n_samples,
            analysis_type="clock_recovery_fft",
            fix_hint="Use edge-based clock recovery for short signals or acquire more data",
        )


def _determine_frequency_range(
    min_freq: float | None,
    max_freq: float | None,
    sample_rate: float,
) -> tuple[float, float]:
    """Determine frequency range for FFT analysis."""
    min_freq_val = min_freq if min_freq is not None else sample_rate / 1000
    max_freq_val = max_freq if max_freq is not None else sample_rate / 2
    return min_freq_val, max_freq_val


def _compute_fft_spectrum(
    data: NDArray[Any],
    sample_rate: float,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute FFT spectrum of signal."""
    n = len(data)
    data_centered = data - np.mean(data)
    nfft = int(2 ** np.ceil(np.log2(n)))
    spectrum = np.fft.rfft(data_centered, n=nfft)
    freq = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    return freq, magnitude


def _find_peak_frequency(
    freq: NDArray[Any],
    magnitude: NDArray[Any],
    min_freq: float,
    max_freq: float,
) -> tuple[float, float, NDArray[Any]]:
    """Find peak frequency in specified range."""
    mask = (freq >= min_freq) & (freq <= max_freq)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No frequency components found in range [{min_freq:.0f} Hz, {max_freq:.0f} Hz]. "
            f"Signal may be constant (DC) or frequency is outside specified range. "
            f"Adjust min_freq/max_freq or check signal integrity."
        )

    local_peak_idx = np.argmax(magnitude[valid_indices])
    peak_idx = valid_indices[local_peak_idx]
    peak_freq = freq[peak_idx]
    peak_mag = magnitude[peak_idx]

    return peak_freq, peak_mag, valid_indices


def _calculate_fft_confidence(
    magnitude: NDArray[Any],
    peak_mag: float,
    valid_indices: NDArray[Any],
) -> float:
    """Calculate confidence score for FFT peak."""
    rms_mag = np.sqrt(np.mean(magnitude[valid_indices] ** 2))
    return min(1.0, (peak_mag / rms_mag - 1) / 10) if rms_mag > 0 else 0.0


def _refine_peak_frequency(
    peak_freq: float,
    magnitude: NDArray[Any],
    freq: NDArray[Any],
    sample_rate: float,
    n_data: int,
) -> float:
    """Refine peak frequency using parabolic interpolation."""
    peak_idx = np.argmin(np.abs(freq - peak_freq))

    if 0 < peak_idx < len(magnitude) - 1:
        alpha = magnitude[peak_idx - 1]
        beta = magnitude[peak_idx]
        gamma = magnitude[peak_idx + 1]

        if beta > alpha and beta > gamma:
            nfft = int(2 ** np.ceil(np.log2(n_data)))
            freq_resolution = sample_rate / nfft
            delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-12)
            refined: float = float(peak_freq + delta * freq_resolution)
            return refined

    return peak_freq


def _check_confidence_and_warn(confidence: float, peak_freq: float) -> None:
    """Warn if confidence is low."""
    if confidence < 0.5:
        import warnings

        warnings.warn(
            f"FFT clock recovery has low confidence ({confidence:.2f}). "
            f"Detected frequency: {peak_freq / 1e6:.3f} MHz. "
            f"Consider using longer signal, edge-based recovery, or verifying signal periodicity.",
            UserWarning,
            stacklevel=3,
        )


def recover_clock_edge(
    trace: WaveformTrace | DigitalTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
    threshold: float | None = None,
) -> ClockRecoveryResult:
    """Recover clock frequency from edge timestamps.

    Computes clock frequency from edge-to-edge timing, also
    providing jitter statistics.

    Args:
        trace: Input trace (analog or digital).
        edge_type: Type of edges to use ("rising" or "falling").
        threshold: Threshold for edge detection (analog traces only).

    Returns:
        ClockRecoveryResult with frequency and jitter statistics.

    Example:
        >>> result = recover_clock_edge(trace)
        >>> print(f"Clock: {result.frequency / 1e6:.3f} MHz")
        >>> print(f"Jitter RMS: {result.jitter_rms * 1e12:.1f} ps")

    References:
        IEEE 2414-2020 Section 4
    """
    # Get edge timestamps
    ref_level = 0.5 if threshold is None else threshold
    edges = _get_edge_timestamps(trace, edge_type, ref_level)

    if len(edges) < 3:
        return ClockRecoveryResult(
            frequency=np.nan,
            period=np.nan,
            method="edge",
            confidence=0.0,
        )

    # Compute periods
    periods = np.diff(edges)

    if len(periods) == 0:
        return ClockRecoveryResult(
            frequency=np.nan,
            period=np.nan,
            method="edge",
            confidence=0.0,
        )

    # Calculate statistics
    mean_period = float(np.mean(periods))
    std_period = float(np.std(periods))
    frequency = 1.0 / mean_period if mean_period > 0 else np.nan

    # Jitter statistics
    jitter_rms = std_period
    jitter_pp = float(np.max(periods) - np.min(periods))

    # Confidence based on period consistency (low jitter = high confidence)
    if mean_period > 0:
        cv = std_period / mean_period  # Coefficient of variation
        confidence = max(0.0, min(1.0, 1.0 - cv * 10))
    else:
        confidence = 0.0

    return ClockRecoveryResult(
        frequency=float(frequency),
        period=mean_period,
        method="edge",
        confidence=float(confidence),
        jitter_rms=jitter_rms,
        jitter_pp=jitter_pp,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _get_edge_timestamps(
    trace: WaveformTrace | DigitalTrace,
    edge_type: Literal["rising", "falling", "both"],
    ref_level: float = 0.5,
) -> NDArray[np.float64]:
    """Get edge timestamps from a trace.

    Args:
        trace: Input trace.
        edge_type: Type of edges to find.
        ref_level: Reference level for analog traces (0.0 to 1.0).

    Returns:
        Array of edge timestamps in seconds.
    """
    if isinstance(trace, DigitalTrace):
        data = trace.data.astype(np.float64)
        sample_rate = trace.metadata.sample_rate
    else:
        data = trace.data
        sample_rate = trace.metadata.sample_rate

    if len(data) < 2:
        return np.array([], dtype=np.float64)

    sample_period = 1.0 / sample_rate

    # Find threshold level
    low, high = _find_levels(data)
    threshold = low + ref_level * (high - low)

    timestamps: list[float] = []

    if edge_type in ("rising", "both"):
        crossings = np.where((data[:-1] < threshold) & (data[1:] >= threshold))[0]
        for idx in crossings:
            # Linear interpolation
            if idx < len(data) - 1:
                v1, v2 = data[idx], data[idx + 1]
                if abs(v2 - v1) > 1e-12:
                    t_offset = (threshold - v1) / (v2 - v1) * sample_period
                    t_offset = max(0, min(sample_period, t_offset))
                else:
                    t_offset = sample_period / 2
                timestamps.append(idx * sample_period + t_offset)

    if edge_type in ("falling", "both"):
        crossings = np.where((data[:-1] >= threshold) & (data[1:] < threshold))[0]
        for idx in crossings:
            if idx < len(data) - 1:
                v1, v2 = data[idx], data[idx + 1]
                if abs(v2 - v1) > 1e-12:
                    t_offset = (threshold - v1) / (v2 - v1) * sample_period
                    t_offset = max(0, min(sample_period, t_offset))
                else:
                    t_offset = sample_period / 2
                timestamps.append(idx * sample_period + t_offset)

    timestamps.sort()
    return np.array(timestamps, dtype=np.float64)


def _find_levels(data: NDArray[np.float64]) -> tuple[float, float]:
    """Find low and high levels using histogram method.

    Args:
        data: Waveform data array.

    Returns:
        Tuple of (low_level, high_level).
    """
    # Use percentiles for robust level detection
    p10, p90 = np.percentile(data, [10, 90])

    # Refine using histogram peaks
    try:
        hist, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in lower and upper halves
        mid_idx = len(hist) // 2
        low_idx = np.argmax(hist[:mid_idx])
        high_idx = mid_idx + np.argmax(hist[mid_idx:])

        low = bin_centers[low_idx]
        high = bin_centers[high_idx]

        # Sanity check
        if high <= low:
            return float(p10), float(p90)

        return float(low), float(high)
    except (ValueError, IndexError):
        return float(p10), float(p90)


def rms_jitter(
    trace: WaveformTrace | DigitalTrace,
    *,
    edge_type: Literal["rising", "falling", "both"] = "rising",
    threshold: float = 0.5,
) -> RMSJitterResult:
    """Measure RMS jitter from edge timing variations.

    Computes root-mean-square jitter as the standard deviation of edge
    timing variations per IEEE 2414-2020. RMS jitter characterizes the
    random component of timing uncertainty.

    Args:
        trace: Input trace (analog or digital).
        edge_type: Type of edges to measure ("rising", "falling", or "both").
        threshold: Threshold for edge detection (0.0 to 1.0).

    Returns:
        RMSJitterResult containing RMS jitter and statistics.

    Example:
        >>> result = rms_jitter(clock_trace)
        >>> print(f"RMS jitter: {result.rms * 1e12:.2f} ps")
        >>> print(f"Uncertainty: +/- {result.uncertainty * 1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 5.1
        TIM-007
    """
    # Get edge timestamps
    edges = _get_edge_timestamps(trace, edge_type, threshold)

    if len(edges) < 3:
        return RMSJitterResult(
            rms=np.nan,
            mean=np.nan,
            samples=0,
            uncertainty=np.nan,
            edge_type=edge_type,
        )

    # Calculate periods
    periods = np.diff(edges)

    if len(periods) < 2:
        return RMSJitterResult(
            rms=np.nan,
            mean=np.nan,
            samples=len(edges),
            uncertainty=np.nan,
            edge_type=edge_type,
        )

    # RMS jitter is the standard deviation of periods
    mean_period = float(np.mean(periods))
    jitter_rms = float(np.std(periods, ddof=1))

    # Measurement uncertainty (1-sigma)
    # For N samples, uncertainty of std estimate is std / sqrt(2*(N-1))
    n = len(periods)
    uncertainty = jitter_rms / np.sqrt(2 * (n - 1)) if n > 1 else np.nan

    return RMSJitterResult(
        rms=jitter_rms,
        mean=mean_period,
        samples=n,
        uncertainty=uncertainty,
        edge_type=edge_type,
    )


def peak_to_peak_jitter(
    trace: WaveformTrace | DigitalTrace,
    *,
    edge_type: Literal["rising", "falling", "both"] = "rising",
    threshold: float = 0.5,
) -> float:
    """Measure peak-to-peak jitter from edge timing variations.

    Pk-Pk jitter is the maximum range of edge timing deviations from
    the ideal periodic timing, measured over the observation window.

    Args:
        trace: Input trace (analog or digital).
        edge_type: Type of edges to measure ("rising", "falling", or "both").
        threshold: Threshold for edge detection (0.0 to 1.0).

    Returns:
        Peak-to-peak jitter in seconds.

    Example:
        >>> jitter_pp = peak_to_peak_jitter(clock_trace)
        >>> print(f"Pk-Pk jitter: {jitter_pp * 1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 5.2
        TIM-008
    """
    # Get edge timestamps
    edges = _get_edge_timestamps(trace, edge_type, threshold)

    if len(edges) < 3:
        return np.nan

    # Calculate periods
    periods = np.diff(edges)

    if len(periods) < 2:
        return np.nan

    # Pk-Pk jitter is the range of period variations
    jitter_pp = float(np.max(periods) - np.min(periods))

    return jitter_pp


def time_interval_error(
    trace: WaveformTrace | DigitalTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
    nominal_period: float | None = None,
    threshold: float = 0.5,
) -> NDArray[np.float64]:
    """Measure Time Interval Error (TIE) from clock signal.

    TIE is the deviation of each edge from its ideal position based on
    the recovered clock period. Provides a time series of jitter values
    for trend analysis and decomposition.

    Args:
        trace: Input trace (analog or digital).
        edge_type: Type of edges to measure ("rising" or "falling").
        nominal_period: Expected period in seconds. If None, computed from data.
        threshold: Threshold for edge detection (0.0 to 1.0).

    Returns:
        Array of TIE values in seconds, one per edge.

    Raises:
        InsufficientDataError: If trace has fewer than 3 edges.

    Example:
        >>> tie = time_interval_error(clock_trace)
        >>> plt.plot(tie * 1e12)
        >>> plt.ylabel("TIE (ps)")
        >>> plt.xlabel("Edge number")

    References:
        IEEE 2414-2020 Section 5.1
        TIM-009
    """
    # Get edge timestamps
    edges = _get_edge_timestamps(trace, edge_type, threshold)

    if len(edges) < 3:
        raise InsufficientDataError(
            "TIE measurement requires at least 3 edges",
            required=3,
            available=len(edges),
            analysis_type="time_interval_error",
        )

    # Calculate actual periods
    periods = np.diff(edges)

    # Use mean period if nominal not provided
    if nominal_period is None:
        nominal_period = np.mean(periods)

    # Calculate ideal edge positions
    n_edges = len(edges)
    start_time = edges[0]
    ideal_positions = start_time + np.arange(n_edges) * nominal_period

    # TIE is actual - ideal
    tie: NDArray[np.float64] = edges - ideal_positions

    return tie


__all__ = [
    "ClockRecoveryResult",
    "RMSJitterResult",
    "TimingViolation",
    "hold_time",
    "peak_to_peak_jitter",
    "phase",
    "propagation_delay",
    "recover_clock_edge",
    "recover_clock_fft",
    "rms_jitter",
    "setup_time",
    "skew",
    "slew_rate",
    "time_interval_error",
]
