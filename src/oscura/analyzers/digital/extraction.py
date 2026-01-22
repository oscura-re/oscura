"""Digital signal extraction and edge detection.

This module provides functions for extracting digital signals from
analog waveforms and detecting edge transitions.


Example:
    >>> from oscura.analyzers.digital import to_digital, detect_edges
    >>> digital = to_digital(analog_trace, threshold=1.4)
    >>> edges = detect_edges(digital, edge_type="rising")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Standard logic family threshold constants
# Reference: Various IC manufacturer datasheets
LOGIC_FAMILIES: dict[str, dict[str, float]] = {
    "TTL": {
        "VIL_max": 0.8,  # Maximum input low voltage
        "VIH_min": 2.0,  # Minimum input high voltage
        "VOL_max": 0.4,  # Maximum output low voltage
        "VOH_min": 2.4,  # Minimum output high voltage
        "VCC": 5.0,
    },
    "CMOS_5V": {
        "VIL_max": 1.5,
        "VIH_min": 3.5,
        "VOL_max": 0.1,
        "VOH_min": 4.9,
        "VCC": 5.0,
    },
    "LVTTL": {
        "VIL_max": 0.8,
        "VIH_min": 1.5,
        "VOL_max": 0.4,
        "VOH_min": 2.4,
        "VCC": 3.3,
    },
    "LVCMOS_3V3": {
        "VIL_max": 0.3 * 3.3,  # 30% of VCC
        "VIH_min": 0.7 * 3.3,  # 70% of VCC
        "VOL_max": 0.1,
        "VOH_min": 3.2,
        "VCC": 3.3,
    },
    "LVCMOS_2V5": {
        "VIL_max": 0.3 * 2.5,
        "VIH_min": 0.7 * 2.5,
        "VOL_max": 0.1,
        "VOH_min": 2.4,
        "VCC": 2.5,
    },
    "LVCMOS_1V8": {
        "VIL_max": 0.3 * 1.8,
        "VIH_min": 0.7 * 1.8,
        "VOL_max": 0.1,
        "VOH_min": 1.7,
        "VCC": 1.8,
    },
    "LVCMOS_1V2": {
        "VIL_max": 0.3 * 1.2,
        "VIH_min": 0.7 * 1.2,
        "VOL_max": 0.1,
        "VOH_min": 1.1,
        "VCC": 1.2,
    },
}


def to_digital(
    trace: WaveformTrace,
    *,
    threshold: float | Literal["auto"] = "auto",
    hysteresis: float | tuple[float, float] | None = None,
) -> DigitalTrace:
    """Extract digital signal from analog waveform.

    Converts an analog waveform to a digital (boolean) signal using
    threshold comparison.

    Args:
        trace: Input analog waveform trace.
        threshold: Voltage threshold for conversion. Can be:
            - A float value for fixed threshold
            - "auto" for adaptive threshold (midpoint of 10th-90th percentile)
        hysteresis: Hysteresis for noise immunity. Can be:
            - None: No hysteresis
            - A float: Symmetric hysteresis band around threshold
            - A tuple (low, high): Explicit low and high thresholds

    Returns:
        DigitalTrace with boolean data and detected edges.

    Raises:
        InsufficientDataError: If trace has insufficient data.

    Example:
        >>> digital = to_digital(analog_trace, threshold=1.4)
        >>> print(f"High samples: {digital.data.sum()}")

        >>> # With hysteresis for noisy signals
        >>> digital = to_digital(analog_trace, threshold=1.4, hysteresis=0.2)

    References:
        TTL Logic thresholds: VIL_max=0.8V, VIH_min=2.0V
    """
    if len(trace.data) < 2:
        raise InsufficientDataError(
            "Trace too short for digital extraction",
            required=2,
            available=len(trace.data),
            analysis_type="digital_extraction",
        )

    # Convert memoryview to ndarray if needed
    data = np.asarray(trace.data)

    # Determine threshold
    if threshold == "auto":
        # Adaptive threshold: midpoint of 10th-90th percentile
        p10, p90 = np.percentile(data, [10, 90])
        thresh_value = (p10 + p90) / 2.0
    else:
        thresh_value = float(threshold)

    # Apply threshold with or without hysteresis
    if hysteresis is not None:
        if isinstance(hysteresis, tuple):
            thresh_low, thresh_high = hysteresis
        else:
            thresh_low = thresh_value - hysteresis / 2
            thresh_high = thresh_value + hysteresis / 2
        digital_data = _apply_hysteresis(data, thresh_low, thresh_high)
    else:
        digital_data = data >= thresh_value

    # Detect edges
    edges = _detect_edges_internal(data, digital_data, trace.metadata.sample_rate, thresh_value)

    return DigitalTrace(
        data=digital_data,
        metadata=trace.metadata,
        edges=edges,
    )


def _apply_hysteresis(
    data: NDArray[np.floating[Any]],
    thresh_low: float,
    thresh_high: float,
) -> NDArray[np.bool_]:
    """Apply Schmitt trigger-style hysteresis thresholding.

    Args:
        data: Input analog data.
        thresh_low: Lower threshold (switch to low when below).
        thresh_high: Upper threshold (switch to high when above).

    Returns:
        Boolean array with hysteresis applied.
    """
    result = np.zeros(len(data), dtype=np.bool_)

    # Initial state based on first sample
    state = data[0] >= (thresh_low + thresh_high) / 2

    for i, value in enumerate(data):
        if state:
            # Currently high, switch low if below thresh_low
            if value < thresh_low:
                state = False
        # Currently low, switch high if above thresh_high
        elif value >= thresh_high:
            state = True
        result[i] = state

    return result


def detect_edges(
    trace: WaveformTrace | DigitalTrace,
    *,
    edge_type: Literal["rising", "falling", "both"] = "both",
    threshold: float | Literal["auto"] = "auto",
) -> NDArray[np.float64]:
    """Detect edge transitions in a signal.

    Finds rising and/or falling edges with sub-sample timestamp
    interpolation for improved accuracy.

    Args:
        trace: Input waveform (analog or digital).
        edge_type: Type of edges to detect:
            - "rising": Low-to-high transitions
            - "falling": High-to-low transitions
            - "both": All transitions
        threshold: Threshold for edge detection (only for analog traces).

    Returns:
        Array of edge timestamps in seconds.

    Raises:
        InsufficientDataError: If trace has insufficient data.

    Example:
        >>> edges = detect_edges(trace, edge_type="rising")
        >>> print(f"Found {len(edges)} rising edges")
    """
    if len(trace.data) < 2:
        raise InsufficientDataError(
            "Trace too short for edge detection",
            required=2,
            available=len(trace.data),
            analysis_type="edge_detection",
        )

    # Convert to digital if analog
    digital = to_digital(trace, threshold=threshold) if isinstance(trace, WaveformTrace) else trace

    # Find transitions - ensure we have a numpy array
    data = np.asarray(digital.data)

    transitions = np.diff(data.astype(np.int8))

    # Get edge indices
    if edge_type == "rising":
        edge_indices = np.where(transitions == 1)[0]
    elif edge_type == "falling":
        edge_indices = np.where(transitions == -1)[0]
    else:  # both
        edge_indices = np.where(transitions != 0)[0]

    # Convert indices to timestamps
    sample_period = digital.metadata.time_base
    timestamps = edge_indices.astype(np.float64) * sample_period

    # Sub-sample interpolation for analog traces
    if isinstance(trace, WaveformTrace) and threshold != "auto":
        thresh_value = float(threshold)
        timestamps = _interpolate_edges(trace.data, edge_indices, sample_period, thresh_value)

    return timestamps


def _detect_edges_internal(
    analog_data: NDArray[np.floating[Any]],
    digital_data: NDArray[np.bool_],
    sample_rate: float,
    threshold: float,
) -> list[tuple[float, bool]]:
    """Detect edges and return as (timestamp, is_rising) tuples.

    Args:
        analog_data: Original analog data for interpolation.
        digital_data: Thresholded digital data.
        sample_rate: Sample rate in Hz.
        threshold: Threshold used for conversion.

    Returns:
        List of (timestamp, is_rising) tuples.
    """
    sample_period = 1.0 / sample_rate
    transitions = np.diff(digital_data.astype(np.int8))

    edges: list[tuple[float, bool]] = []

    # Rising edges
    rising_indices = np.where(transitions == 1)[0]
    for idx in rising_indices:
        # Sub-sample interpolation
        if 0 < idx < len(analog_data) - 1:
            t = _interpolate_crossing(
                analog_data[idx], analog_data[idx + 1], threshold, sample_period
            )
            timestamp = idx * sample_period + t
        else:
            timestamp = idx * sample_period
        edges.append((timestamp, True))

    # Falling edges
    falling_indices = np.where(transitions == -1)[0]
    for idx in falling_indices:
        if 0 < idx < len(analog_data) - 1:
            t = _interpolate_crossing(
                analog_data[idx], analog_data[idx + 1], threshold, sample_period
            )
            timestamp = idx * sample_period + t
        else:
            timestamp = idx * sample_period
        edges.append((timestamp, False))

    # Sort by timestamp
    edges.sort(key=lambda x: x[0])

    return edges


def _interpolate_edges(
    data: NDArray[np.floating[Any]],
    edge_indices: NDArray[np.intp],
    sample_period: float,
    threshold: float,
) -> NDArray[np.float64]:
    """Interpolate edge timestamps for sub-sample accuracy.

    Uses linear interpolation between samples to estimate the
    exact crossing point.

    Args:
        data: Analog data array.
        edge_indices: Indices of detected edges.
        sample_period: Time between samples.
        threshold: Threshold level.

    Returns:
        Array of interpolated timestamps.
    """
    timestamps = np.zeros(len(edge_indices), dtype=np.float64)

    for i, idx in enumerate(edge_indices):
        base_time = idx * sample_period

        if 0 < idx < len(data) - 1:
            # Linear interpolation between samples
            t = _interpolate_crossing(data[idx], data[idx + 1], threshold, sample_period)
            timestamps[i] = base_time + t
        else:
            timestamps[i] = base_time

    return timestamps


def _interpolate_crossing(
    v1: float,
    v2: float,
    threshold: float,
    sample_period: float,
) -> float:
    """Linearly interpolate threshold crossing time.

    Args:
        v1: Voltage at sample before crossing.
        v2: Voltage at sample after crossing.
        threshold: Threshold level.
        sample_period: Time between samples.

    Returns:
        Time offset from v1 to crossing point.
    """
    dv = v2 - v1
    if abs(dv) < 1e-12:
        return sample_period / 2  # Midpoint if no change

    # Linear interpolation: t = (threshold - v1) / (v2 - v1) * period
    t = (threshold - v1) / dv * sample_period
    return max(0.0, min(sample_period, t))


def get_logic_threshold(
    family: str,
    threshold_type: Literal["midpoint", "VIH", "VIL"] = "midpoint",
) -> float:
    """Get threshold voltage for a logic family.

    Args:
        family: Logic family name (e.g., "TTL", "LVCMOS_3V3").
        threshold_type: Type of threshold:
            - "midpoint": Midpoint between VIL_max and VIH_min
            - "VIH": Minimum input high voltage
            - "VIL": Maximum input low voltage

    Returns:
        Threshold voltage.

    Raises:
        ValueError: If family or threshold_type is unknown.

    Example:
        >>> get_logic_threshold("TTL", "midpoint")
        1.4
    """
    if family not in LOGIC_FAMILIES:
        available = ", ".join(LOGIC_FAMILIES.keys())
        raise ValueError(f"Unknown logic family: {family}. Available: {available}")

    levels = LOGIC_FAMILIES[family]

    if threshold_type == "midpoint":
        return (levels["VIL_max"] + levels["VIH_min"]) / 2
    elif threshold_type == "VIH":
        return levels["VIH_min"]
    elif threshold_type == "VIL":
        return levels["VIL_max"]
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")


__all__ = [
    "LOGIC_FAMILIES",
    "detect_edges",
    "get_logic_threshold",
    "to_digital",
]
