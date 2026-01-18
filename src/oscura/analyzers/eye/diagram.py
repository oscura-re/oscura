"""Eye diagram generation from serial data.

This module generates eye diagrams by folding waveform data
at the unit interval boundary.

Example:
    >>> from oscura.analyzers.eye.diagram import generate_eye
    >>> eye = generate_eye(trace, unit_interval=1e-9)
    >>> print(f"Eye diagram: {eye.n_traces} traces, {eye.samples_per_ui} samples/UI")

References:
    IEEE 802.3: Ethernet Physical Layer Specifications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from oscura.core.exceptions import AnalysisError, InsufficientDataError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class EyeDiagram:
    """Eye diagram data structure.

    Attributes:
        data: 2D array of eye traces (n_traces x samples_per_ui).
        time_axis: Time axis in UI (0.0 to 2.0 for 2-UI eye).
        unit_interval: Unit interval in seconds.
        samples_per_ui: Number of samples per unit interval.
        n_traces: Number of overlaid traces.
        sample_rate: Original sample rate in Hz.
        histogram: Optional 2D histogram (voltage x time bins).
        voltage_bins: Bin edges for voltage axis.
        time_bins: Bin edges for time axis.
    """

    data: NDArray[np.float64]
    time_axis: NDArray[np.float64]
    unit_interval: float
    samples_per_ui: int
    n_traces: int
    sample_rate: float
    histogram: NDArray[np.float64] | None = None
    voltage_bins: NDArray[np.float64] | None = None
    time_bins: NDArray[np.float64] | None = None


def generate_eye(
    trace: WaveformTrace,
    unit_interval: float,
    *,
    n_ui: int = 2,
    trigger_level: float = 0.5,
    trigger_edge: str = "rising",
    max_traces: int | None = None,
    generate_histogram: bool = True,
    histogram_bins: tuple[int, int] = (100, 100),
) -> EyeDiagram:
    """Generate eye diagram from waveform data.

    Folds the waveform at unit interval boundaries to create
    an overlaid eye pattern for signal quality analysis.

    Args:
        trace: Input waveform trace.
        unit_interval: Unit interval (bit period) in seconds.
        n_ui: Number of unit intervals to display (1 or 2).
        trigger_level: Trigger level as fraction of amplitude.
        trigger_edge: Trigger on "rising" or "falling" edges.
        max_traces: Maximum number of traces to include.
        generate_histogram: Generate 2D histogram for persistence.
        histogram_bins: (voltage_bins, time_bins) for histogram.

    Returns:
        EyeDiagram with overlaid traces and optional histogram.

    Raises:
        AnalysisError: If unit interval is too short.
        InsufficientDataError: If not enough data for eye generation.

    Example:
        >>> eye = generate_eye(trace, unit_interval=1e-9)
        >>> print(f"Generated {eye.n_traces} traces")

    References:
        OIF CEI: Common Electrical I/O Eye Diagram Methodology
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate
    1.0 / sample_rate

    # Calculate samples per UI
    samples_per_ui = round(unit_interval * sample_rate)

    if samples_per_ui < 4:
        raise AnalysisError(
            f"Unit interval too short: {samples_per_ui} samples/UI. Need at least 4 samples per UI."
        )

    n_samples = len(data)
    total_ui_samples = samples_per_ui * n_ui

    if n_samples < total_ui_samples * 2:
        raise InsufficientDataError(
            f"Need at least {total_ui_samples * 2} samples for eye diagram",
            required=total_ui_samples * 2,
            available=n_samples,
            analysis_type="eye_diagram_generation",
        )

    # Find trigger points
    low = np.percentile(data, 10)
    high = np.percentile(data, 90)
    threshold = low + trigger_level * (high - low)

    if trigger_edge == "rising":
        trigger_mask = (data[:-1] < threshold) & (data[1:] >= threshold)
    else:
        trigger_mask = (data[:-1] >= threshold) & (data[1:] < threshold)

    trigger_indices = np.where(trigger_mask)[0]

    if len(trigger_indices) < 2:
        raise InsufficientDataError(
            "Not enough trigger events for eye diagram",
            required=2,
            available=len(trigger_indices),
            analysis_type="eye_diagram_generation",
        )

    # Extract eye traces
    eye_traces = []
    half_ui = samples_per_ui // 2  # Start half UI before trigger

    for trig_idx in trigger_indices:
        start_idx = trig_idx - half_ui
        end_idx = start_idx + total_ui_samples

        if start_idx >= 0 and end_idx <= n_samples:
            eye_traces.append(data[start_idx:end_idx])

        if max_traces is not None and len(eye_traces) >= max_traces:
            break

    if len(eye_traces) == 0:
        raise InsufficientDataError(
            "Could not extract any complete eye traces",
            required=1,
            available=0,
            analysis_type="eye_diagram_generation",
        )

    # Stack into 2D array
    eye_data = np.array(eye_traces, dtype=np.float64)

    # Generate time axis in UI
    time_axis = np.linspace(0, n_ui, total_ui_samples, endpoint=False)

    # Optional: Generate 2D histogram
    histogram = None
    voltage_bins = None
    time_bins = None

    if generate_histogram:
        # Flatten for histogram
        all_voltages = eye_data.flatten()
        all_times = np.tile(time_axis, len(eye_traces))

        # Create histogram
        voltage_range = (np.min(all_voltages), np.max(all_voltages))
        time_range = (0, n_ui)

        histogram, voltage_edges, time_edges = np.histogram2d(
            all_voltages,
            all_times,
            bins=histogram_bins,
            range=[voltage_range, time_range],
        )

        voltage_bins = voltage_edges
        time_bins = time_edges

    return EyeDiagram(
        data=eye_data,
        time_axis=time_axis,
        unit_interval=unit_interval,
        samples_per_ui=samples_per_ui,
        n_traces=len(eye_traces),
        sample_rate=sample_rate,
        histogram=histogram,
        voltage_bins=voltage_bins,
        time_bins=time_bins,
    )


def generate_eye_from_edges(
    trace: WaveformTrace,
    edge_timestamps: NDArray[np.float64],
    *,
    n_ui: int = 2,
    samples_per_ui: int = 100,
    max_traces: int | None = None,
) -> EyeDiagram:
    """Generate eye diagram using recovered clock edges.

    Uses pre-recovered clock edges for triggering, which can provide
    more accurate alignment than threshold-based triggering.

    Args:
        trace: Input waveform trace.
        edge_timestamps: Array of clock edge timestamps in seconds.
        n_ui: Number of unit intervals to display.
        samples_per_ui: Samples per UI in resampled eye.
        max_traces: Maximum traces to include.

    Returns:
        EyeDiagram with overlaid traces.

    Raises:
        InsufficientDataError: If not enough edge timestamps or traces.

    Example:
        >>> edges = recover_clock_edges(trace)
        >>> eye = generate_eye_from_edges(trace, edges)
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    if len(edge_timestamps) < 3:
        raise InsufficientDataError(
            "Need at least 3 edge timestamps",
            required=3,
            available=len(edge_timestamps),
            analysis_type="eye_diagram_generation",
        )

    # Calculate unit interval from edges
    periods = np.diff(edge_timestamps)
    unit_interval = float(np.median(periods))

    # Create time vector for original data
    original_time = np.arange(len(data)) / sample_rate

    # Extract and resample traces around each edge
    eye_traces = []
    total_samples = samples_per_ui * n_ui
    half_ui = unit_interval / 2

    for edge_time in edge_timestamps:
        # Define window around edge
        start_time = edge_time - half_ui
        end_time = start_time + unit_interval * n_ui

        if start_time < 0 or end_time > original_time[-1]:
            continue

        # Find samples within window
        mask = (original_time >= start_time) & (original_time <= end_time)
        window_time = original_time[mask] - start_time
        window_data = data[mask]

        if len(window_data) < 4:
            continue

        # Resample to consistent samples_per_ui
        resample_time = np.linspace(0, unit_interval * n_ui, total_samples)
        resampled = np.interp(resample_time, window_time, window_data)

        eye_traces.append(resampled)

        if max_traces is not None and len(eye_traces) >= max_traces:
            break

    if len(eye_traces) == 0:
        raise InsufficientDataError(
            "Could not extract any eye traces",
            required=1,
            available=0,
            analysis_type="eye_diagram_generation",
        )

    eye_data = np.array(eye_traces, dtype=np.float64)
    time_axis = np.linspace(0, n_ui, total_samples, endpoint=False)

    return EyeDiagram(
        data=eye_data,
        time_axis=time_axis,
        unit_interval=unit_interval,
        samples_per_ui=samples_per_ui,
        n_traces=len(eye_traces),
        sample_rate=sample_rate,
    )


def auto_center_eye_diagram(
    eye: EyeDiagram,
    *,
    trigger_fraction: float = 0.5,
    symmetric_range: bool = True,
) -> EyeDiagram:
    """Auto-center eye diagram on optimal crossing point.

    Automatically centers eye diagrams on the optimal trigger point
    and scales amplitude for maximum eye opening visibility with
    symmetric vertical centering.

    Args:
        eye: Input EyeDiagram to center.
        trigger_fraction: Trigger level as fraction of amplitude (default 0.5 = 50%).
        symmetric_range: Use symmetric amplitude range ±max(abs(signal)).

    Returns:
        Centered EyeDiagram with adjusted data.

    Raises:
        ValueError: If trigger_fraction is not in [0, 1].

    Example:
        >>> eye = generate_eye(trace, unit_interval=1e-9)
        >>> centered = auto_center_eye_diagram(eye)
        >>> # Centered at 50% crossing with symmetric amplitude

    References:
        VIS-021: Eye Diagram Auto-Centering
    """
    if not 0 <= trigger_fraction <= 1:
        raise ValueError(f"trigger_fraction must be in [0, 1], got {trigger_fraction}")

    data = eye.data

    # Calculate optimal trigger point using histogram-based threshold
    # Find median value (represents middle level)
    np.median(data)

    # Calculate amplitude range
    low = np.percentile(data, 10)
    high = np.percentile(data, 90)
    amplitude_range = high - low

    # Trigger threshold at specified fraction
    threshold = low + trigger_fraction * amplitude_range

    # Find crossing points for each trace
    # A crossing is where signal crosses threshold
    n_traces, samples_per_trace = data.shape
    crossing_indices = []

    for trace_idx in range(n_traces):
        trace = data[trace_idx, :]

        # Find zero-crossings relative to threshold
        crossings = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]

        if len(crossings) > 0:
            # Use first crossing in this trace
            crossing_indices.append(crossings[0])

    if len(crossing_indices) == 0:
        # No crossings found, return original
        import warnings

        warnings.warn(
            "No crossing points found, cannot auto-center eye diagram",
            UserWarning,
            stacklevel=2,
        )
        return eye

    # Calculate median crossing position
    int(np.median(crossing_indices))

    # Align all traces to common crossing point
    # This requires resampling/shifting each trace
    aligned_data = np.zeros_like(data)
    target_crossing = samples_per_trace // 2  # Center of trace

    for trace_idx in range(n_traces):
        trace = data[trace_idx, :]

        # Find crossing for this trace
        crossings = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]

        if len(crossings) > 0:
            crossing = crossings[0]
            shift = target_crossing - crossing

            # Shift trace by interpolation
            if shift != 0:
                # Simple roll (circular shift)
                aligned_data[trace_idx, :] = np.roll(trace, shift)
            else:
                aligned_data[trace_idx, :] = trace
        else:
            # No crossing, keep original
            aligned_data[trace_idx, :] = trace

    # Scale amplitude to symmetric range if requested
    if symmetric_range:
        max_abs = np.max(np.abs(aligned_data))
        if max_abs > 0:
            # Center on zero
            aligned_data = aligned_data - np.mean(aligned_data)
            # Scale to ±max for symmetric range
            # No additional scaling needed, data already centered

    # Create centered eye diagram
    return EyeDiagram(
        data=aligned_data,
        time_axis=eye.time_axis,
        unit_interval=eye.unit_interval,
        samples_per_ui=eye.samples_per_ui,
        n_traces=eye.n_traces,
        sample_rate=eye.sample_rate,
        histogram=None,  # Invalidate histogram after centering
        voltage_bins=None,
        time_bins=None,
    )


__all__ = [
    "EyeDiagram",
    "auto_center_eye_diagram",
    "generate_eye",
    "generate_eye_from_edges",
]
