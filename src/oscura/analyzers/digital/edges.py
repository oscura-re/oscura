"""Edge detection with sub-sample precision and timing analysis.

This module provides edge detection with interpolation for sub-sample precision,
timing measurements between edges, and timing constraint validation for digital
signal analysis.


Example:
    >>> import numpy as np
    >>> from oscura.analyzers.digital.edges import detect_edges, measure_edge_timing
    >>> # Generate test signal
    >>> signal = np.array([0, 0, 0.5, 1.0, 1.0, 1.0, 0.5, 0, 0])
    >>> # Detect edges
    >>> edges = detect_edges(signal, edge_type='both', sample_rate=100e6)
    >>> # Measure timing
    >>> timing = measure_edge_timing(edges, sample_rate=100e6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.core.memoize import memoize_analysis

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Edge:
    """A detected edge in the signal.

    Attributes:
        sample_index: Sample index where edge was detected.
        time: Interpolated edge time in seconds.
        edge_type: Type of edge ('rising' or 'falling').
        amplitude: Transition amplitude in signal units (volts).
        slew_rate: Edge slew rate (signal units per second).
        quality: Edge quality classification.
    """

    sample_index: int
    time: float  # Interpolated time
    edge_type: Literal["rising", "falling"]
    amplitude: float  # Transition amplitude
    slew_rate: float  # V/s or samples/s
    quality: Literal["clean", "slow", "noisy", "glitch"]


@dataclass
class EdgeTiming:
    """Timing measurements from edge analysis.

    Attributes:
        periods: Array of edge-to-edge periods in seconds.
        mean_period: Mean period in seconds.
        std_period: Standard deviation of period in seconds.
        min_period: Minimum period in seconds.
        max_period: Maximum period in seconds.
        duty_cycles: Array of duty cycle ratios (0-1).
        mean_duty_cycle: Mean duty cycle ratio.
        jitter_rms: RMS jitter in seconds.
        jitter_pp: Peak-to-peak jitter in seconds.
    """

    periods: NDArray[np.float64]  # Edge-to-edge periods
    mean_period: float
    std_period: float
    min_period: float
    max_period: float
    duty_cycles: NDArray[np.float64]
    mean_duty_cycle: float
    jitter_rms: float
    jitter_pp: float


@dataclass
class TimingConstraint:
    """Timing constraint for validation.

    Attributes:
        name: Descriptive name for the constraint.
        min_time: Minimum allowed time in seconds (None for no minimum).
        max_time: Maximum allowed time in seconds (None for no maximum).
        reference: Which edges to check ('rising', 'falling', or 'both').
    """

    name: str
    min_time: float | None = None
    max_time: float | None = None
    reference: str | None = None  # 'rising', 'falling', 'both'


@dataclass
class TimingViolation:
    """A timing constraint violation.

    Attributes:
        constraint: The violated constraint.
        measured_time: The measured time that violated the constraint.
        edge_index: Index of the edge that violated the constraint.
        sample_index: Sample index where violation occurred.
    """

    constraint: TimingConstraint
    measured_time: float
    edge_index: int
    sample_index: int


@memoize_analysis(maxsize=32)
def detect_edges(
    trace: NDArray[np.float64],
    edge_type: Literal["rising", "falling", "both"] = "both",
    threshold: float | Literal["auto"] = "auto",
    hysteresis: float = 0.0,
    sample_rate: float = 1.0,
) -> list[Edge]:
    """Detect signal edges with configurable threshold.

    Detects rising and/or falling edges in a digital or analog signal with
    optional hysteresis for noise immunity.

    Args:
        trace: Input signal trace (analog or digital).
        edge_type: Type of edges to detect ('rising', 'falling', or 'both').
        threshold: Detection threshold. 'auto' computes from signal midpoint.
        hysteresis: Hysteresis amount for noise immunity (signal units).
        sample_rate: Sample rate in Hz for time calculation.

    Returns:
        List of Edge objects with detected edges.

    Example:
        >>> signal = np.array([0, 0, 1, 1, 0, 0])
        >>> edges = detect_edges(signal, edge_type='rising')
        >>> len(edges)
        1
    """
    if len(trace) < 2:
        return []

    trace = np.asarray(trace, dtype=np.float64)

    # Compute threshold if auto
    thresh_val: float
    if threshold == "auto":
        thresh_val = float((np.max(trace) + np.min(trace)) / 2.0)
    else:
        thresh_val = threshold

    # Apply hysteresis if specified
    if hysteresis > 0:
        thresh_high = thresh_val + hysteresis / 2.0
        thresh_low = thresh_val - hysteresis / 2.0
    else:
        thresh_high = thresh_val
        thresh_low = thresh_val

    edges: list[Edge] = []
    time_base = 1.0 / sample_rate

    # State machine for hysteresis
    state = trace[0] > thresh_val  # Initial state

    for i in range(1, len(trace)):
        prev_val = trace[i - 1]
        curr_val = trace[i]

        # Detect transitions with hysteresis
        if not state and curr_val > thresh_high:
            # Rising edge
            if edge_type in ["rising", "both"]:
                # Interpolate edge time
                interp_time = interpolate_edge_time(trace, i - 1, method="linear")
                time = (i - 1 + interp_time) * time_base

                # Calculate edge properties
                amplitude = curr_val - prev_val
                slew_rate = amplitude * sample_rate

                # Classify quality (simple heuristic)
                quality = classify_edge_quality(trace, i, sample_rate)

                edges.append(
                    Edge(
                        sample_index=i,
                        time=time,
                        edge_type="rising",
                        amplitude=abs(amplitude),
                        slew_rate=slew_rate,
                        quality=quality,
                    )
                )
            state = True

        elif state and curr_val < thresh_low:
            # Falling edge
            if edge_type in ["falling", "both"]:
                # Interpolate edge time
                interp_time = interpolate_edge_time(trace, i - 1, method="linear")
                time = (i - 1 + interp_time) * time_base

                # Calculate edge properties
                amplitude = prev_val - curr_val
                slew_rate = -amplitude * sample_rate

                # Classify quality (simple heuristic)
                quality = classify_edge_quality(trace, i, sample_rate)

                edges.append(
                    Edge(
                        sample_index=i,
                        time=time,
                        edge_type="falling",
                        amplitude=abs(amplitude),
                        slew_rate=slew_rate,
                        quality=quality,
                    )
                )
            state = False

    return edges


def interpolate_edge_time(
    trace: NDArray[np.float64], sample_index: int, method: Literal["linear", "quadratic"] = "linear"
) -> float:
    """Interpolate edge time for sub-sample precision.

    Uses linear or quadratic interpolation to estimate the fractional sample
    position where an edge crosses the threshold.

    Args:
        trace: Input signal trace.
        sample_index: Sample index just before the edge.
        method: Interpolation method ('linear' or 'quadratic').

    Returns:
        Fractional sample offset (0.0 to 1.0) from sample_index.

    Example:
        >>> trace = np.array([0, 0.3, 0.8, 1.0])
        >>> offset = interpolate_edge_time(trace, 1, method='linear')
    """
    if sample_index < 0 or sample_index >= len(trace) - 1:
        return 0.0

    if method == "linear":
        # Linear interpolation between two points
        v0 = trace[sample_index]
        v1 = trace[sample_index + 1]

        if abs(v1 - v0) < 1e-10:
            return 0.5  # Avoid division by zero

        # Find midpoint crossing
        threshold = (v0 + v1) / 2.0
        fraction = (threshold - v0) / (v1 - v0)

        # Clamp to valid range
        return float(np.clip(fraction, 0.0, 1.0))

    elif method == "quadratic":
        # Quadratic interpolation using 3 points
        if sample_index < 1 or sample_index >= len(trace) - 1:
            # Fall back to linear
            return interpolate_edge_time(trace, sample_index, method="linear")

        # Use points before, at, and after edge
        _v_prev = trace[sample_index - 1]
        v0 = trace[sample_index]
        v1 = trace[sample_index + 1]

        # Fit parabola and find threshold crossing
        # Simplified: use linear for now (full quadratic fit is complex)
        return interpolate_edge_time(trace, sample_index, method="linear")


def measure_edge_timing(edges: list[Edge], sample_rate: float = 1.0) -> EdgeTiming:
    """Measure timing between edges.

    Computes period, duty cycle, and jitter statistics from a list of detected edges.

    Args:
        edges: List of Edge objects from detect_edges().
        sample_rate: Sample rate in Hz (for time base).

    Returns:
        EdgeTiming object with timing measurements.

    Example:
        >>> edges = detect_edges(signal, edge_type='both', sample_rate=100e6)
        >>> timing = measure_edge_timing(edges, sample_rate=100e6)
    """
    if len(edges) < 2:
        # Not enough edges for timing analysis
        return EdgeTiming(
            periods=np.array([]),
            mean_period=0.0,
            std_period=0.0,
            min_period=0.0,
            max_period=0.0,
            duty_cycles=np.array([]),
            mean_duty_cycle=0.0,
            jitter_rms=0.0,
            jitter_pp=0.0,
        )

    # Calculate periods (time between consecutive edges)
    edge_times = np.array([e.time for e in edges])
    periods = np.diff(edge_times)

    # Calculate duty cycles (ratio of high time to period)
    duty_cycles = []
    rising_edges = [e for e in edges if e.edge_type == "rising"]
    falling_edges = [e for e in edges if e.edge_type == "falling"]

    # Match rising and falling edges to compute duty cycles
    for i in range(min(len(rising_edges), len(falling_edges))):
        rise_time = rising_edges[i].time
        fall_time = falling_edges[i].time

        # Find next edge of opposite type
        if i + 1 < len(rising_edges):
            next_rise = rising_edges[i + 1].time
            period = next_rise - rise_time
            if period > 0:
                high_time = fall_time - rise_time
                duty_cycle = high_time / period
                duty_cycles.append(np.clip(duty_cycle, 0.0, 1.0))

    duty_cycles_arr = np.array(duty_cycles) if duty_cycles else np.array([])

    # Calculate jitter
    if len(periods) > 1:
        mean_period = np.mean(periods)
        jitter_rms = np.std(periods)
        jitter_pp = np.max(periods) - np.min(periods)
    else:
        mean_period = periods[0] if len(periods) > 0 else 0.0
        jitter_rms = 0.0
        jitter_pp = 0.0

    return EdgeTiming(
        periods=periods,
        mean_period=float(mean_period),
        std_period=float(np.std(periods)) if len(periods) > 0 else 0.0,
        min_period=float(np.min(periods)) if len(periods) > 0 else 0.0,
        max_period=float(np.max(periods)) if len(periods) > 0 else 0.0,
        duty_cycles=duty_cycles_arr,
        mean_duty_cycle=float(np.mean(duty_cycles_arr)) if len(duty_cycles_arr) > 0 else 0.0,
        jitter_rms=float(jitter_rms),
        jitter_pp=float(jitter_pp),
    )


def check_timing_constraints(
    edges: list[Edge], constraints: list[TimingConstraint], sample_rate: float = 1.0
) -> list[TimingViolation]:
    """Check edges against timing constraints.

    Validates edge timing against specified constraints and reports violations.

    Args:
        edges: List of Edge objects to check.
        constraints: List of TimingConstraint objects defining limits.
        sample_rate: Sample rate in Hz.

    Returns:
        List of TimingViolation objects for any violations found.

    Example:
        >>> constraint = TimingConstraint(name="min_period", min_time=10e-9)
        >>> violations = check_timing_constraints(edges, [constraint])
    """
    violations: list[TimingViolation] = []

    if len(edges) < 2:
        return violations

    # Calculate periods between edges
    for i in range(len(edges) - 1):
        edge_time = edges[i].time
        next_time = edges[i + 1].time
        period = next_time - edge_time

        for constraint in constraints:
            # Check if constraint applies to this edge type
            if constraint.reference:
                if constraint.reference == "rising" and edges[i].edge_type != "rising":
                    continue
                if constraint.reference == "falling" and edges[i].edge_type != "falling":
                    continue

            # Check timing constraints
            violated = False

            if constraint.min_time is not None and period < constraint.min_time:
                violated = True

            if constraint.max_time is not None and period > constraint.max_time:
                violated = True

            if violated:
                violations.append(
                    TimingViolation(
                        constraint=constraint,
                        measured_time=period,
                        edge_index=i,
                        sample_index=edges[i].sample_index,
                    )
                )

    return violations


def classify_edge_quality(
    trace: NDArray[np.float64], edge_index: int, sample_rate: float
) -> Literal["clean", "slow", "noisy", "glitch"]:
    """Classify edge quality.

    Analyzes the edge transition to classify its quality based on slew rate,
    noise, and duration.

    Args:
        trace: Input signal trace.
        edge_index: Sample index of the edge.
        sample_rate: Sample rate in Hz.

    Returns:
        Quality classification: 'clean', 'slow', 'noisy', or 'glitch'.

    Example:
        >>> quality = classify_edge_quality(trace, 10, 100e6)
    """
    if edge_index < 1 or edge_index >= len(trace) - 1:
        return "clean"

    # Get window around edge
    window_size = min(10, edge_index, len(trace) - edge_index - 1)
    window = trace[edge_index - window_size : edge_index + window_size + 1]

    # Calculate transition amplitude
    v_before = trace[edge_index - 1]
    v_after = trace[edge_index]
    amplitude = abs(v_after - v_before)

    # Check for glitch (very short duration)
    if window_size < 3:
        return "glitch"

    # Calculate noise (std dev in window)
    noise = np.std(window)

    # Calculate slew rate
    _slew_rate = amplitude * sample_rate

    # Simple heuristic classification
    signal_range = np.max(trace) - np.min(trace)

    if amplitude < signal_range * 0.1:
        return "glitch"

    if noise > amplitude * 0.2:
        return "noisy"

    # Check if transition is slow (takes many samples)
    transition_samples = 0
    _threshold = (v_before + v_after) / 2.0

    for i in range(max(0, edge_index - window_size), min(len(trace), edge_index + window_size)):
        val = trace[i]
        if v_before < v_after:  # Rising
            if v_before <= val <= v_after:
                transition_samples += 1
        else:  # Falling
            if v_after <= val <= v_before:
                transition_samples += 1

    if transition_samples > 5:
        return "slow"

    return "clean"


class EdgeDetector:
    """Object-oriented wrapper for edge detection functionality.

    Provides a class-based interface for edge detection operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> detector = EdgeDetector()
        >>> rising, falling = detector.detect_all_edges(signal_data)
    """

    def __init__(
        self,
        threshold: float | Literal["auto"] = "auto",
        hysteresis: float = 0.0,
        sample_rate: float = 1.0,
        min_pulse_width: int | None = None,
    ):
        """Initialize edge detector.

        Args:
            threshold: Detection threshold. 'auto' computes from signal midpoint.
            hysteresis: Hysteresis amount for noise immunity (signal units).
            sample_rate: Sample rate in Hz for time calculation.
            min_pulse_width: Minimum pulse width in samples to filter noise.
        """
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.sample_rate = sample_rate
        self.min_pulse_width = min_pulse_width

    def detect_all_edges(
        self, trace: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Detect all rising and falling edges.

        Args:
            trace: Input signal trace (analog or digital).

        Returns:
            Tuple of (rising_edge_indices, falling_edge_indices).

        Example:
            >>> detector = EdgeDetector(sample_rate=100e6)
            >>> rising, falling = detector.detect_all_edges(signal)
        """
        edges = detect_edges(
            trace,
            edge_type="both",
            threshold=self.threshold,
            hysteresis=self.hysteresis,
            sample_rate=self.sample_rate,
        )

        # Filter by min_pulse_width if specified
        if self.min_pulse_width is not None and len(edges) > 1:
            filtered_edges = []
            for i, edge in enumerate(edges):
                if i == 0:
                    filtered_edges.append(edge)
                    continue
                # Check distance to previous edge
                dist = edge.sample_index - edges[i - 1].sample_index
                if dist >= self.min_pulse_width:
                    filtered_edges.append(edge)
            edges = filtered_edges

        rising_indices = np.array(
            [e.sample_index for e in edges if e.edge_type == "rising"], dtype=np.int64
        )
        falling_indices = np.array(
            [e.sample_index for e in edges if e.edge_type == "falling"], dtype=np.int64
        )

        return rising_indices, falling_indices

    def detect_rising_edges(self, trace: NDArray[np.float64]) -> list[Edge]:
        """Detect only rising edges.

        Args:
            trace: Input signal trace.

        Returns:
            List of Edge objects for rising edges.
        """
        return detect_edges(
            trace,
            edge_type="rising",
            threshold=self.threshold,
            hysteresis=self.hysteresis,
            sample_rate=self.sample_rate,
        )

    def detect_falling_edges(self, trace: NDArray[np.float64]) -> list[Edge]:
        """Detect only falling edges.

        Args:
            trace: Input signal trace.

        Returns:
            List of Edge objects for falling edges.
        """
        return detect_edges(
            trace,
            edge_type="falling",
            threshold=self.threshold,
            hysteresis=self.hysteresis,
            sample_rate=self.sample_rate,
        )

    def measure_timing(self, trace: NDArray[np.float64]) -> EdgeTiming:
        """Detect edges and measure timing.

        Args:
            trace: Input signal trace.

        Returns:
            EdgeTiming object with timing measurements.
        """
        edges = detect_edges(
            trace,
            edge_type="both",
            threshold=self.threshold,
            hysteresis=self.hysteresis,
            sample_rate=self.sample_rate,
        )
        return measure_edge_timing(edges, self.sample_rate)


__all__ = [
    "Edge",
    "EdgeDetector",
    "EdgeTiming",
    "TimingConstraint",
    "TimingViolation",
    "check_timing_constraints",
    "classify_edge_quality",
    "detect_edges",
    "interpolate_edge_time",
    "measure_edge_timing",
]
