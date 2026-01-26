"""Edge triggering for Oscura.

Provides edge detection with configurable thresholds, hysteresis,
and edge polarity (rising, falling, or both).

Example:
    >>> from oscura.utils.triggering.edge import EdgeTrigger, find_rising_edges
    >>> # Object-oriented approach
    >>> trigger = EdgeTrigger(level=1.5, edge="rising", hysteresis=0.1)
    >>> events = trigger.find_events(trace)
    >>> # Functional approach
    >>> timestamps = find_rising_edges(trace, level=1.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.types import DigitalTrace, WaveformTrace
from oscura.utils.triggering.base import (
    Trigger,
    TriggerEvent,
    TriggerType,
    interpolate_crossing,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EdgeTrigger(Trigger):
    """Edge trigger with threshold and optional hysteresis.

    Detects signal crossings of a threshold level with configurable
    hysteresis for noise immunity.

    Attributes:
        level: Trigger threshold level.
        edge: Edge type - "rising", "falling", or "either".
        hysteresis: Hysteresis band (Schmitt trigger style).
    """

    def __init__(
        self,
        level: float,
        edge: Literal["rising", "falling", "either"] = "rising",
        hysteresis: float = 0.0,
    ) -> None:
        """Initialize edge trigger.

        Args:
            level: Trigger threshold level in signal units (e.g., volts).
            edge: Edge polarity to trigger on.
            hysteresis: Hysteresis band width. Trigger requires signal to
                       cross level +/- hysteresis/2 before retriggering.
        """
        self.level = level
        self.edge = edge
        self.hysteresis = hysteresis

    def find_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[TriggerEvent]:
        """Find all edge events in the trace.

        Args:
            trace: Input waveform or digital trace.

        Returns:
            List of trigger events for each detected edge.
        """
        if isinstance(trace, DigitalTrace):
            # For digital traces, use edge list if available
            data = trace.data.astype(np.float64)
        else:
            data = trace.data

        sample_period = trace.metadata.time_base
        events: list[TriggerEvent] = []

        if self.hysteresis > 0:
            # Schmitt trigger mode
            events = self._find_edges_with_hysteresis(data, sample_period)
        else:
            # Simple threshold crossing
            events = self._find_edges_simple(data, sample_period)

        return events

    def _find_edges_simple(
        self,
        data: NDArray[np.floating[Any]],
        sample_period: float,
    ) -> list[TriggerEvent]:
        """Find edges using simple threshold crossing."""
        events: list[TriggerEvent] = []

        below = data < self.level
        above = data >= self.level

        if self.edge in ("rising", "either"):
            # Rising: below -> above
            rising_idx = np.where(below[:-1] & above[1:])[0]
            for idx in rising_idx:
                timestamp = interpolate_crossing(data, idx, self.level, sample_period, rising=True)
                events.append(
                    TriggerEvent(
                        timestamp=timestamp,
                        sample_index=int(idx),
                        event_type=TriggerType.RISING_EDGE,
                        level=float(data[idx + 1]),
                    )
                )

        if self.edge in ("falling", "either"):
            # Falling: above -> below
            falling_idx = np.where(above[:-1] & below[1:])[0]
            for idx in falling_idx:
                timestamp = interpolate_crossing(data, idx, self.level, sample_period, rising=False)
                events.append(
                    TriggerEvent(
                        timestamp=timestamp,
                        sample_index=int(idx),
                        event_type=TriggerType.FALLING_EDGE,
                        level=float(data[idx + 1]),
                    )
                )

        # Sort by timestamp if we detected both edge types
        if self.edge == "either":
            events.sort(key=lambda e: e.timestamp)

        return events

    def _find_edges_with_hysteresis(
        self,
        data: NDArray[np.floating[Any]],
        sample_period: float,
    ) -> list[TriggerEvent]:
        """Find edges using Schmitt trigger with hysteresis."""
        events: list[TriggerEvent] = []

        high_thresh = self.level + self.hysteresis / 2
        low_thresh = self.level - self.hysteresis / 2

        # State machine: track if we're currently "high" or "low"
        state = "low" if data[0] < self.level else "high"

        for i in range(1, len(data)):
            if state == "low" and data[i] >= high_thresh:
                # Rising edge detected
                state = "high"
                if self.edge in ("rising", "either"):
                    timestamp = interpolate_crossing(
                        data, i - 1, high_thresh, sample_period, rising=True
                    )
                    events.append(
                        TriggerEvent(
                            timestamp=timestamp,
                            sample_index=i,
                            event_type=TriggerType.RISING_EDGE,
                            level=float(data[i]),
                        )
                    )

            elif state == "high" and data[i] <= low_thresh:
                # Falling edge detected
                state = "low"
                if self.edge in ("falling", "either"):
                    timestamp = interpolate_crossing(
                        data, i - 1, low_thresh, sample_period, rising=False
                    )
                    events.append(
                        TriggerEvent(
                            timestamp=timestamp,
                            sample_index=i,
                            event_type=TriggerType.FALLING_EDGE,
                            level=float(data[i]),
                        )
                    )

        return events


def find_rising_edges(
    trace: WaveformTrace,
    level: float | None = None,
    *,
    hysteresis: float = 0.0,
    return_indices: bool = False,
) -> NDArray[np.float64] | NDArray[np.int64]:
    """Find all rising edge timestamps or indices.

    Args:
        trace: Input waveform trace.
        level: Trigger threshold. If None, uses signal midpoint.
        hysteresis: Hysteresis band for noise immunity.
        return_indices: If True, return sample indices instead of timestamps.

    Returns:
        Array of timestamps (seconds) or sample indices.

    Example:
        >>> edges = find_rising_edges(trace, level=1.5)
        >>> print(f"Found {len(edges)} rising edges")
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = EdgeTrigger(level=level, edge="rising", hysteresis=hysteresis)
    events = trigger.find_events(trace)

    if return_indices:
        return np.array([e.sample_index for e in events], dtype=np.int64)
    return np.array([e.timestamp for e in events], dtype=np.float64)


def find_falling_edges(
    trace: WaveformTrace,
    level: float | None = None,
    *,
    hysteresis: float = 0.0,
    return_indices: bool = False,
) -> NDArray[np.float64] | NDArray[np.int64]:
    """Find all falling edge timestamps or indices.

    Args:
        trace: Input waveform trace.
        level: Trigger threshold. If None, uses signal midpoint.
        hysteresis: Hysteresis band for noise immunity.
        return_indices: If True, return sample indices instead of timestamps.

    Returns:
        Array of timestamps (seconds) or sample indices.

    Example:
        >>> edges = find_falling_edges(trace, level=1.5)
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = EdgeTrigger(level=level, edge="falling", hysteresis=hysteresis)
    events = trigger.find_events(trace)

    if return_indices:
        return np.array([e.sample_index for e in events], dtype=np.int64)
    return np.array([e.timestamp for e in events], dtype=np.float64)


def find_all_edges(
    trace: WaveformTrace,
    level: float | None = None,
    *,
    hysteresis: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Find all edges (rising and falling) with polarity.

    Args:
        trace: Input waveform trace.
        level: Trigger threshold. If None, uses signal midpoint.
        hysteresis: Hysteresis band for noise immunity.

    Returns:
        Tuple of (timestamps, is_rising) where is_rising is True for
        rising edges and False for falling edges.

    Example:
        >>> timestamps, is_rising = find_all_edges(trace, level=1.5)
        >>> rising = timestamps[is_rising]
        >>> falling = timestamps[~is_rising]
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = EdgeTrigger(level=level, edge="either", hysteresis=hysteresis)
    events = trigger.find_events(trace)

    timestamps = np.array([e.timestamp for e in events], dtype=np.float64)
    is_rising = np.array([e.event_type == TriggerType.RISING_EDGE for e in events], dtype=np.bool_)

    return timestamps, is_rising


def edge_count(
    trace: WaveformTrace,
    level: float | None = None,
    edge: Literal["rising", "falling", "either"] = "either",
    *,
    hysteresis: float = 0.0,
) -> int:
    """Count edges in a trace.

    Args:
        trace: Input waveform trace.
        level: Trigger threshold. If None, uses signal midpoint.
        edge: Edge type to count.
        hysteresis: Hysteresis band for noise immunity.

    Returns:
        Number of edges found.

    Example:
        >>> n_rising = edge_count(trace, level=1.5, edge="rising")
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = EdgeTrigger(level=level, edge=edge, hysteresis=hysteresis)
    return trigger.count_events(trace)


def edge_rate(
    trace: WaveformTrace,
    level: float | None = None,
    edge: Literal["rising", "falling", "either"] = "either",
    *,
    hysteresis: float = 0.0,
) -> float:
    """Calculate edge rate (edges per second).

    Args:
        trace: Input waveform trace.
        level: Trigger threshold.
        edge: Edge type to count.
        hysteresis: Hysteresis band for noise immunity.

    Returns:
        Edge rate in Hz.

    Example:
        >>> rate = edge_rate(trace, level=1.5, edge="rising")
        >>> print(f"Toggle rate: {rate} Hz")
    """
    count = edge_count(trace, level, edge, hysteresis=hysteresis)
    duration = trace.duration

    if duration <= 0:
        return 0.0

    return count / duration


__all__ = [
    "EdgeTrigger",
    "edge_count",
    "edge_rate",
    "find_all_edges",
    "find_falling_edges",
    "find_rising_edges",
]
