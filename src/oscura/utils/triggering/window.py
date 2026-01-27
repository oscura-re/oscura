"""Window and zone triggering for Oscura.

Provides window triggering (signal inside/outside voltage window) and
zone triggering (signal enters/exits defined zones) for limit testing.

Example:
    >>> from oscura.utils.triggering.window import WindowTrigger, find_window_violations
    >>> # Trigger when signal exits 0-3.3V window
    >>> trigger = WindowTrigger(low_threshold=0, high_threshold=3.3, trigger_on="exit")
    >>> violations = trigger.find_events(trace)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.core.exceptions import AnalysisError
from oscura.utils.triggering.base import (
    Trigger,
    TriggerEvent,
    TriggerType,
)

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


@dataclass
class Zone:
    """Defines a voltage/time zone for triggering.

    Attributes:
        low: Lower voltage boundary.
        high: Upper voltage boundary.
        start_time: Start time boundary (None for no limit).
        end_time: End time boundary (None for no limit).
        name: Optional zone name for identification.
    """

    low: float
    high: float
    start_time: float | None = None
    end_time: float | None = None
    name: str = ""


class WindowTrigger(Trigger):
    """Window trigger for detecting voltage limit violations.

    Triggers when the signal enters or exits a voltage window defined
    by low and high threshold levels.

    Attributes:
        low_threshold: Lower window boundary.
        high_threshold: Upper window boundary.
        trigger_on: When to trigger - "entry", "exit", or "both".
    """

    def __init__(
        self,
        low_threshold: float,
        high_threshold: float,
        trigger_on: Literal["entry", "exit", "both"] = "exit",
    ) -> None:
        """Initialize window trigger.

        Args:
            low_threshold: Lower window boundary.
            high_threshold: Upper window boundary.
            trigger_on: "entry" triggers when entering window,
                       "exit" triggers when leaving window,
                       "both" triggers on either event.

        Raises:
            AnalysisError: If low_threshold is not less than high_threshold.
        """
        if low_threshold >= high_threshold:
            raise AnalysisError("low_threshold must be less than high_threshold")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.trigger_on = trigger_on

    def find_events(
        self,
        trace: WaveformTrace,  # type: ignore[override]
    ) -> list[TriggerEvent]:
        """Find all window entry/exit events.

        Args:
            trace: Input waveform trace.

        Returns:
            List of trigger events for window crossings.
        """
        data = trace.data
        sample_period = trace.metadata.time_base
        events: list[TriggerEvent] = []

        # Determine if each sample is inside the window
        inside = (data >= self.low_threshold) & (data <= self.high_threshold)

        # Find transitions
        for i in range(1, len(inside)):
            if inside[i] and not inside[i - 1]:
                # Entry event
                if self.trigger_on in ("entry", "both"):
                    events.append(
                        TriggerEvent(
                            timestamp=i * sample_period,
                            sample_index=i,
                            event_type=TriggerType.WINDOW_ENTRY,
                            level=float(data[i]),
                            data={
                                "window": (self.low_threshold, self.high_threshold),
                                "direction": "entering",
                            },
                        )
                    )

            elif not inside[i] and inside[i - 1]:
                # Exit event
                if self.trigger_on in ("exit", "both"):
                    # Determine which boundary was crossed
                    boundary = "high" if data[i] > self.high_threshold else "low"
                    events.append(
                        TriggerEvent(
                            timestamp=i * sample_period,
                            sample_index=i,
                            event_type=TriggerType.WINDOW_EXIT,
                            level=float(data[i]),
                            data={
                                "window": (self.low_threshold, self.high_threshold),
                                "direction": "exiting",
                                "boundary": boundary,
                            },
                        )
                    )

        return events


class ZoneTrigger(Trigger):
    """Zone trigger for multiple defined voltage/time zones.

    Triggers when signal enters any of the defined zones. Useful for
    mask testing and compliance checking.

    Attributes:
        zones: List of Zone definitions.
        trigger_on: When to trigger - "entry", "exit", or "violation".
    """

    def __init__(
        self,
        zones: list[Zone],
        trigger_on: Literal["entry", "exit", "violation"] = "violation",
    ) -> None:
        """Initialize zone trigger.

        Args:
            zones: List of zones to monitor.
            trigger_on: "entry" for entering zones, "exit" for leaving,
                       "violation" is alias for "entry" (common use case).
        """
        self.zones = zones
        self.trigger_on = trigger_on

    def find_events(
        self,
        trace: WaveformTrace,  # type: ignore[override]
    ) -> list[TriggerEvent]:
        """Find all zone-related events.

        Args:
            trace: Input waveform trace.

        Returns:
            List of trigger events.
        """
        data = trace.data
        sample_period = trace.metadata.time_base
        time_vector = np.arange(len(data)) * sample_period
        events: list[TriggerEvent] = []

        for zone in self.zones:
            # Check time limits
            if zone.start_time is not None:
                time_mask = time_vector >= zone.start_time
            else:
                time_mask = np.ones(len(data), dtype=bool)

            if zone.end_time is not None:
                time_mask &= time_vector <= zone.end_time

            # Check voltage limits
            in_zone = (data >= zone.low) & (data <= zone.high) & time_mask

            # Find transitions
            for i in range(1, len(in_zone)):
                if in_zone[i] and not in_zone[i - 1]:
                    # Entry event
                    if self.trigger_on in ("entry", "violation"):
                        events.append(
                            TriggerEvent(
                                timestamp=i * sample_period,
                                sample_index=i,
                                event_type=TriggerType.ZONE_VIOLATION,
                                level=float(data[i]),
                                data={
                                    "zone_name": zone.name,
                                    "zone_bounds": (zone.low, zone.high),
                                    "direction": "entering",
                                },
                            )
                        )

                elif not in_zone[i] and in_zone[i - 1]:
                    # Exit event
                    if self.trigger_on == "exit":
                        events.append(
                            TriggerEvent(
                                timestamp=i * sample_period,
                                sample_index=i,
                                event_type=TriggerType.ZONE_VIOLATION,
                                level=float(data[i]),
                                data={
                                    "zone_name": zone.name,
                                    "zone_bounds": (zone.low, zone.high),
                                    "direction": "exiting",
                                },
                            )
                        )

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        return events


def find_window_violations(
    trace: WaveformTrace,
    low: float,
    high: float,
) -> list[TriggerEvent]:
    """Find all window violations (signal outside limits).

    Args:
        trace: Input waveform trace.
        low: Lower limit.
        high: Upper limit.

    Returns:
        List of trigger events for each exit from the window.

    Example:
        >>> # Check if signal stays within 0-3.3V
        >>> violations = find_window_violations(trace, low=0, high=3.3)
        >>> if violations:
        ...     print(f"Signal violated limits {len(violations)} times")
    """
    trigger = WindowTrigger(
        low_threshold=low,
        high_threshold=high,
        trigger_on="exit",
    )
    return trigger.find_events(trace)


def find_zone_events(
    trace: WaveformTrace,
    zones: list[tuple[float, float] | Zone],
) -> list[TriggerEvent]:
    """Find events where signal enters defined zones.

    Args:
        trace: Input waveform trace.
        zones: List of zones as (low, high) tuples or Zone objects.

    Returns:
        List of trigger events.

    Example:
        >>> # Define forbidden zones
        >>> zones = [
        ...     (0.8, 1.2),   # Metastable region around 1V
        ...     (3.5, 5.0),   # Overvoltage region
        ... ]
        >>> events = find_zone_events(trace, zones)
    """
    zone_objs: list[Zone] = []
    for i, z in enumerate(zones):
        if isinstance(z, Zone):
            zone_objs.append(z)
        else:
            zone_objs.append(Zone(low=z[0], high=z[1], name=f"zone_{i}"))

    trigger = ZoneTrigger(zones=zone_objs, trigger_on="violation")
    return trigger.find_events(trace)


def check_limits(
    trace: WaveformTrace,
    low: float,
    high: float,
) -> dict:  # type: ignore[type-arg]
    """Check if trace stays within voltage limits.

    Args:
        trace: Input waveform trace.
        low: Lower limit.
        high: Upper limit.

    Returns:
        Dictionary with:
        - passed: True if no violations
        - violations: List of violation events
        - min_value: Minimum value in trace
        - max_value: Maximum value in trace
        - time_in_spec: Percentage of time within limits
        - time_out_of_spec: Percentage of time outside limits

    Example:
        >>> result = check_limits(trace, low=0, high=3.3)
        >>> if result['passed']:
        ...     print("Signal within limits")
        >>> else:
        ...     print(f"{result['time_out_of_spec']:.1f}% of time out of spec")
    """
    violations = find_window_violations(trace, low, high)

    data = trace.data
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    # Calculate time in/out of spec
    in_spec = (data >= low) & (data <= high)
    pct_in_spec = np.sum(in_spec) / len(data) * 100
    pct_out_spec = 100 - pct_in_spec

    return {
        "passed": len(violations) == 0 and min_val >= low and max_val <= high,
        "violations": violations,
        "min_value": min_val,
        "max_value": max_val,
        "time_in_spec": pct_in_spec,
        "time_out_of_spec": pct_out_spec,
    }


class MaskTrigger(Trigger):
    """Mask trigger for eye diagram and waveform mask testing.

    Tests waveform against a defined mask (polygonal region).
    Triggers on any mask violation.
    """

    def __init__(
        self,
        mask_points: list[tuple[float, float]],
        mode: Literal["inside", "outside"] = "inside",
    ) -> None:
        """Initialize mask trigger.

        Args:
            mask_points: List of (time, voltage) points defining mask polygon.
            mode: "inside" triggers when signal is inside mask,
                  "outside" triggers when signal is outside mask.

        Raises:
            AnalysisError: If mask has fewer than 3 points.
        """
        if len(mask_points) < 3:
            raise AnalysisError("Mask must have at least 3 points")

        self.mask_points = mask_points
        self.mode = mode

    def find_events(
        self,
        trace: WaveformTrace,  # type: ignore[override]
    ) -> list[TriggerEvent]:
        """Find mask violations.

        Args:
            trace: Input waveform trace.

        Returns:
            List of trigger events for mask violations.
        """
        from matplotlib.path import Path

        # Create polygon path
        mask_path = Path(self.mask_points)

        data = trace.data
        sample_period = trace.metadata.time_base
        time_vector = np.arange(len(data)) * sample_period

        # Create points array for containment test
        points = np.column_stack([time_vector, data])

        # Check which points are inside the mask
        inside = mask_path.contains_points(points)

        events: list[TriggerEvent] = []

        # Find violations based on mode
        if self.mode == "inside":
            # Trigger when inside mask (mask defines forbidden region)
            violation_indices = np.where(inside)[0]
        else:
            # Trigger when outside mask (mask defines required region)
            violation_indices = np.where(~inside)[0]

        # Group consecutive violations into events
        if len(violation_indices) > 0:
            # Find starts of violation regions
            starts = [violation_indices[0]]
            for i in range(1, len(violation_indices)):
                if violation_indices[i] != violation_indices[i - 1] + 1:
                    starts.append(violation_indices[i])

            for start_idx in starts:
                events.append(
                    TriggerEvent(
                        timestamp=start_idx * sample_period,
                        sample_index=int(start_idx),
                        event_type=TriggerType.ZONE_VIOLATION,
                        level=float(data[start_idx]),
                        data={
                            "mask_mode": self.mode,
                            "violation_type": "inside_forbidden"
                            if self.mode == "inside"
                            else "outside_required",
                        },
                    )
                )

        return events


__all__ = [
    "MaskTrigger",
    "WindowTrigger",
    "Zone",
    "ZoneTrigger",
    "check_limits",
    "find_window_violations",
    "find_zone_events",
]
