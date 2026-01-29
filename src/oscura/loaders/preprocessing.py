"""Idle and padding detection and removal.

This module provides functions to detect and optionally remove idle regions,
padding, and non-data samples from loaded binary captures.


Example:
    >>> from oscura.loaders.preprocessing import detect_idle_regions, trim_idle
    >>> regions = detect_idle_regions(trace, pattern='zeros', min_duration=100)
    >>> print(f"Found {len(regions)} idle regions")
    >>> trimmed_trace = trim_idle(trace, trim_start=True, trim_end=True)
    >>> print(f"Trimmed {len(trace.data) - len(trimmed_trace.data)} samples")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from oscura.core.types import DigitalTrace, TraceMetadata

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Logger for debug output
logger = logging.getLogger(__name__)


@dataclass
class IdleRegion:
    """Idle region in a trace.



    Attributes:
        start: Start sample index.
        end: End sample index (exclusive).
        pattern: Detected idle pattern.
        duration_samples: Duration in samples.
    """

    start: int
    end: int
    pattern: str
    duration_samples: int

    @property
    def length(self) -> int:
        """Get region length in samples.

        Returns:
            Number of samples in region.
        """
        return self.end - self.start

    def get_duration_seconds(self, sample_rate: float) -> float:
        """Get region duration in seconds.

        Args:
            sample_rate: Sample rate in Hz.

        Returns:
            Duration in seconds.
        """
        return self.length / sample_rate


@dataclass
class IdleStatistics:
    """Statistics about idle regions in a trace.



    Attributes:
        total_samples: Total number of samples in trace.
        idle_samples: Total number of idle samples.
        active_samples: Total number of active samples.
        idle_regions: List of idle regions.
        dominant_pattern: Most common idle pattern.
    """

    total_samples: int
    idle_samples: int
    active_samples: int
    idle_regions: list[IdleRegion]
    dominant_pattern: str

    @property
    def idle_fraction(self) -> float:
        """Fraction of trace that is idle.

        Returns:
            Idle fraction (0.0 to 1.0).
        """
        if self.total_samples == 0:
            return 0.0
        return self.idle_samples / self.total_samples

    @property
    def active_fraction(self) -> float:
        """Fraction of trace that is active.

        Returns:
            Active fraction (0.0 to 1.0).
        """
        return 1.0 - self.idle_fraction


def detect_idle_regions(
    trace: DigitalTrace,
    pattern: str = "auto",
    min_duration: int = 100,
) -> list[IdleRegion]:
    """Detect idle regions in a digital trace.



    Identifies regions where the signal is idle (constant pattern) for
    a minimum duration. Supports auto-detection and explicit patterns.

    Args:
        trace: Digital trace to analyze.
        pattern: Idle pattern to detect ("auto", "zeros", "ones", or byte value).
        min_duration: Minimum duration in samples to consider as idle.

    Returns:
        List of detected idle regions.

    Example:
        >>> regions = detect_idle_regions(trace, pattern='zeros', min_duration=100)
        >>> for region in regions:
        ...     print(f"Idle from {region.start} to {region.end}")
    """
    data = trace.data

    if len(data) < min_duration:
        return []

    idle_regions: list[IdleRegion] = []

    if pattern == "auto":
        # Auto-detect pattern from start/end of trace
        pattern = _auto_detect_pattern(data)
        logger.debug("Auto-detected idle pattern: %s", pattern)

    # Detect idle runs
    if pattern == "zeros":
        idle_mask = ~data  # Invert: True where data is False (zero)
    elif pattern == "ones":
        idle_mask = data  # True where data is True (one)
    else:
        # For specific byte values, would need multi-bit comparison
        # For now, default to zeros
        logger.warning("Pattern '%s' not fully supported, using zeros", pattern)
        idle_mask = ~data

    # Find runs of idle samples
    # Pad mask to detect transitions at boundaries
    padded = np.concatenate(([False], idle_mask, [False]))
    transitions = np.diff(padded.astype(np.int8))

    # Rising edges (start of idle region)
    starts = np.where(transitions == 1)[0]
    # Falling edges (end of idle region)
    ends = np.where(transitions == -1)[0]

    # Filter by minimum duration
    for start, end in zip(starts, ends, strict=False):
        duration = end - start
        if duration >= min_duration:
            idle_regions.append(
                IdleRegion(
                    start=int(start),
                    end=int(end),
                    pattern=pattern,
                    duration_samples=int(duration),
                )
            )

    logger.info(
        "Detected %d idle regions (pattern: %s, min_duration: %d)",
        len(idle_regions),
        pattern,
        min_duration,
    )

    return idle_regions


def _auto_detect_pattern(data: NDArray[np.bool_]) -> str:
    """Auto-detect idle pattern from trace data.

    Looks at the start and end of the trace to determine the
    most likely idle pattern.

    Args:
        data: Boolean trace data.

    Returns:
        Detected pattern ("zeros", "ones", or "unknown").
    """
    if len(data) == 0:
        return "zeros"

    # Check first and last 100 samples (or 10% of trace, whichever is smaller)
    check_len = min(100, len(data) // 10, len(data))

    if check_len == 0:
        return "zeros"

    start_samples = data[:check_len]
    end_samples = data[-check_len:]

    # Count zeros in start/end regions
    start_zeros = np.sum(~start_samples)
    end_zeros = np.sum(~end_samples)

    # If majority are zeros, pattern is zeros
    if start_zeros > check_len // 2 or end_zeros > check_len // 2:
        return "zeros"

    # If majority are ones, pattern is ones
    if start_zeros < check_len // 4 and end_zeros < check_len // 4:
        return "ones"

    # Default to zeros
    return "zeros"


def trim_idle(
    trace: DigitalTrace,
    trim_start: bool = True,
    trim_end: bool = True,
    pattern: str = "auto",
    min_duration: int = 100,
) -> DigitalTrace:
    """Trim idle regions from trace.



    Removes idle regions from the start and/or end of a trace.

    Args:
        trace: Digital trace to trim.
        trim_start: Remove idle from start of trace.
        trim_end: Remove idle from end of trace.
        pattern: Idle pattern to detect ("auto", "zeros", "ones").
        min_duration: Minimum idle duration to trim.

    Returns:
        New DigitalTrace with idle regions removed.

    Example:
        >>> trimmed = trim_idle(trace, trim_start=True, trim_end=True)
        >>> print(f"Removed {len(trace.data) - len(trimmed.data)} idle samples")
    """
    if len(trace.data) == 0:
        return trace

    # Detect idle regions
    idle_regions = detect_idle_regions(trace, pattern=pattern, min_duration=min_duration)

    if not idle_regions:
        return trace

    # Find start and end trim points
    start_idx = 0
    end_idx = len(trace.data)

    if trim_start and idle_regions:
        # Check if first region starts at beginning
        first_region = idle_regions[0]
        if first_region.start == 0:
            start_idx = first_region.end
            logger.info("Trimming %d idle samples from start", first_region.length)

    if trim_end and idle_regions:
        # Check if last region ends at end
        last_region = idle_regions[-1]
        if last_region.end == len(trace.data):
            end_idx = last_region.start
            logger.info("Trimming %d idle samples from end", last_region.length)

    # Create trimmed trace
    if start_idx > 0 or end_idx < len(trace.data):
        trimmed_data = trace.data[start_idx:end_idx]

        # Preserve metadata
        new_metadata = TraceMetadata(
            sample_rate=trace.metadata.sample_rate,
            vertical_scale=trace.metadata.vertical_scale,
            vertical_offset=trace.metadata.vertical_offset,
            acquisition_time=trace.metadata.acquisition_time,
            trigger_info=trace.metadata.trigger_info,
            source_file=trace.metadata.source_file,
            channel_name=trace.metadata.channel_name,
        )

        return DigitalTrace(data=trimmed_data, metadata=new_metadata, edges=None)

    return trace


def get_idle_statistics(
    trace: DigitalTrace,
    pattern: str = "auto",
    min_duration: int = 100,
) -> IdleStatistics:
    """Get statistics about idle regions in trace.



    Computes comprehensive statistics about idle vs. active samples.

    Args:
        trace: Digital trace to analyze.
        pattern: Idle pattern to detect ("auto", "zeros", "ones").
        min_duration: Minimum idle duration to count.

    Returns:
        IdleStatistics with analysis results.

    Example:
        >>> stats = get_idle_statistics(trace)
        >>> print(f"Idle fraction: {stats.idle_fraction:.1%}")
        >>> print(f"Found {len(stats.idle_regions)} idle regions")
    """
    idle_regions = detect_idle_regions(trace, pattern=pattern, min_duration=min_duration)

    total_samples = len(trace.data)
    idle_samples = sum(region.length for region in idle_regions)
    active_samples = total_samples - idle_samples

    # Determine dominant pattern
    if idle_regions:
        # Count pattern occurrences
        pattern_counts: dict[str, int] = {}
        for region in idle_regions:
            pattern_counts[region.pattern] = pattern_counts.get(region.pattern, 0) + region.length

        dominant_pattern = max(pattern_counts, key=pattern_counts.get)  # type: ignore[arg-type]
    else:
        dominant_pattern = "none"

    return IdleStatistics(
        total_samples=total_samples,
        idle_samples=idle_samples,
        active_samples=active_samples,
        idle_regions=idle_regions,
        dominant_pattern=dominant_pattern,
    )


# Type alias for backward compatibility
IdleStats = IdleStatistics
"""Type alias for IdleStatistics."""

__all__ = [
    "IdleRegion",
    "IdleStatistics",
    "IdleStats",
    "detect_idle_regions",
    "get_idle_statistics",
    "trim_idle",
]
