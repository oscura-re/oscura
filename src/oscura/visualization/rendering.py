"""Rendering optimization for large datasets and streaming updates.

This module provides level-of-detail rendering, progressive rendering,
and memory-efficient plot updates for high-performance visualization.


Example:
    >>> from oscura.visualization.rendering import render_with_lod
    >>> time_lod, data_lod = render_with_lod(time, data, screen_width=1920)

References:
    - Level-of-detail (LOD) rendering techniques
    - Min-max envelope for waveform rendering
    - Progressive rendering algorithms
    - Streaming data visualization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def render_with_lod(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    *,
    screen_width: int = 1920,
    samples_per_pixel: float = 2.0,
    max_points: int = 100_000,
    method: Literal["minmax", "lttb", "uniform"] = "minmax",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Render signal with level-of-detail decimation./VIS-019.

    Reduces number of points while preserving visual appearance using
    intelligent downsampling. Target: <100k points at any zoom level.

    Args:
        time: Time array.
        data: Signal data array.
        screen_width: Screen width in pixels.
        samples_per_pixel: Target samples per pixel (2.0 recommended).
        max_points: Maximum points to render (default: 100k).
        method: Decimation method ("minmax", "lttb", "uniform").

    Returns:
        Tuple of (decimated_time, decimated_data).

    Raises:
        ValueError: If arrays are invalid or method unknown.

    Example:
        >>> # 1M sample signal decimated for 1920px display
        >>> time_lod, data_lod = render_with_lod(time, data, screen_width=1920)
        >>> print(len(data_lod))  # ~3840 samples (2 per pixel)

    References:
        VIS-017: Performance - LOD Rendering
        VIS-019: Memory-Efficient Plot Rendering
    """
    if len(time) == 0 or len(data) == 0:
        raise ValueError("Time or data array is empty")

    if len(time) != len(data):
        raise ValueError(f"Time and data length mismatch: {len(time)} vs {len(data)}")

    # Calculate target point count
    target_points = min(
        int(screen_width * samples_per_pixel),
        max_points,
    )

    # Skip decimation if already below target
    if len(data) <= target_points:
        return (time, data)

    # Apply decimation
    if method == "uniform":
        return _decimate_uniform(time, data, target_points)
    elif method == "minmax":
        return _decimate_minmax_envelope(time, data, target_points)
    elif method == "lttb":
        return _decimate_lttb(time, data, target_points)
    else:
        raise ValueError(f"Unknown decimation method: {method}")


def _decimate_uniform(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    target_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Uniform stride decimation (simple but loses peaks).

    Args:
        time: Time array.
        data: Signal data array.
        target_points: Target number of points after decimation.

    Returns:
        Tuple of (decimated_time, decimated_data).
    """
    stride = len(data) // target_points
    stride = max(stride, 1)

    indices = np.arange(0, len(data), stride)[:target_points]
    return (time[indices], data[indices])


def _decimate_minmax_envelope(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    target_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Min-max envelope decimation - preserves peaks and valleys.

    This method ensures all signal extrema are preserved in the decimated view.

    Args:
        time: Time array.
        data: Signal data array.
        target_points: Target number of points after decimation.

    Returns:
        Tuple of (decimated_time, decimated_data).
    """
    # Calculate bucket size (each bucket contributes 2 points: min and max)
    bucket_size = len(data) // (target_points // 2)

    if bucket_size < 1:
        return (time, data)

    decimated_time = []
    decimated_data = []

    for i in range(0, len(data), bucket_size):
        bucket_data = data[i : i + bucket_size]
        bucket_time = time[i : i + bucket_size]

        if len(bucket_data) == 0:
            continue

        # Find min and max in bucket
        min_idx = np.argmin(bucket_data)
        max_idx = np.argmax(bucket_data)

        # Add in chronological order
        if min_idx < max_idx:
            decimated_time.extend([bucket_time[min_idx], bucket_time[max_idx]])
            decimated_data.extend([bucket_data[min_idx], bucket_data[max_idx]])
        else:
            decimated_time.extend([bucket_time[max_idx], bucket_time[min_idx]])
            decimated_data.extend([bucket_data[max_idx], bucket_data[min_idx]])

    return (np.array(decimated_time), np.array(decimated_data))


def _decimate_lttb(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    target_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Largest Triangle Three Buckets decimation.

    Preserves visual shape by maximizing triangle areas.

    Args:
        time: Time array.
        data: Signal data array.
        target_points: Target number of points after decimation.

    Returns:
        Tuple of (decimated_time, decimated_data).
    """
    if len(data) <= target_points:
        return (time, data)

    # Always include first and last points
    sampled_time = [time[0]]
    sampled_data = [data[0]]

    bucket_size = (len(data) - 2) / (target_points - 2)

    prev_idx = 0

    for i in range(target_points - 2):
        # Average point of next bucket
        avg_range_start = int((i + 1) * bucket_size) + 1
        avg_range_end = int((i + 2) * bucket_size) + 1
        avg_range_end = min(avg_range_end, len(data))

        if avg_range_start < avg_range_end:
            avg_time = np.mean(time[avg_range_start:avg_range_end])
            avg_data = np.mean(data[avg_range_start:avg_range_end])
        else:
            avg_time = time[-1]
            avg_data = data[-1]

        # Current bucket range
        range_start = int(i * bucket_size) + 1
        range_end = int((i + 1) * bucket_size) + 1
        range_end = min(range_end, len(data) - 1)

        # Find point in bucket that forms largest triangle
        max_area = -1.0
        max_idx = range_start

        for idx in range(range_start, range_end):
            # Calculate triangle area
            area = abs(
                (time[prev_idx] - avg_time) * (data[idx] - data[prev_idx])
                - (time[prev_idx] - time[idx]) * (avg_data - data[prev_idx])
            )

            if area > max_area:
                max_area = area
                max_idx = idx

        sampled_time.append(time[max_idx])
        sampled_data.append(data[max_idx])
        prev_idx = max_idx

    # Always include last point
    sampled_time.append(time[-1])
    sampled_data.append(data[-1])

    return (np.array(sampled_time), np.array(sampled_data))


def progressive_render(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    *,
    viewport: tuple[float, float] | None = None,
    priority: Literal["viewport", "full"] = "viewport",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Progressive rendering - render visible viewport first.

    Args:
        time: Time array.
        data: Signal data array.
        viewport: Visible viewport (t_min, t_max). None = full range.
        priority: Rendering priority ("viewport" = visible first, "full" = all data).

    Returns:
        Tuple of (time, data) for priority rendering.

    Example:
        >>> # Render only visible portion for fast initial display
        >>> time_vis, data_vis = progressive_render(
        ...     time, data, viewport=(0, 0.001), priority="viewport"
        ... )

    References:
        VIS-019: Memory-Efficient Plot Rendering (progressive rendering)
    """
    if viewport is None or priority == "full":
        return (time, data)

    t_min, t_max = viewport

    # Find indices within viewport
    mask = (time >= t_min) & (time <= t_max)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        # Viewport is outside data range
        return (time, data)

    # Return viewport data first
    viewport_time = time[indices]
    viewport_data = data[indices]

    return (viewport_time, viewport_data)


def estimate_memory_usage(
    n_samples: int,
    n_channels: int = 1,
    dtype: type = np.float64,
) -> float:
    """Estimate memory usage for plot rendering.

    Args:
        n_samples: Number of samples per channel.
        n_channels: Number of channels.
        dtype: Data type for arrays.

    Returns:
        Estimated memory usage in MB.

    Example:
        >>> mem_mb = estimate_memory_usage(1_000_000, n_channels=4)
        >>> print(f"Memory: {mem_mb:.1f} MB")

    References:
        VIS-019: Memory-Efficient Plot Rendering
    """
    # Bytes per sample
    if dtype == np.float64:
        bytes_per_sample = 8
    elif dtype == np.float32 or dtype == np.int32:
        bytes_per_sample = 4
    elif dtype == np.int16:
        bytes_per_sample = 2
    else:
        bytes_per_sample = 8  # Default

    # Total memory: time + data arrays per channel
    # Time array: n_samples * bytes_per_sample
    # Data arrays: n_channels * n_samples * bytes_per_sample
    total_bytes = (1 + n_channels) * n_samples * bytes_per_sample

    # Convert to MB
    return total_bytes / (1024 * 1024)


def downsample_for_memory(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    *,
    target_memory_mb: float = 50.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Downsample signal to meet memory target.

    Args:
        time: Time array.
        data: Signal data array.
        target_memory_mb: Target memory usage in MB.

    Returns:
        Tuple of (decimated_time, decimated_data).

    Example:
        >>> # Reduce 100MB dataset to 50MB
        >>> time_ds, data_ds = downsample_for_memory(time, data, target_memory_mb=50.0)

    References:
        VIS-019: Memory-Efficient Plot Rendering (memory target <50MB per subplot)
    """
    current_memory = estimate_memory_usage(len(data), n_channels=1)

    if current_memory <= target_memory_mb:
        # Already within target
        return (time, data)

    # Calculate required decimation factor
    decimation_factor = current_memory / target_memory_mb
    target_samples = int(len(data) / decimation_factor)

    # Use min-max to preserve features
    return _decimate_minmax_envelope(time, data, target_samples)


class StreamingRenderer:
    """Streaming plot renderer for real-time data updates.

    Handles incremental data updates without full redraws for performance.

    Example:
        >>> renderer = StreamingRenderer(max_samples=10000)
        >>> renderer.append(new_time, new_data)
        >>> time, data = renderer.get_render_data()

    References:
        VIS-018: Streaming Plot Updates
    """

    def __init__(
        self,
        *,
        max_samples: int = 10_000,
        decimation_method: Literal["minmax", "lttb", "uniform"] = "minmax",
    ):
        """Initialize streaming renderer.

        Args:
            max_samples: Maximum samples to keep in buffer.
            decimation_method: Decimation method for buffer management.
        """
        self.max_samples = max_samples
        self.decimation_method = decimation_method

        self._time: list[float] = []
        self._data: list[float] = []

    def append(
        self,
        time: NDArray[np.float64],
        data: NDArray[np.float64],
    ) -> None:
        """Append new data to streaming buffer.

        Args:
            time: New time samples.
            data: New data samples.
        """
        self._time.extend(time.tolist())
        self._data.extend(data.tolist())

        # Decimate if buffer exceeds limit
        if len(self._data) > self.max_samples:
            self._decimate_buffer()

    def _decimate_buffer(self) -> None:
        """Decimate internal buffer to max_samples."""
        time_arr = np.array(self._time)
        data_arr = np.array(self._data)

        time_dec, data_dec = render_with_lod(
            time_arr,
            data_arr,
            max_points=self.max_samples,
            method=self.decimation_method,
        )

        self._time = time_dec.tolist()
        self._data = data_dec.tolist()

    def get_render_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get current data for rendering.

        Returns:
            Tuple of (time, data) arrays.
        """
        return (np.array(self._time), np.array(self._data))

    def clear(self) -> None:
        """Clear streaming buffer."""
        self._time.clear()
        self._data.clear()


__all__ = [
    "StreamingRenderer",
    "downsample_for_memory",
    "estimate_memory_usage",
    "progressive_render",
    "render_with_lod",
]
